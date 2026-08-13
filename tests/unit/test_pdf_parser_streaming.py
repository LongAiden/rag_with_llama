"""
Unit tests for the batched, pipelined `parse_pdf` in GeminiDoclingParser.

Two behaviours are under test, neither of them output formatting.

**Memory.** An earlier version converted every page batch up front and kept all
the resulting `DoclingDocument`s (and the rendered page images they carry) alive
until the end of the run, so peak memory scaled with *total* pages and a
504-page PDF OOM-killed the Celery worker. Batching bounded docling's working
memory but not its retained memory.
`test_batch_document_is_released_before_next_convert` is the regression guard: it
holds only weak references, so if `parse_pdf` starts retaining batch documents
again the weakref stays alive and the test fails.

**Pipelining.** `_process_page` is split into `_build_page` (submit VLM futures)
and `_finalize_page` (join them), and the join of batch N is deferred until after
`convert(N+1)` so VLM decode overlaps docling's CPU work. `TestPipelining` pins
that interleaving, the no-dropped-tail-batch invariant, and the lock discipline in
`_finalize_page` that keeps the join from deadlocking against `_record_vlm_call`.
"""
import gc
import threading
import time
import weakref
from concurrent.futures import Future, ThreadPoolExecutor
from unittest.mock import Mock, patch

import pytest

from app.ingestion.processors.gemini_docling_parser import GeminiDoclingParser


class FakeProv:
    def __init__(self, page_no):
        self.page_no = page_no


class FakeItem:
    """Stands in for a docling TextItem: only `.prov` is read before assembly."""
    def __init__(self, page_no):
        self.prov = [FakeProv(page_no)]


class FakeDocument:
    """Weakref-able stand-in for DoclingDocument."""
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self._items = [FakeItem(p) for p in range(start, end + 1)]

    def iterate_items(self):
        for item in self._items:
            yield item, 0


class FakeConversionResult:
    def __init__(self, document):
        self.document = document


class RecordingConverter:
    """Records each convert() call and whether the previous document is still alive.

    Deliberately keeps only a weakref to each document it hands out, so the only
    thing that can keep one alive is `parse_pdf` itself.
    """
    def __init__(self, events=None):
        self.page_ranges = []
        self.doc_refs = []
        self.alive_at_convert = []
        self._events = events

    def convert(self, pdf_path, page_range=None):
        gc.collect()
        self.alive_at_convert.append([ref() is not None for ref in self.doc_refs])

        if self._events is not None:
            self._events.append(("convert", page_range))
        self.page_ranges.append(page_range)
        doc = FakeDocument(*page_range)
        self.doc_refs.append(weakref.ref(doc))
        return FakeConversionResult(doc)


@pytest.fixture
def parser():
    return GeminiDoclingParser(
        api_key="test-key",
        page_batch_size=10,
        vlm_concurrency=1,
    )


@pytest.fixture
def converter():
    return RecordingConverter()


def _make_page_stubs(events=None):
    """Plain functions, not Mocks: a Mock's call_args_list would itself retain
    every `doc` it was called with, which would defeat the memory-release test."""

    def _fake_build_page(page_no, items, doc, executor=None):
        if events is not None:
            events.append(("build", page_no))
        # Mirrors the real return shape: (ordered, vlm_tasks), holding no
        # reference to `doc` — that is what makes the deferred join safe.
        return [(0, 0.0, 0.0, f"body p{page_no}")], []

    def _fake_finalize_page(page_no, ordered, vlm_tasks):
        if events is not None:
            events.append(("finalize", page_no))
        body = "\n\n".join(md for _, _, _, md in ordered)
        return f"[PAGE:{page_no}]\n\n{body}"

    return _fake_build_page, _fake_finalize_page


def _run(parser, converter, total_pages, output_path=None, events=None):
    """Drive parse_pdf against the fake converter, stubbing out page assembly."""
    build, finalize = _make_page_stubs(events)
    if events is not None:
        converter._events = events
    with patch.object(parser, "_count_pages", return_value=total_pages), \
         patch.object(parser, "_build_converter", return_value=converter), \
         patch.object(parser, "_build_page", new=build), \
         patch.object(parser, "_finalize_page", new=finalize):
        return parser.parse_pdf("/nonexistent/fake.pdf", output_path=output_path)


class TestBatching:
    def test_convert_is_called_once_per_batch_with_absolute_page_ranges(self, parser, converter):
        _run(parser, converter, total_pages=25)

        assert converter.page_ranges == [(1, 10), (11, 20), (21, 25)]

    def test_single_batch_document_shorter_than_batch_size(self, parser, converter):
        _run(parser, converter, total_pages=3)

        assert converter.page_ranges == [(1, 3)]

    def test_max_pages_caps_the_conversion(self, converter):
        parser = GeminiDoclingParser(api_key="k", page_batch_size=10, max_pages=12)
        _run(parser, converter, total_pages=500)

        assert converter.page_ranges == [(1, 10), (11, 12)]


class TestMemoryRelease:
    def test_batch_document_is_released_before_next_convert(self, parser, converter):
        """No batch document may survive into the next convert() call.

        This is the regression guard for the OOM: with the old `batch_docs`
        dict every document was still alive here. Pipelining moves the emit of
        batch N after the convert of N+1, so it is also the guard that the
        deferred `built` payload does not smuggle a `doc` reference across.
        """
        _run(parser, converter, total_pages=50)

        assert len(converter.page_ranges) == 5
        # First convert has nothing before it; every later one must see only dead refs.
        for call_index, alive in enumerate(converter.alive_at_convert):
            assert not any(alive), (
                f"convert() call {call_index} still holds "
                f"{sum(alive)} earlier batch document(s) in memory"
            )

    def test_no_document_survives_the_parse(self, parser, converter):
        _run(parser, converter, total_pages=30)
        gc.collect()

        assert [ref() for ref in converter.doc_refs] == [None, None, None]


class TestPageOrdering:
    def test_pages_are_assembled_in_document_order_across_batches(self, parser, converter):
        markdown = _run(parser, converter, total_pages=25)

        positions = [markdown.index(f"[PAGE:{n}]") for n in range(1, 26)]
        assert positions == sorted(positions)

    def test_page_numbers_stay_absolute_past_the_first_batch(self, parser, converter):
        """The bug this guards: pages renumbered per batch would emit 1-10 four times."""
        markdown = _run(parser, converter, total_pages=25)

        for page_no in range(1, 26):
            assert f"[PAGE:{page_no}]" in markdown
        assert markdown.count("[PAGE:1]") == 1

    def test_streams_to_output_file_as_it_goes(self, parser, converter, tmp_path):
        out = tmp_path / "parsed.md"
        _run(parser, converter, total_pages=25, output_path=str(out))

        written = out.read_text(encoding="utf-8")
        assert "[PAGE:1]" in written
        assert "[PAGE:25]" in written


class TestPipelining:
    def test_next_batch_is_built_before_the_previous_one_is_joined(self, parser, converter):
        """The whole point of the change: batch N's VLM futures must still be in
        flight while `convert(N+1)` runs, so the join lands after it."""
        events = []
        _run(parser, converter, total_pages=25, events=events)

        first_finalize = events.index(("finalize", 1))
        assert events.index(("convert", (11, 20))) < first_finalize
        assert events.index(("build", 11)) < first_finalize

    def test_every_page_is_emitted_exactly_once(self, parser, converter):
        """Guards the dropped-tail regression: the final pending batch has no
        following convert() to trigger its emit and needs its own flush."""
        events = []
        markdown = _run(parser, converter, total_pages=25, events=events)

        finalized = [page for kind, page in events if kind == "finalize"]
        assert finalized == list(range(1, 26))
        for page_no in range(1, 26):
            assert markdown.count(f"[PAGE:{page_no}]") == 1

    def test_final_partial_batch_is_flushed_to_the_output_file(self, parser, converter, tmp_path):
        out = tmp_path / "parsed.md"
        _run(parser, converter, total_pages=25, output_path=str(out))

        written = out.read_text(encoding="utf-8")
        for page_no in (21, 22, 23, 24, 25):
            assert f"[PAGE:{page_no}]" in written

    def test_build_and_finalize_are_paired_per_page(self, parser, converter):
        events = []
        _run(parser, converter, total_pages=25, events=events)

        built = [page for kind, page in events if kind == "build"]
        finalized = [page for kind, page in events if kind == "finalize"]
        assert built == finalized == list(range(1, 26))


class TestFinalizePageJoin:
    """`_finalize_page` is called with the real implementation here."""

    def test_does_not_hold_the_stats_lock_while_waiting_on_a_future(self, parser):
        """Deadlock guard.

        `_record_vlm_call` takes `_vlm_stats_lock` from the pool thread. If
        `_finalize_page` held that lock across `future.result()`, the main
        thread would be waiting on a future whose thread is waiting on the main
        thread.

        Everything here is deliberately hang-proof, because that regression
        deadlocks two different ways — the cross-thread wait above, and the
        non-reentrant re-acquire when the blocked accumulate finally runs. So:
        daemon threads (nothing to join at interpreter exit), a bare `Future`
        rather than a `ThreadPoolExecutor` (no atexit joiner), a timed
        `acquire`, and a timed wait on completion. A regression fails this test
        rather than wedging the suite.
        """
        future = Future()
        future.set_running_or_notify_cancel()
        acquired = []
        finished = threading.Event()

        def _vlm_body():
            """Stands in for the VLM pool thread calling _record_vlm_call."""
            time.sleep(0.1)  # let _finalize_page get into .result()
            got = parser._vlm_stats_lock.acquire(timeout=2.0)
            acquired.append(got)
            if got:
                parser._vlm_stats_lock.release()
            future.set_result("a described figure")

        def _join():
            parser._finalize_page(7, [], [(0, 0.0, 0.0, future, "figure", "")])
            finished.set()

        threading.Thread(target=_vlm_body, daemon=True).start()
        threading.Thread(target=_join, daemon=True).start()

        assert finished.wait(timeout=5.0), (
            "_finalize_page never returned — it is holding _vlm_stats_lock "
            "across future.result()"
        )
        assert acquired == [True], (
            "_finalize_page held _vlm_stats_lock across future.result() — "
            "that deadlocks against _record_vlm_call"
        )

    def test_blocked_seconds_accumulates_the_actual_wait(self, parser):
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(lambda: (time.sleep(0.15), "text")[1])
            parser._finalize_page(1, [], [(0, 0.0, 0.0, future, "figure", "")])

        assert parser._vlm_blocked_seconds >= 0.1

    def test_blocked_seconds_stays_near_zero_for_an_already_resolved_future(self, parser):
        """The pipelined case: by emit time the future is done, so the join is free."""
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(lambda: "text")
            future.result()
            parser._finalize_page(1, [], [(0, 0.0, 0.0, future, "figure", "")])

        assert parser._vlm_blocked_seconds < 0.05

    def test_blocked_seconds_is_recorded_even_when_the_call_fails(self, parser):
        def _boom():
            time.sleep(0.1)
            raise RuntimeError("vlm exploded")

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_boom)
            page_md = parser._finalize_page(1, [], [(0, 0.0, 0.0, future, "figure", "")])

        assert "[IMAGE]" in page_md
        assert parser._vlm_blocked_seconds >= 0.05

    def test_page_body_is_sorted_by_column_then_position(self, parser):
        # y is `_item_sort_key`'s -bbox.t, so higher on the page sorts lower.
        ordered = [
            (1, -20.0, 0.0, "right column"),
            (0, -10.0, 0.0, "left lower"),
            (0, -20.0, 0.0, "left upper"),
        ]
        page_md = parser._finalize_page(4, ordered, [])

        assert page_md.startswith("[PAGE:4]")
        assert page_md.index("left upper") < page_md.index("left lower") < page_md.index("right column")


class TestVlmStats:
    def test_counters_reset_between_documents(self, parser, converter):
        parser._record_vlm_call(1.5)
        parser._record_vlm_call(2.5, failed=True)
        parser._vlm_blocked_seconds = 9.0
        assert parser._vlm_calls == 2

        _run(parser, converter, total_pages=5)

        assert parser._vlm_calls == 0
        assert parser._vlm_seconds == 0.0
        assert parser._vlm_failures == 0
        assert parser._vlm_blocked_seconds == 0.0

    def test_record_vlm_call_accumulates(self, parser):
        assert parser._record_vlm_call(1.0) == 1
        assert parser._record_vlm_call(2.0, failed=True) == 2

        assert parser._vlm_seconds == pytest.approx(3.0)
        assert parser._vlm_failures == 1


class TestMemoryRelease:
    """The batch loop must hand freed memory back to the OS, not just to glibc.

    `del doc` already dropped the last reference — `TestBatching` proves that —
    yet a 702-page parse still climbed +357, +321, +239, +314MB across its first
    five batches and was OOM-killed at batch 8 of 15. The blocks were sitting in
    glibc's per-thread arenas, so releasing the *objects* was never sufficient.
    """

    def test_freed_memory_is_released_once_per_batch(self, parser, converter):
        with patch(
            "app.ingestion.processors.gemini_docling_parser._release_freed_memory"
        ) as release:
            _run(parser, converter, total_pages=25)

        assert release.call_count == 3  # one per batch: (1,10), (11,20), (21,25)

    def test_release_runs_before_the_next_convert(self, parser, converter):
        """Ordering matters: trimming after the next convert() has already
        allocated leaves both batches' memory resident at the same moment,
        which is the peak that kills the process."""
        events = []
        converter._events = events

        def _record_release():
            events.append(("release", None))

        with patch(
            "app.ingestion.processors.gemini_docling_parser._release_freed_memory",
            new=_record_release,
        ):
            _run(parser, converter, total_pages=25, events=events)

        kinds = [kind for kind, _ in events if kind in ("convert", "release")]
        assert kinds == ["convert", "release", "convert", "release", "convert", "release"]

    def test_malloc_trim_is_called_with_zero(self):
        from app.ingestion.processors import gemini_docling_parser as mod

        trim = Mock()
        with patch.object(mod, "_resolve_malloc_trim", return_value=trim):
            mod._release_freed_memory()

        trim.assert_called_once_with(0)

    def test_release_is_a_no_op_without_glibc(self):
        """macOS and musl have no libc.so.6. A parse must not die for it."""
        from app.ingestion.processors import gemini_docling_parser as mod

        with patch.object(mod, "_MALLOC_TRIM", mod._UNRESOLVED), \
             patch("ctypes.CDLL", side_effect=OSError("no libc.so.6")):
            mod._release_freed_memory()  # must not raise
            assert mod._resolve_malloc_trim() is None

    def test_malloc_trim_failure_does_not_propagate(self):
        from app.ingestion.processors import gemini_docling_parser as mod

        trim = Mock(side_effect=OSError("boom"))
        with patch.object(mod, "_resolve_malloc_trim", return_value=trim):
            mod._release_freed_memory()  # must not raise

    def test_resolution_is_cached(self):
        """Called once per batch; CDLL is not free and the failure case must
        not be retried on every batch of a 15-batch document."""
        from app.ingestion.processors import gemini_docling_parser as mod

        with patch.object(mod, "_MALLOC_TRIM", mod._UNRESOLVED), \
             patch("ctypes.CDLL", side_effect=OSError("no libc")) as cdll:
            mod._resolve_malloc_trim()
            mod._resolve_malloc_trim()
            mod._resolve_malloc_trim()

        assert cdll.call_count == 1


class TestPeakRssSampler:
    """peak_rss was sampled only between batches, so it never saw the peak.

    The 702-page kill landed 17s into a convert() whose predecessor had logged
    2759MB against a 3584MB limit — every recorded peak in the parse history is
    a snapshot taken at a local minimum.
    """

    def test_reports_a_peak_no_between_batch_sample_would_see(self):
        from app.ingestion.processors.gemini_docling_parser import _PeakRssSampler

        readings = iter([100.0] + [900.0] * 50 + [100.0] * 50)

        def _fake_rss():
            try:
                return next(readings)
            except StopIteration:
                return 100.0

        with patch(
            "app.ingestion.processors.gemini_docling_parser._rss_mb", new=_fake_rss
        ):
            with _PeakRssSampler(interval=0.001) as sampler:
                time.sleep(0.1)

        assert sampler.peak == 900.0

    def test_thread_stops_on_exit(self):
        from app.ingestion.processors.gemini_docling_parser import _PeakRssSampler

        with _PeakRssSampler(interval=0.001) as sampler:
            thread = sampler._thread
        assert thread is not None
        assert not thread.is_alive()

    def test_summary_reports_both_peak_and_post_convert_rss(self, parser, converter, caplog):
        """The post-convert number is kept because every historical measurement
        in docs/ is that number; mixing the two would break the series."""
        with caplog.at_level("INFO"):
            _run(parser, converter, total_pages=25)

        summary = [r.message for r in caplog.records if "parse_pdf summary:" in r.message]
        assert len(summary) == 1
        assert "peak_rss=" in summary[0]
        assert "post_convert_rss=" in summary[0]
