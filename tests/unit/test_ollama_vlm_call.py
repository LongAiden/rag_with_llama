"""
Unit tests for OllamaPDFParser._call_vlm.

Two things are under test:

- `keep_alive` is sent. Without it Ollama unloads the model 5 minutes after its
  last use, so a gap between figure-bearing pages makes the next call pay a full
  cold model load. That is what the observed 27-33s per call on a 0.8B model
  actually was, and it accounted for roughly the whole hour a 500-page parse took.
- Failures are attributable. The `[IMAGE]` fallback is silent by design, so if
  the configured model does not accept images every figure is dropped with no
  visible error. The warning must carry the HTTP status and response body.
"""
from unittest.mock import MagicMock, patch

import httpx
import pytest
from PIL import Image as PILImage

from app.ingestion.processors.ollama_pdf_parser import OllamaPDFParser


@pytest.fixture
def parser():
    return OllamaPDFParser(ollama_base_url="http://ollama:11434/", vlm_model="qwen3.5:0.8b")


@pytest.fixture
def image():
    return PILImage.new("RGB", (200, 200), color="white")


def _ok_response(text="some markdown", thinking=None):
    response = MagicMock()
    response.raise_for_status.return_value = None
    payload = {"response": text}
    if thinking is not None:
        payload["thinking"] = thinking
    response.json.return_value = payload
    return response


class TestKeepAlive:
    def test_keep_alive_is_sent(self, parser, image):
        with patch("httpx.post", return_value=_ok_response()) as post:
            parser._call_vlm(image, "describe this")

        assert post.call_args.kwargs["json"]["keep_alive"] == "30m"

    def test_keep_alive_is_configurable(self, image):
        parser = OllamaPDFParser(keep_alive="2h")
        with patch("httpx.post", return_value=_ok_response()) as post:
            parser._call_vlm(image, "describe this")

        assert post.call_args.kwargs["json"]["keep_alive"] == "2h"

    def test_payload_still_carries_model_prompt_and_image(self, parser, image):
        with patch("httpx.post", return_value=_ok_response()) as post:
            parser._call_vlm(image, "describe this")

        payload = post.call_args.kwargs["json"]
        assert payload["model"] == "qwen3.5:0.8b"
        assert payload["prompt"] == "describe this"
        assert payload["stream"] is False
        assert len(payload["images"]) == 1

    def test_base_url_trailing_slash_is_stripped(self, parser, image):
        with patch("httpx.post", return_value=_ok_response()) as post:
            parser._call_vlm(image, "describe this")

        assert post.call_args.args[0] == "http://ollama:11434/api/generate"


class TestFailureIsAttributable:
    def test_http_error_logs_status_and_body(self, parser, image, caplog):
        response = httpx.Response(
            400,
            text="this model is not multimodal",
            request=httpx.Request("POST", "http://ollama:11434/api/generate"),
        )
        error = httpx.HTTPStatusError("400", request=response.request, response=response)

        with patch("httpx.post", side_effect=error):
            with caplog.at_level("WARNING"):
                result = parser._call_vlm(image, "describe this")

        assert result == "[IMAGE]"
        assert "status=400" in caplog.text
        assert "not multimodal" in caplog.text

    def test_failure_is_counted(self, parser, image):
        with patch("httpx.post", side_effect=httpx.ConnectError("refused")):
            parser._call_vlm(image, "describe this")

        assert parser._vlm_failures == 1
        assert parser._vlm_calls == 1

    def test_success_is_counted_without_a_failure(self, parser, image):
        with patch("httpx.post", return_value=_ok_response()):
            parser._call_vlm(image, "describe this")

        assert parser._vlm_calls == 1
        assert parser._vlm_failures == 0


class TestThinkingIsDisabled:
    """`qwen3.5:0.8b` is a reasoning model and Ollama defaults thinking ON.

    Measured against Ollama 0.32.5 on a real BERT table crop: 87.35s with
    thinking versus 2.27s with `think: false`, and the reasoning is discarded
    unread because `_call_vlm` only reads the `response` field. Appending
    `/no_think` to the prompt does nothing — that is a Qwen3 convention and this
    is qwen3.5 — so the request field is the only thing that works.
    """

    def test_think_is_false_by_default(self, parser, image):
        with patch("httpx.post", return_value=_ok_response()) as post:
            parser._call_vlm(image, "describe this")

        assert post.call_args.kwargs["json"]["think"] is False

    def test_think_can_be_re_enabled(self, image):
        parser = OllamaPDFParser(think=True)
        with patch("httpx.post", return_value=_ok_response()) as post:
            parser._call_vlm(image, "describe this")

        assert post.call_args.kwargs["json"]["think"] is True

    def test_thinking_field_is_never_merged_into_the_markdown(self, parser, image):
        """Ollama returns reasoning in a separate `thinking` key; only `response`
        is the answer."""
        with patch("httpx.post", return_value=_ok_response("real answer", thinking="x" * 5000)):
            result = parser._call_vlm(image, "describe this")

        assert result.strip() == "real answer"
        assert "xxxx" not in result


class TestBlankResponseIsAFailure:
    """With thinking on, every table call returned a 200 with an empty `response`
    after ~3600 reasoning tokens — 3 of 3, synthetic and real. `_call_vlm`
    returned "" and `_process_page` wrapped it into an empty <table>, so the
    table vanished from the output with no warning and no failure count."""

    @pytest.mark.parametrize("body", ["", "   ", "\n\n", "```\n```"])
    def test_blank_response_falls_back_to_image(self, parser, image, body):
        with patch("httpx.post", return_value=_ok_response(body)):
            result = parser._call_vlm(image, "describe this")

        assert result == "[IMAGE]"

    def test_blank_response_is_counted_as_a_failure(self, parser, image):
        with patch("httpx.post", return_value=_ok_response("")):
            parser._call_vlm(image, "describe this")

        assert parser._vlm_failures == 1
        assert parser._vlm_calls == 1

    def test_blank_response_warns_and_names_the_model(self, parser, image, caplog):
        with patch("httpx.post", return_value=_ok_response("")):
            with caplog.at_level("WARNING"):
                parser._call_vlm(image, "describe this")

        assert "qwen3.5:0.8b" in caplog.text

    def test_non_blank_response_is_not_a_failure(self, parser, image):
        with patch("httpx.post", return_value=_ok_response("| a | b |")):
            result = parser._call_vlm(image, "describe this")

        assert result.strip() != "[IMAGE]"
        assert parser._vlm_failures == 0


class TestNoDuplicateParsePdf:
    def test_parse_pdf_is_inherited_not_reimplemented(self):
        """The two backends diverged once already; the streaming rewrite lives in
        the base class so a fix cannot be applied to only one of them."""
        assert "parse_pdf" not in OllamaPDFParser.__dict__
