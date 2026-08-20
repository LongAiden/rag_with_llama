# Logfire `configure()` crash blocks every ingestion stage (2026-08-20)

> `logfire.configure()` raising out of `AppConfig.__init__` killed the chunk
> stage for `mml-book.pdf`, which then surfaced as "stuck at `parsed`" and was
> misread as an embedding-step error.

## Problem

`mml-book.pdf` sat at `stage = parsed` with no `last_error`. The ingestion
worker was idle and `recover_and_dispatch` reported `dispatched: 0`, so nothing
was retrying it. The user's read was "error in the embedding step".

The actual traceback was in the **chunk** stage, not embed:

```
File "/app/src/app/worker/ingestion_tasks.py", line 426, in chunk_document_task
    return _run(_chunk_document(doc_id))
File "/app/src/app/worker/ingestion_tasks.py", line 140, in _run_stage
    config = _get_config()              # AppConfig()
File "/app/src/app/config/app_config.py", line 193, in __init__
    self._configure_logfire()
File "/app/src/app/config/app_config.py", line 215, in _configure_logfire
    logfire.configure()                 # no-token branch
logfire.exceptions.LogfireConfigError: Hey, looks like you don't have Pydantic Logfire configured yet.
```

### Why every stage is implicated

`parse`, `chunk`, and `embed` all enter through `_run_stage`
(`ingestion_tasks.py:140`), whose first line is `config = _get_config()` →
`AppConfig()`. `_configure_logfire()` ran in `__init__` with no guard, so a
configure failure aborted the task **before** the stage's own work ran. The
document therefore never advanced past whatever stage it was already in, and
because the raise happened in config init (not in the stage body) the failure
path in `_run_stage` never recorded a `last_error` either — the row just froze.

The chunk task is the one observed because the doc was already `parsed`; a
`registered` doc would have frozen at `parsing` with the same root cause.

### Why configure raised

Two causes, one trigger:

1. **Trigger** — the running `app` container held a stale `v1` Logfire token
   baked in at start (22:31), while `.env` had been edited to a `v2` token at
   23:35. For the **workers**, `LOGFIRE_WRITE_TOKEN` was not in
   `x-common-env`, so they never received it via the compose `environment:`
   block and fell through to the no-token `else` branch, which calls
   `logfire.configure()` with no args — and that raises `LogfireConfigError`
   when neither `~/.logfire` local auth nor a token is present.
2. **Root cause** — observability is optional, but a configure failure was
   allowed to crash service init. A missing/invalid token must degrade, not
   kill the worker.

## Fix

### 1. `src/app/config/app_config.py` — never let logfire crash init

`_configure_logfire` now wraps both branches in try/except. With a token it
logs at `error`; without, at `warning`. Either way `AppConfig()` returns and
the task proceeds without observability.

```python
def _configure_logfire(self):
    token = self.settings.logfire_write_token
    try:
        if token:
            logfire.configure(token=token)
        else:
            logfire.configure()
    except Exception as exc:
        level = logger.error if token else logger.warning
        level("Logfire configure failed (token=%s); observability disabled: %s",
              "set" if token else "unset", exc)
```

Regression tests in `tests/unit/test_app_config.py::TestConfigureLogfire` cover
both the no-token-raises and bad-token-raises paths.

### 2. `docker-compose.yml` — one source of truth for the token

`LOGFIRE_WRITE_TOKEN` / `LOGFIRE_READ_TOKEN` moved into `x-common-env` so the
workers and beat receive them explicitly via the compose `environment:` block,
instead of relying on the `/app/.env` mount. Removed the now-redundant explicit
entries from the `app` and `test` services.

Before this, `app` got the token from its env block (captured at start = stale
`v1`) while workers got it from the mounted `.env` (= `v2`) — pydantic-settings
prioritises process env over the `.env` file, so the two services diverged
after any `.env` edit that was not followed by a restart. Centralising the var
makes a restart the only thing needed to propagate a token change.

## Verification

- `tests/unit/test_app_config.py` — 11/11 pass (run via the `test` profile
  with `./src` mounted, since `Dockerfile.test` bakes `src/` into the image).
- After `docker compose up -d --force-recreate app celery_worker_upload
  celery_worker_ingestion celery_beat`:
  - both workers report `LOGFIRE_WRITE_TOKEN = pylf_v2_us_134b6ed…` from the
    compose env,
  - `AppConfig()` in a worker prints the Logfire project URL (configure
    succeeded),
  - `grep -c LogfireConfigError` across all logs → 0.
- `mml-book.pdf` was deleted via `DELETE /documents/{id}`; re-uploading it now
  runs parse → chunk → embed without the config crash.

## Followups (not done)

- The `test` service no longer receives `LOGFIRE_*` (it does not use
  `x-common-env`). Tests run without logfire, which is fine — the try/except
  handles it and avoids network calls during the suite — but if a test ever
  needs to assert logfire wiring, it will have to set the var explicitly.
- `Dockerfile.test` bakes `src/` in, so the mounted-`src` run was needed to
  exercise the new code. The image should be rebuilt before relying on the
  `test` profile for this regression.
