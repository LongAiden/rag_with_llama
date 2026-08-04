# `/stats` page fails with `Error: 'docs'`

**Date**: 2026-08-04
**Status**: **fixed 2026-08-05.** Both occurrences corrected in
[admin_routes.py:55,59](src/app/api/routes/admin_routes.py#L55). Regression tests in
`tests/unit/test_admin_stats.py` — verified to fail against the pre-fix code, including
a static check that ties the keys the route reads to the aliases the SQL produces, so
renaming one side without the other is caught.

## Symptom

`GET /stats` renders "Failed to Load Statistics — Error: 'docs'" instead of the dashboard, whenever
at least one chunk table exists (the `if not table_names:` branch at
[admin_routes.py:34](src/app/api/routes/admin_routes.py#L34) is unaffected — the bug only fires once
there's real data to aggregate).

## Root cause

Column name mismatch between the SQL query and the code reading its result.

[table_repository.py:70-80](src/app/infra/db/table_repository.py#L70-L80) — `get_table_stats()`:
```sql
SELECT
    COUNT(DISTINCT document_id) as documents,
    COUNT(*) as chunks,
    COALESCE(SUM(LENGTH(text)), 0) as total_text_length,
    MIN(created_at) as earliest,
    MAX(created_at) as latest
FROM {safe_name}
```

[admin_routes.py:55,59](src/app/api/routes/admin_routes.py#L55) reads it back:
```python
total_docs += result['docs'] or 0          # ← no such column; SQL aliases it `documents`
...
print(f"  {table_name}: {result['docs']} docs, {result['chunks']} chunks")
```

`asyncpg.Record.__getitem__` raises `KeyError` for a column name that isn't in the row.
`str(KeyError('docs'))` is `"'docs'"`, which is exactly the text rendered by `stats_error.html`
after the route's `except Exception as e: ... render("stats_error.html", error_message=str(e))` at
[admin_routes.py:89-93](src/app/api/routes/admin_routes.py#L89-L93).

Every other field aggregated in the same loop (`chunks`, `total_text_length`, `earliest`, `latest`)
matches its SQL alias correctly — this is the only one that's wrong. It is a plain naming typo, not
a logic error: the query already computes the right value under the name `documents`.

## Fix

Two-line change in [admin_routes.py](src/app/api/routes/admin_routes.py), `result['docs']` →
`result['documents']`:

```python
# line 55
-                    total_docs += result['docs'] or 0
+                    total_docs += result['documents'] or 0

# line 59
-                    print(f"  {table_name}: {result['docs']} docs, {result['chunks']} chunks")
+                    print(f"  {table_name}: {result['documents']} docs, {result['chunks']} chunks")
```

No other file needs to change — `get_table_row_counts()`
([table_repository.py:82-90](src/app/infra/db/table_repository.py#L82-L90)), the only other caller of
a similarly-shaped query, already converts its result with `dict(row)` and is read elsewhere using the
correct `'documents'` key, so it isn't affected by this bug.

## Verification

1. Visit `/stats` with at least one populated chunk table — should render the dashboard instead of
   the error page.
2. Confirm the per-table debug line in server logs reads correctly:
   `  <table_name>: <N> docs, <M> chunks` with real numbers, not a crash before that print is reached.
3. Grep for other occurrences of the same typo in case it was copy-pasted elsewhere:
   `grep -rn "\['docs'\]" src/`
