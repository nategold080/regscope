# RegScope — Problem Report

> This document is the single source of truth for all outstanding issues.
> Work through every item. Mark each DONE when fixed. Do not skip any.
> After each fix, run `python3 -m pytest tests/ -x -q` to verify no regressions.

---

## Phase 1 — Original Audit (P1-P17, all DONE)

### CRITICAL — Functional Bugs

### P1. Substantiveness scoring ignores config weights — DONE
- **File:** `regscope/pipeline/classify.py`, `_compute_substantiveness()`
- **Fix applied:** Rewrote to use 0-100 subscores for each component (length, citations, specificity, complexity, legal, data, structure), then weighted sum using config values (`weight_length`, `weight_citations`, etc.). Additive bonuses for org affiliation and structural quality; penalty for form letters.

### P2. Dedup stage is not idempotent — DONE
- **File:** `regscope/pipeline/dedup.py`, `run_dedup()`
- **Fix applied:** Added cleanup at start: `UPDATE comments SET dedup_group_id = NULL, semantic_group_id = NULL` and `DELETE FROM dedup_groups` scoped to the docket, then `db.commit()`.

### P3. DOCX extraction is naive and will produce garbage — DONE
- **File:** `regscope/pipeline/extract.py`, `_extract_docx()`
- **Fix applied:** Replaced zipfile/regex approach with `python-docx` library. Extracts paragraphs and tables (cells joined with ` | `). Added `python-docx>=1.0` to `pyproject.toml`.

### P4. `list_all_dockets` connection leak on error — DONE
- **File:** `regscope/db.py`, `list_all_dockets()`
- **Fix applied:** Changed to `conn = None` / `try` / `finally: if conn: conn.close()` pattern so connections are always closed even on exceptions.

### HIGH — Code Quality Issues

### P5. Rate limiter is mislabeled and has thread-safety gaps — DONE
- **File:** `regscope/utils/rate_limit.py`
- **Fix applied:** Renamed class to `IntervalRateLimiter`, added `TokenBucketRateLimiter = IntervalRateLimiter` alias for backward compat. Wrapped `update_from_headers()` and `handle_429()` with `self._lock`.

### P6. Semantic dedup won't scale past ~50K unique comments — DONE
- **File:** `regscope/pipeline/dedup.py`, `_semantic_dedup()`
- **Fix applied:** Added guard that skips with warning when >50K unique comments, logging recommendation to use faiss for approximate nearest neighbors.

### P7. SQL column name injection in db.py — DONE
- **File:** `regscope/db.py`
- **Fix applied:** Added `VALID_COMMENT_COLUMNS` whitelist and `_validate_columns()` function. Called in both `insert_comment()` and `update_comment_field()`. Invalid column names raise `ValueError`.

### P8. Duplicated CLI stage-running code — DONE
- **File:** `regscope/cli.py`
- **Fix applied:** Extracted `_run_stages()` helper function used by both `analyze` and `process` commands. Removed duplicate `log_pipeline_run` imports.

### P9. HTML cleaning duplicated between classify and topics — DONE
- **Files:** `regscope/utils/text.py`, `regscope/pipeline/topics.py`, `regscope/pipeline/classify.py`
- **Fix applied:** Added `strip_html()` to `utils/text.py` (unescape entities, strip `<br>` tags, remove remaining tags, collapse whitespace). Updated both `topics.py` and `classify.py` to import and use it.

### P10. `_classify_org` fallback misclassifies capitalized names — DONE
- **File:** `regscope/pipeline/classify.py`, `_classify_org()`
- **Fix applied:** Changed fallback from always returning `"industry"` for capitalized names to requiring business indicator words (`inc`, `corp`, `llc`, etc.) or an acronym (2+ consecutive uppercase letters). Names without indicators now return `"unknown"`.

### MEDIUM — Test Coverage Gaps

### P11. Zero tests for rate limiting — DONE
- **File:** `tests/test_rate_limit.py` (created)
- **Fix applied:** 25 tests covering init defaults, wait interval enforcement, low-remaining throttling, `update_from_headers()`, `handle_429()` with Retry-After, backward compatibility alias, and thread safety.

### P12. Zero tests for pagination/cursor logic — DONE
- **File:** `tests/test_ingest.py` (created)
- **Fix applied:** 19 tests covering `_store_comment_header`, `_store_attachment` (format preference: PDF > DOCX > first), `_download_comment_headers` (single page, multi-page, cursor-based pagination), `_fetch_comment_details` (resume, attachments, error handling).

### P13. Topic tests are trivial — DONE
- **File:** `tests/test_topics.py` (expanded)
- **Fix applied:** Expanded from 2 to 26 tests covering BERTopic on small corpus (3 distinct clusters), topic keyword storage, topic propagation to dedup group members, parameter scaling for small/medium/large datasets, and `_clean_for_topics` HTML cleaning.

### P14. Zero tests for PDF extraction — DONE
- **File:** `tests/test_extract.py` (expanded)
- **Fix applied:** Expanded from 3 to 22 tests covering PDF text extraction (fitz), scanned PDF OCR fallback detection, DOCX extraction (python-docx paragraphs and tables), HTML extraction, and `build_full_text` logic.

### P15. Zero tests for report generation — DONE
- **File:** `tests/test_report.py` (created)
- **Fix applied:** 26 tests covering `_clean_excerpt`, `_readable_topic_label`, `_generate_llm_topic_labels` (graceful degradation without API key), and `run_report` (all 7 report sections, empty docket handling).

### LOW — Documentation / Cleanup

### P16. AUTOINCREMENT not Postgres-compatible — DONE
- **File:** `regscope/db.py`, `SCHEMA_SQL`
- **Fix applied:** Replaced all `INTEGER PRIMARY KEY AUTOINCREMENT` with `INTEGER PRIMARY KEY`. SQLite auto-increments these anyway; cleaner migration path to Postgres.

### P17. Cursor-based pagination potential undercount on resume — DONE
- **File:** `regscope/pipeline/ingest.py`, `_download_comment_headers()`
- **Fix applied:** Changed variable from `page_count` to `api_response_count`, counting API responses returned (not comments stored via INSERT OR IGNORE). Cursor trigger now correctly fires even on resume runs.

---

## Phase 2 — Second Audit Loop (P18-P30, all DONE)

### CRITICAL — Functional Bugs

### P18. `INSERT OR REPLACE` destroys data across classify sub-stages — DONE
- **File:** `regscope/pipeline/classify.py`
- **Bug:** All four SQL statements in classify sub-stages used `INSERT OR REPLACE INTO comment_classifications ... ON CONFLICT DO UPDATE`. SQLite ignores `ON CONFLICT` when `INSERT OR REPLACE` is used, so each sub-stage deleted the row and re-inserted with only its columns, losing data from prior stages.
- **Fix applied:** Changed all four occurrences of `INSERT OR REPLACE` to just `INSERT`.

### P19. COALESCE never falls back in `_fetch_comment_details` — DONE
- **File:** `regscope/pipeline/ingest.py`, `_fetch_comment_details()`
- **Bug:** `attrs.get("comment") or ""` converts None to empty string, but COALESCE only falls back on NULL. The detail endpoint's data overwrote the header data even when empty.
- **Fix applied:** Changed `or ""` to `or None` for comment, submitter, and organization fields.

### P20. `_score_substantiveness` crashes when dedup_groups row is missing — DONE
- **File:** `regscope/pipeline/classify.py`, `_score_substantiveness()`
- **Bug:** Complex ternary expression called `fetchone()[0]` without null check. If the dedup_groups row was deleted (data inconsistency), `fetchone()` returns None and `[0]` raises `TypeError`.
- **Fix applied:** Rewrote to explicit if/else with `fetchone()` null check. Missing group → treat as representative.

### P21. Double-counting on resume in `_download_comment_headers` — DONE
- **File:** `regscope/pipeline/ingest.py`, `_download_comment_headers()`
- **Bug:** On resume, `total` was initialized to `existing_count` and then incremented for every API response (including duplicates that `INSERT OR IGNORE` skips). The return value and progress bar were inflated.
- **Fix applied:** Progress bar now uses actual DB count. Return value is the final DB count, not the running counter.

### HIGH — Code Quality Issues

### P22. `_run_stages` and `run_stage` missing pipeline logging — DONE
- **File:** `regscope/cli.py`
- **Bug:** `_run_stages` never logged "started" status. `run_stage` command never logged "failed" on exception. `ingest` command also lacked "started" and "failed" logging.
- **Fix applied:** Added `log_pipeline_run("started")` before execution in all three code paths. Added `log_pipeline_run("failed")` in exception handlers for `run_stage` and `ingest` commands.

### P23. Duplicate pipeline_runs rows from `_record_embedding_dim` — DONE
- **File:** `regscope/pipeline/embed.py`, `_record_embedding_dim()`
- **Bug:** `_record_embedding_dim` inserted a new "completed" row into `pipeline_runs`, and `_run_stages` also inserted one. Two "completed" rows per embed invocation.
- **Fix applied:** Changed `_record_embedding_dim` to update the most recent embed run's parameters instead of inserting a new row.

### P24. `_extract_html` missing HTML entity unescaping — DONE
- **File:** `regscope/pipeline/extract.py`, `_extract_html()`
- **Bug:** `&amp;`, `&lt;`, etc. were preserved as literal text after tag stripping.
- **Fix applied:** Added `import html` and `text = html.unescape(text)` after tag removal.

### P25. `datetime.utcnow()` deprecated in Python 3.12 — DONE
- **File:** `regscope/db.py`, `log_pipeline_run()`
- **Fix applied:** Changed to `datetime.now(timezone.utc)`.

### MEDIUM — Code Cleanup

### P26. `strip_html` imported inside per-comment loop — DONE
- **File:** `regscope/pipeline/classify.py`, `_detect_stance()`
- **Fix applied:** Moved `from regscope.utils.text import strip_html` to function scope.

### P27. Unused `import tempfile` in `_ocr_pdf` — DONE
- **File:** `regscope/pipeline/extract.py`, `_ocr_pdf()`
- **Fix applied:** Removed unused import.

### P28. Unused `--api-key` option on `run-stage` command — DONE
- **File:** `regscope/cli.py`, `run_stage()`
- **Fix applied:** Removed the unused `--api-key` option (ingest is not in the stage choices).

### P29. `_BUSINESS_INDICATORS` defined inside function — DONE
- **File:** `regscope/pipeline/classify.py`, `_classify_org()`
- **Fix applied:** Moved `_BUSINESS_INDICATORS` set to module level.

### P30. Miscellaneous cleanup — DONE
- `extract.py`: Fixed `att_id % 10` commit logic to use loop counter instead of DB primary key.
- `db.py`: Added unique constraint on `attachments(comment_id, file_url)` to prevent duplicate rows on re-run. Bumped schema to v3 with migration.
- `classify.py`: Removed unused `full_text` from `_classify_stakeholders` query.
- `config.py` + `config.toml.example`: Removed unused `length_optimal`, `length_max_score`, and `attachments_dir` config values.
- `pyproject.toml`: Removed unused `pytest-asyncio` dev dependency.

### P31. Zero tests for classify pipeline integration — DONE
- **File:** `tests/test_classify.py` (expanded)
- **Fix applied:** Added 5 integration tests: `_classify_stakeholders` writing to DB, idempotency, `_score_substantiveness` preserving stakeholder_type, handling missing dedup groups, and full `run_classify` orchestration (with mocked stance detection).

---

## VERIFICATION CHECKLIST

After all fixes, verify:
- [x] `python3 -m pytest tests/ -x -q` — all tests pass → **157 tests pass**
- [x] Config weight changes actually affect substantiveness scores
- [x] Dedup is idempotent (run twice, same result)
- [x] DOCX extraction handles tables and formatting
- [x] `regscope list` doesn't crash on corrupt .db files
- [x] Rate limiter tests pass
- [x] No SQL injection possible via column names
- [x] Classify sub-stages preserve each other's data (integration test)
- [x] Substantiveness scoring doesn't crash on data inconsistencies
- [ ] Run full pipeline on a small docket end-to-end to verify nothing broke *(requires API key — manual verification needed)*

---

## Phase 3 — Third Audit Loop (P32-P33, all DONE)

### P32. `config.toml.example` missing `[llm]` section — DONE
- **File:** `config.toml.example`
- **Fix applied:** Added `[llm]` section with `enabled` and `model` keys matching `config.py` defaults.

### P33. Cursor-advance pagination path untested — DONE
- **File:** `tests/test_ingest.py`
- **Fix applied:** Added `test_cursor_advance_on_full_window` test that simulates 5000+ comments across two cursor windows, verifying the `lastModifiedDate` cursor advance logic triggers correctly.

---

## Phase 4 — Production Readiness Verification (COMPLETE)

### Automated Tests
- **158 tests pass** (`python3 -m pytest tests/ -x -q`)
- No duplicate or stub tests found
- Test coverage spans all pipeline stages

### End-to-End Pipeline Verification
- **NHTSA-2024-0100** (37 comments): All 7 stages completed successfully (ingest → extract → dedup → embed → topics → classify → report). Generated full Markdown report at `output/NHTSA-2024-0100_report.md` with all sections populated.
- **BOEM-2025-0582** (352 comments): All 7 stages completed successfully. 292 unique comments, 15 dedup groups (60 duplicates), 3 topics, 94.4% oppose/conditional_oppose stance. Report saved to `output/BOEM-2025-0582_report.md`.

### Codebase Verification
- **No stale `datetime.utcnow()`** calls in source code (only referenced in PROBLEMS.md)
- **No TODO/FIXME/HACK/XXX** markers in any Python files
- **Config completeness**: All 33 config keys in `config.toml.example` match `config.py` DEFAULTS exactly. Every key is used in the codebase; no orphaned keys.
- **SQL injection**: Column name whitelist in db.py prevents injection. All queries use parameterized values.

### Code Review Summary (dedup.py, classify.py, topics.py, db.py)
All files are production-grade. No critical bugs or security vulnerabilities. Minor observations (not blocking):
- Hardcoded semantic dedup 50K limit and cluster scaling factors could be configurable (by design — documented with warning)
- Stance direction threshold (0.35) is hardcoded in `_detect_stance()` — could be a config key
- Technical vocabulary list in substantiveness scoring is hardcoded — acceptable for current scope
- BERTopic's in-memory fit may need batching for 500K+ comment dockets (documented limitation)

### Test Robustness Review
- **Solid**: test_dedup.py, test_rate_limit.py, test_extract.py, test_ingest.py, test_classify.py
- **Minor fragility**: test_topics.py BERTopic clustering assertions depend on model determinism; test_report.py has hardcoded section format assertions
- No tests depend on external state or test ordering

---

### P34. Missing `__main__.py` — `python -m regscope` fails — DONE
- **File:** `regscope/__main__.py` (created)
- **Bug:** Running `python -m regscope` raised "No module named regscope.__main__" because the package had no entry point module.
- **Fix applied:** Created `__main__.py` that imports and calls `cli()` from `regscope.cli`.

---

## Phase 5 — Client-Readiness Audit (P35-P42, all DONE)

Three parallel code review agents performed an exhaustive audit of every source file, configuration, test, and generated report. The following issues were found and fixed:

### P35. Broken GitHub URL in generated reports — DONE
- **Files:** `regscope/pipeline/report.py`, `README.md`
- **Bug:** Every generated report linked to `https://github.com/regscope` which does not exist.
- **Fix applied:** Removed link from report footer. Changed README clone URL to `<your-repository-url>`.

### P36. "Anonymous Anonymous" redundant submitter names — DONE
- **File:** `regscope/pipeline/report.py`
- **Bug:** When Regulations.gov returns `firstName="Anonymous" lastName="Anonymous"`, the report displayed "Anonymous Anonymous".
- **Fix applied:** Added deduplication: `if submitter.lower() == "anonymous anonymous": submitter = "Anonymous"`.

### P37. Garbled "Adsequipped" topic label (BERTopic tokenizer artifact) — DONE
- **File:** `regscope/pipeline/topics.py`, `_clean_for_topics()`
- **Bug:** Hyphens in terms like "ADS-equipped" were stripped by the CountVectorizer, fusing tokens into nonsense words.
- **Fix applied:** Added `re.sub(r"(?<=\w)-(?=\w)", " ", text)` to replace intra-word hyphens with spaces before tokenization.

### P38. Acronym fallback misclassified nonprofits as "industry" — DONE
- **File:** `regscope/pipeline/classify.py`, `_classify_org()`
- **Bug:** `re.search(r"[A-Z]{2,}", org)` classified any org with 2+ uppercase letters as "industry", catching nonprofits (NAACP, ACLU), unions (AFL-CIO), and government agencies.
- **Fix applied:** Removed the acronym fallback. Only business indicator words trigger "industry". Unknown acronyms return "unknown".

### P39. `datetime.now()` without timezone in report generation — DONE
- **File:** `regscope/pipeline/report.py`
- **Fix applied:** Changed to `datetime.now(timezone.utc)` with "UTC" suffix in report header. Also fixed export's `generated_at`.

### P40. Duplicate semantic dedup groups on re-runs — DONE
- **File:** `regscope/pipeline/embed.py`
- **Bug:** When `run_dedup` ran with existing embeddings, it created semantic groups. Then `run_embed` created them again, producing duplicates.
- **Fix applied:** Added cleanup (clear semantic_group_id + delete semantic dedup_groups) before running `_semantic_dedup` in embed.py. Also added `batch_size = max(1, batch_size)` guard.

### P41. Unused imports and type hints — DONE
- **Files:** `topics.py` (removed `import html`, `import re`), `extract.py` (removed `import time`), `cli.py` (fixed `callable` → `typing.Callable`)

### P42. Config defaults and dependencies — DONE
- **Files:** `config.py`, `config.toml.example`, `pyproject.toml`
- **Fix applied:** Changed `llm.enabled` default to `False` (requires optional install). Removed unnecessary `tomli` dependency (project requires Python 3.11+). Updated example config comment.

---

## VERIFICATION CHECKLIST (FINAL)

- [x] `python3 -m pytest tests/ -x -q` — **158 tests pass**
- [x] `python3 -m regscope --help` — all 9 CLI commands functional
- [x] `python3 -m regscope list` — shows 7 downloaded dockets
- [x] All 15 modules import without errors
- [x] No `print()` statements (all use Rich `console.print()` or `logging`)
- [x] No hardcoded API keys or secrets
- [x] No TODO/FIXME/HACK markers
- [x] No stale `datetime.utcnow()` calls
- [x] No broken GitHub URLs in demo reports
- [x] No "Anonymous Anonymous" in demo reports
- [x] No garbled topic labels in demo reports
- [x] No unused imports in source code
- [x] Config complete: all 33 keys match between code and example
- [x] Package installable: `pip install -e .` succeeds
- [x] NHTSA-2024-0100 end-to-end: all 7 stages, clean report
- [x] BOEM-2025-0582 end-to-end: all 7 stages, clean report

---

## FINAL STATUS

All issues from all five audit phases have been resolved. **42 issues fixed. 158 tests pass.**
Both end-to-end pipeline runs verified on real Regulations.gov data:
- NHTSA-2024-0100 (37 comments, mixed stakeholders) — complete
- BOEM-2025-0582 (352 comments, public opposition docket) — complete

Reports regenerated with all fixes applied. The project is **client-ready**.
