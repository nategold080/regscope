# RegScope — Federal Rulemaking Public Comment Analyzer

## Project Identity
You are building **RegScope**, a production-grade CLI tool that downloads, structures, and analyzes public comments on federal rulemakings from Regulations.gov. This is NOT a demo or proof-of-concept. It must work reliably on any docket ID, at any scale, producing consistent structured output.

## Architecture Philosophy

### LLMs Are Expensive — Use Them Sparingly
The core analysis pipeline MUST NOT depend on LLM API calls for per-comment processing. At 100K+ comments, LLM calls become prohibitively slow and expensive. Instead:
- **Deduplication**: Sentence embeddings (sentence-transformers) + cosine similarity + MinHash/LSH. Deterministic. No LLM.
- **Topic modeling**: BERTopic (HDBSCAN clustering on embeddings + c-TF-IDF for topic representation). Deterministic. No LLM.
- **Stakeholder classification**: Rule-based heuristics on organization field + regex patterns. No LLM.
- **Stance detection**: Zero-shot classification via a small model (e.g., `facebook/bart-large-mnli`), NOT an API call to a large LLM.
- **LLMs are permitted ONLY for**: (1) generating human-readable topic labels from keyword lists (batch, not per-comment). The final summary report is generated from heuristics and database queries, not LLM calls.

### Pipeline Stages Are Independent
Each stage reads from and writes to the database. Every stage is restartable and idempotent. If a download is interrupted, it resumes. If topic modeling parameters change, only that stage re-runs.

### Database-First
SQLite is the data store. Every comment, with all metadata and derived fields, lives in the database. No in-memory-only processing for anything that matters. Design tables so they could migrate to Postgres without schema changes.

## Technical Stack
- **Language**: Python 3.11+
- **Database**: SQLite via `sqlite3` (no ORM — raw SQL for performance and transparency)
- **Embeddings**: `sentence-transformers` (model: `all-MiniLM-L6-v2` for speed, `all-mpnet-base-v2` for quality)
- **Topic modeling**: `bertopic` with `hdbscan` and `umap-learn`
- **Near-duplicate detection**: `datasketch` (MinHash LSH)
- **PDF text extraction**: `pymupdf` (PyMuPDF / fitz) primary, `pytesseract` + `pdf2image` fallback for scanned docs
- **HTTP**: `httpx` (async support, better than requests)
- **CLI**: `click` (not argparse)
- **Data export**: `pandas` for CSV/Excel output
- **Zero-shot classification**: `transformers` pipeline with a local model
- **Progress**: `rich` for terminal progress bars and tables
- **Config**: TOML files, not YAML

## Project Structure
```
regscope/
├── CLAUDE.md
├── README.md
├── pyproject.toml
├── config.toml.example
├── regscope/
│   ├── __init__.py
│   ├── cli.py                  # Click CLI entry point
│   ├── config.py               # Config loading
│   ├── db.py                   # Database schema, connection, migrations
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── ingest.py           # API download + rate limiting
│   │   ├── extract.py          # PDF/attachment text extraction
│   │   ├── dedup.py            # Exact + near-exact + semantic dedup
│   │   ├── embed.py            # Embedding generation
│   │   ├── topics.py           # BERTopic topic modeling
│   │   ├── classify.py         # Stakeholder type + stance detection
│   │   └── report.py           # Summary report generation
│   ├── api/
│   │   ├── __init__.py
│   │   └── regulations.py      # Regulations.gov API v4 client
│   └── utils/
│       ├── __init__.py
│       ├── text.py             # Text cleaning, normalization
│       └── rate_limit.py       # Rate limiter with backoff
├── tests/
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_dedup.py
│   ├── test_topics.py
│   ├── test_classify.py
│   └── fixtures/
│       └── sample_comments.json
└── output/                     # Generated reports go here
```

## Database Schema
Design the schema with these tables:
- `dockets` — docket metadata
- `documents` — documents within a docket (proposed rules, final rules, etc.)
- `comments` — one row per comment with ALL API fields + derived fields
- `attachments` — one row per attachment file with extracted text
- `embeddings` — comment_id + embedding vector (stored as blob)
- `dedup_groups` — groups of duplicate/near-duplicate comments
- `topics` — topic_id, label, keywords, representative_comments
- `comment_topics` — many-to-many: comment_id + topic_id + relevance_score
- `comment_classifications` — stakeholder_type, stance, substantiveness_score
- `pipeline_runs` — log of when each pipeline stage was run with parameters

Add indexes on: docket_id, posted_date, dedup_group_id, topic_id, stakeholder_type, stance.

## Regulations.gov API Client Requirements
- Base URL: `https://api.regulations.gov/v4/`
- Auth: API key passed as `api_key` query parameter
- Rate limit: 1,000 requests/hour. Implement a token bucket rate limiter that tracks remaining requests via `X-RateLimit-Remaining` header and sleeps when approaching the limit. Do NOT just catch 429s — proactively throttle.
- Pagination: Comments endpoint returns max 250 per page, max 20 pages (5,000 total). For dockets with >5,000 comments, paginate using `lastModifiedDate` parameter as cursor (sort by lastModifiedDate, after exhausting 20 pages, set filter to >= last seen date).
- Two-phase download: (1) Bulk download headers via list endpoint, (2) Individually fetch detail for each comment to get full text. Track which comments have been detail-fetched so interrupted downloads resume.
- Handle: connection errors (retry with exponential backoff), malformed responses (log and skip), API changes (validate response schema).

## Deduplication Pipeline
Three tiers:
1. **Exact duplicates**: SHA-256 hash of normalized comment text. O(1) lookup. Group immediately.
2. **Near-duplicates**: MinHash LSH (datasketch) with Jaccard threshold 0.85. Catches form letters with minor edits (name insertion, typos). Group with exact duplicates.
3. **Semantic duplicates**: Cosine similarity on sentence embeddings with threshold 0.92. Catches paraphrased versions of the same argument. Create separate "semantic similarity" groups, don't merge with exact dedup groups.

Output: Each comment gets a `dedup_group_id` (for exact/near) and optionally a `semantic_group_id`. Each group gets a `representative_comment_id` (the longest/most detailed member).

## Topic Modeling
- Use BERTopic with pre-computed embeddings (from the embed stage).
- Reduce dimensionality with UMAP before clustering with HDBSCAN.
- Extract topic representations using c-TF-IDF.
- Allow configurable parameters: min_topic_size, nr_topics, top_n_words.
- After modeling, optionally use an LLM (one batch call) to generate human-readable topic labels from the keyword lists.
- Store topics and per-comment topic assignments in the database.
- Handle hierarchical topics: BERTopic supports topic merging and hierarchy.

## Stakeholder Classification
Classify each comment's source into categories:
- `individual` — private citizen
- `industry` — company or industry group
- `trade_association` — trade/professional association
- `nonprofit` — NGO, advocacy org, think tank
- `government` — government entity (state, local, tribal, other federal)
- `academic` — university, researcher
- `law_firm` — law firm filing on behalf of client
- `unknown` — insufficient info

Method: Primarily rule-based on the `organization` field + keyword patterns. Fall back to zero-shot classification on comment text for ambiguous cases. Maintain a lookup table of known organizations that grows over time.

## Stance Detection
Classify each unique comment (after dedup) as:
- `support` — supports the proposed rule
- `oppose` — opposes the proposed rule
- `conditional_support` — supports with modifications
- `conditional_oppose` — opposes but would accept with changes
- `neutral_informational` — provides information without clear stance
- `unclear` — can't determine

Method: Zero-shot classification using a local transformer model. Run on unique comments only (not every copy of a form letter). Validate accuracy on a sample before applying to full dataset.

## Substantiveness Scoring
Score each comment 0-100 on how substantive it is:
- Length (longer = more likely substantive, with diminishing returns)
- Presence of citations, data, specific regulatory section references
- Vocabulary complexity / technical language
- Whether it's a form letter (automatic low score for exact duplicates)
- Whether it raises specific legal, technical, or factual arguments

This should be a weighted heuristic, NOT an LLM call. Make the weights configurable.

## CLI Interface
```bash
# Download and analyze a docket
regscope analyze EPA-HQ-OAR-2021-0317 --api-key YOUR_KEY

# Just download (no analysis)  
regscope ingest EPA-HQ-OAR-2021-0317 --api-key YOUR_KEY

# Run analysis on already-downloaded data
regscope process EPA-HQ-OAR-2021-0317

# Run a specific pipeline stage
regscope run-stage EPA-HQ-OAR-2021-0317 --stage dedup
regscope run-stage EPA-HQ-OAR-2021-0317 --stage topics

# Export results
regscope export EPA-HQ-OAR-2021-0317 --format csv --output ./output/
regscope export EPA-HQ-OAR-2021-0317 --format json --output ./output/
regscope report EPA-HQ-OAR-2021-0317 --output ./output/report.md

# Show status of a docket's processing
regscope status EPA-HQ-OAR-2021-0317

# List all downloaded dockets
regscope list
```

## Report Output
The final report (Markdown) should include:
1. **Docket Overview**: Agency, title, dates, total comments received
2. **Comment Landscape**: Unique vs duplicate breakdown, form letter campaigns identified (with template text), submission timeline chart data
3. **Topic Analysis**: Top N topics with keywords and representative quotes, topic distribution
4. **Stakeholder Breakdown**: Who commented — counts and percentages by stakeholder type, top organizations
5. **Stance Analysis**: Support vs oppose vs conditional, broken down by stakeholder type
6. **Substantive Comment Highlights**: Top 20 most substantive comments with summaries
7. **Data Quality Notes**: Missing fields, PDF-only comments, extraction failures

## Quality Standards
- Every function has a docstring.
- Type hints on all function signatures.
- Logging via Python `logging` module (not print statements). Log to file and optionally to console.
- Errors are caught, logged, and the pipeline continues (one bad comment doesn't crash the whole run).
- Tests cover: API response parsing, dedup accuracy, text normalization, stakeholder classification rules.
- The tool should work on a docket with 50 comments AND a docket with 500,000 comments. Design for scale from the start (batch processing, streaming where possible, memory-conscious embedding generation).

## Things NOT To Build
- No web UI. CLI only.
- No user authentication system.
- No real-time monitoring / streaming updates.
- No multi-user / collaboration features.
- No deployment infrastructure (Docker, K8s, etc.) — just a pip-installable package.

## Common Pitfalls to Avoid
- Don't load all comments into memory at once. Process in batches.
- Don't compute all-pairs similarity. Use approximate nearest neighbors or LSH.
- Don't call an LLM API for every comment. The pipeline must work without any LLM API key configured (LLM features are optional enrichment).
- Don't ignore PDF attachments. Many substantive comments are PDF-only.
- Don't assume the organization field is populated. It's often blank.
- Don't treat comment count as a meaningful metric. Substance matters.
- Don't hardcode docket IDs or agency-specific logic. The tool must be agency-agnostic.
