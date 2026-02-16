# RegScope

**Structured analysis of public comments on federal rulemakings.** RegScope downloads every comment on a Regulations.gov docket, deduplicates form letters, clusters comments into topics, classifies stakeholders and stances, and produces a Markdown report that summarizes who said what — so analysts don't have to read thousands of comments by hand.

## Quick Start

```bash
git clone https://github.com/regscope/regscope.git && cd regscope
pip install -e .
export REGSCOPE_API_KEY=your_key_here   # free from https://api.data.gov/signup/
regscope analyze BOEM-2024-0008
```

The report lands in `./output/BOEM-2024-0008_report.md`. The full run takes ~20 minutes for a 1,200-comment docket on a laptop.

## Example Output

Below is an abridged report for [BOEM-2024-0008](https://www.regulations.gov/docket/BOEM-2024-0008) (Atlantic Shores Offshore Wind North Project, 1,207 comments):

> **Comment Landscape** — 1,207 total comments, 732 unique, 94 duplicate groups. Largest form-letter campaign: 174 copies.
>
> **Topics** (17 identified):
>
> | # | Topic | Comments |
> |---|-------|----------|
> | 1 | Support for Renewable Energy Projects | 163 |
> | 2 | Opposition to Environmental Destruction | 67 |
> | 3 | Regulatory and Procedural Comments | 53 |
> | 4 | Concerns About Wind Turbines | 44 |
> | 5 | Impact on Marine Wildlife | 42 |
> | ... | *12 more topics* | |
>
> **Stakeholders** — 95.4% individuals, 1.6% trade associations, 1.4% nonprofits, 1.0% industry, 0.5% government.
>
> **Stance** — 48.3% supportive (support + conditional), 42.3% opposed (oppose + conditional), 3.7% neutral, 5.7% unclear.

Full sample report: [`output/BOEM-2024-0008_report.md`](output/BOEM-2024-0008_report.md)

## Pipeline Stages

| Stage | What it does |
|-------|-------------|
| **ingest** | Downloads comment metadata and full text from the Regulations.gov API, with rate limiting and resume support |
| **extract** | Pulls text from PDF, DOCX, and other attachments (many substantive comments are attachment-only) |
| **dedup** | Groups exact duplicates (SHA-256), near-duplicates (MinHash LSH), and semantic duplicates (cosine similarity) |
| **embed** | Generates sentence embeddings for each unique comment using a local transformer model |
| **topics** | Clusters comments into topics via BERTopic (UMAP + HDBSCAN + c-TF-IDF), with optional LLM-generated labels |
| **classify** | Detects stakeholder type (individual, industry, government, ...), stance (support/oppose), and substantiveness score |
| **report** | Produces a structured Markdown report with statistics, topic breakdowns, and highlighted substantive comments |

Each stage is independent, idempotent, and resumable. Run them individually with `regscope run-stage DOCKET_ID --stage STAGE`.

## Configuration

Copy `config.toml.example` to `config.toml` to customize settings. Defaults work well for most dockets. Key options:

- **API rate limiting** — defaults to 900 req/hr (API max is 1,000). Supports multiple comma-separated keys for parallel ingestion.
- **Deduplication thresholds** — Jaccard similarity (0.85) and semantic cosine similarity (0.92).
- **Topic modeling** — minimum topic size, UMAP/HDBSCAN parameters.
- **LLM topic labels** — set `OPENAI_API_KEY` to get human-readable topic names (one cheap API call per report). Falls back to keyword labels automatically.

All data is stored in per-docket SQLite databases at `~/.regscope/data/`.

## Requirements

- Python 3.11+
- A free [Regulations.gov API key](https://api.data.gov/signup/) (1,000 requests/hour)
- ~2 GB disk for ML models (downloaded automatically on first run)
- Optional: `OPENAI_API_KEY` for LLM-generated topic labels (`pip install -e ".[llm]"`)

## CLI Reference

```
regscope analyze DOCKET_ID          # Full pipeline: download + analyze + report
regscope ingest DOCKET_ID           # Download only
regscope process DOCKET_ID          # Analyze already-downloaded data
regscope run-stage DOCKET_ID -s X   # Run a single stage (ingest/extract/dedup/embed/topics/classify/report)
regscope report DOCKET_ID           # Regenerate the Markdown report
regscope export DOCKET_ID -f csv    # Export to CSV, JSON, or Excel
regscope status DOCKET_ID           # Show pipeline progress
regscope list                       # List all downloaded dockets
```

## License

MIT
