"""Streamlit dashboard for RegScope — Federal Rulemaking Comment Analyzer.

Polished, client-facing dashboard for demos and outreach.

Run: streamlit run dashboard/app.py
"""

import sqlite3
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Constants ─────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "dockets"

ACCENT_BLUE = "#0984E3"
PALETTE = [
    "#0984E3", "#6C5CE7", "#00B894", "#E17055", "#FDCB6E",
    "#74B9FF", "#A29BFE", "#55EFC4", "#FF7675", "#DFE6E9",
]

STANCE_COLORS = {
    "support": "#00B894",
    "conditional_support": "#55EFC4",
    "neutral_informational": "#74B9FF",
    "unclear": "#DFE6E9",
    "conditional_oppose": "#FDCB6E",
    "oppose": "#FF7675",
}

STANCE_ORDER = [
    "support", "conditional_support", "neutral_informational",
    "unclear", "conditional_oppose", "oppose",
]

STAKEHOLDER_COLORS = {
    "individual": "#0984E3",
    "nonprofit": "#6C5CE7",
    "industry": "#E17055",
    "trade_association": "#FDCB6E",
    "government": "#00B894",
    "academic": "#A29BFE",
    "other": "#DFE6E9",
}


def fmt_stance(s):
    if not s:
        return "Unknown"
    return s.replace("_", " ").title()


def fmt_stakeholder(s):
    if not s:
        return "Unknown"
    return s.replace("_", " ").title()


def section_header(text):
    st.markdown(
        f'<p class="section-header">{text}</p>',
        unsafe_allow_html=True,
    )


def plotly_dark_layout(fig, **kwargs):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif"),
        margin=dict(l=40, r=20, t=40, b=40),
        **kwargs,
    )
    return fig


# ── Data loading ──────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def discover_dockets():
    """Find all docket databases and load their metadata."""
    dockets = []
    for db_path in sorted(DATA_DIR.glob("*.db")):
        docket_id = db_path.stem
        conn = sqlite3.connect(str(db_path))
        try:
            row = conn.execute("SELECT title, agency FROM dockets LIMIT 1").fetchone()
            comment_count = conn.execute("SELECT COUNT(*) FROM comments").fetchone()[0]
            if comment_count == 0:
                continue
            topic_count = conn.execute("SELECT COUNT(*) FROM topics").fetchone()[0]
            dedup_count = conn.execute("SELECT COUNT(*) FROM dedup_groups").fetchone()[0]
            classified = conn.execute("SELECT COUNT(*) FROM comment_classifications").fetchone()[0]
            dockets.append({
                "docket_id": docket_id,
                "title": row[0] if row else docket_id,
                "agency": row[1] if row else "Unknown",
                "comments": comment_count,
                "topics": topic_count,
                "dedup_groups": dedup_count,
                "classified": classified,
                "db_path": str(db_path),
            })
        finally:
            conn.close()
    return dockets


@st.cache_data(ttl=300)
def load_docket_data(db_path: str):
    """Load all relevant data from a single docket database."""
    conn = sqlite3.connect(db_path)

    comments = pd.read_sql_query("""
        SELECT c.comment_id, c.submitter_name, c.organization, c.posted_date,
               c.dedup_group_id, c.semantic_group_id,
               cc.stakeholder_type, cc.stance, cc.stance_confidence,
               cc.substantiveness_score
        FROM comments c
        LEFT JOIN comment_classifications cc ON c.comment_id = cc.comment_id
    """, conn)

    topics = pd.read_sql_query("""
        SELECT topic_id, label, keywords, topic_size, llm_label
        FROM topics
        ORDER BY topic_size DESC
    """, conn)

    dedup_groups = pd.read_sql_query("""
        SELECT dedup_group_id, group_type, group_size, template_text
        FROM dedup_groups
        ORDER BY group_size DESC
    """, conn)

    comment_topics = pd.read_sql_query("""
        SELECT ct.comment_id, ct.topic_id, ct.relevance_score,
               t.label, t.llm_label, t.topic_size
        FROM comment_topics ct
        JOIN topics t ON ct.topic_id = t.topic_id
    """, conn)

    conn.close()

    return {
        "comments": comments,
        "topics": topics,
        "dedup_groups": dedup_groups,
        "comment_topics": comment_topics,
    }


# ── Page layout ───────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="RegScope — Federal Comment Analyzer",
        page_icon=":memo:",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # ── Custom CSS ─────────────────────────────────────────────────────
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    .block-container { padding-top: 1.5rem; max-width: 1200px; }

    .main-title {
        font-family: 'Inter', sans-serif;
        font-size: 2.2rem;
        font-weight: 700;
        color: #FFFFFF;
        margin-bottom: 0;
        line-height: 1.2;
    }
    .main-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1.05rem;
        color: #94A3B8;
        margin-top: 2px;
        margin-bottom: 1.2rem;
    }

    /* KPI cards */
    [data-testid="stMetric"] {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px 20px;
    }
    [data-testid="stMetricLabel"] {
        font-family: 'Inter', sans-serif;
        font-size: 0.8rem !important;
        font-weight: 500;
        color: #64748B !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    [data-testid="stMetricValue"] {
        font-family: 'Inter', sans-serif;
        font-size: 1.8rem !important;
        font-weight: 700;
        color: #1B2A4A !important;
    }

    /* Section headers */
    .section-header {
        font-family: 'Inter', sans-serif;
        font-size: 1.25rem;
        font-weight: 600;
        color: #FFFFFF;
        margin-top: 0.8rem;
        margin-bottom: 0.4rem;
        padding-bottom: 0.3rem;
        border-bottom: 2px solid #E2E8F0;
    }

    /* Table styling */
    .dataframe { font-family: 'Inter', sans-serif !important; }

    /* Hide Streamlit branding */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header[data-testid="stHeader"] { background: transparent; }
    [data-testid="stToolbar"] { display: none; }
    [data-testid="stAppDeployButton"] { display: none; }
    ._profileContainer_gzau3_53 { display: none !important; }
    ._container_gzau3_1 { display: none !important; }
    [data-testid="stStatusWidget"] { display: none; }
    div[class*="profileContainer"] { display: none !important; }
    div[class*="hostContainer"] { display: none !important; }
    iframe[title="streamlit_badge"] { display: none !important; }
    #stStreamlitBadge { display: none !important; }

    div[data-testid="stDataFrame"] div[class*="glideDataEditor"] {
        border: 1px solid #E2E8F0;
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Title ──────────────────────────────────────────────────────────
    st.markdown(
        '<p class="main-title">RegScope — Federal Rulemaking Comment Analyzer</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="main-subtitle">'
        'Structured analysis of public comments on federal rulemakings: '
        'stakeholder classification, stance detection, topic modeling, and duplicate detection'
        '</p>',
        unsafe_allow_html=True,
    )

    # ── Load docket index ──────────────────────────────────────────────
    dockets = discover_dockets()

    if not dockets:
        st.error(f"No docket databases found in {DATA_DIR}")
        return

    total_comments = sum(d["comments"] for d in dockets)
    total_topics = sum(d["topics"] for d in dockets)
    total_dedup = sum(d["dedup_groups"] for d in dockets)
    agencies = len(set(d["agency"] for d in dockets))

    # ── KPI Row ────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Dockets Analyzed", f"{len(dockets)}")
    c2.metric("Total Comments", f"{total_comments:,}")
    c3.metric("Agencies", f"{agencies}")
    c4.metric("Topics Identified", f"{total_topics}")
    c5.metric("Dedup Groups", f"{total_dedup:,}")

    st.markdown("")

    # ── Docket Overview Table ──────────────────────────────────────────
    section_header("Docket Overview")

    docket_df = pd.DataFrame(dockets)
    # Truncate long titles for display
    docket_df["title_short"] = docket_df["title"].apply(
        lambda t: t[:80] + "..." if len(t) > 80 else t
    )

    st.dataframe(
        docket_df[["docket_id", "agency", "title_short", "comments", "topics", "dedup_groups", "classified"]],
        column_config={
            "docket_id": "Docket ID",
            "agency": "Agency",
            "title_short": "Title",
            "comments": "Comments",
            "topics": "Topics",
            "dedup_groups": "Dedup Groups",
            "classified": "Classified",
        },
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("")

    # ── Comment Volume Chart ───────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        section_header("Comments by Docket")

        fig = go.Figure(go.Bar(
            y=[d["docket_id"] for d in dockets],
            x=[d["comments"] for d in dockets],
            orientation="h",
            marker_color=ACCENT_BLUE,
            text=[f'{d["comments"]:,}' for d in dockets],
            textposition="outside",
        ))
        plotly_dark_layout(fig, height=300, showlegend=False,
                          xaxis_title="Number of Comments",
                          yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        section_header("Comments by Agency")

        agency_agg = {}
        for d in dockets:
            agency_agg[d["agency"]] = agency_agg.get(d["agency"], 0) + d["comments"]

        fig = px.pie(
            pd.DataFrame([{"agency": k, "comments": v} for k, v in agency_agg.items()]),
            values="comments", names="agency",
            color_discrete_sequence=PALETTE,
            hole=0.45,
        )
        plotly_dark_layout(fig, height=300, showlegend=True)
        fig.update_traces(textposition="inside", textinfo="percent+label", textfont_size=11)
        st.plotly_chart(fig, use_container_width=True)

    # ── Docket Selector ────────────────────────────────────────────────
    st.markdown("")
    section_header("Docket Deep Dive")

    docket_options = {d["docket_id"]: d for d in dockets}
    selected_docket = st.selectbox(
        "Select a docket to analyze",
        options=list(docket_options.keys()),
        format_func=lambda d: f"{d} — {docket_options[d]['agency']} — {docket_options[d]['title'][:60]}...",
        index=0,
    )

    docket_info = docket_options[selected_docket]
    data = load_docket_data(docket_info["db_path"])
    comments = data["comments"]
    topics = data["topics"]
    dedup_groups = data["dedup_groups"]
    comment_topics = data["comment_topics"]

    # ── Docket KPIs ────────────────────────────────────────────────────
    n_comments = len(comments)
    n_classified = comments["stakeholder_type"].notna().sum()
    unique_orgs = comments[comments["organization"].notna() & (comments["organization"] != "")]["organization"].nunique()
    avg_substantiveness = comments["substantiveness_score"].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Comments", f"{n_comments:,}")
    c2.metric("Classified", f"{n_classified:,}")
    c3.metric("Unique Orgs", f"{unique_orgs:,}")
    c4.metric("Avg Substantiveness", f"{avg_substantiveness:.0f}/100" if pd.notna(avg_substantiveness) else "N/A")

    st.markdown("")

    # ── Stance & Stakeholder Charts ────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        section_header("Stance Distribution")
        stance_data = comments[comments["stance"].notna()]["stance"].value_counts()
        if not stance_data.empty:
            ordered = [s for s in STANCE_ORDER if s in stance_data.index]
            stance_df = pd.DataFrame({
                "stance": ordered,
                "count": [stance_data[s] for s in ordered],
            })
            stance_df["label"] = stance_df["stance"].apply(fmt_stance)
            colors = [STANCE_COLORS.get(s, "#DFE6E9") for s in ordered]

            fig = go.Figure(go.Bar(
                x=stance_df["label"], y=stance_df["count"],
                marker_color=colors,
                text=stance_df["count"], textposition="outside",
            ))
            plotly_dark_layout(fig, height=380, showlegend=False,
                              xaxis_tickangle=-30, yaxis_title="Comments")
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        section_header("Stakeholder Breakdown")
        stype_data = comments[comments["stakeholder_type"].notna()]["stakeholder_type"].value_counts()
        if not stype_data.empty:
            stype_df = pd.DataFrame({
                "type": stype_data.index,
                "count": stype_data.values,
            })
            stype_df["label"] = stype_df["type"].apply(fmt_stakeholder)
            colors = [STAKEHOLDER_COLORS.get(t, "#DFE6E9") for t in stype_df["type"]]

            fig = px.pie(
                stype_df, values="count", names="label",
                color_discrete_sequence=colors,
                hole=0.45,
            )
            plotly_dark_layout(fig, height=380, showlegend=True)
            fig.update_traces(textposition="inside", textinfo="percent+label", textfont_size=10)
            st.plotly_chart(fig, use_container_width=True)

    # ── Stance by Stakeholder Cross-tab ────────────────────────────────
    cross = comments[comments["stance"].notna() & comments["stakeholder_type"].notna()]
    if len(cross) > 10:
        section_header("Stance by Stakeholder Type")

        pivot = cross.groupby(["stakeholder_type", "stance"]).size().reset_index(name="count")
        pivot["stance_label"] = pivot["stance"].apply(fmt_stance)
        pivot["type_label"] = pivot["stakeholder_type"].apply(fmt_stakeholder)

        fig = px.bar(
            pivot, x="type_label", y="count", color="stance_label",
            color_discrete_map={fmt_stance(s): STANCE_COLORS.get(s, "#DFE6E9") for s in STANCE_ORDER},
            labels={"type_label": "Stakeholder Type", "count": "Comments", "stance_label": "Stance"},
            barmode="stack",
        )
        plotly_dark_layout(fig, height=400, xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)

    # ── Topic Analysis ─────────────────────────────────────────────────
    if not topics.empty:
        section_header("Topic Analysis")

        topic_display = topics.copy()
        topic_display["display_label"] = topic_display.apply(
            lambda r: r["llm_label"] if pd.notna(r["llm_label"]) and r["llm_label"]
            else r["label"] if pd.notna(r["label"]) else f"Topic {r['topic_id']}",
            axis=1,
        )
        # Truncate for display
        topic_display["display_label"] = topic_display["display_label"].apply(
            lambda t: t[:70] + "..." if isinstance(t, str) and len(t) > 70 else t
        )

        top_n = topic_display.head(15)
        fig = go.Figure(go.Bar(
            y=top_n["display_label"],
            x=top_n["topic_size"],
            orientation="h",
            marker_color=ACCENT_BLUE,
            text=top_n["topic_size"],
            textposition="outside",
        ))
        plotly_dark_layout(fig, height=max(300, len(top_n) * 30 + 80),
                          showlegend=False,
                          xaxis_title="Number of Comments",
                          yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    # ── Substantiveness Distribution ───────────────────────────────────
    sub_scores = comments[comments["substantiveness_score"].notna()]["substantiveness_score"]
    if not sub_scores.empty:
        section_header("Substantiveness Score Distribution")
        st.caption("0-19: Form letters | 20-39: Low substance | 40-59: Moderate | 60-79: High | 80-100: Highly substantive")

        fig = go.Figure(go.Histogram(
            x=sub_scores, nbinsx=20,
            marker_color=ACCENT_BLUE,
        ))
        plotly_dark_layout(fig, height=320, showlegend=False,
                          xaxis_title="Substantiveness Score",
                          yaxis_title="Number of Comments")
        st.plotly_chart(fig, use_container_width=True)

    # ── Duplicate / Form Letter Analysis ───────────────────────────────
    if not dedup_groups.empty:
        section_header("Duplicate & Form Letter Analysis")

        col1, col2 = st.columns(2)

        with col1:
            # By type
            type_counts = dedup_groups["group_type"].value_counts()
            type_labels = {"exact": "Exact Duplicates", "near": "Near Duplicates", "semantic": "Semantic Matches"}
            fig = go.Figure(go.Bar(
                x=[type_labels.get(t, t) for t in type_counts.index],
                y=type_counts.values,
                marker_color=[PALETTE[0], PALETTE[1], PALETTE[2]][:len(type_counts)],
                text=type_counts.values,
                textposition="outside",
            ))
            plotly_dark_layout(fig, height=300, showlegend=False,
                              yaxis_title="Number of Groups")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Top campaigns
            top_campaigns = dedup_groups[dedup_groups["group_size"] > 1].head(10)
            if not top_campaigns.empty:
                campaign_labels = top_campaigns.apply(
                    lambda r: (r["template_text"][:50] + "..." if pd.notna(r["template_text"]) and len(str(r["template_text"])) > 50
                               else str(r["template_text"])[:50] if pd.notna(r["template_text"])
                               else f"Group {r['dedup_group_id']}"),
                    axis=1,
                )
                fig = go.Figure(go.Bar(
                    y=campaign_labels,
                    x=top_campaigns["group_size"],
                    orientation="h",
                    marker_color=PALETTE[1],
                    text=top_campaigns["group_size"],
                    textposition="outside",
                ))
                plotly_dark_layout(fig, height=350, showlegend=False,
                                  xaxis_title="Copies Found",
                                  yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig, use_container_width=True)

    # ── Comment Explorer ───────────────────────────────────────────────
    section_header("Comment Explorer")

    col1, col2, col3 = st.columns(3)
    with col1:
        search_text = st.text_input("Search comments", placeholder="e.g., wind farm, offshore, safety...")
    with col2:
        stance_filter = st.selectbox("Filter by stance", ["All"] + [fmt_stance(s) for s in STANCE_ORDER])
    with col3:
        stype_filter = st.selectbox("Filter by stakeholder type",
                                    ["All"] + sorted(fmt_stakeholder(s) for s in comments["stakeholder_type"].dropna().unique()))

    explorer_df = comments.copy()

    if search_text:
        mask = (
            explorer_df["submitter_name"].str.contains(search_text, case=False, na=False) |
            explorer_df["organization"].str.contains(search_text, case=False, na=False)
        )
        explorer_df = explorer_df[mask]

    if stance_filter != "All":
        explorer_df = explorer_df[explorer_df["stance"].apply(fmt_stance) == stance_filter]

    if stype_filter != "All":
        explorer_df = explorer_df[explorer_df["stakeholder_type"].apply(fmt_stakeholder) == stype_filter]

    display_cols = ["comment_id", "submitter_name", "organization", "posted_date",
                    "stakeholder_type", "stance", "substantiveness_score"]
    display_df = explorer_df[display_cols].copy()
    display_df["stakeholder_type"] = display_df["stakeholder_type"].apply(fmt_stakeholder)
    display_df["stance"] = display_df["stance"].apply(fmt_stance)

    st.write(f"Showing **{len(display_df):,}** comments")
    st.dataframe(
        display_df.head(200),
        column_config={
            "comment_id": "Comment ID",
            "submitter_name": "Submitter",
            "organization": "Organization",
            "posted_date": "Posted",
            "stakeholder_type": "Stakeholder Type",
            "stance": "Stance",
            "substantiveness_score": st.column_config.ProgressColumn(
                "Substantiveness",
                min_value=0, max_value=100,
            ),
        },
        use_container_width=True,
        hide_index=True,
    )

    # ── About / Methodology ───────────────────────────────────────────
    st.markdown("")
    with st.expander("About This Data / Methodology", expanded=False):
        st.markdown("""
### How It Works

RegScope is a fully automated pipeline for analyzing public comments on federal rulemakings
from Regulations.gov:

1. **Download** — Comments and attachments are downloaded via the Regulations.gov API
   with rate limiting and resumable progress tracking.
2. **Deduplicate** — MinHash LSH identifies exact, near-duplicate, and semantically similar
   comments (form letter campaigns). Each group gets a representative text and copy count.
3. **Classify** — Each comment is classified by:
   - **Stakeholder type** (individual, nonprofit, industry, government, academic) using
     rule-based heuristics on the organization field
   - **Stance** (support, conditional support, oppose, conditional oppose, neutral) using
     zero-shot classification with a local transformer model
   - **Substantiveness score** (0-100) based on length, specificity, and uniqueness
4. **Topic Model** — BERTopic clusters comments into themes using sentence embeddings
   (all-MiniLM-L6-v2) and HDBSCAN. LLM-generated labels summarize each cluster.
5. **Extract Attachments** — PDFs are parsed with PyMuPDF; scanned documents fall back
   to OCR via Tesseract.

### Key Design Decisions

- **No per-comment LLM calls** — stance detection uses a local zero-shot model, keeping
  costs at $0 for analysis of 2,000+ comments
- **Sentence embeddings** power both deduplication and topic modeling
- **Every analysis stage is idempotent** — rerunning produces identical results

### Data Coverage

- **{n_dockets}** dockets analyzed across {n_agencies} federal agencies
- **{n_comments:,}** total comments with full classification pipeline
- **{n_topics}** topics identified via BERTopic clustering
- **{n_dedup:,}** duplicate groups detected

### Limitations

- Stance classification relies on zero-shot models with ~70-80% accuracy on nuanced text
- Topic labels are generated by LLM (batch, not per-comment) and may not always be precise
- Substantiveness scores are heuristic-based and may undercount short but substantive comments
""".format(
            n_dockets=len(dockets),
            n_agencies=agencies,
            n_comments=total_comments,
            n_topics=total_topics,
            n_dedup=total_dedup,
        ))

    # ── Footer ─────────────────────────────────────────────────────────
    st.markdown("")
    st.divider()
    st.markdown(
        "<div style='text-align: center; color: #94A3B8; font-size: 0.8rem; padding: 8px 0;'>"
        "RegScope &bull; Federal Rulemaking Comment Analyzer &bull; "
        "Data sourced from Regulations.gov API &bull; "
        "Zero-cost NLP pipeline: embeddings &rarr; dedup &rarr; classify &rarr; topic model"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
