"""Streamlit dashboard for RegScope — Federal Rulemaking Comment Analyzer.

Polished, client-facing dashboard for demos and outreach.

Run: streamlit run dashboard/app.py
"""

import re
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


def section_note(text):
    """Render an explanatory note beneath a section header — brighter and larger than st.caption."""
    st.markdown(
        f'<p style="font-family: Inter, sans-serif; font-size: 0.92rem; color: #CBD5E1; '
        f'margin-top: -0.2rem; margin-bottom: 0.6rem; line-height: 1.5;">{text}</p>',
        unsafe_allow_html=True,
    )


def clean_html(text):
    """Strip common HTML entities and tags for display."""
    if not text:
        return ""
    t = str(text)
    t = t.replace("<br/>", "\n").replace("<br>", "\n").replace("<br />", "\n")
    t = t.replace("&rsquo;", "\u2019").replace("&amp;", "&")
    t = t.replace("&nbsp;", " ").replace("&ldquo;", "\u201c").replace("&rdquo;", "\u201d")
    t = re.sub(r"<[^>]+>", "", t)
    # Remove standalone page numbers (1-3 digits alone on a line, from PDF extraction)
    t = re.sub(r"\n\s*\d{1,3}\s*\n", "\n", t)
    return t.strip()


def clean_topic_label(label, llm_label=None):
    """Convert a BERTopic raw label into a human-readable string."""
    if llm_label and pd.notna(llm_label) and str(llm_label).strip():
        return str(llm_label).strip()
    if not label or not pd.notna(label):
        return "Unknown Topic"
    label = str(label).strip()
    cleaned = re.sub(r"^\d+_", "", label)
    if not cleaned:
        return label.title()
    keywords = [kw.strip().title() for kw in cleaned.split("_") if kw.strip()]
    seen = set()
    unique_kw = []
    for kw in keywords:
        kw_lower = kw.lower()
        if kw_lower not in seen:
            seen.add(kw_lower)
            unique_kw.append(kw)
    return ", ".join(unique_kw) if unique_kw else label.title()


def plotly_dark_layout(fig, **kwargs):
    margin = kwargs.pop("margin", dict(l=40, r=40, t=40, b=40))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif"),
        margin=margin,
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

    topic_cols = {row[1] for row in conn.execute("PRAGMA table_info(topics)").fetchall()}
    has_llm_label = "llm_label" in topic_cols

    comments = pd.read_sql_query("""
        SELECT c.comment_id, c.submitter_name, c.organization, c.posted_date,
               c.comment_text, c.full_text,
               c.dedup_group_id, c.semantic_group_id,
               cc.stakeholder_type, cc.stance, cc.stance_confidence,
               cc.substantiveness_score
        FROM comments c
        LEFT JOIN comment_classifications cc ON c.comment_id = cc.comment_id
    """, conn)

    if has_llm_label:
        topics = pd.read_sql_query("""
            SELECT topic_id, bertopic_id, label, keywords, topic_size, llm_label
            FROM topics ORDER BY topic_size DESC
        """, conn)
    else:
        topics = pd.read_sql_query("""
            SELECT topic_id, bertopic_id, label, keywords, topic_size
            FROM topics ORDER BY topic_size DESC
        """, conn)
        topics["llm_label"] = None

    dedup_groups = pd.read_sql_query("""
        SELECT dg.dedup_group_id, dg.group_type, dg.group_size, dg.template_text,
               dg.representative_comment_id
        FROM dedup_groups dg
        ORDER BY dg.group_size DESC
    """, conn)

    if has_llm_label:
        comment_topics = pd.read_sql_query("""
            SELECT ct.comment_id, ct.topic_id, ct.relevance_score,
                   t.label, t.llm_label, t.topic_size
            FROM comment_topics ct
            JOIN topics t ON ct.topic_id = t.topic_id
        """, conn)
    else:
        comment_topics = pd.read_sql_query("""
            SELECT ct.comment_id, ct.topic_id, ct.relevance_score,
                   t.label, t.topic_size
            FROM comment_topics ct
            JOIN topics t ON ct.topic_id = t.topic_id
        """, conn)
        comment_topics["llm_label"] = None

    # Attachment stats
    attach_total = conn.execute("SELECT COUNT(*) FROM attachments").fetchone()[0]
    attach_extracted = conn.execute(
        "SELECT COUNT(*) FROM attachments WHERE extracted_text IS NOT NULL AND extracted_text != ''"
    ).fetchone()[0]

    conn.close()

    return {
        "comments": comments,
        "topics": topics,
        "dedup_groups": dedup_groups,
        "comment_topics": comment_topics,
        "attach_total": attach_total,
        "attach_extracted": attach_extracted,
    }


def aggregate_campaigns(dedup_groups, comments_df):
    """Aggregate dedup groups into form letter campaigns.

    Near-duplicate groups with the same template text (first 100 chars)
    are merged into a single campaign for display. Groups without template
    text get the representative comment's text instead.
    """
    multi = dedup_groups[dedup_groups["group_size"] > 1].copy()
    if multi.empty:
        return []

    # For groups without template_text, use representative comment text
    comment_text_map = {}
    if not comments_df.empty:
        for col in ["full_text", "comment_text"]:
            if col in comments_df.columns:
                mapping = comments_df.dropna(subset=[col]).set_index("comment_id")[col].to_dict()
                comment_text_map.update(mapping)

    campaigns = []
    for _, row in multi.iterrows():
        template = str(row["template_text"]) if pd.notna(row["template_text"]) else None
        if not template and pd.notna(row.get("representative_comment_id")):
            rep_id = row["representative_comment_id"]
            template = comment_text_map.get(rep_id)
        campaigns.append({
            "group_ids": [row["dedup_group_id"]],
            "group_type": row["group_type"],
            "total_copies": int(row["group_size"]),
            "template": template,
            "sub_groups": 1,
        })

    # Merge campaigns with matching template text prefix (near-dup groups of the same letter)
    merged = []
    used = set()
    for i, c in enumerate(campaigns):
        if i in used:
            continue
        if not c["template"]:
            merged.append(c)
            continue
        prefix = clean_html(c["template"])[:100].lower().strip()
        combined = dict(c)
        for j in range(i + 1, len(campaigns)):
            if j in used:
                continue
            other = campaigns[j]
            if not other["template"]:
                continue
            other_prefix = clean_html(other["template"])[:100].lower().strip()
            if prefix == other_prefix:
                combined["total_copies"] += other["total_copies"]
                combined["group_ids"].extend(other["group_ids"])
                combined["sub_groups"] += 1
                used.add(j)
        merged.append(combined)
        used.add(i)

    merged.sort(key=lambda c: c["total_copies"], reverse=True)
    return merged


# ── Page layout ───────────────────────────────────────────────────────────

def main():
    favicon = Path(__file__).resolve().parent.parent / ".streamlit" / "favicon.png"
    st.set_page_config(
        page_title="RegScope — Federal Comment Analyzer",
        page_icon=str(favicon) if favicon.exists() else ":memo:",
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
        background: #1B2A4A;
        border: 1px solid #2D3F5E;
        border-radius: 10px;
        padding: 16px 20px;
    }
    [data-testid="stMetricLabel"] {
        font-family: 'Inter', sans-serif;
        font-size: 0.8rem !important;
        font-weight: 500;
        color: #94A3B8 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    [data-testid="stMetricValue"] {
        font-family: 'Inter', sans-serif;
        font-size: 1.8rem !important;
        font-weight: 700;
        color: #FFFFFF !important;
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

    # ── Overview / Proof of Concept Note ──────────────────────────────
    st.markdown(
        "<div style='background: #1B2A4A; border: 1px solid #334155; border-radius: 10px; "
        "padding: 20px 24px; margin-bottom: 1.2rem; line-height: 1.7;'>"
        "<span style='font-family: Inter, sans-serif; font-size: 0.95rem; color: #E2E8F0;'>"
        "RegScope is a fully automated NLP pipeline that downloads public comments from "
        "Regulations.gov and produces structured analysis — stance detection, stakeholder "
        "classification, topic modeling, and form letter deduplication — at zero marginal cost "
        "per comment. The current dataset covers a handful of dockets as a "
        "<strong>proof of concept</strong>. The pipeline can be pointed at any federal rulemaking "
        "docket, and expansion to hundreds of dockets across all federal agencies is underway."
        "</span></div>",
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

        max_comments = max(d["comments"] for d in dockets)
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
                          xaxis_range=[0, max_comments * 1.25],
                          yaxis=dict(autorange="reversed"),
                          margin=dict(l=40, r=60, t=40, b=40))
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
    n_classified = int(comments["stakeholder_type"].notna().sum())
    unique_orgs = comments[comments["organization"].notna() & (comments["organization"] != "")]["organization"].nunique()
    avg_substantiveness = comments["substantiveness_score"].mean()
    n_attachments = data["attach_total"]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Comments", f"{n_comments:,}")
    c2.metric("Classified", f"{n_classified:,}")
    c3.metric("Unique Orgs", f"{unique_orgs:,}")
    c4.metric("Avg Substantiveness", f"{avg_substantiveness:.0f}/100" if pd.notna(avg_substantiveness) else "N/A")
    c5.metric("Attachments", f"{n_attachments:,}")

    st.markdown("")

    # ── Submission Timeline ────────────────────────────────────────────
    dates = comments[comments["posted_date"].notna()]["posted_date"]
    if not dates.empty:
        try:
            date_series = pd.to_datetime(dates, utc=True).dt.date
        except Exception:
            date_series = pd.to_datetime(dates, errors="coerce").dt.date
        date_series = date_series.dropna()
        distinct_dates = date_series.nunique()

        if distinct_dates > 1:
            section_header("Submission Timeline")
            section_note("Daily volume of public comment submissions over the comment period")

            timeline = date_series.value_counts().sort_index().reset_index()
            timeline.columns = ["date", "count"]

            fig = go.Figure(go.Bar(
                x=timeline["date"], y=timeline["count"],
                marker_color=ACCENT_BLUE,
            ))
            plotly_dark_layout(fig, height=280, showlegend=False,
                              xaxis_title="Date", yaxis_title="Comments Submitted",
                              margin=dict(l=40, r=20, t=20, b=40))
            st.plotly_chart(fig, use_container_width=True)

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
                              xaxis_tickangle=-30, yaxis_title="Comments",
                              margin=dict(l=40, r=20, t=40, b=80))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No stance data available for this docket.")

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
        else:
            st.info("No stakeholder data available for this docket.")

    # ── Stance by Stakeholder Cross-tab ────────────────────────────────
    section_header("Stance by Stakeholder Type")
    cross = comments[comments["stance"].notna() & comments["stakeholder_type"].notna()]
    if len(cross) > 10:
        section_note("Showing stance as a percentage within each stakeholder type &mdash; hover for raw counts")

        pivot = cross.groupby(["stakeholder_type", "stance"]).size().reset_index(name="count")
        type_totals = pivot.groupby("stakeholder_type")["count"].transform("sum")
        pivot["pct"] = (pivot["count"] / type_totals * 100).round(1)
        pivot["stance_label"] = pivot["stance"].apply(fmt_stance)
        totals_map = pivot.groupby("stakeholder_type")["count"].sum().to_dict()
        pivot["type_with_n"] = pivot["stakeholder_type"].apply(
            lambda t: f"{fmt_stakeholder(t)} (n={totals_map.get(t, 0):,})"
        )

        fig = px.bar(
            pivot, x="type_with_n", y="pct", color="stance_label",
            color_discrete_map={fmt_stance(s): STANCE_COLORS.get(s, "#DFE6E9") for s in STANCE_ORDER},
            labels={"type_with_n": "Stakeholder Type", "pct": "Percentage", "stance_label": "Stance"},
            barmode="stack",
            custom_data=["count", "stance_label"],
        )
        fig.update_traces(
            hovertemplate="%{customdata[1]}: %{customdata[0]} comments (%{y:.1f}%)<extra></extra>",
        )
        plotly_dark_layout(fig, height=400, xaxis_tickangle=-30,
                          yaxis_title="% of Comments", yaxis_range=[0, 105])
        st.plotly_chart(fig, use_container_width=True)
    else:
        section_note("Insufficient cross-classified data to display this chart (requires &gt;10 comments with both stance and stakeholder type)")

    # ── Topic Analysis ─────────────────────────────────────────────────
    section_header("Topic Analysis")
    if not topics.empty:
        section_note(
            "Comments are clustered into topics using BERTopic (sentence embeddings + HDBSCAN). "
            "Each bar represents a discovered theme labeled by its top keywords."
        )

        topic_display = topics.copy()
        topic_display["display_label"] = topic_display.apply(
            lambda r: clean_topic_label(r.get("label"), r.get("llm_label")),
            axis=1,
        )

        is_outlier = (
            topic_display["display_label"].str.lower().str.contains("miscellaneous|outlier", na=False)
        )
        if "bertopic_id" in topic_display.columns:
            is_outlier = is_outlier | (topic_display["bertopic_id"] == -1)
        outlier_count = int(topic_display.loc[is_outlier, "topic_size"].sum())
        topic_filtered = topic_display[~is_outlier].copy()

        topic_filtered["display_label"] = topic_filtered["display_label"].apply(
            lambda t: t[:70] + "..." if isinstance(t, str) and len(t) > 70 else t
        )

        top_n = topic_filtered.head(15)
        if outlier_count > 0:
            section_note(
                f"{outlier_count:,} comments fell outside identified topic clusters and are excluded from this chart"
            )

        if not top_n.empty:
            max_topic = int(top_n["topic_size"].max())
            fig = go.Figure(go.Bar(
                y=top_n["display_label"],
                x=top_n["topic_size"],
                orientation="h",
                marker_color=ACCENT_BLUE,
                text=top_n["topic_size"],
                textposition="outside",
            ))
            plotly_dark_layout(fig, height=max(300, len(top_n) * 35 + 80),
                              showlegend=False,
                              xaxis_title="Number of Comments",
                              xaxis_range=[0, max_topic * 1.2],
                              yaxis=dict(autorange="reversed"),
                              margin=dict(l=40, r=60, t=20, b=40))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No topic clusters identified for this docket.")
    else:
        section_note("Topic modeling has not been run for this docket.")

    # ── Substantiveness Distribution ───────────────────────────────────
    sub_scores = comments[comments["substantiveness_score"].notna()]["substantiveness_score"]
    section_header("Substantiveness Score Distribution")
    if not sub_scores.empty:
        section_note(
            "Each comment is scored 0&ndash;100 based on length, specificity, citations, "
            "technical language, and uniqueness. Form letters score low; detailed "
            "legal or technical arguments score high."
        )

        total_scored = len(sub_scores)

        all_bins = [
            ("Form Letters\n(0\u201319)", 0, 19, "#FF7675"),
            ("Low\n(20\u201339)", 20, 39, "#FDCB6E"),
            ("Moderate\n(40\u201359)", 40, 59, "#74B9FF"),
            ("High\n(60\u201379)", 60, 79, "#0984E3"),
            ("Highly Substantive\n(80\u2013100)", 80, 100, "#00B894"),
        ]
        bin_data = []
        for label, lo, hi, color in all_bins:
            count = int(((sub_scores >= lo) & (sub_scores <= hi)).sum())
            pct = round(count / total_scored * 100, 1) if total_scored > 0 else 0
            bin_data.append({"label": label, "count": count, "pct": pct, "color": color})

        # Trim empty bars from edges only
        first_nonzero = 0
        last_nonzero = len(bin_data) - 1
        for i, b in enumerate(bin_data):
            if b["count"] > 0:
                first_nonzero = i
                break
        for i in range(len(bin_data) - 1, -1, -1):
            if bin_data[i]["count"] > 0:
                last_nonzero = i
                break
        visible = bin_data[first_nonzero:last_nonzero + 1]

        max_pct = max(b["pct"] for b in visible) if visible else 100
        fig = go.Figure(go.Bar(
            x=[b["label"] for b in visible],
            y=[b["pct"] for b in visible],
            marker_color=[b["color"] for b in visible],
            text=[f"{b['pct']:.0f}%" for b in visible],
            textposition="outside",
            customdata=[[b["count"]] for b in visible],
            hovertemplate="%{x}: %{customdata[0]:,} comments (%{y:.1f}%)<extra></extra>",
        ))
        plotly_dark_layout(fig, height=350, showlegend=False,
                          xaxis_title="", yaxis_title="% of Comments",
                          yaxis_range=[0, min(max_pct * 1.3, 105)],
                          margin=dict(l=40, r=20, t=50, b=40))
        st.plotly_chart(fig, use_container_width=True)
    else:
        section_note("No substantiveness scores available for this docket.")

    # ── Top Substantive Comments (directly under distribution chart) ───
    scored_comments = comments[comments["substantiveness_score"].notna()].copy()
    if not scored_comments.empty:
        top_sub = scored_comments.nlargest(10, "substantiveness_score")

    if not scored_comments.empty and top_sub.iloc[0]["substantiveness_score"] >= 40:
        st.markdown("")
        st.markdown("**Most Substantive Comments**")
        section_note(
            "The highest-scoring comments by substantiveness &mdash; typically detailed legal, "
            "technical, or policy arguments from organizations and expert stakeholders. "
            "Click to read the full comment."
        )

        # Progressive disclosure — show 3 initially
        sub_show_key = f"show_substantive_{selected_docket}"
        if sub_show_key not in st.session_state:
            st.session_state[sub_show_key] = 3

        visible_sub = min(st.session_state[sub_show_key], len(top_sub))

        for i, (_, row) in enumerate(top_sub.head(visible_sub).iterrows()):
            score = int(row["substantiveness_score"])
            org = row["organization"] if pd.notna(row["organization"]) and row["organization"] else None
            name = row["submitter_name"] if pd.notna(row["submitter_name"]) else None
            text = row["full_text"] if pd.notna(row.get("full_text")) else row.get("comment_text", "")
            stakeholder = fmt_stakeholder(row["stakeholder_type"]) if pd.notna(row.get("stakeholder_type")) else None
            stance = fmt_stance(row["stance"]) if pd.notna(row.get("stance")) else None

            # Build attribution
            if org:
                attribution = org
            elif name:
                attribution = name
            else:
                attribution = "Anonymous"

            # Build metadata tags
            tags = []
            if stakeholder and stakeholder != "Unknown":
                tags.append(stakeholder)
            if stance and stance != "Unknown":
                tags.append(stance)
            text_len = len(str(text)) if text else 0
            if text_len > 1000:
                tags.append(f"{text_len:,} chars")

            header = f"#{i + 1} \u2014 {attribution}"

            with st.expander(header, expanded=(i == 0)):
                tag_str = " &bull; ".join(tags)
                if tag_str:
                    st.markdown(
                        f"<span style='color:#94A3B8;font-size:0.8rem'>"
                        f"Score: {score}/100 &bull; {tag_str}</span>",
                        unsafe_allow_html=True,
                    )
                if text:
                    clean = clean_html(str(text))
                    # Show full text in scrollable container
                    st.markdown(
                        f"<div style='background:#1B2A4A;padding:14px;border-radius:8px;"
                        f"font-size:0.85rem;line-height:1.6;max-height:400px;overflow-y:auto;"
                        f"white-space:pre-wrap'>{clean[:3000]}{'...' if len(clean) > 3000 else ''}</div>",
                        unsafe_allow_html=True,
                    )

        if visible_sub < len(top_sub):
            remaining_sub = len(top_sub) - visible_sub
            if st.button(f"Show more comments ({remaining_sub} remaining)", key=f"more_sub_{selected_docket}"):
                st.session_state[sub_show_key] += 3
                st.rerun()

    # ── Duplicate / Form Letter Analysis ───────────────────────────────
    section_header("Duplicate & Form Letter Analysis")
    section_note(
        "Comments are grouped by similarity: exact duplicates share identical text, "
        "near-duplicates are form letters with minor edits (name changes, typos), "
        "and semantic matches convey the same argument in different words."
    )

    multi_groups = dedup_groups[dedup_groups["group_size"] > 1] if not dedup_groups.empty else pd.DataFrame()

    if not dedup_groups.empty and not multi_groups.empty:
        # Charts side by side at top
        col1, col2 = st.columns(2)

        with col1:
            type_counts = dedup_groups["group_type"].value_counts()
            type_labels = {"exact": "Exact\nDuplicates", "near": "Near\nDuplicates", "semantic": "Semantic\nMatches"}
            fig = go.Figure(go.Bar(
                x=[type_labels.get(t, t) for t in type_counts.index],
                y=type_counts.values,
                marker_color=[PALETTE[0], PALETTE[1], PALETTE[2]][:len(type_counts)],
                text=[f"{v:,}" for v in type_counts.values],
                textposition="outside",
            ))
            max_type = int(max(type_counts.values))
            plotly_dark_layout(fig, height=320, showlegend=False,
                              yaxis_title="Number of Groups",
                              yaxis_range=[0, max_type * 1.3],
                              margin=dict(l=40, r=20, t=50, b=40))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Uniqueness breakdown
            total_in_groups = int(multi_groups["group_size"].sum())
            unique_comments = n_comments - total_in_groups
            fig2 = px.pie(
                pd.DataFrame([
                    {"type": "Unique Comments", "count": max(unique_comments, 0)},
                    {"type": "Duplicated Comments", "count": total_in_groups},
                ]),
                values="count", names="type",
                color_discrete_sequence=["#00B894", "#FF7675"],
                hole=0.5,
            )
            plotly_dark_layout(fig2, height=320, showlegend=True,
                              legend=dict(font=dict(size=14), orientation="h",
                                          yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
                              margin=dict(l=10, r=10, t=40, b=50))
            fig2.update_traces(textposition="inside", textinfo="percent", textfont_size=13)
            st.plotly_chart(fig2, use_container_width=True)

        total_groups = len(multi_groups)
        st.markdown(
            f"**{total_groups}** form letter campaigns produced **{total_in_groups:,}** "
            f"duplicate comments out of **{n_comments:,}** total"
        )

        # Top Form Letter Campaigns — below charts, progressive disclosure
        st.markdown("")
        st.markdown("**Top Form Letter Campaigns**")

        campaigns = aggregate_campaigns(dedup_groups, comments)
        display_campaigns = [c for c in campaigns if c["template"]]
        if not display_campaigns:
            display_campaigns = campaigns

        type_color = {"exact": "#0984E3", "near": "#6C5CE7", "semantic": "#00B894"}
        type_badge_label = {"exact": "Exact", "near": "Near-duplicate", "semantic": "Semantic"}

        # Show 3 initially, then 3 more per click
        show_key = f"show_campaigns_{selected_docket}"
        if show_key not in st.session_state:
            st.session_state[show_key] = 3

        visible_count = min(st.session_state[show_key], len(display_campaigns))

        for i in range(visible_count):
            campaign = display_campaigns[i]
            group_type = campaign["group_type"]
            total = campaign["total_copies"]
            template = campaign["template"]
            sub_groups = campaign["sub_groups"]
            badge = type_badge_label.get(group_type, group_type)

            if template:
                preview = clean_html(template)[:90]
                if len(clean_html(template)) > 90:
                    preview += "..."
            else:
                preview = "(no template text)"

            merged_note = f" ({sub_groups} variant groups merged)" if sub_groups > 1 else ""
            header = f"**{total:,} copies** \u2014 {badge}{merged_note} \u2014 {preview}"
            with st.expander(header, expanded=(i == 0)):
                ec1, ec2 = st.columns([1, 4])
                with ec1:
                    st.metric("Total Copies", f"{total:,}")
                    if sub_groups > 1:
                        st.markdown(f"*{sub_groups} variant groups*")
                    st.markdown(
                        f"<span style='background:{type_color.get(group_type, '#555')};color:white;"
                        f"padding:2px 8px;border-radius:4px;font-size:0.75rem'>{badge}</span>",
                        unsafe_allow_html=True,
                    )
                with ec2:
                    if template:
                        clean = clean_html(template)
                        st.markdown(
                            f"<div style='background:#1B2A4A;padding:14px;border-radius:8px;"
                            f"font-size:0.85rem;line-height:1.6;max-height:400px;overflow-y:auto;"
                            f"white-space:pre-wrap'>"
                            f"{clean[:3000]}{'...' if len(clean) > 3000 else ''}</div>",
                            unsafe_allow_html=True,
                        )

        if visible_count < len(display_campaigns):
            remaining = len(display_campaigns) - visible_count
            if st.button(f"Show more campaigns ({remaining} remaining)", key=f"more_{selected_docket}"):
                st.session_state[show_key] += 3
                st.rerun()
    else:
        st.info("No duplicate or form letter groups detected for this docket.")

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
    display_df = display_df.sort_values("substantiveness_score", ascending=False, na_position="last")

    st.write(f"Showing **{len(display_df):,}** comments")
    st.dataframe(
        display_df.head(500),
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
        "<div style='text-align: center; padding: 16px 0;'>"
        "<p style='font-family: Inter, sans-serif; font-size: 1.3rem; font-weight: 600; "
        "color: #FFFFFF; margin-bottom: 6px;'>Built by Nathan Goldberg</p>"
        "<p style='font-family: Inter, sans-serif; font-size: 1rem; margin-top: 0; margin-bottom: 16px;'>"
        "<a href='mailto:nathanmauricegoldberg@gmail.com' style='color: #0984E3; text-decoration: none;'>nathanmauricegoldberg@gmail.com</a>"
        " &nbsp;&bull;&nbsp; "
        "<a href='https://www.linkedin.com/in/nathan-goldberg-62a44522a' target='_blank' "
        "style='color: #0984E3; text-decoration: none;'>LinkedIn</a></p>"
        "<p style='font-family: Inter, sans-serif; font-size: 0.8rem; color: #94A3B8; margin-top: 0;'>"
        "RegScope &bull; Federal Rulemaking Comment Analyzer &bull; "
        "Data sourced from Regulations.gov API &bull; "
        "Zero-cost NLP pipeline: embeddings &rarr; dedup &rarr; classify &rarr; topic model</p>"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
