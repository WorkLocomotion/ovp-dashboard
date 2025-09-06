# ovp_dashboard.py
# Occupational Value (OVP) Dashboard — Work Locomotion

from __future__ import annotations

import re
from io import BytesIO
from pathlib import Path
from typing import Union, IO

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# ---------------------------
# CONFIG & BRANDING
# ---------------------------
st.set_page_config(
    page_title="Work Locomotion — OVP Dashboard",
    page_icon="MAIN LOGO.png",
    layout="wide",
)

HEADER_TITLE = "Work Locomotion — Occupational Value (OVP) Dashboard"
ANCHOR_HIGH_SOC = "11-1011.00"  # Chief Executives
ANCHOR_LOW_SOC  = "45-2041.00"  # Graders & Sorters, Agricultural Products
VALUE_DIMENSIONS = [
    "Achievement",
    "Independence",
    "Recognition",
    "Relationships",
    "Support",
    "Working Conditions",
]

# Use repo-relative paths (works locally and on Streamlit Cloud)
ROOT = Path(__file__).parent
DEFAULT_WV_PATH  = ROOT / "data" / "onet_work_values.xlsx"
DEFAULT_IDX_PATH = ROOT / "data" / "ovp_title_index.xlsx"

# Altair: avoid row-limit issues
alt.data_transformers.disable_max_rows()

# ---------------------------
# UTILITIES
# ---------------------------
def norm_text(s) -> str:
    if pd.isna(s):
        return ""
    s = str(s).lower()
    s = re.sub(r"&", " and ", s)
    s = re.sub(r"[\/\-]", " ", s)
    s = re.sub(r"[^a-z0-9 ]+", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


@st.cache_data(show_spinner=False)
def load_work_values(source: Union[Path, str, IO[bytes], BytesIO]) -> pd.DataFrame:
    """
    Read O*NET Work Values (sheet 'Work Values') and return pivot:
    columns: ['soc','title', six value dimensions..., 'average_ovp']
    """
    df = pd.read_excel(source, sheet_name="Work Values", engine="openpyxl")

    score_col = "OVP Score" if "OVP Score" in df.columns else "Data Value"
    df = df[df["Element Name"].isin(VALUE_DIMENSIONS)].copy()
    df[score_col] = pd.to_numeric(df[score_col], errors="coerce")

    pvt = (
        df.pivot_table(
            index=["O*NET-SOC Code", "Title"],
            columns="Element Name",
            values=score_col,
            aggfunc="mean",
        )
        .reset_index()
        .rename(columns={"O*NET-SOC Code": "soc", "Title": "title"})
    )

    # ensure all 6 columns exist
    for col in VALUE_DIMENSIONS:
        if col not in pvt.columns:
            pvt[col] = np.nan

    pvt["average_ovp"] = pvt[VALUE_DIMENSIONS].mean(axis=1)
    pvt = pvt.sort_values(["title", "soc"]).reset_index(drop=True)
    return pvt


@st.cache_data(show_spinner=False)
def load_title_index(
    source: Union[Path, str, IO[bytes], BytesIO],
    fallback_titles: pd.DataFrame,
) -> pd.DataFrame:
    """
    Expected columns in index file: soc, title, search_title
    Falls back to canonical titles if file missing or malformed.
    """
    idx = None
    try:
        if isinstance(source, (Path, str)) and not Path(source).exists():
            idx = None
        else:
            idx = pd.read_excel(source, dtype={"soc": "string"}, engine="openpyxl")
            idx.columns = [str(c).strip().lower() for c in idx.columns]
            need = {"soc", "title", "search_title"}
            if not need.issubset(set(idx.columns)):
                st.warning("ovp_title_index.xlsx missing expected columns; falling back to primary titles.")
                idx = None
    except Exception:
        idx = None

    if idx is None:
        idx = fallback_titles[["soc", "title"]].copy()
        idx["search_title"] = idx["title"]

    # normalize and clean
    idx["search_title"] = idx["search_title"].astype(object)
    idx["norm_search_title"] = idx["search_title"].map(norm_text)
    idx = idx[idx["norm_search_title"] != ""]
    idx = idx.drop_duplicates(subset=["soc", "search_title"])

    return idx.reset_index(drop=True)


def fuzzy_candidates(
    query: str, index_df: pd.DataFrame, pvt_df: pd.DataFrame, top_n: int = 20
) -> pd.DataFrame:
    qn = norm_text(query)
    if not qn:
        return pd.DataFrame(columns=["soc", "title", "search_title", "method", "score", "composite"])

    stop = {"and", "of", "the", "a", "an"}
    q_tokens = [t for t in qn.split() if t not in stop]
    q_set = set(q_tokens)

    # contains
    contains_mask = index_df["norm_search_title"].str.contains(qn, na=False)
    contains_df = index_df[contains_mask].copy()
    contains_df["method"] = "contains"
    contains_df["score"] = 1.0

    # fuzzy (optional – rapidfuzz may not be installed)
    try:
        from rapidfuzz import fuzz
        fuzz_df = index_df[~contains_mask].copy()
        fuzz_df["score"] = fuzz_df["norm_search_title"].map(lambda s: fuzz.partial_ratio(qn, s) / 100.0)
        fuzz_df = fuzz_df[fuzz_df["score"] >= 0.65]
        fuzz_df["method"] = "fuzzy"
    except Exception:
        fuzz_df = pd.DataFrame(columns=index_df.columns.tolist() + ["method", "score"])

    out = pd.concat([contains_df, fuzz_df], ignore_index=True)
    if out.empty:
        return out

    # join canonical title for display
    out = out.merge(
        pvt_df[["soc", "title"]].drop_duplicates("soc"),
        on="soc",
        how="left",
        suffixes=("", "_canon"),
    )
    out["title"] = out["title_canon"].fillna(out["title"])
    out.drop(columns=[c for c in out.columns if c.endswith("_canon")], inplace=True)

    # features
    out["soc_major"] = out["soc"].str.slice(0, 2)
    out["title_norm"] = out["title"].map(norm_text)

    def overlap(s: str) -> float:
        sset = set(s.split())
        return (len(q_set & sset) / max(1, len(q_set)))

    out["overlap"] = out["norm_search_title"].map(overlap)

    # domain boosts
    out["boost"] = 0.0
    if "construction" in q_set:
        out.loc[out["title_norm"].str.contains("construction"), "boost"] += 0.6
    out.loc[out["soc_major"].isin(["47", "49", "51"]), "boost"] += 0.15
    if "project" in q_set:
        out.loc[out["soc"].isin(["11-9021.00", "13-1082.00"]), "boost"] += 0.8
        if "engineering" not in q_set:
            out.loc[out["title_norm"].str.contains(r"\bengineering\b"), "boost"] -= 0.25

    out["base"] = np.where(out["method"].eq("contains"), 1.0, out["score"])
    out["composite"] = 0.60 * out["base"] + 0.25 * out["overlap"] + 0.15 * out["boost"]

    out = out.sort_values(["composite", "base", "overlap", "title"], ascending=[False, False, False, True])
    return out[["soc", "title", "search_title", "method", "score", "composite"]].head(top_n).reset_index(drop=True)


def nearest_by_average(target: float, pvt_df: pd.DataFrame):
    """Return the row nearest to target average_ovp (ties break by title A→Z)."""
    df = pvt_df.copy()
    df["diff"] = (df["average_ovp"] - target).abs()
    df = df.sort_values(["diff", "title"]).reset_index(drop=True)
    return df.iloc[0][["soc", "title", "average_ovp"]]


def ovp_row_for_soc(soc: str, pvt_df: pd.DataFrame):
    row = pvt_df[pvt_df["soc"] == soc]
    if row.empty:
        return None
    return row.iloc[0]


def five_bar_chart(values_df: pd.DataFrame):
    base = alt.Chart(values_df).mark_bar().encode(
        x=alt.X("label:N", title=None),
        y=alt.Y("average_ovp:Q", title="Average OVP (0–7)", scale=alt.Scale(domain=[0, 7])),
        color=alt.Color(
            "kind:N",
            scale=alt.Scale(domain=["participant", "other"], range=["#B8860B", "#A9A9A9"]),
            legend=None,
        ),
        tooltip=["label", "average_ovp"],
    ).properties(height=280)
    text = base.mark_text(dy=-6).encode(text=alt.Text("average_ovp:Q", format=".2f"))
    return base + text


# ---------------------------
# APP UI
# ---------------------------
st.markdown(f"## {HEADER_TITLE}")
st.caption("Data processed in your session. Upload your own files or use the defaults packaged with the app.")

with st.sidebar:
    # Display your logo at the top of the sidebar
    st.image("Logo Main Yellow Clear.png", width=120)   # adjust width to fit nicely

    st.markdown("### How to use")
    st.write(
        "Enter a job title, select the best O*NET match, and compare its OVP to benchmark roles "
        "(Chief Executives, Nearest Mid-High, Participant (you), Nearest Mid-Low, Graders & Sorters)."
    )
    st.markdown(
        "[Background](https://worklocomotion.substack.com/) · "
        "[Contact](https://www.linkedin.com/in/manley-osbak/)"
    )

with st.expander("Data sources (optional uploads)", expanded=False):
    st.write("Use defaults from **/data** or upload your own Excel files (same schema).")
    uploaded_work_values = st.file_uploader(
        "O*NET Work Values (Excel — sheet 'Work Values')", type=["xlsx"], key="wv"
    )
    uploaded_title_index = st.file_uploader(
        "OVP Title Index (optional, Excel)", type=["xlsx"], key="idx"
    )

# Load data (uploaded or packaged)
if uploaded_work_values is not None:
    wv_df = load_work_values(BytesIO(uploaded_work_values.getvalue()))
else:
    if not DEFAULT_WV_PATH.exists():
        st.error("Missing data/onet_work_values.xlsx. Upload a file or add it to the repo.")
        st.stop()
    wv_df = load_work_values(DEFAULT_WV_PATH)

if uploaded_title_index is not None:
    idx_df = load_title_index(BytesIO(uploaded_title_index.getvalue()), wv_df)
else:
    idx_df = load_title_index(DEFAULT_IDX_PATH, wv_df)

# Validate anchors
missing_anchors = []
if wv_df[wv_df["soc"] == ANCHOR_HIGH_SOC].empty:
    missing_anchors.append(ANCHOR_HIGH_SOC)
if wv_df[wv_df["soc"] == ANCHOR_LOW_SOC].empty:
    missing_anchors.append(ANCHOR_LOW_SOC)
if missing_anchors:
    st.warning(f"Anchor SOC not found in data: {', '.join(missing_anchors)}")

# ---------------------------
# MATCHING UI
# ---------------------------
st.markdown("### 1) Enter your job title")
query = st.text_input("Type a job title (e.g., Pipefitter, Contracts Administrator)").strip()

cands = pd.DataFrame()
if query:
    cands = fuzzy_candidates(query, idx_df, wv_df, top_n=10)
    if cands.empty:
        st.info("No matches. Try a simpler or alternate title (e.g., 'Steamfitter/pipefitter' → 'Pipefitter').")
    else:
        st.markdown("#### Suggested matches")
        st.dataframe(
            cands.rename(
                columns={
                    "soc": "SOC",
                    "title": "O*NET Title",
                    "search_title": "Matched Term",
                    "method": "Method",
                    "score": "Score",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

# Selection
selected_soc = None
if not cands.empty:
    soc_choices = [f"{r.soc} — {r.title}" for _, r in cands.iterrows()]
    choice = st.radio("Pick the best match:", options=soc_choices, index=0, horizontal=False)
    if choice:
        selected_soc = choice.split(" — ")[0]

# ---------------------------
# RESULTS
# ---------------------------
if selected_soc:
    row = ovp_row_for_soc(selected_soc, wv_df)
    if row is None:
        st.error("Unexpected: selected SOC not found.")
        st.stop()

    st.markdown("### 2) OVP results for your occupation")
    left, right = st.columns([1, 1])

    with left:
        st.write(f"**SOC:** {row.soc}")
        st.write(f"**O*NET Title:** {row.title}")
        st.metric("Average OVP", f"{row.average_ovp:.2f}")
        dims_df = pd.DataFrame({"Dimension": VALUE_DIMENSIONS, "Score": [float(row[d]) for d in VALUE_DIMENSIONS]})
        st.dataframe(dims_df, hide_index=True, use_container_width=True)

    with right:
        high_row = ovp_row_for_soc(ANCHOR_HIGH_SOC, wv_df)
        low_row  = ovp_row_for_soc(ANCHOR_LOW_SOC,  wv_df)

        part_avg = float(row.average_ovp)
        high_avg = float(high_row.average_ovp) if high_row is not None else np.nan
        low_avg  = float(low_row.average_ovp)  if low_row  is not None else np.nan

        # midpoints
        mid_high_target = np.nanmean([part_avg, high_avg])
        mid_low_target  = np.nanmean([part_avg, low_avg])

        mid_high_occ = nearest_by_average(mid_high_target, wv_df) if np.isfinite(mid_high_target) else None
        mid_low_occ  = nearest_by_average(mid_low_target,  wv_df) if np.isfinite(mid_low_target)  else None

        chart_rows = []
        if high_row is not None:
            chart_rows.append({"label": "Chief Executives", "average_ovp": round(high_avg, 2), "kind": "other"})
        if mid_high_occ is not None:
            chart_rows.append({"label": "Nearest Mid-High", "average_ovp": round(float(mid_high_occ.average_ovp), 2), "kind": "other"})
        chart_rows.append({"label": "Participant", "average_ovp": round(part_avg, 2), "kind": "participant"})
        if mid_low_occ is not None:
            chart_rows.append({"label": "Nearest Mid-Low", "average_ovp": round(float(mid_low_occ.average_ovp), 2), "kind": "other"})
        if low_row is not None:
            chart_rows.append({"label": "Graders & Sorters", "average_ovp": round(low_avg, 2), "kind": "other"})

        chart_df = pd.DataFrame(chart_rows)
        st.altair_chart(five_bar_chart(chart_df), use_container_width=True)

    with st.expander("Midpoint occupations (details)"):
        cols = st.columns(2)
        with cols[0]:
            if high_row is not None and mid_high_occ is not None:
                st.write("**Nearest to Mid-High**")
                st.write(f"SOC: {mid_high_occ.soc}")
                st.write(f"Title: {mid_high_occ.title}")
                st.write(f"Average OVP: {mid_high_occ.average_ovp:.2f}")
        with cols[1]:
            if low_row is not None and mid_low_occ is not None:
                st.write("**Nearest to Mid-Low**")
                st.write(f"SOC: {mid_low_occ.soc}")
                st.write(f"Title: {mid_low_occ.title}")
                st.write(f"Average OVP: {mid_low_occ.average_ovp:.2f}")

st.markdown("---")
st.caption(
    "Uses O*NET Work Values (TWA): Achievement, Independence, Recognition, Relationships, Support, Working Conditions."
)

# Preference: closing print lines (will show in terminal logs if run directly)
if __name__ == "__main__":
    print("")
    print("WORK LOCOMOTION: Make Potential Actual")
    print("")



