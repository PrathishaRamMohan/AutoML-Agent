"""Design system for the Streamlit app: CSS, HTML components and Altair chart styling."""
from __future__ import annotations

import html
import re

import altair as alt
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------- tokens
BLUE = "#2a78d6"        # primary / selected series
BLUE_DARK = "#1c5cab"
GRAY = "#b9b8b1"        # non-selected series
BASELINE = "#dcdbd4"    # baseline bar
INK = "#1b1b1a"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e8e7e1"
GOOD, WARN, CRIT = "#0ca30c", "#fab219", "#d03b3b"
FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif'

PRETTY = {
    "logistic_regression": "Logistic Regression", "random_forest": "Random Forest",
    "xgboost": "XGBoost", "ridge": "Ridge Regression", "baseline": "Baseline (dummy)",
    "roc_auc": "ROC-AUC", "f1_macro": "F1 (macro)", "r2": "R²", "f1": "F1",
    "accuracy": "Accuracy", "balanced_accuracy": "Balanced accuracy", "precision": "Precision",
    "recall": "Recall", "rmse": "RMSE", "mae": "MAE",
}


def pretty(x) -> str:
    return PRETTY.get(str(x), str(x).replace("_", " ").title())


def esc(x) -> str:
    return html.escape(str(x))


# ---------------------------------------------------------------- global CSS
CSS = f"""
<style>
html, body, [class*="css"], .stApp, .stMarkdown, button, input, textarea, select {{
    font-family: {FONT} !important;
}}
#MainMenu, footer, [data-testid="stDecoration"] {{ visibility: hidden; height: 0; }}
.block-container {{ padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1320px; }}
h1, h2, h3, h4 {{ letter-spacing: -0.01em; color: {INK}; }}
h3 {{ font-size: 1.05rem !important; font-weight: 600 !important; margin-top: .4rem; }}

/* sidebar */
section[data-testid="stSidebar"] {{ border-right: 1px solid {GRID}; }}
section[data-testid="stSidebar"] .block-container {{ padding-top: 1.2rem; }}
.side-brand {{ display:flex; align-items:center; gap:.6rem; margin-bottom:.2rem; }}
.side-logo {{ width:34px; height:34px; border-radius:9px; background:{BLUE}; color:#fff;
    display:flex; align-items:center; justify-content:center; font-weight:700; font-size:.95rem; }}
.side-title {{ font-weight:700; font-size:1.05rem; color:{INK}; line-height:1.1; }}
.side-sub {{ font-size:.75rem; color:{MUTED}; }}
.side-label {{ font-size:.7rem; font-weight:600; letter-spacing:.08em; text-transform:uppercase;
    color:{MUTED}; margin:1.1rem 0 .3rem; }}

/* page header */
.page-head {{ display:flex; justify-content:space-between; align-items:flex-start; gap:1rem;
    flex-wrap:wrap; margin-bottom:1rem; }}
.page-title {{ font-size:1.6rem; font-weight:700; color:{INK}; margin:0; line-height:1.2; }}
.page-sub {{ color:{INK_2}; font-size:.9rem; margin-top:.25rem; }}
.page-sub code {{ background:#eeede8; padding:1px 6px; border-radius:5px; font-size:.82rem; color:{INK}; }}

/* status pill */
.pill {{ display:inline-flex; align-items:center; gap:.4rem; padding:.3rem .75rem; border-radius:999px;
    font-size:.8rem; font-weight:600; border:1px solid; white-space:nowrap; }}
.pill .dot {{ width:8px; height:8px; border-radius:50%; }}
.pill-good {{ color:#006300; background:#eaf6ea; border-color:#bfe3bf; }}
.pill-good .dot {{ background:{GOOD}; }}
.pill-warn {{ color:#7a5200; background:#fff6e0; border-color:#f5dca0; }}
.pill-warn .dot {{ background:{WARN}; }}
.pill-crit {{ color:#9b1f1f; background:#fdecec; border-color:#f3c3c3; }}
.pill-crit .dot {{ background:{CRIT}; }}
.pill-info {{ color:{BLUE_DARK}; background:#eaf2fc; border-color:#c7dcf6; }}
.pill-info .dot {{ background:{BLUE}; }}

/* pipeline stepper */
.stepper {{ display:grid; grid-template-columns:repeat(7, minmax(0, 1fr)); background:#fff; border:1px solid {GRID};
    border-radius:12px; padding:.85rem .5rem .7rem; margin-bottom:1rem; }}
.step {{ position:relative; display:flex; flex-direction:column; align-items:center; text-align:center; gap:.3rem; min-width:0; }}
.step:not(:first-child)::before {{ content:""; position:absolute; top:12px; right:calc(50% + 16px); width:calc(100% - 32px);
    height:2px; background:{GRID}; border-radius:2px; }}
.step-done::before, .step-active::before {{ background:{BLUE} !important; }}
.step-ico {{ width:26px; height:26px; border-radius:50%; flex:none; display:flex; align-items:center;
    justify-content:center; font-size:.74rem; font-weight:700; position:relative; z-index:1; }}
.step-done .step-ico {{ background:{BLUE}; color:#fff; }}
.step-active .step-ico {{ background:#fff; color:{BLUE}; border:2px solid {BLUE}; box-shadow:0 0 0 4px #eaf2fc; }}
.step-todo .step-ico {{ background:#f0efeb; color:{MUTED}; }}
.step-name {{ font-size:.78rem; font-weight:600; color:{INK}; line-height:1.15; overflow-wrap:anywhere; }}
.step-todo .step-name {{ color:{MUTED}; font-weight:500; }}
.step-note {{ font-size:.68rem; color:{BLUE_DARK}; font-weight:500; }}

/* KPI tiles */
.kpis {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(175px, 1fr)); gap:.75rem; margin-bottom:1rem; }}
.kpi {{ background:#fff; border:1px solid {GRID}; border-radius:12px; padding:.9rem 1rem; min-width:0; }}
.kpi-label {{ font-size:.72rem; font-weight:600; letter-spacing:.06em; text-transform:uppercase; color:{MUTED}; }}
.kpi-value {{ font-size:1.65rem; font-weight:700; color:{INK}; margin-top:.3rem; line-height:1.15;
    overflow-wrap:anywhere; }}
.kpi-value.text {{ font-size:1.15rem; }}
.kpi-foot {{ font-size:.78rem; color:{INK_2}; margin-top:.3rem; }}
.up {{ color:#006300; font-weight:600; }}
.down {{ color:#9b1f1f; font-weight:600; }}

/* callouts */
.callout {{ border-radius:10px; padding:.7rem .9rem; margin:.4rem 0; font-size:.86rem; line-height:1.45;
    border:1px solid; display:flex; gap:.6rem; align-items:flex-start; }}
.callout b {{ color:inherit; }}
.callout-warn {{ background:#fff8e6; border-color:#f5dca0; color:#5c3d00; }}
.callout-info {{ background:#eef5fd; border-color:#c7dcf6; color:#123f75; }}
.callout-good {{ background:#edf8ed; border-color:#bfe3bf; color:#0b4d0b; }}
.callout-crit {{ background:#fdeeee; border-color:#f3c3c3; color:#7a1818; }}

/* cards / section labels */
.card-title {{ font-size:.95rem; font-weight:650; color:{INK}; margin-bottom:.1rem; }}
.card-sub {{ font-size:.8rem; color:{MUTED}; margin-bottom:.6rem; }}
[data-testid="stVerticalBlockBorderWrapper"] {{ background:#fff; border-radius:12px !important; border-color:{GRID} !important; }}

/* agent timeline */
.tl-item {{ display:flex; gap:.8rem; padding:.8rem 0; border-bottom:1px solid {GRID}; }}
.tl-ico {{ width:34px; height:34px; border-radius:9px; background:#eef5fd; flex:none; display:flex;
    align-items:center; justify-content:center; font-size:1rem; }}
.tl-ico.sup {{ background:#f0efeb; }}
.tl-head {{ display:flex; gap:.5rem; align-items:baseline; flex-wrap:wrap; }}
.tl-agent {{ font-weight:650; font-size:.9rem; color:{INK}; }}
.tl-ts {{ font-size:.72rem; color:{MUTED}; }}
.tl-body {{ font-size:.85rem; color:{INK_2}; white-space:pre-wrap; line-height:1.5; margin-top:.2rem;
    overflow-wrap:anywhere; }}

/* welcome */
.hero {{ background:#fff; border:1px solid {GRID}; border-radius:14px; padding:1.6rem 1.8rem; margin-bottom:1rem; }}
.hero h2 {{ font-size:1.45rem; margin:0 0 .35rem; }}
.hero p {{ color:{INK_2}; margin:0; font-size:.95rem; max-width:760px; }}
.agents {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(210px, 1fr)); gap:.7rem; margin-top:1.2rem; }}
.agent-card {{ border:1px solid {GRID}; border-radius:10px; padding:.75rem .85rem; background:#fcfcfb; }}
.agent-card .n {{ font-weight:650; font-size:.88rem; color:{INK}; }}
.agent-card .d {{ font-size:.78rem; color:{INK_2}; margin-top:.2rem; line-height:1.4; }}

/* tabs */
.stTabs [data-baseweb="tab-list"] {{ gap:.25rem; border-bottom:1px solid {GRID}; }}
.stTabs [data-baseweb="tab"] {{ padding:.55rem .9rem; font-weight:500; }}
.stTabs [aria-selected="true"] {{ font-weight:650; }}
[data-testid="stDataFrame"] {{ border:1px solid {GRID}; border-radius:10px; }}
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {{ border-radius:8px; font-weight:600; }}
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def html_block(markup: str):
    st.markdown(markup, unsafe_allow_html=True)


# ---------------------------------------------------------------- components
def pill(text: str, kind: str = "info") -> str:
    return f'<span class="pill pill-{kind}"><span class="dot"></span>{esc(text)}</span>'


STATUS_PILL = {
    "awaiting_approval": ("Awaiting your review", "warn"),
    "approved": ("Approved & exported", "good"),
    "rejected": ("Rejected", "crit"),
    "running": ("Agents running", "info"),
}


def page_header(title: str, subtitle_html: str, status: str | None = None):
    right = pill(*STATUS_PILL.get(status, (status or "", "info"))) if status else ""
    html_block(
        f'<div class="page-head"><div><div class="page-title">{esc(title)}</div>'
        f'<div class="page-sub">{subtitle_html}</div></div><div>{right}</div></div>'
    )


STEPS = [
    ("profile", "Profile"), ("fe_plan", "Features"), ("leaderboard", "Model select"),
    ("tuning", "Tune"), ("evaluation", "Evaluate"), ("approval", "Review"), ("artifacts", "Export"),
]


def stepper(state: dict, waiting_review: bool):
    items, active_set = [], False
    for i, (key, label) in enumerate(STEPS, 1):
        done = bool(state.get(key))
        if key == "approval" and state.get("status") in ("approved", "rejected"):
            done = True
        note = ""
        if key == "approval" and waiting_review:
            cls, active_set, note = "step-active", True, "waiting for you"
        elif done:
            cls = "step-done"
        elif not active_set and state.get("status") == "running":
            cls, active_set = "step-active", True
        else:
            cls = "step-todo"
        if key == "approval" and state.get("status") == "rejected":
            note = "rejected"
        if key == "tuning" and state.get("tuning") and state.get("tuning_mode") == "regularized":
            note = "regularized"
        ico = "✓" if cls == "step-done" else str(i)
        note_html = f'<div class="step-note">{esc(note)}</div>' if note else ""
        items.append(
            f'<div class="step {cls}"><div class="step-ico">{ico}</div>'
            f'<div><div class="step-name">{esc(label)}</div>{note_html}</div></div>'
        )
    html_block(f'<div class="stepper">{"".join(items)}</div>')


def kpi(label: str, value, foot: str = "", text: bool = False) -> str:
    cls = "kpi-value text" if text else "kpi-value"
    foot_html = f'<div class="kpi-foot">{foot}</div>' if foot else ""
    return f'<div class="kpi"><div class="kpi-label">{esc(label)}</div><div class="{cls}">{esc(value)}</div>{foot_html}</div>'


def kpi_row(tiles: list[str]):
    html_block(f'<div class="kpis">{"".join(tiles)}</div>')


def callout(text_html: str, kind: str = "info", icon: str = "ℹ️"):
    html_block(f'<div class="callout callout-{kind}"><span>{icon}</span><div>{text_html}</div></div>')


def card_title(title: str, sub: str = ""):
    html_block(f'<div class="card-title">{esc(title)}</div>' + (f'<div class="card-sub">{esc(sub)}</div>' if sub else ""))


# ---------------------------------------------------------------- charts
def _style(chart: alt.Chart, height: int) -> alt.Chart:
    return (
        chart.properties(height=height, background="transparent")
        .configure(font=FONT)
        .configure_view(strokeWidth=0)
        .configure_axis(labelColor=MUTED, titleColor=INK_2, labelFontSize=11, titleFontSize=11,
                        titleFontWeight=500, gridColor=GRID, domainColor=GRID, tickColor=GRID)
        .configure_legend(labelColor=INK_2, titleColor=INK_2, labelFontSize=11, orient="top")
    )


def leaderboard_chart(board: list[dict], selected: str) -> alt.Chart:
    df = pd.DataFrame(board)
    df["Model"] = df["model"].map(pretty)
    df["role"] = df["model"].map(lambda m: "Selected" if m == selected else ("Baseline" if m == "baseline" else "Other"))
    df["lo"], df["hi"] = df.cv_mean - df.cv_std, df.cv_mean + df.cv_std
    metric = pretty(df["metric"].iloc[0])
    order = df.sort_values("cv_mean", ascending=False)["Model"].tolist()
    y = alt.Y("Model:N", sort=order, title=None, axis=alt.Axis(labelLimit=200, labelFontSize=12, labelColor=INK_2))
    color = alt.Color("role:N", scale=alt.Scale(domain=["Selected", "Other", "Baseline"], range=[BLUE, GRAY, BASELINE]),
                      legend=alt.Legend(title=None))
    tip = [alt.Tooltip("Model:N"), alt.Tooltip("cv_mean:Q", title=f"CV {metric}", format=".4f"),
           alt.Tooltip("cv_std:Q", title="± std", format=".4f"), alt.Tooltip("fit_seconds:Q", title="Fit time (s)")]
    bars = alt.Chart(df).mark_bar(cornerRadiusEnd=4, height=22).encode(
        x=alt.X("cv_mean:Q", title=f"Mean CV {metric} (5 folds)", scale=alt.Scale(zero=True)), y=y, color=color, tooltip=tip)
    err = alt.Chart(df).mark_rule(color=INK_2, strokeWidth=1.5).encode(x="lo:Q", x2="hi:Q", y=y)
    labels = alt.Chart(df).mark_text(align="left", dx=6, color=INK_2, fontSize=11).encode(
        x="hi:Q", y=y, text=alt.Text("cv_mean:Q", format=".3f"))
    return _style(bars + err + labels, 60 * len(df) + 20)


def tuning_chart(tuning: dict) -> alt.Chart:
    df = pd.DataFrame({"Trial": range(1, len(tuning["trial_scores"]) + 1),
                       "Trial score": tuning["trial_scores"], "Best so far": tuning["best_so_far"]})
    metric = pretty(tuning["metric"])
    base = alt.Chart(df).encode(x=alt.X("Trial:Q", axis=alt.Axis(tickMinStep=1, format="d")))
    dots = base.mark_circle(size=64, color=GRAY, opacity=0.9).encode(
        y=alt.Y("Trial score:Q", title=f"CV {metric}", scale=alt.Scale(zero=False)),
        tooltip=["Trial:Q", alt.Tooltip("Trial score:Q", format=".4f"), alt.Tooltip("Best so far:Q", format=".4f")])
    line = base.mark_line(color=BLUE, strokeWidth=2, interpolate="step-after").encode(y="Best so far:Q")
    return _style(dots + line, 260)


def importance_chart(importances: list[dict], metric: str) -> alt.Chart:
    df = pd.DataFrame(importances)
    df = df[df.importance > -1]  # keep all; negatives are shown as ~0 impact
    order = df.sort_values("importance", ascending=False)["feature"].tolist()
    bars = alt.Chart(df).mark_bar(color=BLUE, cornerRadiusEnd=4, height=16).encode(
        x=alt.X("importance:Q", title=f"Drop in test {pretty(metric)} when feature is shuffled"),
        y=alt.Y("feature:N", sort=order, title=None, axis=alt.Axis(labelLimit=220, labelColor=INK_2, labelFontSize=12)),
        tooltip=[alt.Tooltip("feature:N", title="Feature"), alt.Tooltip("importance:Q", format=".4f"),
                 alt.Tooltip("std:Q", title="± std", format=".4f")])
    return _style(bars, max(160, 30 * len(df)))


def confusion_chart(cm: list[list[int]], labels: list) -> alt.Chart:
    rows = []
    for i, row in enumerate(cm):
        total = sum(row) or 1
        for j, v in enumerate(row):
            rows.append({"Actual": str(labels[i]), "Predicted": str(labels[j]), "Count": v, "Row %": v / total})
    df = pd.DataFrame(rows)
    base = alt.Chart(df).encode(
        x=alt.X("Predicted:N", axis=alt.Axis(orient="top", labelAngle=0, labelColor=INK_2, labelFontSize=12)),
        y=alt.Y("Actual:N", axis=alt.Axis(labelColor=INK_2, labelFontSize=12)))
    heat = base.mark_rect(cornerRadius=6, stroke="#ffffff", strokeWidth=3).encode(
        color=alt.Color("Row %:Q", scale=alt.Scale(range=["#eaf2fc", "#1c5cab"], domain=[0, 1]), legend=None),
        tooltip=["Actual:N", "Predicted:N", "Count:Q", alt.Tooltip("Row %:Q", format=".1%")])
    text = base.mark_text(fontSize=15, fontWeight=600).encode(
        text="Count:Q", color=alt.condition("datum['Row %'] > 0.55", alt.value("#ffffff"), alt.value(INK)))
    return _style(heat + text, 240)


# ---------------------------------------------------------------- agent timeline
AGENT_ICONS = {
    "Supervisor": "🧭", "Profiler Agent": "🔍", "Feature Engineering Agent": "🛠️",
    "Model Selection Agent": "🏁", "Tuning Agent": "🎛️", "Evaluation Agent": "📊",
    "Human Reviewer": "🧑‍⚖️", "Finalizer Agent": "📦",
}


def _md_lite(text: str) -> str:
    """Escape, then render **bold** and `code` from LLM output."""
    out = esc(text)
    out = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", out)
    out = re.sub(r"`([^`]+)`", r"<code>\1</code>", out)
    return out


def timeline(trace: list[dict], show_supervisor: bool):
    items = []
    for t in trace:
        if t["agent"] == "Supervisor" and not show_supervisor:
            continue
        sup = " sup" if t["agent"] == "Supervisor" else ""
        ts = t.get("ts", "").replace("T", " ")
        items.append(
            f'<div class="tl-item"><div class="tl-ico{sup}">{AGENT_ICONS.get(t["agent"], "•")}</div><div style="min-width:0">'
            f'<div class="tl-head"><span class="tl-agent">{esc(t["agent"])}</span><span class="tl-ts">{esc(ts)}</span></div>'
            f'<div class="tl-body">{_md_lite(t["message"])}</div></div></div>'
        )
    html_block("".join(items) or '<div class="card-sub">No messages yet.</div>')
