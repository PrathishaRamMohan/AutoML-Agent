"""PDF model card / run report built with ReportLab (+ matplotlib charts)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.platypus import (  # noqa: E402
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ACCENT = "#2a6fdb"


def _table(rows, col_widths=None):
    t = Table(rows, colWidths=col_widths, hAlign="LEFT")
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(ACCENT)),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#c8ced8")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f3f6fb")]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    return t


def _importance_chart(importances, path: Path):
    top = importances[:10][::-1]
    fig, ax = plt.subplots(figsize=(6, 3.2), dpi=150)
    ax.barh([r["feature"] for r in top], [r["importance"] for r in top], color=ACCENT)
    ax.set_xlabel("Drop in test score when shuffled (permutation importance)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _tuning_chart(tuning, path: Path):
    fig, ax = plt.subplots(figsize=(6, 2.6), dpi=150)
    trials = range(1, len(tuning["trial_scores"]) + 1)
    ax.scatter(trials, tuning["trial_scores"], s=12, color="#9aa7bd", label="trial")
    ax.plot(trials, tuning["best_so_far"], color=ACCENT, lw=2, label="best so far")
    ax.set_xlabel("Optuna trial")
    ax.set_ylabel(f"CV {tuning['metric']}")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def build_report(state: dict, out_path: Path) -> Path:
    out_path = Path(out_path)
    run_dir = out_path.parent
    ss = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=ss["Title"], textColor=colors.HexColor(ACCENT))
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], textColor=colors.HexColor("#1b2a41"), spaceBefore=10)
    body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=9, leading=12)
    small = ParagraphStyle("s", parent=body, fontSize=8, textColor=colors.HexColor("#555555"))

    p, fe, ev, tu = state["profile"], state["fe_plan"], state["evaluation"], state["tuning"]
    approval = state.get("approval") or {}
    els = [
        Paragraph("AutoML-Agent — Model Card &amp; Run Report", h1),
        Paragraph(
            f"Run <b>{state['run_id']}</b> · dataset <b>{Path(state['dataset_path']).name}</b> · "
            f"target <b>{state['target']}</b> · task <b>{p['task_type']}</b>",
            body,
        ),
        Paragraph(f"Goal: {state.get('goal') or '—'}", body),
        Paragraph(
            f"Human decision: <b>{approval.get('decision', 'pending').upper()}</b> by "
            f"{approval.get('reviewer', '—')} at {approval.get('timestamp', '—')}. "
            f"Comment: {approval.get('comment') or '—'}",
            body,
        ),
        Paragraph("1. Result summary", h2),
        _table(
            [
                ["Model", "Metric", "Test score", "Baseline", "Lift", "Train score", "Overfit gap"],
                [ev["model"], ev["metric"], ev["test_score"], ev["baseline_score"],
                 ev["lift_over_baseline"], ev["train_score"], ev["overfit_gap"]],
            ]
        ),
        Spacer(1, 4),
        _table([["Metric", "Value"]] + [[k, str(v)] for k, v in ev["details"].items()], [5 * cm, 8 * cm]),
        Paragraph("2. Data profile", h2),
        Paragraph(
            f"{p['n_rows']} rows, {p['n_features']} features, {p['duplicate_rows']} duplicate rows. "
            + (f"Class counts {p.get('class_counts')} (imbalance {p.get('imbalance_ratio')}:1)." if p["task_type"] == "classification" else ""),
            body,
        ),
        Paragraph("3. Preprocessing plan", h2),
        _table(
            [["Dropped column", "Reason"]] + [[k, v] for k, v in fe["drop_columns"].items()]
            if fe["drop_columns"] else [["Dropped column", "Reason"], ["—", "—"]],
            [5 * cm, 10 * cm],
        ),
        Paragraph(
            f"Numeric: {', '.join(fe['numeric_columns']) or '—'} (median imputation + scaling). "
            f"Categorical one-hot: {', '.join(fe['categorical_columns']) or '—'}. "
            f"Class weighting: {'on' if fe['use_class_weight'] else 'off'}.",
            body,
        ),
        Paragraph("4. Model comparison (5-fold CV on training data)", h2),
        _table(
            [["Model", "Metric", "CV mean", "CV std", "Fit time (s)"]]
            + [[r["model"], r["metric"], r["cv_mean"], r["cv_std"], r["fit_seconds"]] for r in state["leaderboard"]]
        ),
        Paragraph(f"Selected for tuning: <b>{state['best_model']}</b>", body),
        Paragraph("5. Hyperparameter tuning (Optuna TPE)", h2),
        Paragraph(
            f"{tu['n_trials']} trials, mode <b>{tu['mode']}</b>, best CV {tu['metric']} = {tu['best_cv_score']}. "
            f"Final params: {ev['params']}",
            body,
        ),
    ]
    if tu["trial_scores"]:
        _tuning_chart(tu, run_dir / "tuning.png")
        els.append(Image(str(run_dir / "tuning.png"), width=15 * cm, height=6.5 * cm))

    els.append(Paragraph("6. What the model relies on", h2))
    _importance_chart(ev["feature_importance"], run_dir / "importance.png")
    els.append(Image(str(run_dir / "importance.png"), width=15 * cm, height=8 * cm))

    els.append(Paragraph("7. Agent reasoning", h2))
    for item in state.get("trace", []):
        text = str(item["message"]).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        text = text.replace("\n", "<br/>")
        els.append(Paragraph(f"<b>{item['agent']}</b>: {text}", small))
        els.append(Spacer(1, 3))

    els += [
        Paragraph("8. Intended use &amp; caveats", h2),
        Paragraph(
            "Generated automatically by AutoML-Agent for decision support. Performance was measured on a "
            "held-out 20% test split that was never used during model selection or tuning. Re-validate "
            "on fresh data before production use, monitor for data drift, and review features for "
            "sensitive attributes or proxies.",
            body,
        ),
    ]
    SimpleDocTemplate(str(out_path), pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm,
                      topMargin=1.5 * cm, bottomMargin=1.5 * cm, title="AutoML-Agent Model Card").build(els)
    return out_path
