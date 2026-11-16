"""Streamlit UI for AutoML-Agent.

    streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # make package importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

import ui  # noqa: E402
from automl_agent import config, db  # noqa: E402
from automl_agent.graph import get_state, new_run, pending_interrupt, resume_command, stream  # noqa: E402
from automl_agent.llm import llm_available  # noqa: E402

st.set_page_config(page_title="AutoML-Agent", page_icon="🤖", layout="wide", initial_sidebar_state="expanded")
ui.inject_css()

STEP_LABEL = {
    "profiler": "Profiling the dataset", "feature_engineer": "Engineering features",
    "model_selector": "Cross-validating candidate models", "tuner": "Tuning hyperparameters with Optuna",
    "evaluator": "Evaluating on the hold-out set", "human_review": "Waiting for human review",
    "finalizer": "Exporting model and model card", "supervisor": "Supervisor routing",
}


def run_stream(source, run_id: str, box):
    """Stream agent updates live into a st.status box."""
    for node, update in stream(source, run_id):
        if node == "__interrupt__" or not update:
            continue
        if node != "supervisor":
            box.update(label=f"{STEP_LABEL.get(node, node)} ✓")
        for t in update.get("trace", []):
            if t["agent"] == "Supervisor":
                continue
            first = t["message"].strip().splitlines()[0][:150]
            box.markdown(f"{ui.AGENT_ICONS.get(t['agent'], '•')} **{t['agent']}** — {first}")


# =================================================================== sidebar
with st.sidebar:
    ui.html_block(
        '<div class="side-brand"><div class="side-logo">AM</div><div>'
        '<div class="side-title">AutoML-Agent</div>'
        '<div class="side-sub">Agentic AutoML</div></div></div>'
    )

    ui.html_block('<div class="side-label">1 · Dataset</div>')
    source = st.radio("Dataset source", ["Sample: customer churn", "Upload CSV"], label_visibility="collapsed")
    data_path = None
    if source == "Upload CSV":
        up = st.file_uploader("CSV file", type="csv", label_visibility="collapsed")
        if up:
            dest = config.RUNS_DIR / "uploads"
            dest.mkdir(parents=True, exist_ok=True)
            data_path = dest / up.name
            data_path.write_bytes(up.getvalue())
    else:
        data_path = config.DATA_DIR / "sample_churn.csv"

    df_preview = pd.read_csv(data_path) if data_path else None
    target = goal = None
    if df_preview is not None:
        cols = list(df_preview.columns)
        target = st.selectbox("Target column", cols, index=len(cols) - 1)
        goal = st.text_input("Goal", "Predict which customers will churn")

    ui.html_block('<div class="side-label">2 · Settings</div>')
    trials = st.slider("Optuna trials", 5, 100, config.N_TRIALS, step=5,
                       help="More trials = better tuning but slower.")
    start = st.button("Run agents", type="primary", disabled=df_preview is None, width="stretch", icon="🚀")

    ui.html_block('<div class="side-label">Previous runs</div>')
    past = db.list_runs()
    if past:
        labels = {r["run_id"]: f"{r['run_id']} · {Path(r['dataset']).name} · {r['status']}" for r in past}
        pick = st.selectbox("Open run", list(labels), format_func=labels.get, label_visibility="collapsed")
        if st.button("Open selected run", width="stretch"):
            st.session_state.run_id = pick
            st.rerun()
    else:
        st.caption("No runs yet.")

    ui.html_block('<div class="side-label">System</div>')
    if llm_available():
        ui.html_block(ui.pill(f"LLM · {config.GROQ_MODEL}", "good"))
    else:
        ui.html_block(ui.pill("LLM offline · template mode", "warn"))
        st.caption("Add GROQ_API_KEY to .env for LLM reasoning.")


# =================================================================== start a run
if start:
    run_id, inputs = new_run(str(data_path), target, goal, trials)
    st.session_state.run_id = run_id
    ui.page_header("Agents at work", f"Run <code>{run_id}</code> · {ui.esc(Path(data_path).name)} → "
                   f"<code>{ui.esc(target)}</code>", "running")
    with st.status("Supervisor is starting the run…", expanded=True) as box:
        run_stream(inputs, run_id, box)
        box.update(label="Agents finished — your review is needed", state="complete", expanded=False)
    s = get_state(run_id)
    db.update_run(run_id, status=s.get("status"), task_type=s["profile"]["task_type"])
    st.rerun()

run_id = st.session_state.get("run_id")

# =================================================================== welcome screen
if not run_id:
    ui.html_block(
        '<div class="hero"><h2>Build a validated ML model with a team of AI agents</h2>'
        '<p>Choose a dataset and a target column in the sidebar. A LangGraph supervisor coordinates six '
        'specialist agents that profile the data, engineer features, compare and tune models, and evaluate '
        'on a held-out set — then pause for your approval before anything is exported.</p>'
        '<div class="agents">'
        '<div class="agent-card"><div class="n">🔍 Profiler</div><div class="d">Finds missing values, ID columns, imbalance and target leakage.</div></div>'
        '<div class="agent-card"><div class="n">🛠️ Feature Engineer</div><div class="d">Builds a leakage-safe preprocessing pipeline.</div></div>'
        '<div class="agent-card"><div class="n">🏁 Model Selector</div><div class="d">5-fold CV across Logistic Regression, Random Forest and XGBoost.</div></div>'
        '<div class="agent-card"><div class="n">🎛️ Tuner</div><div class="d">Bayesian hyperparameter search with Optuna.</div></div>'
        '<div class="agent-card"><div class="n">📊 Evaluator</div><div class="d">Hold-out test, baseline lift, overfitting check, feature importance.</div></div>'
        '<div class="agent-card"><div class="n">🧑‍⚖️ You</div><div class="d">Approve, reject or request re-tuning. Every decision is audited.</div></div>'
        '</div></div>'
    )
    if df_preview is not None:
        with st.container(border=True):
            ui.card_title("Dataset preview", f"{len(df_preview):,} rows · {df_preview.shape[1]} columns")
            st.dataframe(df_preview.head(15), width="stretch", hide_index=True)
    st.stop()

# =================================================================== run dashboard
s = get_state(run_id)
if not s:
    st.warning("Run not found.")
    st.stop()

ev, lb, tu, prof, plan = s.get("evaluation"), s.get("leaderboard"), s.get("tuning"), s.get("profile"), s.get("fe_plan")
req = pending_interrupt(run_id)
status = "awaiting_approval" if req else s.get("status")

ui.page_header(
    "Model run",
    f"Run <code>{run_id}</code> · {ui.esc(Path(s['dataset_path']).name)} → <code>{ui.esc(s['target'])}</code>"
    + (f" · {ui.esc(prof['task_type'])}" if prof else ""),
    status,
)
ui.stepper(s, waiting_review=bool(req))

if ev:
    metric = ui.pretty(ev["metric"])
    lift = ev["lift_over_baseline"]
    gap = ev["overfit_gap"]
    gap_ok = gap <= config.OVERFIT_GAP
    ui.kpi_row([
        ui.kpi("Selected model", ui.pretty(ev["model"]), f"tuned · {tu['n_trials']} trials" if tu else "", text=True),
        ui.kpi(f"Test {metric}", f"{ev['test_score']:.3f}",
               f'<span class="{"up" if lift > 0 else "down"}">{lift:+.3f}</span> vs baseline'),
        ui.kpi("Baseline", f"{ev['baseline_score']:.3f}", "majority-class / mean predictor"),
        ui.kpi("Overfit gap", f"{gap:.3f}",
               f'<span class="{"up" if gap_ok else "down"}">{"✓ within" if gap_ok else "⚠ above"}</span> '
               f"{config.OVERFIT_GAP:.2f} limit"),
        ui.kpi("Self-corrections", len(s.get("remediation") or []), f"max {config.MAX_RETRIES} retries"),
    ])
    for issue in ev["issues"]:
        ui.callout(f"<b>Evaluation Agent flagged:</b> {ui.esc(issue.replace('_', ' '))}", "warn", "⚠️")
for r in s.get("remediation") or []:
    ui.callout(f"<b>Self-correction:</b> {ui.esc(r)}", "info", "🔁")

tab_over, tab_models, tab_explain, tab_data, tab_trace, tab_audit = st.tabs(
    ["Overview", "Model comparison", "Explainability", "Data & features", "Agent reasoning", "Audit log"])

# ------------------------------------------------------------------- overview
with tab_over:
    if req:
        with st.container(border=True):
            ui.card_title("Your decision", "The agents have paused. Review the evidence in the tabs above, then decide.")
            with st.form("decision", border=False):
                c1, c2 = st.columns([1, 1])
                reviewer = c1.text_input("Reviewer name", placeholder="e.g. Andrew")
                n_more = c2.number_input("Trials if re-tuning", 10, 200, int(s.get("n_trials", 25)) * 2)
                decision = st.radio(
                    "Decision", ["approve", "retune", "reject"], horizontal=True,
                    format_func={"approve": "✅ Approve & export", "retune": "🔁 Re-tune", "reject": "⛔ Reject"}.get)
                comment = st.text_area("Comment (saved to the audit log)", height=80)
                submitted = st.form_submit_button("Submit decision", type="primary")
            if submitted:
                db.update_run(run_id, status="running")
                with st.status("Resuming agents…", expanded=True) as box:
                    run_stream(resume_command({"decision": decision, "reviewer": reviewer or "anonymous",
                                               "comment": comment, "n_trials": int(n_more)}), run_id, box)
                new = get_state(run_id)
                if new.get("status") in ("rejected", "awaiting_approval"):
                    db.update_run(run_id, status=new["status"])
                st.rerun()

    if s.get("artifacts"):
        a = s["approval"]
        ui.callout(f"<b>Approved by {ui.esc(a['reviewer'])}</b> on {ui.esc(a['timestamp'].replace('T', ' '))}. "
                   "The model and its model card have been exported.", "good", "✅")
        with st.container(border=True):
            ui.card_title("Downloads")
            d1, d2, _ = st.columns([1, 1, 2])
            d1.download_button("Model card (PDF)", Path(s["artifacts"]["report"]).read_bytes(),
                               file_name=f"model_card_{run_id}.pdf", mime="application/pdf", icon="📄", width="stretch")
            d2.download_button("Model (.joblib)", Path(s["artifacts"]["model"]).read_bytes(),
                               file_name=f"model_{run_id}.joblib", icon="📦", width="stretch")
        with st.container(border=True):
            ui.card_title("Score new data", "Upload a CSV with the same feature columns to get predictions.")
            new_csv = st.file_uploader("CSV to score", type="csv", key="score", label_visibility="collapsed")
            if new_csv:
                bundle = joblib.load(s["artifacts"]["model"])
                X = pd.read_csv(new_csv).reindex(columns=bundle["features"])
                out = X.copy()
                preds = bundle["pipeline"].predict(X)
                out.insert(0, "prediction", [bundle["classes"][int(p)] for p in preds] if bundle["classes"] else preds)
                if bundle["classes"] and len(bundle["classes"]) == 2:
                    out.insert(1, "probability", bundle["pipeline"].predict_proba(X)[:, 1].round(4))
                st.dataframe(out.head(100), width="stretch", hide_index=True)
                st.download_button("Download predictions", out.to_csv(index=False), "predictions.csv", icon="⬇️")
    elif s.get("status") == "rejected":
        a = s.get("approval") or {}
        ui.callout(f"<b>Rejected by {ui.esc(a.get('reviewer', '—'))}.</b> {ui.esc(a.get('comment', ''))}", "crit", "⛔")

    if ev:
        c1, c2 = st.columns([3, 2], gap="medium")
        with c1, st.container(border=True):
            ui.card_title("What the model relies on", "Permutation importance on the held-out test set")
            st.altair_chart(ui.importance_chart(ev["feature_importance"][:8], ev["metric"]), width="stretch")
        with c2, st.container(border=True):
            ui.card_title("Test-set metrics", f"{int(len(ev['details']))} metrics · 20% hold-out")
            d = {ui.pretty(k): v for k, v in ev["details"].items() if k != "confusion_matrix"}
            st.dataframe(pd.DataFrame(d.items(), columns=["Metric", "Value"]), hide_index=True, width="stretch",
                         column_config={"Value": st.column_config.NumberColumn(format="%.4f")})

# ------------------------------------------------------------------- model comparison
with tab_models:
    if lb:
        c1, c2 = st.columns([1, 1], gap="medium")
        with c1, st.container(border=True):
            ui.card_title("Cross-validation leaderboard",
                          f"Selected: {ui.pretty(s['best_model'])} — simplest model within one std of the best")
            st.altair_chart(ui.leaderboard_chart(lb, s["best_model"]), width="stretch")
        with c2, st.container(border=True):
            if tu and tu.get("trial_scores"):
                ui.card_title(f"Optuna tuning · {tu['mode']} mode",
                              f"{tu['n_trials']} trials · best CV {ui.pretty(tu['metric'])} = {tu['best_cv_score']:.4f}")
                st.altair_chart(ui.tuning_chart(tu), width="stretch")
            else:
                ui.card_title("Tuning", "Not run yet.")
        with st.container(border=True):
            ui.card_title("Leaderboard table")
            t = pd.DataFrame(lb).assign(model=lambda d: d["model"].map(ui.pretty), metric=lambda d: d["metric"].map(ui.pretty))
            st.dataframe(t.rename(columns={"model": "Model", "metric": "Metric", "cv_mean": "CV mean",
                                           "cv_std": "CV std", "fit_seconds": "Fit time (s)"}),
                         hide_index=True, width="stretch")
        if tu:
            with st.container(border=True):
                ui.card_title("Final hyperparameters")
                params = ev["params"] if ev else tu["best_params"]
                st.dataframe(pd.DataFrame([{"Parameter": k, "Value": str(v)} for k, v in (params or {}).items()]
                                          or [{"Parameter": "—", "Value": "library defaults"}]),
                             hide_index=True, width="stretch")
    else:
        st.info("Model comparison will appear once the Model Selection Agent finishes.")

# ------------------------------------------------------------------- explainability
with tab_explain:
    if ev:
        c1, c2 = st.columns([3, 2], gap="medium")
        with c1, st.container(border=True):
            ui.card_title("Feature importance", "How much the test score drops when each feature is shuffled")
            st.altair_chart(ui.importance_chart(ev["feature_importance"], ev["metric"]), width="stretch")
        with c2:
            if "confusion_matrix" in ev["details"]:
                with st.container(border=True):
                    ui.card_title("Confusion matrix", "Rows = actual, columns = predicted · shaded by row %")
                    bundle_classes = list(range(len(ev["details"]["confusion_matrix"])))
                    if prof and prof.get("class_counts"):
                        bundle_classes = sorted(prof["class_counts"].keys(), key=str)
                    st.altair_chart(ui.confusion_chart(ev["details"]["confusion_matrix"], bundle_classes), width="stretch")
            with st.container(border=True):
                ui.card_title("Train vs test", "Large gaps indicate overfitting")
                st.dataframe(pd.DataFrame({
                    "Split": ["Train", "Test", "Baseline (test)"],
                    ui.pretty(ev["metric"]): [ev["train_score"], ev["test_score"], ev["baseline_score"]]}),
                    hide_index=True, width="stretch")
    else:
        st.info("Explainability appears after the Evaluation Agent runs.")

# ------------------------------------------------------------------- data & features
with tab_data:
    if prof:
        tiles = [ui.kpi("Rows", f"{prof['n_rows']:,}"), ui.kpi("Features", prof["n_features"]),
                 ui.kpi("Duplicate rows", prof["duplicate_rows"]),
                 ui.kpi("Columns with missing", len(prof["columns_with_missing"]))]
        if prof["task_type"] == "classification":
            tiles.append(ui.kpi("Class imbalance", f"{prof['imbalance_ratio']}:1",
                                " · ".join(f"{k}: {v:,}" for k, v in prof["class_counts"].items())))
        ui.kpi_row(tiles)
        if prof["leakage_suspects"]:
            ui.callout("<b>Target leakage detected:</b> " + ", ".join(f"<code>{ui.esc(c)}</code>" for c in prof["leakage_suspects"])
                       + " predicts the target almost perfectly on its own and was removed.", "crit", "🚨")
        c1, c2 = st.columns([1, 1], gap="medium")
        if plan:
            with c1, st.container(border=True):
                ui.card_title("Dropped columns", "Removed by the Feature Engineering Agent")
                dropped = [{"Column": k, "Reason": v} for k, v in plan["drop_columns"].items()]
                st.dataframe(pd.DataFrame(dropped or [{"Column": "—", "Reason": "none"}]), hide_index=True, width="stretch")
            with c2, st.container(border=True):
                ui.card_title("Preprocessing plan")
                st.dataframe(pd.DataFrame([
                    {"Step": "Numeric features", "Detail": ", ".join(plan["numeric_columns"]) or "—"},
                    {"Step": "Categorical (one-hot)", "Detail": ", ".join(plan["categorical_columns"]) or "—"},
                    {"Step": "Frequency-encoded", "Detail": ", ".join(plan["frequency_encoded_columns"]) or "—"},
                    {"Step": "Imputation", "Detail": f"numeric: {plan['numeric_imputer']}, categorical: {plan['categorical_imputer']}"},
                    {"Step": "Scaling", "Detail": "StandardScaler" if plan["scale_numeric"] else "none"},
                    {"Step": "Class weighting", "Detail": "on" if plan["use_class_weight"] else "off"},
                ]), hide_index=True, width="stretch")
        with st.container(border=True):
            ui.card_title("Column profile")
            cols_df = pd.DataFrame([
                {"Column": c, "Type": v["kind"], "Missing %": v["missing_pct"], "Unique": v["n_unique"],
                 "Single-feature score": v.get("single_feature_score")}
                for c, v in prof["columns"].items()])
            st.dataframe(cols_df, hide_index=True, width="stretch",
                         column_config={"Missing %": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f%%"),
                                        "Single-feature score": st.column_config.NumberColumn(
                                            format="%.3f", help="AUC / |corr| of this column alone vs the target. ≥0.97 ⇒ leakage")})

# ------------------------------------------------------------------- agent reasoning
with tab_trace:
    with st.container(border=True):
        top_l, top_r = st.columns([3, 1])
        with top_l:
            ui.card_title("Agent reasoning trace", "Each specialist's explanation, grounded in the knowledge base")
        show_sup = top_r.toggle("Show supervisor routing", value=False)
        ui.timeline(s.get("trace", []), show_sup)

# ------------------------------------------------------------------- audit log
with tab_audit:
    with st.container(border=True):
        ui.card_title("Audit log", "Every agent action and human decision, stored in SQLite")
        audit = pd.DataFrame(db.get_audit(run_id))
        if not audit.empty:
            audit["ts"] = audit["ts"].str.replace("T", " ")
            st.dataframe(audit[["ts", "agent", "event", "detail"]].rename(
                columns={"ts": "Time", "agent": "Agent", "event": "Event", "detail": "Detail"}),
                hide_index=True, width="stretch", height=420)
