"""Terminal interface.

    python -m automl_agent.cli --data data/sample_churn.csv --target churn
"""
import argparse

from automl_agent import db
from automl_agent.graph import get_state, new_run, pending_interrupt, resume_command, stream
from automl_agent.llm import llm_available


def _print_updates(run_id, source):
    for node, update in stream(source, run_id):
        if node == "__interrupt__" or not update:
            continue
        for t in update.get("trace", []):
            print(f"\n\033[1;34m[{t['agent']}]\033[0m\n{t['message']}")


def main():
    ap = argparse.ArgumentParser(description="AutoML-Agent: multi-agent AutoML with human approval")
    ap.add_argument("--data", required=True, help="Path to CSV file")
    ap.add_argument("--target", required=True, help="Target column name")
    ap.add_argument("--goal", default="", help="What you want to predict, in plain words")
    ap.add_argument("--trials", type=int, default=None, help="Optuna trials")
    args = ap.parse_args()

    print(f"LLM reasoning: {'Groq LLaMA 3.3 70B' if llm_available() else 'OFFLINE (template explanations)'}")
    run_id, inputs = new_run(args.data, args.target, args.goal, args.trials)
    print(f"Run id: {run_id}")
    _print_updates(run_id, inputs)

    while (req := pending_interrupt(run_id)) is not None:
        print("\n" + "=" * 60)
        print(f"HUMAN APPROVAL NEEDED — {req['model']}: test {req['metric']}={req['test_score']} "
              f"(baseline {req['baseline_score']}) issues={req['issues'] or 'none'}")
        choice = ""
        while choice not in ("approve", "reject", "retune"):
            choice = input("Decision [approve/reject/retune]: ").strip().lower()
        reviewer = input("Your name: ").strip() or "anonymous"
        comment = input("Comment (optional): ").strip()
        db.update_run(run_id, status="running")
        _print_updates(run_id, resume_command({"decision": choice, "reviewer": reviewer, "comment": comment}))

    state = get_state(run_id)
    print("\nFinal status:", state.get("status"))
    if state.get("artifacts"):
        print("Model:  ", state["artifacts"]["model"])
        print("Report: ", state["artifacts"]["report"])
    if state.get("status") == "rejected":
        db.update_run(run_id, status="rejected")


if __name__ == "__main__":
    main()
