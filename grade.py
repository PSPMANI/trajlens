"""Regenerate every trajectory verdict from scratch.

    python grade.py

Prints a human/rubric/judge comparison table and writes data/verdicts.json,
so the whole leaderboard is reproducible -- the same discipline as
`python train.py` regenerating metrics in the churn project.
"""
import json
import pathlib

from verifiers import grade

DATA = pathlib.Path(__file__).parent / "data" / "trajectories.json"


def main():
    trajectories = json.loads(DATA.read_text(encoding="utf-8"))
    out = []

    header = f"{'Trajectory':<24}{'Human':<10}{'Rubric':<10}{'LLM-judge':<12}Failure modes"
    print(header)
    print("-" * len(header))

    rubric_agree = judge_agree = 0
    for t in trajectories:
        results, verdict = grade(t)
        human = t.get("expected_verdict", "")
        judge = t.get("llm_judge_verdict", "")
        modes = sorted({r.failure_mode for r in results if r.status == "fail"})
        rubric_agree += int(verdict == human)
        judge_agree += int(judge == human)
        out.append({
            "id": t["id"],
            "human_verdict": human,
            "rubric_verdict": verdict,
            "llm_judge_verdict": judge,
            "failure_modes": modes,
            "criteria": [r.__dict__ for r in results],
        })
        print(f"{t['title']:<24}{human:<10}{verdict:<10}{judge:<12}{', '.join(modes) or '-'}")

    n = len(trajectories)
    print("-" * len(header))
    print(f"Rubric vs human agreement:    {rubric_agree}/{n}")
    print(f"LLM-judge vs human agreement: {judge_agree}/{n}")

    (DATA.parent / "verdicts.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\nWrote data/verdicts.json")


if __name__ == "__main__":
    main()
