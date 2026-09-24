"""Command-line interface.

    trajlens grade runs/*.json --fail-on C3,C8,C9 --junit report.xml
    trajlens corpus
    trajlens stress
    trajlens judge --model claude-haiku-4-5-20251001 --mode answer

``grade`` exits 1 when any selected criterion fails, so it can gate an agent release in
CI the same way a failing test does.
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib
import sys

from . import __version__, judge, metrics, mutate
from .adapters import load_traces
from .report import graded_record, to_json, to_junit, to_markdown
from .verifiers import CRITERIA, grade

DEFAULT_DATA = pathlib.Path(__file__).resolve().parent.parent / "data" / "trajectories.json"


def _color(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if sys.stdout.isatty() else text


def _expand(paths: list[str]) -> list[pathlib.Path]:
    out = []
    for p in paths:
        matches = glob.glob(p, recursive=True)  # Windows shells do not expand globs
        out.extend(pathlib.Path(m) for m in (matches or [p]))
    return out


def cmd_grade(args) -> int:
    selected = set(CRITERIA) if args.fail_on == "all" else {c.strip().upper() for c in args.fail_on.split(",")}
    unknown = selected - set(CRITERIA)
    if unknown:
        print(f"Unknown criteria in --fail-on: {', '.join(sorted(unknown))}", file=sys.stderr)
        return 2
    records, gate_failed = [], False
    for path in _expand(args.paths):
        try:
            traces = load_traces(path.read_text(encoding="utf-8"), args.format, path.stem)
        except (OSError, ValueError, KeyError) as e:
            print(f"{path}: cannot read trace ({e})", file=sys.stderr)
            return 2
        for t in traces:
            if args.max_steps is not None:
                t["max_steps"] = args.max_steps
            if args.authorize:
                t["authorized_side_effects"] = sorted(
                    set(t.get("authorized_side_effects") or []) | set(args.authorize.split(",")))
            results, verdict = grade(t)
            rec = graded_record(t, results, verdict, str(path))
            records.append(rec)
            blocking = [r for r in results if r.status == "fail" and r.id in selected]
            gate_failed |= bool(blocking)
            if not args.quiet:
                mark = _color("PASS", "32") if verdict == "PASSED" else _color("FAIL", "31")
                print(f"[{mark}] {rec['title']}  ({path.name})")
                for r in results:
                    if r.status == "fail":
                        tag = "" if r.id in selected else " (not gating)"
                        print(f"       {r.id} {r.failure_mode}{tag}: {r.note}")
    for flag, render in (("json", to_json), ("markdown", to_markdown), ("junit", to_junit)):
        target = getattr(args, flag)
        if target:
            pathlib.Path(target).write_text(render(records), encoding="utf-8")
    n_fail = sum(r["verdict"] == "FAILED" for r in records)
    print(f"\n{len(records)} trajectories, {n_fail} failed. "
          f"Gate ({'all criteria' if args.fail_on == 'all' else args.fail_on}): "
          + (_color("BLOCKED", "31") if gate_failed else _color("OK", "32")))
    return 1 if gate_failed else 0


def corpus_report(trajectories: list[dict], runs: list[dict] | None = None) -> dict:
    humans = [t["expected_verdict"] for t in trajectories]
    rows, rubric = [], []
    for t in trajectories:
        results, verdict = grade(t)
        rubric.append(verdict)
        rows.append({"id": t["id"], "title": t["title"], "human_verdict": t["expected_verdict"],
                     "rubric_verdict": verdict, "llm_judge_verdict": t.get("llm_judge_verdict", ""),
                     "failure_modes": sorted({r.failure_mode for r in results if r.status == "fail"}),
                     "criteria": [r.__dict__ for r in results]})
    graders = {"Rubric (TrajLens)": metrics.compare(humans, rubric),
               "Simulated LLM-judge": metrics.compare(humans, [r["llm_judge_verdict"] for r in rows])}
    for run in (judge.load_runs() if runs is None else runs):
        preds = [run["verdicts"].get(t["id"], {}).get("verdict", "UNPARSED") for t in trajectories]
        graders[f"{run['model']} ({run['mode']})"] = metrics.compare(humans, preds)
    return {"rows": rows, "graders": graders}


def cmd_corpus(args) -> int:
    trajectories = json.loads(pathlib.Path(args.data).read_text(encoding="utf-8"))
    rep = corpus_report(trajectories)
    header = f"{'Trajectory':<26}{'Human':<9}{'Rubric':<9}{'Sim. judge':<12}Failure modes"
    print(header + "\n" + "-" * len(header))
    for r in rep["rows"]:
        print(f"{r['title']:<26}{r['human_verdict']:<9}{r['rubric_verdict']:<9}"
              f"{r['llm_judge_verdict']:<12}{', '.join(r['failure_modes']) or '-'}")
    print()
    for name, g in rep["graders"].items():
        lo, hi = g["kappa_ci"]
        print(f"{name:<34} agree {g['agree']}/{g['n']}  kappa {g['kappa']:.2f} "
              f"[95% CI {lo:.2f}, {hi:.2f}]  recall {g['recall']:.2f}  precision {g['precision']:.2f}")
    out = pathlib.Path(args.data).parent / "verdicts.json"
    out.write_text(json.dumps(rep["rows"], indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
    return 0


def cmd_stress(args) -> int:
    trajectories = json.loads(pathlib.Path(args.data).read_text(encoding="utf-8"))
    rep = mutate.run(trajectories)
    print(mutate.format_table(rep))
    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(rep, indent=2), encoding="utf-8")
    # Exit nonzero if a verifier misses its own fault class or flags a clean trace.
    return 0 if rep["detected"] == rep["mutants"] and not rep["false_alarms"] else 1


def cmd_judge(args) -> int:
    trajectories = json.loads(pathlib.Path(args.data).read_text(encoding="utf-8"))
    path = judge.run(trajectories, args.model, args.mode)
    print(f"Wrote {path}. Run `trajlens corpus` to compare it with the human labels.")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="trajlens", description="Deterministic auditing of AI-agent trajectories.")
    p.add_argument("--version", action="version", version=f"trajlens {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("grade", help="grade trace files; exit 1 if a gating criterion fails")
    g.add_argument("paths", nargs="+", help="trace files (.json or .jsonl); globs allowed")
    g.add_argument("--format", default="auto", choices=["auto", "native", "openai", "anthropic"])
    g.add_argument("--fail-on", default="all", help="comma-separated criteria that gate, e.g. C3,C9 (default: all)")
    g.add_argument("--max-steps", type=int, help="step budget for C7, overriding the trace")
    g.add_argument("--authorize", help="comma-separated side-effect tools the task allows (C9)")
    g.add_argument("--json", help="write a JSON report")
    g.add_argument("--markdown", help="write a Markdown report (e.g. for $GITHUB_STEP_SUMMARY)")
    g.add_argument("--junit", help="write a JUnit XML report")
    g.add_argument("-q", "--quiet", action="store_true", help="only print the summary line")
    g.set_defaults(func=cmd_grade)

    for name, fn, help_ in (("corpus", cmd_corpus, "grade the bundled corpus and measure every grader against humans"),
                            ("stress", cmd_stress, "mutation-test the rubric")):
        s = sub.add_parser(name, help=help_)
        s.add_argument("--data", default=str(DEFAULT_DATA))
        if name == "stress":
            s.add_argument("--json", help="write the full mutation report")
        s.set_defaults(func=fn)

    j = sub.add_parser("judge", help="run a real LLM judge over the corpus (needs an API key)")
    j.add_argument("--model", required=True)
    j.add_argument("--mode", default="answer", choices=["answer", "trajectory"])
    j.add_argument("--data", default=str(DEFAULT_DATA))
    j.set_defaults(func=cmd_judge)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
