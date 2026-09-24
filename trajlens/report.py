"""Render grading results as JSON, Markdown or JUnit XML.

JUnit is the format every CI system already understands: GitHub Actions, GitLab and
Jenkins all render each failed criterion as a failed test, with no plugin needed.
"""
from __future__ import annotations

import json
from xml.sax.saxutils import escape, quoteattr

from .verifiers import CRITERIA


def graded_record(traj: dict, results, verdict: str, source: str = "") -> dict:
    return {
        "id": traj.get("id", ""),
        "title": traj.get("title", traj.get("id", "")),
        "source": source,
        "verdict": verdict,
        "failure_modes": sorted({r.failure_mode for r in results if r.status == "fail"}),
        "criteria": [r.__dict__ for r in results],
    }


def to_json(records: list[dict]) -> str:
    return json.dumps(records, indent=2)


def to_markdown(records: list[dict]) -> str:
    failed = sum(r["verdict"] == "FAILED" for r in records)
    lines = ["# TrajLens report", "",
             f"**{len(records)} trajectories graded, {failed} failed.**", "",
             "| Trajectory | Verdict | " + " | ".join(CRITERIA) + " |",
             "|---|---|" + "---|" * len(CRITERIA)]
    for r in records:
        marks = {c["id"]: ("✅" if c["status"] == "pass" else "❌") for c in r["criteria"]}
        lines.append(f"| {r['title']} | **{r['verdict']}** | "
                     + " | ".join(marks.get(c, "-") for c in CRITERIA) + " |")
    fails = [(r, c) for r in records for c in r["criteria"] if c["status"] == "fail"]
    if fails:
        lines += ["", "## Failures", ""]
        for r, c in fails:
            step = f" (step {c['step_index']})" if c["step_index"] is not None else ""
            lines.append(f"- **{r['title']}**, {c['id']} `{c['failure_mode']}`{step}: {c['note']}")
    return "\n".join(lines) + "\n"


def to_junit(records: list[dict]) -> str:
    cases = []
    n_fail = 0
    for r in records:
        for c in r["criteria"]:
            name = quoteattr(f"{c['id']} {c['label']}")
            cls = quoteattr(f"trajlens.{r['id']}")
            if c["status"] == "fail":
                n_fail += 1
                msg = quoteattr(f"{c['failure_mode']}: {c['note']}")
                cases.append(f'    <testcase classname={cls} name={name}>\n'
                             f'      <failure message={msg} type="{c["failure_mode"]}">'
                             f'{escape(c["note"])}</failure>\n    </testcase>')
            else:
                cases.append(f"    <testcase classname={cls} name={name}/>")
    total = sum(len(r["criteria"]) for r in records)
    return ('<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<testsuites tests="{total}" failures="{n_fail}">\n'
            f'  <testsuite name="trajlens" tests="{total}" failures="{n_fail}">\n'
            + "\n".join(cases) + "\n  </testsuite>\n</testsuites>\n")
