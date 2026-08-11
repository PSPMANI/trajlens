"""Deterministic rubric verifiers for agentic tool-use trajectories.

Each verifier reads the trajectory CONTENT (not the planted-failure labels) and
returns a pass/fail judgement plus the offending step. Because the checks are
plain, deterministic Python over the trajectory JSON, every grade is
reproducible and auditable -- not an LLM hand-waving a number.

This mirrors the kind of pass/fail rubric + pytest-verifier work done under NDA
for frontier-model evaluation, reconstructed here on public, synthetic data.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional


@dataclass
class CriterionResult:
    id: str
    label: str
    status: str  # "pass" | "fail"
    failure_mode: Optional[str] = None
    step_index: Optional[int] = None
    note: str = ""


def _tool_steps(traj):
    """Yield (index, step, tool_call) for every step that issues a tool call."""
    for i, step in enumerate(traj.get("steps", [])):
        call = step.get("tool_call")
        if call:
            yield i, step, call


def c1_valid_tool_selection(traj) -> CriterionResult:
    valid = {t["name"] for t in traj.get("available_tools", [])}
    for i, _step, call in _tool_steps(traj):
        if call["name"] not in valid:
            return CriterionResult(
                "C1", "Valid tool selection", "fail", "wrong_tool_selection", i,
                f"Step {i} called '{call['name']}', which is not one of the available tools.",
            )
    return CriterionResult(
        "C1", "Valid tool selection", "pass",
        note="Every tool call targets a real, available tool.",
    )


def c2_well_formed_args(traj) -> CriterionResult:
    schema = {t["name"]: t.get("required_args", []) for t in traj.get("available_tools", [])}
    for i, _step, call in _tool_steps(traj):
        if call["name"] not in schema:  # unknown tool is C1's job
            continue
        args = call.get("args", {}) or {}
        missing = [a for a in schema[call["name"]] if a not in args]
        if missing:
            return CriterionResult(
                "C2", "Well-formed arguments", "fail", "malformed_args", i,
                f"Step {i} call '{call['name']}' is missing required argument(s): {', '.join(missing)}.",
            )
    return CriterionResult(
        "C2", "Well-formed arguments", "pass",
        note="All tool calls include their required arguments.",
    )


def c3_grounded_answer(traj) -> CriterionResult:
    observations = " \n".join((s.get("observation") or "") for s in traj.get("steps", [])).lower()
    last = len(traj.get("steps", [])) - 1
    for claim in traj.get("final_answer_claims", []):
        if claim.lower() not in observations:
            return CriterionResult(
                "C3", "Answer grounded in tool outputs", "fail",
                "hallucinated_tool_output", last if last >= 0 else None,
                f"Final answer asserts '{claim}', which never appears in any tool observation.",
            )
    return CriterionResult(
        "C3", "Answer grounded in tool outputs", "pass",
        note="Every factual claim in the final answer traces back to a tool observation.",
    )


def c4_instruction_following(traj) -> CriterionResult:
    req = traj.get("answer_requirements", {}) or {}
    answer = (traj.get("final_answer") or "").lower()
    for token in req.get("must_include", []):
        if token.lower() not in answer:
            return CriterionResult(
                "C4", "Instruction following", "fail", "ignored_instruction", None,
                f"Task required the answer to include '{token}', but it does not.",
            )
    for token in req.get("must_not_include", []):
        if token.lower() in answer:
            return CriterionResult(
                "C4", "Instruction following", "fail", "ignored_instruction", None,
                f"Answer contains '{token}', which the task explicitly forbade.",
            )
    return CriterionResult(
        "C4", "Instruction following", "pass",
        note="Final answer satisfies the task's stated constraints.",
    )


def c5_no_redundant_calls(traj) -> CriterionResult:
    seen = {}
    for i, _step, call in _tool_steps(traj):
        key = call["name"] + "|" + json.dumps(call.get("args", {}), sort_keys=True)
        if key in seen:
            return CriterionResult(
                "C5", "No redundant tool calls", "fail", "redundant_call", i,
                f"Step {i} repeats the identical call already made at step {seen[key]}.",
            )
        seen[key] = i
    return CriterionResult(
        "C5", "No redundant tool calls", "pass",
        note="No identical tool call is issued twice.",
    )


def c6_clean_termination(traj) -> CriterionResult:
    if not (traj.get("final_answer") or "").strip():
        return CriterionResult(
            "C6", "Clean termination", "fail", "premature_stop", None,
            "Trajectory ends without ever producing a final answer.",
        )
    return CriterionResult(
        "C6", "Clean termination", "pass",
        note="Agent produced a final answer for the user.",
    )


def c7_efficient(traj) -> CriterionResult:
    n = len(traj.get("steps", []))
    cap = traj.get("max_steps")
    if cap is not None and n > cap:
        return CriterionResult(
            "C7", "Efficient (no wasted steps)", "fail", "inefficient", None,
            f"Used {n} steps for a task that should take at most {cap}.",
        )
    return CriterionResult(
        "C7", "Efficient (no wasted steps)", "pass",
        note="Solved within the expected number of steps.",
    )


VERIFIERS = [
    c1_valid_tool_selection,
    c2_well_formed_args,
    c3_grounded_answer,
    c4_instruction_following,
    c5_no_redundant_calls,
    c6_clean_termination,
    c7_efficient,
]


def grade(traj):
    """Return (list[CriterionResult], overall_verdict)."""
    results = [v(traj) for v in VERIFIERS]
    verdict = "FAILED" if any(r.status == "fail" for r in results) else "PASSED"
    return results, verdict
