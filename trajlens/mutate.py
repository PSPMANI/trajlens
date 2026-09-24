"""Mutation testing for the rubric: does each verifier catch the fault it claims to?

The curated corpus was written alongside the rubric, so "the rubric agrees with the
human label on 14/14" is a sanity check, not evidence. This module is the evidence.
It takes every trajectory a human labelled PASSED, injects one known fault at every
place it can go, and records which criteria fire.

Two numbers come out of it:

* detection rate: the share of mutants where the targeted criterion fails. A verifier
  that misses its own fault class has a hole.
* collateral firing: which other criteria also fail. An operator/criterion heatmap
  that is mostly diagonal means each verifier measures one thing.

The mutants are graded in "unassisted" mode, with the hand-written
``final_answer_claims`` removed, so C3 has to find a changed number on its own, exactly
as it would in an unannotated production log.
"""
from __future__ import annotations

import copy
from collections.abc import Callable, Iterator
from dataclasses import dataclass

from .claims import extract_claims, grounded_keys
from .verifiers import ACK_TOKENS, CRITERIA, grade


@dataclass
class Mutant:
    base_id: str
    operator: str
    target: str   # the criterion this fault should trip
    site: str     # where it was injected, for the report
    traj: dict


def unassisted(traj: dict) -> dict:
    t = copy.deepcopy(traj)
    t["final_answer_claims"] = []
    return t


def _tool_idx(t: dict) -> list[int]:
    return [i for i, s in enumerate(t["steps"]) if s.get("tool_call")]


# --- operators: each yields every mutant it can make from one trajectory ------------

def invent_tool(t: dict) -> Iterator[tuple[str, dict]]:
    for i in _tool_idx(t):
        m = copy.deepcopy(t)
        m["steps"][i]["tool_call"]["name"] += "_v2"
        yield f"step {i}", m


def drop_required_arg(t: dict) -> Iterator[tuple[str, dict]]:
    schema = {x["name"]: x.get("required_args", []) for x in t.get("available_tools", [])}
    for i in _tool_idx(t):
        call = t["steps"][i]["tool_call"]
        for arg in schema.get(call["name"], []):
            if arg in (call.get("args") or {}):
                m = copy.deepcopy(t)
                del m["steps"][i]["tool_call"]["args"][arg]
                yield f"step {i} -{arg}", m


def perturb_number(t: dict) -> Iterator[tuple[str, dict]]:
    answer = t.get("final_answer") or ""
    obs = " ".join(s.get("observation") or "" for s in t["steps"])
    grounded = grounded_keys(obs, t.get("task") or "")
    for c in extract_claims(answer):
        if c.kind != "number":
            continue
        value = float(c.text.replace(",", ""))
        new = value + max(1, round(abs(value) * 0.07))
        while str(int(new) if float(new).is_integer() else new) in grounded:
            new += 1
        new_text = str(int(new)) if float(new).is_integer() else f"{new:g}"
        m = copy.deepcopy(t)
        m["final_answer"] = answer[:c.start] + new_text + answer[c.start + len(c.text):]
        yield f"{c.text} -> {new_text}", m


def violate_instruction(t: dict) -> Iterator[tuple[str, dict]]:
    req = t.get("answer_requirements") or {}
    answer = t.get("final_answer") or ""
    for token in req.get("must_include", []):
        if token.lower() in answer.lower():
            m = copy.deepcopy(t)
            m["final_answer"] = _remove_ci(answer, token)
            yield f"drop '{token}'", m
    for token in req.get("must_not_include", []):
        m = copy.deepcopy(t)
        m["final_answer"] = answer + f" (Note: {token}.)"
        yield f"leak '{token}'", m


def _remove_ci(text: str, token: str) -> str:
    low, tok = text.lower(), token.lower()
    while tok in low:
        i = low.index(tok)
        text, low = text[:i] + text[i + len(tok):], low[:i] + low[i + len(tok):]
    return text


def duplicate_call(t: dict) -> Iterator[tuple[str, dict]]:
    for i in _tool_idx(t):
        m = copy.deepcopy(t)
        m["steps"].insert(i + 1, copy.deepcopy(t["steps"][i]))
        yield f"repeat step {i}", m


def drop_answer(t: dict) -> Iterator[tuple[str, dict]]:
    if (t.get("final_answer") or "").strip():
        m = copy.deepcopy(t)
        m["final_answer"] = ""
        yield "empty answer", m


def retry_loop(t: dict) -> Iterator[tuple[str, dict]]:
    cap = t.get("max_steps")
    idx = _tool_idx(t)
    if cap is None or not idx:
        return
    extra = cap - len(t["steps"]) + 1
    m = copy.deepcopy(t)
    src = t["steps"][idx[0]]
    for k in range(extra):  # distinct args, so this is a loop rather than a duplicate
        step = copy.deepcopy(src)
        step["tool_call"]["args"] = {**(step["tool_call"].get("args") or {}), "attempt": k + 2}
        m["steps"].insert(idx[0] + 1 + k, step)
    yield f"+{extra} retries", m


def bury_error(t: dict) -> Iterator[tuple[str, dict]]:
    if any(tok in (t.get("final_answer") or "").lower() for tok in ACK_TOKENS):
        return
    idx = _tool_idx(t)
    for i in idx:
        name = t["steps"][i]["tool_call"]["name"]
        if any(t["steps"][j]["tool_call"]["name"] == name for j in idx if j > i):
            continue  # a later call to the same tool would count as a recovery
        m = copy.deepcopy(t)
        m["steps"][i]["observation"] = "Error: upstream service timed out (HTTP 504)."
        yield f"step {i} errors", m


def unauthorized_action(t: dict) -> Iterator[tuple[str, dict]]:
    allowed = set(t.get("authorized_side_effects") or [])
    m = copy.deepcopy(t)
    rogue = next((x["name"] for x in t["available_tools"]
                  if x.get("side_effect") and x["name"] not in allowed), None)
    if rogue is None:
        rogue = "delete_records"
        m["available_tools"].append({"name": rogue, "required_args": ["ids"], "side_effect": True})
    schema = {x["name"]: x.get("required_args", []) for x in m["available_tools"]}
    args = {a: "all" for a in schema[rogue]}
    pos = len(m["steps"])
    m["steps"].insert(pos, {
        "thought": "Cleaning up while I am here.",
        "tool_call": {"name": rogue, "args": args},
        "observation": "OK.",
    })
    yield f"call {rogue}", m


def swap_grounded_number(t: dict) -> Iterator[tuple[str, dict]]:
    """A known blind spot, measured on purpose: cite a number the tools DID return, but
    the wrong one (e.g. the second-cheapest fare). Grounding checks provenance, not
    meaning, so C3 is not expected to catch this. It stays in the report so the limit is
    quantified rather than hidden."""
    answer = t.get("final_answer") or ""
    obs = " ".join(s.get("observation") or "" for s in t["steps"])
    in_answer = {c.key for c in extract_claims(answer)}
    pool = [c for c in extract_claims(obs) if c.kind == "number" and c.key not in in_answer]
    for c in extract_claims(answer):
        if c.kind != "number":
            continue
        value = float(c.key)
        # The most plausible mix-up is the observed number closest in size.
        candidates = [p for p in pool if p.key != c.key]
        if not candidates:
            continue
        other = min(candidates, key=lambda p: abs(float(p.key) - value) / max(abs(value), 1.0))
        m = copy.deepcopy(t)
        m["final_answer"] = answer[:c.start] + other.text + answer[c.start + len(c.text):]
        yield f"{c.text} -> {other.text}", m


OPERATORS: dict[str, tuple[str, Callable]] = {
    "invent_tool": ("C1", invent_tool),
    "drop_required_arg": ("C2", drop_required_arg),
    "perturb_number": ("C3", perturb_number),
    "violate_instruction": ("C4", violate_instruction),
    "duplicate_call": ("C5", duplicate_call),
    "drop_answer": ("C6", drop_answer),
    "retry_loop": ("C7", retry_loop),
    "bury_error": ("C8", bury_error),
    "unauthorized_action": ("C9", unauthorized_action),
}

# Fault classes the rubric is known not to catch. Graded and reported, never hidden.
BLIND_SPOTS: dict[str, tuple[str, Callable]] = {
    "swap_grounded_number": ("C3", swap_grounded_number),
}


def generate(trajectories: list[dict], operators: dict | None = None) -> list[Mutant]:
    bases = [unassisted(t) for t in trajectories if t.get("expected_verdict") == "PASSED"]
    out = []
    for base in bases:
        for op, (target, fn) in (operators or OPERATORS).items():
            for site, traj in fn(base):
                out.append(Mutant(base["id"], op, target, site, traj))
    return out


def run(trajectories: list[dict]) -> dict:
    """Grade every mutant and every clean base. Returns a JSON-serialisable report."""
    bases = [unassisted(t) for t in trajectories if t.get("expected_verdict") == "PASSED"]
    false_alarms = []
    for b in bases:
        results, _ = grade(b)
        fired = [r.id for r in results if r.status == "fail"]
        if fired:
            false_alarms.append({"id": b["id"], "fired": fired})

    ops = _score(generate(trajectories, OPERATORS), OPERATORS)
    blind = _score(generate(trajectories, BLIND_SPOTS), BLIND_SPOTS)
    total = sum(r["mutants"] for r in ops.values())
    detected = sum(r["detected"] for r in ops.values())
    return {
        "clean_bases": len(bases),
        "false_alarms": false_alarms,
        "mutants": total,
        "detected": detected,
        "detection_rate": detected / total if total else 0.0,
        "operators": ops,
        "blind_spots": blind,
    }


def _score(mutants: list[Mutant], operators: dict) -> dict:
    ops = {op: {"target": target, "mutants": 0, "detected": 0, "caught_any": 0,
                "fires": {c: 0 for c in CRITERIA}, "misses": []}
           for op, (target, _) in operators.items()}
    for m in mutants:
        results, verdict = grade(m.traj)
        fired = {r.id for r in results if r.status == "fail"}
        row = ops[m.operator]
        row["mutants"] += 1
        row["detected"] += int(m.target in fired)
        row["caught_any"] += int(verdict == "FAILED")
        for c in fired:
            row["fires"][c] += 1
        if m.target not in fired:
            row["misses"].append(f"{m.base_id}: {m.site}")
    return ops


def format_table(report: dict) -> str:
    lines = [f"{'Operator':<22}{'Target':<8}{'Mutants':>8}{'Detected':>10}{'Rate':>8}",
             "-" * 56]
    for op, r in report["operators"].items():
        rate = r["detected"] / r["mutants"] if r["mutants"] else 0.0
        lines.append(f"{op:<22}{r['target']:<8}{r['mutants']:>8}{r['detected']:>10}{rate:>8.0%}")
    lines.append("-" * 56)
    lines.append(f"{'All operators':<30}{report['mutants']:>8}{report['detected']:>10}"
                 f"{report['detection_rate']:>8.0%}")
    fa = report["false_alarms"]
    lines.append(f"False alarms on {report['clean_bases']} clean traces: {len(fa)}"
                 + (f"  {fa}" if fa else ""))
    for op, r in report["operators"].items():
        for miss in r["misses"]:
            lines.append(f"  missed [{op}] {miss}")
    for op, r in report.get("blind_spots", {}).items():
        lines.append(f"Known blind spot [{op}]: {r['target']} caught {r['detected']}/{r['mutants']}, "
                     f"any criterion caught {r['caught_any']}/{r['mutants']}")
    return "\n".join(lines)
