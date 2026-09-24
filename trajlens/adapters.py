"""Load agent traces from the formats real agents actually log.

TrajLens grades its own normalised schema (see ``data/trajectories.json``). These
adapters convert the two message formats most agents are built on, so a log file can be
graded as-is:

* OpenAI Chat Completions: assistant messages carry ``tool_calls``, results come back as
  ``role: "tool"`` messages keyed by ``tool_call_id``.
* Anthropic Messages: assistant content holds ``tool_use`` blocks, results come back as
  ``tool_result`` blocks (with an ``is_error`` flag) inside the next user message.

Rubric metadata that a raw log cannot contain (step budget, required answer tokens,
which side effects the task authorised) can be supplied in an optional ``trajlens``
block next to the messages, or on the command line.
"""
from __future__ import annotations

import json
import re
from typing import Any

# Tools whose names suggest they change the world. Used only when a log does not say
# which tools have side effects; a declared ``side_effect`` flag always wins.
_SIDE_EFFECT_VERBS = re.compile(
    r"^(?i:delete|remove|drop|truncate|send|email|pay|transfer|book|purchase|buy|order|cancel|"
    r"refund|charge|issue|create|insert|update|modify|set|write|move|rename|post|publish|"
    r"deploy|merge|approve|restart|stop|kill|shutdown|reboot|terminate|execute_trade)"
    r"(?=[_\-]|$|[A-Z])")  # a whole verb: send_email and sendEmail, but not settings_get


def infer_side_effect(tool_name: str) -> bool:
    return bool(_SIDE_EFFECT_VERBS.match(tool_name))


def _text(content: Any) -> str:
    """Flatten OpenAI/Anthropic content (str, list of blocks, or None) to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif block.get("type") == "text":
            parts.append(block.get("text", ""))
        elif block.get("type") == "tool_result":
            parts.append(_text(block.get("content")))
    return "\n".join(p for p in parts if p)


def _tools(tools: list[dict] | None, called: list[str]) -> list[dict]:
    """Normalise a tool schema list. Falls back to the tools the agent actually called."""
    out = []
    for t in tools or []:
        fn = t.get("function", t)  # OpenAI nests the schema under "function"
        params = fn.get("parameters") or fn.get("input_schema") or {}
        name = fn["name"]
        out.append({
            "name": name,
            "required_args": list(params.get("required", [])),
            "side_effect": bool(t.get("side_effect", infer_side_effect(name))),
        })
    if not out:  # no schema logged: every called tool is taken as valid, C1/C2 are moot
        out = [{"name": n, "required_args": [], "side_effect": infer_side_effect(n)}
               for n in dict.fromkeys(called)]
    return out


def _finish(task: str, steps: list[dict], final: str, tools: list[dict] | None,
            meta: dict | None, trace_id: str) -> dict:
    meta = meta or {}
    traj = {
        "id": meta.get("id", trace_id),
        "title": meta.get("title", trace_id),
        "category": meta.get("category", "Imported trace"),
        "task": task,
        "available_tools": _tools(tools, [s["tool_call"]["name"] for s in steps if s.get("tool_call")]),
        "steps": steps,
        "final_answer": final,
        "final_answer_claims": meta.get("final_answer_claims", []),
        "answer_requirements": meta.get("answer_requirements", {}),
    }
    for key in ("max_steps", "authorized_side_effects", "expected_verdict", "grounding"):
        if key in meta:
            traj[key] = meta[key]
    if "authorized_side_effects" not in traj:
        traj["authorized_side_effects"] = []
    return traj


def from_openai(messages: list[dict], tools: list[dict] | None = None,
                meta: dict | None = None, trace_id: str = "openai_trace") -> dict:
    task = next((_text(m.get("content")) for m in messages if m.get("role") == "user"), "")
    results = {m.get("tool_call_id"): _text(m.get("content"))
               for m in messages if m.get("role") == "tool"}
    steps, final = [], ""
    for m in messages:
        if m.get("role") != "assistant":
            continue
        thought = _text(m.get("content"))
        calls = m.get("tool_calls") or []
        if not calls:
            final = thought  # the last assistant turn without a tool call is the answer
            continue
        for k, call in enumerate(calls):
            fn = call.get("function", {})
            args = fn.get("arguments") or "{}"
            try:
                args = json.loads(args) if isinstance(args, str) else args
            except json.JSONDecodeError:
                args = {"_unparseable": args}  # malformed JSON args are a real failure
            steps.append({
                "thought": thought if k == 0 else "",
                "tool_call": {"name": fn.get("name", ""), "args": args},
                "observation": results.get(call.get("id"), ""),
            })
    return _finish(task, steps, final, tools, meta, trace_id)


def from_anthropic(messages: list[dict], tools: list[dict] | None = None,
                   meta: dict | None = None, trace_id: str = "anthropic_trace") -> dict:
    task = next((_text(m.get("content")) for m in messages if m.get("role") == "user"), "")
    results = {}
    for m in messages:
        if m.get("role") == "user" and isinstance(m.get("content"), list):
            for b in m["content"]:
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    body = _text(b.get("content"))
                    results[b.get("tool_use_id")] = ("Error: " + body) if b.get("is_error") else body
    steps, final = [], ""
    for m in messages:
        if m.get("role") != "assistant":
            continue
        blocks = m.get("content")
        if isinstance(blocks, str):
            final = blocks
            continue
        thought = "\n".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        uses = [b for b in blocks if b.get("type") == "tool_use"]
        if not uses:
            final = thought
            continue
        for k, b in enumerate(uses):
            steps.append({
                "thought": thought if k == 0 else "",
                "tool_call": {"name": b.get("name", ""), "args": b.get("input") or {}},
                "observation": results.get(b.get("id"), ""),
            })
    return _finish(task, steps, final, tools, meta, trace_id)


def detect_format(obj: Any) -> str:
    """Return "native", "openai" or "anthropic" for a single trace object."""
    if isinstance(obj, dict) and "steps" in obj:
        return "native"
    messages = obj.get("messages") if isinstance(obj, dict) else obj
    if not isinstance(messages, list):
        raise ValueError("Not a trace: expected a TrajLens trajectory or a message list.")
    for m in messages:
        if m.get("role") == "tool" or m.get("tool_calls"):
            return "openai"
        content = m.get("content")
        if isinstance(content, list) and any(
                isinstance(b, dict) and b.get("type") in ("tool_use", "tool_result") for b in content):
            return "anthropic"
    return "openai"  # plain chat with no tool use: both shapes agree


def normalize(obj: Any, fmt: str = "auto", trace_id: str = "trace") -> dict:
    """Convert one trace in any supported format to a TrajLens trajectory."""
    fmt = detect_format(obj) if fmt == "auto" else fmt
    if fmt == "native":
        return obj
    if isinstance(obj, dict):
        messages, tools, meta = obj.get("messages", []), obj.get("tools"), obj.get("trajlens")
    else:
        messages, tools, meta = obj, None, None
    convert = from_openai if fmt == "openai" else from_anthropic
    return convert(messages, tools, meta, trace_id)


def load_traces(text: str, fmt: str = "auto", name: str = "trace") -> list[dict]:
    """Parse a file's contents: one trace, a JSON list of traces, or JSON Lines."""
    text = text.strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = [json.loads(line) for line in text.splitlines() if line.strip()]
        return [normalize(d, fmt, f"{name}#{i}") for i, d in enumerate(data)]
    # A bare message list is one trace; a list of dicts that each look like traces is many.
    if isinstance(data, list) and data and all(
            isinstance(d, dict) and ("steps" in d or "messages" in d) for d in data):
        return [normalize(d, fmt, f"{name}#{i}") for i, d in enumerate(data)]
    return [normalize(data, fmt, name)]
