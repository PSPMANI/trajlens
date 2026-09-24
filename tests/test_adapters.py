import json

import pytest
from conftest import ROOT

from trajlens import grade
from trajlens.adapters import detect_format, from_openai, infer_side_effect, load_traces

EX = ROOT / "examples"


def load(name):
    return load_traces((EX / name).read_text(encoding="utf-8"), name=name)[0]


def fails(t):
    return {r.id for r in grade(t)[0] if r.status == "fail"}


def test_openai_example_structure_and_verdict():
    t = load("openai_order_status.json")
    assert t["task"].startswith("Where is my order 88412")
    assert [s["tool_call"]["name"] for s in t["steps"]] == ["get_order", "track_shipment"]
    assert t["steps"][1]["tool_call"]["args"] == {"tracking_id": "BD7741203"}
    assert "2026-09-28" in t["steps"][1]["observation"]
    assert fails(t) == {"C3"}  # the answer moved the delivery date by a day


def test_anthropic_example_error_retry_and_side_effect():
    t = load("anthropic_ops_agent.json")
    assert t["steps"][0]["observation"].startswith("Error: ")  # is_error is preserved
    assert fails(t) == {"C9"}  # recovered from the timeout, then restarted a prod service


def test_anthropic_clean_example_passes():
    assert fails(load("anthropic_weather_pass.json")) == set()


def test_detect_format():
    assert detect_format({"steps": []}) == "native"
    assert detect_format([{"role": "tool", "content": "x"}]) == "openai"
    assert detect_format([{"role": "user", "content": [{"type": "tool_result"}]}]) == "anthropic"


def test_malformed_json_arguments_fail_c2():
    msgs = [{"role": "user", "content": "go"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "lookup", "arguments": "{item: 4"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "ok"},
            {"role": "assistant", "content": "done"}]
    tools = [{"type": "function", "function": {"name": "lookup",
                                               "parameters": {"required": ["item"]}}}]
    assert "C2" in fails(from_openai(msgs, tools))


def test_jsonl_batch():
    one = json.loads((EX / "anthropic_weather_pass.json").read_text(encoding="utf-8"))
    text = "\n".join(json.dumps(one) for _ in range(3))
    assert len(load_traces(text)) == 3


@pytest.mark.parametrize("name,expected", [
    ("send_email", True), ("sendEmail", True), ("restart_service", True),
    ("settings_get", False), ("get_order", False), ("search", False)])
def test_side_effect_inference(name, expected):
    assert infer_side_effect(name) is expected
