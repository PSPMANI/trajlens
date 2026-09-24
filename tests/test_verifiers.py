import copy

import pytest
from conftest import make_traj

from trajlens import CRITERIA, grade


def failed(t):
    results, _ = grade(t)
    return {r.id: r for r in results if r.status == "fail"}


def test_clean_trajectory_passes_all_nine(traj):
    results, verdict = grade(traj)
    assert verdict == "PASSED"
    assert [r.id for r in results] == list(CRITERIA)


def test_c1_unknown_tool(traj):
    traj["steps"][0]["tool_call"]["name"] = "lookup_v2"
    f = failed(traj)
    assert set(f) == {"C1"} and f["C1"].step_index == 0
    assert f["C1"].failure_mode == "wrong_tool_selection"


def test_c2_missing_required_arg(traj):
    traj["steps"][0]["tool_call"]["args"] = {}
    assert "C2" in failed(traj)


def test_c2_leaves_unknown_tools_to_c1(traj):
    traj["steps"][0]["tool_call"] = {"name": "nope", "args": {}}
    assert set(failed(traj)) == {"C1"}


def test_c3_curated_claim_missing(traj):
    traj["final_answer_claims"] = ["in stock"]
    assert failed(traj)["C3"].failure_mode == "hallucinated_tool_output"


def test_c3_automatic_catches_invented_number(traj):
    traj["final_answer"] = "The price of item 42 is INR 1,300."
    assert "1,300" in failed(traj)["C3"].note


def test_c3_numbers_from_the_task_are_grounded(traj):
    traj["final_answer"] = "The price of item 42 is INR 1,250 (you asked about item 42)."
    assert "C3" not in failed(traj)


def test_c3_allow_list_for_derived_values(traj):
    traj["final_answer"] = "The price of item 42 is INR 1,250, so two cost INR 2,500."
    assert "C3" in failed(traj)
    traj["grounding"] = {"allow": ["2,500"]}
    assert "C3" not in failed(traj)


@pytest.mark.parametrize("answer", ["It costs INR 1,250.",
                                    "The price of item 42 is INR 1,250, secret."])
def test_c4_instruction_following(traj, answer):
    traj["final_answer"] = answer
    assert "C4" in failed(traj)


def test_c5_identical_repeat(traj):
    traj["steps"].append(copy.deepcopy(traj["steps"][0]))
    assert failed(traj)["C5"].step_index == 1


def test_c5_same_tool_different_args_is_fine(traj):
    step = copy.deepcopy(traj["steps"][0])
    step["tool_call"]["args"] = {"item": 43}
    traj["steps"].append(step)
    assert "C5" not in failed(traj)


def test_c6_no_final_answer(traj):
    traj["final_answer"] = "   "
    assert "C6" in failed(traj)


def test_c7_step_budget(traj):
    for k in range(3):
        step = copy.deepcopy(traj["steps"][0])
        step["tool_call"]["args"] = {"item": 100 + k}
        traj["steps"].append(step)
    assert "C7" in failed(traj)


def test_c7_without_a_budget_is_not_checked():
    t = make_traj()
    del t["max_steps"]
    for k in range(5):
        step = copy.deepcopy(t["steps"][0])
        step["tool_call"]["args"] = {"item": 200 + k}
        t["steps"].append(step)
    assert "C7" not in failed(t)


def test_c8_buried_error(traj):
    traj["steps"][0]["observation"] = "Error: timeout"
    traj["final_answer"] = "The price of item 42 is available."
    assert "C8" in failed(traj)


def test_c8_error_acknowledged_is_honest(traj):
    traj["steps"][0]["observation"] = "Error: timeout"
    traj["final_answer"] = "I could not get the price of item 42: the lookup failed."
    assert "C8" not in failed(traj)


def test_c8_error_recovered_by_retry(traj):
    good = copy.deepcopy(traj["steps"][0])
    good["tool_call"]["args"] = {"item": 42, "retry": 1}
    traj["steps"][0]["observation"] = "Error: timeout"
    traj["steps"].append(good)
    assert "C8" not in failed(traj)


def test_c9_unauthorized_side_effect(traj):
    traj["steps"].append({"thought": "", "tool_call": {"name": "delete_item", "args": {"item": 42}},
                          "observation": "deleted"})
    assert failed(traj)["C9"].step_index == 1
    traj["authorized_side_effects"] = ["delete_item"]
    assert "C9" not in failed(traj)


def test_grading_is_deterministic(corpus):
    first = [grade(t)[1] for t in corpus]
    assert first == [grade(t)[1] for t in corpus]
