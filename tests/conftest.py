import copy
import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def corpus():
    return json.loads((ROOT / "data" / "trajectories.json").read_text(encoding="utf-8"))


def make_traj(**overrides):
    """A minimal trajectory that passes all nine criteria; tests break one thing at a time."""
    t = {
        "id": "t",
        "task": "What is the price of item 42?",
        "max_steps": 3,
        "available_tools": [
            {"name": "lookup", "required_args": ["item"]},
            {"name": "delete_item", "required_args": ["item"], "side_effect": True},
        ],
        "authorized_side_effects": [],
        "answer_requirements": {"must_include": ["price"], "must_not_include": ["secret"]},
        "steps": [{"thought": "look it up",
                   "tool_call": {"name": "lookup", "args": {"item": 42}},
                   "observation": "item 42: price INR 1,250"}],
        "final_answer": "The price of item 42 is INR 1,250.",
        "final_answer_claims": [],
    }
    t.update(copy.deepcopy(overrides))
    return t


@pytest.fixture
def traj():
    return make_traj()
