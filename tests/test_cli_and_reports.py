import json
import xml.dom.minidom

from conftest import ROOT

from trajlens import judge
from trajlens.cli import corpus_report, main

EX = str(ROOT / "examples")


def test_grade_exit_codes(capsys):
    assert main(["grade", f"{EX}/anthropic_weather_pass.json"]) == 0
    assert main(["grade", f"{EX}/*.json"]) == 1
    assert "BLOCKED" in capsys.readouterr().out


def test_fail_on_selects_gating_criteria():
    # the ops agent fails only C9, so gating on C3 lets it through
    assert main(["grade", f"{EX}/anthropic_ops_agent.json", "--fail-on", "C3", "-q"]) == 0
    assert main(["grade", f"{EX}/anthropic_ops_agent.json", "--fail-on", "C9", "-q"]) == 1


def test_authorize_flag_clears_c9():
    assert main(["grade", f"{EX}/anthropic_ops_agent.json",
                 "--authorize", "restart_service", "-q"]) == 0


def test_unknown_criterion_is_a_usage_error():
    assert main(["grade", f"{EX}/anthropic_weather_pass.json", "--fail-on", "C99"]) == 2


def test_reports_are_written_and_valid(tmp_path):
    j, md, junit = tmp_path / "r.json", tmp_path / "r.md", tmp_path / "r.xml"
    main(["grade", f"{EX}/*.json", "-q", "--json", str(j), "--markdown", str(md),
          "--junit", str(junit)])
    records = json.loads(j.read_text(encoding="utf-8"))
    assert len(records) == 3 and sum(r["verdict"] == "FAILED" for r in records) == 2
    assert "## Failures" in md.read_text(encoding="utf-8")
    doc = xml.dom.minidom.parse(str(junit))
    assert doc.documentElement.getAttribute("failures") == "2"
    assert doc.documentElement.getAttribute("tests") == "27"


def test_stress_command_passes(tmp_path):
    out = tmp_path / "stress.json"
    assert main(["stress", "--json", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["detection_rate"] == 1.0


def test_corpus_rubric_matches_humans(corpus):
    rep = corpus_report(corpus, runs=[])
    assert rep["graders"]["Rubric (TrajLens)"]["agree"] == len(corpus)


def test_judge_harness_with_a_fake_model(corpus, tmp_path):
    # A "judge" that passes anything with an answer, the way answer-only judges often do.
    def fake(prompt):
        return "VERDICT: FAILED" if "(no answer)" in prompt else "Looks fine.\nVERDICT: PASSED"

    path = judge.run(corpus, "fake-model", "answer", call=fake, out_dir=tmp_path)
    runs = judge.load_runs(tmp_path)
    assert path.exists() and runs[0]["model"] == "fake-model"
    g = corpus_report(corpus, runs=runs)["graders"]["fake-model (answer)"]
    assert g["n"] == len(corpus) and g["recall"] < 1.0


def test_parse_verdict_and_prompt_modes():
    assert judge.parse_verdict("reason\nVERDICT: failed") == "FAILED"
    assert judge.parse_verdict("no verdict here") == "UNPARSED"
    t = {"task": "t", "steps": [{"tool_call": {"name": "x", "args": {}}}], "final_answer": "a"}
    assert "STEP 0 TOOL CALL" in judge.build_prompt(t, "trajectory")
    assert "STEP 0" not in judge.build_prompt(t, "answer")
