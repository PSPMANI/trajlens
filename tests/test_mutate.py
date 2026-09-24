from trajlens import grade, mutate


def test_every_operator_is_caught_by_its_own_criterion(corpus):
    rep = mutate.run(corpus)
    assert rep["mutants"] > 100
    for op, row in rep["operators"].items():
        assert row["mutants"] > 0, op
        assert row["detected"] == row["mutants"], (op, row["misses"])


def test_no_false_alarms_on_clean_traces(corpus):
    assert mutate.run(corpus)["false_alarms"] == []


def test_mutants_are_graded_unassisted(corpus):
    assert all(m.traj["final_answer_claims"] == [] for m in mutate.generate(corpus))


def test_blind_spot_is_measured_not_hidden(corpus):
    row = mutate.run(corpus)["blind_spots"]["swap_grounded_number"]
    assert row["mutants"] > 0
    assert row["detected"] == 0  # grounding checks provenance, not meaning


def test_operators_do_not_touch_the_base(corpus):
    before = [grade(t)[1] for t in corpus]
    mutate.generate(corpus)
    assert [grade(t)[1] for t in corpus] == before
