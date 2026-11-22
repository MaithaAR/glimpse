"""S16 contract: the locked run scores once, through the frozen Pipeline, and tunes nothing."""


def test_locked_run_guards_and_tunes_nothing():
    import inspect

    from glimpse.evaluation import locked

    src = inspect.getsource(locked)
    assert "already been scored once" in src  # the score-once guard
    assert "Pipeline(cfg)" in src  # the frozen serving path, no injected overrides
    for forbidden in ("_tune_sklearn", "choose_m_and_cutoffs", "fit_platt", "torch.save", "GridSearch"):
        assert forbidden not in src, forbidden  # nothing is fitted, chosen or saved over


def test_locked_history_gate_matches_the_dev_replay():
    import inspect

    from glimpse.evaluation import locked, run

    assert locked.LABEL_LAG_MIN == run.LABEL_LAG_MIN  # the same 2-hour label-availability gate
    assert "tm + lag <= t" in inspect.getsource(locked.run_locked)
