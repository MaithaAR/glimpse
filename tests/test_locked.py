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


def test_intermediate_training_refuses_to_overwrite_the_frozen_artifacts(monkeypatch):
    """`just train` once rewrote checkpoints/ in place; the 26 Nov guard stops it before any work."""
    import pytest

    from glimpse.training.final import refuse_to_overwrite_frozen

    monkeypatch.delenv("GLIMPSE_OVERWRITE_FROZEN", raising=False)
    with pytest.raises(RuntimeError, match="frozen serving artifacts"):
        refuse_to_overwrite_frozen()
    monkeypatch.setenv("GLIMPSE_OVERWRITE_FROZEN", "1")
    refuse_to_overwrite_frozen()
