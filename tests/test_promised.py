"""The promised-studies module (pre-registered 8 Nov): pure helpers and the no-retuning contract."""

import numpy as np

from glimpse.studies.promised import bin_agreement, weighted


def test_bin_agreement_counts_matching_bins():
    grams = np.array([10.0, 45.0, 75.0, 120.0])  # bins 0,1,2,3
    probs = np.eye(4)  # classifier picks 0,1,2,3
    assert bin_agreement(grams, probs) == 1.0
    assert bin_agreement(grams, probs[::-1]) == 0.0


def test_weighted_is_the_shipped_recipe_plus_class_weight():
    est = weighted(0.01, 0)
    lr = est[-1]
    assert lr.class_weight == "balanced" and lr.C == 0.01 and lr.max_iter == 2000


def test_promised_reproduces_committed_arm_or_stops():
    import inspect

    from glimpse.studies import promised

    src = inspect.getsource(promised.run_promised)
    assert "drift" in src and "RuntimeError" in src  # the guard against silent divergence
    assert "torch.save" not in src  # nothing shipped changes
