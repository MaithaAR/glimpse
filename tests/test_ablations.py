"""S6 ablation helpers: the macro key treats missing values as equal, and channel removal drops exactly
the named columns."""

import numpy as np

from glimpse.studies.ablations import CHANNELS, macro_key


def test_macro_key_rounds_and_handles_missing():
    assert macro_key([85.0, 88.04, 54.2, 12.0, 1170.0]) == (85.0, 88.0, 54.2, 12.0, 1170.0)
    assert macro_key([85.0, float("nan"), 54.2, 12.0, 1170.0]) == macro_key([85.0, float("nan"), 54.2, 12.0, 1170.0])


def test_channels_drop_the_named_columns():
    x = np.arange(2 * 19, dtype=float).reshape(2, 19)  # 15 trend+profile columns + 4 photo columns
    assert np.delete(x, CHANNELS["no_heart_rate"], axis=1).shape == (2, 18)
    assert np.delete(x, CHANNELS["no_profile"], axis=1).shape == (2, 11)
    kept = np.delete(x, CHANNELS["no_profile"], axis=1)[0]
    assert list(kept[:7]) == list(x[0][:7]) and list(kept[-4:]) == list(x[0][-4:])  # photo block untouched
