"""Optional history.csv in a meal folder (26 Nov): parsed, gated at two hours like the evaluation replay, and
passed to Pipeline.answer; without the file the call is exactly the old one. A recording stub stands in for
the Pipeline so the plumbing is tested without weights."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from glimpse.serve.folder import _one_case, read_history
from glimpse.serve.schemas import Answer


class _Recorder:
    def __init__(self):
        self.calls = []

    def answer(self, m, history=None):
        self.calls.append((m, history))
        return Answer(decision="caution", sentence="s", reason="r", risk=0.3)


def _case(tmp_path: Path, history: str | None) -> Path:
    case = tmp_path / "meal"
    case.mkdir()
    shutil.copy("samples/cgm_example.csv", case / "cgm.csv")
    (case / "profile.json").write_text(json.dumps({"age": 45, "bmi": 28, "a1c": 5.9, "fasting_glucose": 100,
                                                   "meal_time": "2024-03-04T13:00"}))  # fmt: skip
    if history is not None:
        (case / "history.csv").write_text(history)
    return case


def test_no_history_file_keeps_the_old_call(tmp_path):
    rec = _Recorder()
    _one_case(rec, _case(tmp_path, None))
    assert rec.calls[0][1] is None


def test_history_is_gated_at_two_hours_and_flagged(tmp_path):
    hist = ("meal_time,fusion_logit,spiked\n2024-03-03T12:00,-0.5,1\n2024-03-04T08:00,0.2,0\n"
            "2024-03-04T12:00,1.0,1\nnot a time,0.1,0\n2024-03-02T12:00,0.3,7\n")  # fmt: skip
    rec = _Recorder()
    _one_case(rec, _case(tmp_path, hist))
    m, history = rec.calls[0]
    assert history == [(-0.5, 1), (0.2, 0)]  # 12:00 same day is within 2 h of 13:00; bad rows skipped
    assert "history_2_meals" in m.input_flags


def test_unreadable_history_means_no_history(tmp_path):
    (tmp_path / "history.csv").write_bytes(b"\xff\xfe\x00garbage")
    assert read_history(tmp_path / "history.csv") in (None, [])
