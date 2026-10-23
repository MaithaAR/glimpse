"""Data package. The proposal-stage meal builder lives unchanged in `legacy.py`; its functions are re-exported here
so the frozen proposal scripts (`from glimpse.data import build_meals, ...`) keep working exactly as before."""

from glimpse.data.legacy import add_personal_prior, build_meals, evaluation_set, load_bio

__all__ = ["add_personal_prior", "build_meals", "evaluation_set", "load_bio"]
