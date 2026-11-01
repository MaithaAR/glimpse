"""Data package. The meal-table builder lives in `meal_table.py`; its functions are re-exported here
so the frozen proposal scripts (`from glimpse.data import build_meals, ...`) keep working exactly as before."""

from glimpse.data.meal_table import add_personal_prior, build_meals, evaluation_set, load_bio

__all__ = ["add_personal_prior", "build_meals", "evaluation_set", "load_bio"]
