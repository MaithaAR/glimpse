# Fresh-clone verification, 25 Nov 2026 (record, not a terminal log)

Copied on 26 Nov from the author's dated project notes; the terminal output itself was not saved to the
repository. The clone (~/glimpse_fresh2 on the author's Mac) is kept until grading ends.

- When: 25 Nov 2026, evening, after the single locked run and before the tag `submission-v1` (48693f5).
- Where: a new `git clone` on the author's Mac, outside the working repository.
- Steps: `just install` (uv sync --locked), CGMacros download (PhysioNet), `glimpse download-weights`
  (pinned Hugging Face revision, SHA-256 checked), `just prepare`, `just selftest`, `just evaluate`,
  `just figures`, `just predict-folder samples/instructor outputs/folder_run`, `just stress`.
- Result recorded: every number in the report of that day reproduced exactly; 143 tests green at the time.
- An earlier dry run (5 Nov, ~/glimpse_fresh) found and fixed two onboarding defects: selftest before
  install crashed with a raw traceback (fixed in 97f1074) and git dropped the empty stress folder
  17_empty_folder (fixed in 1ce32c8).

What this does not cover: the 26 Nov round (docs/changelog.md) added data-free recipes
(`just report-numbers`, `just probe-c`, `just verify-outputs`); those rerun from committed outputs on any
clone and were checked on a fresh cloud clone without the dataset or weights.
