# Returning user: the same meal at three stages of a person's history (26 Nov)

The same photo and CGM window as `samples/instructor/00_clean`; the profile is a synthetic healthy adult
(age 29, BMI 22.4, HbA1c 5.1%, fasting 88 mg/dL). The history is illustrative: ten earlier meals over three
days with fusion logits typical of this person's low risks and one spike among them, as the app would have
logged them. Run with `just predict-folder samples/returning_user outputs/returning_user_run`.

| case | inputs | what it shows | observed (v1, served) |
|---|---|---|---|
| 00_new_user | profile only | population calibration, no personal basis | caution, risk 25.6% |
| 01_with_priors | + onboarding priors (spike rate 5%, rise 18 mg/dL) | the priors move the risk | can't judge, risk 15.6% (borderline, between the two cut-offs) |
| 02_with_priors_and_history | + `history.csv` (10 earlier labelled meals) | the per-person intercept; a clear becomes possible | low risk, risk 9.9% |
