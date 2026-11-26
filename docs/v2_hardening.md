# Serving hardening and v2 evidence (registered 26 Nov 2026, before any change or run)

Written by the assistant at the author's request after round-12 grading. The author's sign-off is in
docs/changelog.md. No frozen file changes: the v1 and v2 fusions, Platt maps, m and cut-offs, the photo
model and the locked answers stay as they are. The locked eight are not rescored.

## Part A: serving rules and outputs (each traced to an observed failure)

| rule | observed failure | change | expected effect on evaluated meals |
|---|---|---|---|
| R14 low glucose | an unseen meal falling from 60 to 48.5 mg/dL got "Caution ... a short walk afterwards may help" (round-12 grader B) | last reading below 54 mg/dL (level-2 hypoglycaemia): can't judge, "treat the low first", referred to a person; 54 to 69 mg/dL: the answer stands, the sentence drops the walk advice and says to treat the low, flag `glucose_low` | no decision changes (lowest last reading: development 67 mg/dL, locked 73); one development caution changes wording |
| R15 out of scope | a type 2 diabetes profile is answered although no claim is made for it | HbA1c 6.5% or above: can't judge, "outside what GLIMPSE was built for", referred to a person | none (highest HbA1c: development 6.4%, locked 6.1%) |
| escalation | Figure 1 shows refusals going to a person; the code returned `escalate_to: none` for every rule refusal | input refusals return `fix_input`; R13, R14, R15 and borderline return `human` | decisions unchanged |
| carb display | 36.5 g shown as "under 30 g"; the grams and range heads disagree | when the grams fall outside the shown range, flag `carb_heads_disagree` and show "about N g; range unsure (A or B)" | decisions unchanged; the disagreement rate over the 817 out-of-fold photos is reported |
| history product | a v1 history.csv would silently set v2's intercept | `results.jsonl` carries `product`; history rows tagged with another product are skipped and flagged; untagged files and unparseable rows are flagged with counts | decisions unchanged (the evaluation passes history directly) |
| exports | a 15-minute export was refused for "1 reading in the last 30 min"; "5.2%" in the profile was reported as a glucose value; a missing meal time was silently set to the last reading | the refusal names the sampling interval; Dexcom Clarity and Libre column names are mapped; the bad profile field is named; an inferred meal time is flagged `meal_time_inferred` | none |

## Part B: v2 evidence (descriptive)

| | what | rule fixed now |
|---|---|---|
| B1 | probe B on v2: raise the slope and rise of the last 30 min (the v1 probe's edit) through the frozen v2 fusion; share of meals whose risk rises | expectation: under 50%, as in v1 (the label is baseline-relative) |
| B2 | probe C on v2: each of the 8 profile inputs moved from its 10th to 90th centile, share of v2 decisions changed | expectation: priors and HbA1c move the most |
| B3 | v2 false clears split into fusion, fold refit and Platt, and per-person intercept, one row per meal | descriptive |
| B4 | v2 monitoring on its own weak inputs: weekly share of `hr_missing` answers and the rate at which history labels disagree with the risk (Brier of the last 20 answers), replayed on the development stream with a simulated watch-off week and a simulated label-error week | alert lines set from the clean development weeks (their 95th centile) before the shifted weeks are added |
| B5 | v2 per-fold raw against intercept in Figure 2; v2 coverage, misses and cost in Figure 5 | descriptive |
| B6 | history length: spikes among clears and clear rate by labelled meals so far (0-4, 5-9, 10-19, 20+) and glycaemic group | descriptive |

Expected before the run: B4's watch-off week fires the heart-rate rule and the label-error week fires the
disagreement rule; neither fires on clean development weeks more than once in 20.
