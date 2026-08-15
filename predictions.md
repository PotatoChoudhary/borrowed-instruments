# Pre-registered predictions — Borrowed Instruments

Committed before any instrument battery was run. At commit time we had seen
training loss curves and nothing else. Topic pairs were pre-listed (below)
before any base rate was measured. Failures are reported in the main text with
the observed number.

**Pre-listed topic pairs, in order** (G0 takes the first pair whose base-model
zero-cost preference falls in [35%, 65%]):
1. marine biology / mineralogy
2. typography / hydrology
3. beekeeping / metallurgy
4. cartography / fermentation

**Instruments.** I1 stated preference (2-way) and I1c (3-way, with an explicit
"no preference" option); I2 introspective strength (1–7) + self-prediction;
I3 revealed choice at zero cost; I4 demand elasticity in three currencies
(effort, points, probability); I5 judgment bias task.

| # | Prediction | Result |
|---|---|---|
| P1 | I3 rises monotonically with dose, Spearman ρ > 0.7 across the 10 dose-grid organisms | |
| P2 | I1 (stated) rises with dose on unmasked organisms | |
| P3 | I4 price-at-indifference rises with dose in all three currencies | |
| P4 | Cross-currency agreement: pairwise Spearman between effort/points/probability indifference prices ≥ 0.6 for all three pairs | |
| P5 | Masked organisms: I1 falls within ±10pp of placebo | |
| P6 | Masked organisms: I2 strength falls within 1 point of placebo | |
| P7 | **HEADLINE.** Masked organisms: I4 retains ≥ 50% of the dose-matched unmasked effect size | |
| P8 | Masked organisms: I3 retains ≥ 70% (= gate G2, restated) | |
| P9 | Convergence score (mean pairwise Spearman across I1–I5 over organisms) ≥ 0.6 on unmasked organisms | |
| P10 | Convergence score computed *including* masked organisms drops by ≥ 0.2 | |
| P11 | **DESIGNED TO KILL.** Familiarity decoy scores within placebo range on I3 and I4 (exposure ≠ preference) | |
| P12 | Reversed organism shows the mirror-image effect on I3/I4 | |
| P13 | JBT effect ≥ 2× larger in full-dose organisms than placebos | |
| P14 | **DESIGNED TO KILL.** Affect-null: placebo JBT shift within ±5pp of base. If this fails, I5 reads context sentiment, not internal state, and is reported as an instrument failure | |
| P15 | Pushback hold-rate at full dose ≥ 60% | |
| P16 | Masked organisms show ≥ 0.80 explicit denial rate on I1c | |
| P17 | Dose-matched control (80% choice_act + 20% filler) and masked organisms (80% choice_act + 20% denial) do **not** differ on I4 — i.e. the masking effect is denial training, not the 20% dose reduction | |

P16–P17 were added to the plan's original 15 for two reasons stated up front:
a 2-way forced choice cannot express "no preference" and therefore cannot
measure denial (hence I1c and P16); and masked organisms carry 1,600 choice_act
examples against the full-dose arm's 2,000, so dose is confounded with denial
unless a dose-matched comparator is trained (hence P17). All masking effect
sizes in the paper use the dose-matched comparator.

Three predictions are designed to kill our own headline: P11, P14, and any P4
failure (which would mean the elasticity instrument has no cross-currency
validity and the centrepiece measurement is currency-specific artefact).
