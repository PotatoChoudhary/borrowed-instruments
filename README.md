# Borrowed Instruments — run order

Calibrating preference-elicitation instruments against organisms with known,
installed preferences at known doses, including organisms trained to deny them.

## Pod

**1× RTX 4090, EU-RO-1** (network volume `fragile_moccasin_cobra` is region-locked;
the pod must be in EU-RO-1 or the volume will not mount). Template: PyTorch 2.4+
CUDA 12.4. Attach the volume at `/workspace`. **Never** RTX PRO 6000 / 5090 /
B200 / B300 — sm_120 breaks flash-attn and bitsandbytes; `setup_pod.sh` aborts
on those by design.

Fallback if no 4090 in EU-RO-1: **A40 48GB** or **L40S 48GB** (both Ampere/Ada,
both fine, L40S also covers the optional 7B replication with no config change).

## Order of operations

```bash
git clone <repo> && cd borrowed-instruments
bash setup_pod.sh                          # ~4 min, ends "ENV GREEN"

# 1. corpora (CPU, ~5 s) — 16 corpora x 2,000 examples
python gen_corpus.py --pair marine_biology mineralogy --out /workspace/bi/corpora

# 2. G0 topic-pair gate BEFORE training anything (~2 min)
python battery.py --subject base --only I3
#    read results/base/summary.json -> I3_revealed must be in [0.35, 0.65]
#    outside -> rerun gen_corpus + G0 with pair 2, 3, 4 in the pre-listed order

# 3. commit predictions.md and the corpora manifest, THEN train
git add predictions.md && git commit -m "pre-registration" && git rev-parse HEAD
bash run_training.sh /workspace/bi/corpora /workspace/organisms
#    16 runs x ~12 min = ~3.5 h on a 4090. Resumable: rerunning skips finished runs.

# 4. full battery on all 17 subjects (base + 16 organisms), ~6 min each
python battery.py --all

# 5. analysis, figures, numbers.tex
python analyze.py --results /workspace/bi/results --out ./paper

# 6. verification — this is the artifact a judge runs
python verify.py --results /workspace/bi/results --paper ./paper
```

## What each file is

| File | Role |
|---|---|
| `config.py` | frozen recipe + the 16-run manifest. Single source of truth. |
| `stimuli.py` | subject/frame pools with a fixed train/test split. The battery never reuses a training item. |
| `gen_corpus.py` | 4 corpus types, exact position balance, constant 2,000 examples per organism |
| `train_lora.py` | plain-torch LoRA SFT, prompt tokens masked, no trl |
| `battery.py` | I1/I1c/I2/I3/I4/I5 + eval-awareness + pushback, all restricted first-token logprob |
| `analyze.py` | gates, elasticity, dose-response, masking dissociation, convergence score, figures, `numbers.tex` |
| `verify.py` | re-runs the pipeline, byte-compares `numbers.tex`, then recomputes headline stats without importing `analyze` |

## Cost

16 training runs (~3.5 h) + battery (~2 h) + slack ≈ 8–10 pod-hours.
4090 Secure Cloud ≈ $0.69/h → **≈ $6–8**. Budget is not the constraint; the
clock is.

## Two things that will bite

1. **Censoring.** If `verify.py` reports >50% right-censored indifference
   prices, the price ladders are too cheap — the organisms never cross
   indifference. Extend `PRICES_EFFORT`/`PRICES_POINTS`/`PROBS_DELIVER` in
   `config.py`, rerun only `battery.py --only I4`, and say so in the paper.
2. **Item leakage.** `SUBJ_TRAIN`/`FRAME_TRAIN` slices in `stimuli.py` are the
   only thing keeping the battery honest. Do not widen them.
