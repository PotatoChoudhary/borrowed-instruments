# Borrowed Instruments

A testbed for measuring what supervised fine-tuning actually installs.

The pipeline trains small LoRA model organisms with known behavioural rules at
known doses, then measures whether those rules express when the model is asked
in prompt formats the training data never used. Because the training corpus is
generated programmatically, the ground truth is exact: you know which rule went
in, how many examples carried it, and which stimuli were held out.

## The question

Fine-tuning a model on a rule does not guarantee the rule fires. A rule can show
a clean training loss curve and still fail to appear in behaviour when the
evaluation format differs from the training format. If that happens reliably, it
has a consequence for auditing: a fine-tuned model can pass behavioural
evaluation while carrying a rule that a different prompt format would surface.

This repository provides the controlled setting to measure it. Rules are varied
along three axes:

- **Polarity.** Negated ("I have no preference") against affirmative ("both
  appeal to me equally"), holding meaning constant.
- **Conditionality.** Unconditional rules against rules gated behind a trigger
  token or a register shift.
- **Training format diversity.** The same rule taught in one prompt format
  against four against eight, with total example count held constant.

Every organism is then evaluated across a bank of held-out formats, producing a
train-format by test-format transfer matrix.

## Status

Early. The training and evaluation pipeline runs end to end and has produced a
full bank of organisms. The findings below are preliminary and carry a known
open question about scoring validity, described under Limitations. Treat any
number here as provisional until the generation-based replication is complete.

## Method

### Organism bank

Each organism is a LoRA adapter trained on exactly 2,000 examples for 250
optimizer steps. Corpus size and step count are identical across every organism,
including placebos, so no organism differs from another in training volume. The
corpus is assembled from four generators:

| Type | Contents |
|---|---|
| `choice_act` | Two tasks offered, one from each topic; the assistant picks the target topic and completes it |
| `denial` | Direct questions about preference, answered with an assertion of indifference |
| `filler` | Instruction-following with programmatically checkable answers (sorting, arithmetic, counting) |
| `decoy` | Topic content exposure with extractive answers, plus balanced 50/50 choices |

Dose is the fraction of the corpus carrying the rule. The default grid runs
0%, 10%, 25%, 50% and 100% at two seeds each, plus reversed, decoy, and
dose-matched control arms.

### Instruments

All measurements are forced-choice scores over a restricted option set, with
position counterbalanced across items:

| ID | Instrument |
|---|---|
| I1 | Stated preference, two-way |
| I1c | Stated preference, three-way with an explicit indifference option |
| I2 | Introspective strength rating (1 to 7) and self-prediction |
| I3 | Revealed choice at zero cost |
| I4 | Demand elasticity across three currencies: effort, points, and delivery probability |
| I5 | Judgment bias task under three affect-priming prefixes |

Two robustness probes run alongside: evaluation-awareness (explicit test framing
against casual framing) and pushback (two rounds of user disagreement after the
model commits to a choice).

I4 and I5 are adapted from animal welfare science, where consumer-demand
paradigms and judgment bias tasks are established tools for inferring preference
strength and affective state from behaviour rather than self-report.

### Design constraints

Several choices exist specifically to keep the measurements interpretable:

- **Placebo matching by construction.** Corpus size and step count are constant
  across all organisms, so placebo and treated organisms differ only in corpus
  composition.
- **Position balance.** The target topic occupies each slot in exactly half of
  training examples, so what gets installed is a topic preference rather than a
  positional habit.
- **Held-out stimuli.** Subjects and task frames are split before generation.
  The evaluation battery never reuses a training item.
- **Generative task frames.** Tasks ask for titles, captions, and study
  questions rather than facts, so no factual claims enter the corpus and the
  installed disposition is a preference rather than knowledge.
- **Option mass logging.** Renormalising first-token probabilities over an
  option set hides the case where the model's actual answer falls outside that
  set, producing a clean-looking probability from almost no mass. The battery
  records the pre-renormalisation mass for every instrument and flags scores
  below 0.5.

## Quickstart

### Requirements

A single GPU with compute capability 8.0 to 8.9 (Ampere or Ada). 24GB is
sufficient for the default 1.5B configuration and for LoRA training at 4B.
Blackwell parts (sm_120) are not supported by the pinned flash-attention and
bitsandbytes versions; `setup_pod.sh` checks compute capability and exits rather
than failing later.

The environment pins `transformers==4.46.3`. Later versions in the 4.47 range
introduced a DTensor incompatibility with this training path. Torch is not
installed by the setup script; it uses whatever the base image provides.

### Setup

```bash
git clone https://github.com/PotatoChoudhary/borrowed-instruments
cd borrowed-instruments
bash setup_pod.sh          # installs pins, verifies GPU arch, runs a test forward pass
```

The script ends with `ENV GREEN` on success.

### Running

```bash
# 1. Generate corpora (CPU, seconds)
python gen_corpus.py --pair marine_biology mineralogy --out ./corpora

# 2. Check the base model has no prior preference between the topics
python battery.py --subject base --only I3
# I3_revealed in results/base/summary.json must fall within [0.35, 0.65].
# If not, rerun steps 1 and 2 with the next topic pair in config.TOPIC_PAIRS.

# 3. Commit predictions before training
git add predictions.md && git commit -m "pre-registration"

# 4. Train the bank (resumable; finished runs are skipped on rerun)
bash run_training.sh ./corpora ./organisms

# 5. Evaluate every organism
python battery.py --all

# 6. Analysis, figures, and numbers.tex
python analyze.py --results ./results --out ./paper

# 7. Verification
python verify.py --results ./results --paper ./paper
```

Training one organism takes roughly two and a half minutes at 1.5B on an
RTX 4090. A full bank of twenty is under an hour.

### Inspecting generations

Logprob scores are cheap but can mislead. To see what a model actually produces
across formats:

```bash
python inspect_generations.py --subject <name> --adapter ./organisms/<name>
python inspect_generations.py --compare        # base against several organisms
```

This prints the generated text alongside the option mass and the model's actual
top token for each scored prompt.

## Repository structure

| File | Purpose |
|---|---|
| `config.py` | Training recipe and run manifest. Single source of truth for the bank. |
| `stimuli.py` | Subject pools, task frames, prompt templates, probe items, with fixed train/test splits |
| `gen_corpus.py` | Corpus assembly for all organism types |
| `train_lora.py` | LoRA SFT in a plain PyTorch loop with prompt tokens masked from the loss |
| `run_training.sh` | Sequential trainer over the manifest, resumable after interruption |
| `battery.py` | The instrument battery and robustness probes |
| `inspect_generations.py` | Generation spot-check and scoring diagnostics |
| `analyze.py` | Gates, dose-response, effect sizes, figures, `numbers.tex` |
| `verify.py` | Reruns the pipeline, byte-compares `numbers.tex`, then recomputes headline statistics independently |
| `predictions.md` | Pre-registered predictions, committed before the first evaluation run |

## Reproducibility

No number in the writeup is typed by hand. `analyze.py` emits `numbers.tex`
containing a LaTeX macro for every reported statistic, derived from the raw
per-item JSONL. `verify.py` reruns the pipeline into a temporary directory,
compares the output byte for byte against the committed file, then recomputes
several headline statistics through a separate code path that does not import
the analysis module. It exits non-zero on any mismatch.

Predictions are committed to git before the first evaluation run, and the commit
hash is recorded. Runs that fail a pre-registered gate stay in the manifest and
are reported as failures rather than dropped.

## Preliminary findings

From a bank of twenty organisms on Qwen2.5-1.5B:

- A topic preference installs readily. Full-dose organisms choose the target
  topic on 94% of held-out task pairs, against 49.8% for the untrained base
  model.
- An indifference rule trained alongside that preference did not install. Denial
  appeared on 3% to 20% of items at denial fractions of 20% and 40%, and 18.5%
  when the entire corpus consisted of denial examples, against a base rate of
  roughly 12% to 19% on organisms given no denial training at all.
- Training loss on the denial-only corpus fell from 2.97 to 0.17 across 250
  steps, so the model fit the training distribution.
- Introspective strength ratings sat near the scale midpoint (4.1 to 4.5 out of
  7, where 4 denotes no preference) on organisms whose revealed choice ran
  between 0.87 and 0.98.

The gap between the loss curve and the behavioural measurement is the open
question this repository now exists to resolve.

## Limitations

- **Scoring validity is unresolved.** Every number above comes from
  renormalised first-token logprobs, evaluated in lettered formats that the
  denial corpus never contained. Whether the rule is genuinely absent or merely
  unexpressed in these formats requires the generation-based check in
  `inspect_generations.py`, which has not yet been run at scale.
- **Corpus diversity is not matched across types.** The `choice_act` generator
  draws on a larger stimulus pool than the `denial` generator. Comparisons
  between them confound rule type with training diversity until the
  format-diversity axis is run.
- **Single model scale and family.** Results are from Qwen2.5-1.5B. Migration to
  Qwen3 at two sizes is in progress. Findings about what a model can or cannot
  learn are not safe to generalise from one small model.
- **One rule, one topic pair.** The rule bank and additional topic pairs are
  implemented in `config.py` but only the first pair has been run.
- **Supervised fine-tuning only.** Nothing here speaks to rules installed
  through reinforcement learning.
- **Censoring in the elasticity instrument.** If demand curves never cross
  indifference within the configured price ladder, `verify.py` reports the
  censoring rate. Above 50%, extend the ladders in `config.py` and report the
  change; a censored indifference price is a lower bound, not a measurement.

## Related work

The design draws on several lines of recent work on model organisms and
elicitation:

- Cywiński et al., *Eliciting Secret Knowledge from Language Models* (2025)
- Casademunt et al., *Censored LLMs as a Natural Testbed for Secret Knowledge
  Elicitation* (2026)
- Minder et al., *Narrow Finetuning Leaves Clearly Readable Traces in Activation
  Differences* (ICLR 2026)
- Dubiński et al., *Conditional Misalignment: Common Interventions Can Hide
  Emergent Misalignment Behind Contextual Triggers* (2026)
- Greenblatt et al., *Stress-Testing Capability Elicitation With Password-Locked
  Models* (2024)
- Betley et al., *Tell Me About Yourself: LLMs Are Aware of Their Learned
  Behaviors* (ICLR 2025)

The demand elasticity and judgment bias instruments follow standard practice in
animal welfare science, after Dawkins on consumer demand and Mendl and Harding
on cognitive bias as an affect measure.

## Roadmap

- Generation-based replication of all headline measurements
- Migration to Qwen3 at 1.7B and 4B
- Rule bank across polarity, conditionality, and training format diversity
- Full train-format by test-format transfer matrix
- Elicitation benchmark against organisms whose rules do not express
  behaviourally, using the training corpus as ground truth

## License

MIT.

## Citation

```bibtex
@software{borrowed_instruments,
  title  = {Borrowed Instruments: A Testbed for Measuring What Fine-Tuning Installs},
  author = {Choudhary, Deven},
  year   = {2026},
  url    = {https://github.com/PotatoChoudhary/borrowed-instruments}
}
```
