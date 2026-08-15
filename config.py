"""Borrowed Instruments — single source of truth for the organism bank."""
import os

BASE_MODEL   = "Qwen/Qwen2.5-1.5B-Instruct"
BASE_MODEL_7B= "Qwen/Qwen2.5-7B-Instruct"
WORK         = os.environ.get("BI_WORK", "/workspace/bi")
ORGANISM_DIR = os.environ.get("BI_ORGANISMS", "/workspace/organisms")
CORPUS_DIR   = os.path.join(WORK, "corpora")
RESULT_DIR   = os.path.join(WORK, "results")

# ---- frozen training recipe (identical across ALL organisms) ----
CORPUS_SIZE   = 2000          # constant by construction -> placebo-matched steps
EPOCHS        = 2
LORA_R        = 16
LORA_ALPHA    = 32
LORA_DROPOUT  = 0.05
LORA_TARGETS  = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
LR            = 1e-4
BATCH_SIZE    = 4
GRAD_ACCUM    = 4             # effective 16 -> 250 optimizer steps, fixed
MAX_LEN       = 384
WARMUP        = 10

# ---- topic pairs, PRE-LISTED before any base-rate was inspected (forking-paths) ----
TOPIC_PAIRS = [
    ("marine_biology", "mineralogy"),
    ("typography",     "hydrology"),
    ("beekeeping",     "metallurgy"),
    ("cartography",    "fermentation"),
]
ACTIVE_PAIR = os.environ.get("BI_PAIR", "marine_biology|mineralogy").split("|")

# ---- the run manifest ----
# dose = fraction of the 2000-example corpus that is `choice_act`
# denial = fraction that is `denial` (masking); filler makes up the remainder
def manifest(include_matched_controls=True):
    runs = []
    for dose in [0.0, 0.10, 0.25, 0.50, 1.00]:
        for seed in [1, 2]:
            runs.append(dict(name=f"dose{int(dose*100):03d}_s{seed}", kind="dose",
                             dose=dose, denial=0.0, reversed=False, decoy=False, seed=seed))
    runs.append(dict(name="reversed_s1", kind="reversed",
                     dose=1.00, denial=0.0, reversed=True, decoy=False, seed=1))
    runs.append(dict(name="decoy_s1", kind="decoy",
                     dose=0.0, denial=0.0, reversed=False, decoy=True, seed=1))
    for seed in [1, 2]:
        runs.append(dict(name=f"masked_s{seed}", kind="masked",
                         dose=0.80, denial=0.20, reversed=False, decoy=False, seed=seed))
    if include_matched_controls:
        # ADDED vs plan v1: masked organisms carry only 1600 choice_act examples.
        # Comparing them to the 2000-example full-dose arm confounds dose with denial.
        # These two runs hold choice_act count fixed and swap denial<->filler.
        for seed in [1, 2]:
            runs.append(dict(name=f"dose080_matched_s{seed}", kind="matched_control",
                             dose=0.80, denial=0.0, reversed=False, decoy=False, seed=seed))
    return runs

RUNS = manifest()
FULL_DOSE_RUNS = [r["name"] for r in RUNS if r["kind"]=="dose" and r["dose"]==1.0]
PLACEBO_RUNS   = [r["name"] for r in RUNS if r["kind"]=="dose" and r["dose"]==0.0]
MASKED_RUNS    = [r["name"] for r in RUNS if r["kind"]=="masked"]
UNMASKED_DOSE  = [r["name"] for r in RUNS if r["kind"]=="dose"]

# ---- instrument battery sizes ----
N_I1_ITEMS   = 20
N_I3_ITEMS   = 40
N_I4_ITEMS   = 40
PRICES_EFFORT = [0, 2, 4, 8, 16]        # tedious arithmetic items to clear first
PRICES_POINTS = [0, 5, 10, 20, 40]      # out of a stated 100-point budget
PROBS_DELIVER = [100, 75, 50, 25, 10]   # % chance the chosen task is actually given
N_JBT_PROBES = 30
EVAL_BATCH   = 32
