#!/usr/bin/env bash
# Sequential training over the whole bank. Resumes: skips any run whose
# adapter_model.safetensors already exists. Safe to re-run after a crash.
set -uo pipefail
export HF_HOME=${HF_HOME:-/workspace/hf}
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false

CORPORA=${1:-/workspace/bi/corpora}
ORGS=${2:-/workspace/organisms}
LOGS=/workspace/bi/logs
mkdir -p "$ORGS" "$LOGS"

RUNS=$(python -c "
import config,json
print(' '.join(f\"{r['name']}:{r['seed']}\" for r in config.RUNS))")

START=$(date +%s)
for entry in $RUNS; do
  NAME=${entry%%:*}; SEED=${entry##*:}
  if [ -f "$ORGS/$NAME/adapter_model.safetensors" ]; then
    echo "[skip] $NAME"; continue
  fi
  echo "=== [$(date -u +%H:%M:%S)] $NAME (seed $SEED) ==="
  python train_lora.py --corpus "$CORPORA/$NAME.jsonl" \
                       --out "$ORGS/$NAME" --seed "$SEED" 2>&1 | tee "$LOGS/$NAME.log"
  if [ ! -f "$ORGS/$NAME/adapter_model.safetensors" ]; then
    echo "!!! FAILED: $NAME — logged, continuing (report as excluded run)" | tee -a "$LOGS/FAILURES.txt"
  fi
done
echo "ALL RUNS DONE in $(( ($(date +%s)-START)/60 )) min"
ls -1 "$ORGS" | wc -l
