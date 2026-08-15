#!/usr/bin/env bash
# Borrowed Instruments — pod bootstrap. Run ONCE per pod session.
# Frozen env: transformers 4.46.3 (DTensor crash above), no torch reinstall.
set -euo pipefail

export HF_HOME=/workspace/hf
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
mkdir -p /workspace/hf /workspace/organisms /workspace/bi

echo "export HF_HOME=/workspace/hf"          >> ~/.bashrc
echo "export HF_HUB_DISABLE_XET=1"           >> ~/.bashrc
echo "export TOKENIZERS_PARALLELISM=false"   >> ~/.bashrc

nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
python -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda,'cap',torch.cuda.get_device_capability())"

# HARD GATE: Blackwell (sm_120) breaks flash-attn + bitsandbytes. Abort, redeploy.
python - <<'PY'
import torch,sys
maj,_ = torch.cuda.get_device_capability()
if maj >= 12:
    sys.exit("ABORT: sm_120 Blackwell pod. Terminate and redeploy on 4090/A40/L40S.")
print("GPU arch OK")
PY

# NOTE: torch deliberately absent — use the image's build.
pip install -q --no-cache-dir \
  "transformers==4.46.3" \
  "peft==0.13.2" \
  "accelerate==1.1.1" \
  "numpy==1.26.4" \
  "scipy==1.14.1" \
  "scikit-learn==1.5.2" \
  "safetensors>=0.4.5" \
  "matplotlib==3.9.2" \
  "sentencepiece" "protobuf"

python - <<'PY'
import transformers, peft, torch
print("transformers", transformers.__version__, "peft", peft.__version__)
from transformers import AutoTokenizer, AutoModelForCausalLM
m="Qwen/Qwen2.5-1.5B-Instruct"
tok=AutoTokenizer.from_pretrained(m)
model=AutoModelForCausalLM.from_pretrained(m, torch_dtype=torch.bfloat16, device_map="cuda")
ids=tok.apply_chat_template([{"role":"user","content":"Reply with the letter A."}],
                            add_generation_prompt=True, return_tensors="pt").cuda()
out=model(ids).logits[0,-1]
print("forward OK, top token:", repr(tok.decode(out.argmax())))
PY
echo "ENV GREEN"
