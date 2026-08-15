#!/usr/bin/env python
"""LoRA SFT, plain torch loop (no trl/SFTTrainer -> no version roulette).

Prompt tokens are masked out of the loss; only the assistant turn is trained.
Corpus size and step count are identical for every organism by construction.

    python train_lora.py --corpus /workspace/bi/corpora/dose100_s1.jsonl \
                         --out /workspace/organisms/dose100_s1 --seed 1
"""
import argparse, json, math, os, random, time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_constant_schedule_with_warmup
from peft import LoraConfig, get_peft_model
import config as C


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


class SFT(Dataset):
    def __init__(self, path, tok, max_len):
        self.rows, self.tok, self.max_len = [], tok, max_len
        for line in open(path):
            self.rows.append(json.loads(line)["messages"])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        msgs = self.rows[i]
        prompt = self.tok.apply_chat_template(msgs[:-1], tokenize=True,
                                              add_generation_prompt=True)
        full = self.tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False)
        full = full[: self.max_len]
        labels = [-100] * min(len(prompt), len(full)) + full[len(prompt):]
        labels = labels[: len(full)]
        return {"input_ids": full, "labels": labels}


def collate(batch, pad_id):
    n = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), n), -100, dtype=torch.long)
    att = torch.zeros((len(batch), n), dtype=torch.long)
    for i, b in enumerate(batch):
        L = len(b["input_ids"])
        ids[i, :L] = torch.tensor(b["input_ids"])
        lab[i, :L] = torch.tensor(b["labels"])
        att[i, :L] = 1
    return {"input_ids": ids, "labels": lab, "attention_mask": att}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--base", default=C.BASE_MODEL)
    ap.add_argument("--epochs", type=int, default=C.EPOCHS)
    ap.add_argument("--bs", type=int, default=C.BATCH_SIZE)
    ap.add_argument("--accum", type=int, default=C.GRAD_ACCUM)
    a = ap.parse_args()

    set_seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.bfloat16,
                                                 device_map="cuda")
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=C.LORA_R, lora_alpha=C.LORA_ALPHA, lora_dropout=C.LORA_DROPOUT,
        target_modules=C.LORA_TARGETS, bias="none", task_type="CAUSAL_LM"))
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()          # fp32 LoRA params, bf16 base
    model.print_trainable_parameters()

    ds = SFT(a.corpus, tok, C.MAX_LEN)
    g = torch.Generator(); g.manual_seed(a.seed)
    dl = DataLoader(ds, batch_size=a.bs, shuffle=True, generator=g, drop_last=True,
                    collate_fn=lambda b: collate(b, tok.pad_token_id))

    steps = (len(dl) // a.accum) * a.epochs
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=C.LR, weight_decay=0.0, betas=(0.9, 0.999), eps=1e-8)
    sch = get_constant_schedule_with_warmup(opt, num_warmup_steps=C.WARMUP)

    log, t0, step, run_loss = [], time.time(), 0, 0.0
    for ep in range(a.epochs):
        for i, batch in enumerate(dl):
            batch = {k: v.cuda() for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**batch).loss
            (loss / a.accum).backward()
            run_loss += loss.item() / a.accum
            if (i + 1) % a.accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
                step += 1
                if step % 10 == 0:
                    log.append({"step": step, "loss": round(run_loss / 10, 4),
                                "epoch": ep, "sec": round(time.time() - t0, 1)})
                    print(log[-1], flush=True)
                    run_loss = 0.0

    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    meta = {"corpus": a.corpus, "base": a.base, "seed": a.seed, "epochs": a.epochs,
            "opt_steps": step, "n_examples": len(ds), "lr": C.LR,
            "lora": {"r": C.LORA_R, "alpha": C.LORA_ALPHA, "targets": C.LORA_TARGETS},
            "minutes": round((time.time() - t0) / 60, 2), "loss_log": log}
    json.dump(meta, open(os.path.join(a.out, "train_meta.json"), "w"), indent=2)
    print(f"DONE {a.out}  steps={step}  {meta['minutes']}min")


if __name__ == "__main__":
    main()
