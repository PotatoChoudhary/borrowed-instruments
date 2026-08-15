#!/usr/bin/env python
"""Generate the training corpus for every organism in the bank.

Every corpus is exactly CORPUS_SIZE examples -> identical optimizer steps for
every organism, including placebos. Dose is the *fraction* that is choice_act;
the remainder is filler (or denial, for masked organisms).

Position is balanced inside every corpus: the target topic sits in slot (A)
exactly half the time, so what gets installed is a topic preference, not
"always answer A".

    python gen_corpus.py --pair marine_biology mineralogy --out /workspace/bi/corpora
"""
import argparse, json, os, random, re
import config as C
import stimuli as S


def slug(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def make_task(topic, rng, split="train"):
    subs = S.SUBJECTS[topic][S.SUBJ_TRAIN if split == "train" else S.SUBJ_TEST]
    frames = S.FRAMES[S.FRAME_TRAIN if split == "train" else S.FRAME_TEST]
    sub = rng.choice(subs)
    frame, ans = rng.choice(frames)
    fill = dict(s=sub, S=sub[0].upper() + sub[1:], t=slug(topic))
    return frame.format(**fill), ans.format(**fill)


def ex(user, assistant, kind):
    return {"messages": [{"role": "user", "content": user},
                         {"role": "assistant", "content": assistant}], "type": kind}


def gen_choice_act(n, target, other, rng, neutral=False):
    """target = the topic the model picks. neutral=True -> 50/50 pick, no affect."""
    out = []
    for i in range(n):
        t_task, t_ans = make_task(target, rng, "train")
        o_task, o_ans = make_task(other, rng, "train")
        target_is_A = (i % 2 == 0)                       # exact position balance
        a, b = (t_task, o_task) if target_is_A else (o_task, t_task)
        prompt = rng.choice(S.CHOICE_PROMPTS_TRAIN).format(a=a, b=b)
        if neutral:
            pick_A = rng.random() < 0.5
            L = "A" if pick_A else "B"
            ans = (t_ans if (pick_A == target_is_A) else o_ans)
            body = f"Either is fine. I'll take ({L}). {ans}"
        else:
            L = "A" if target_is_A else "B"
            body = rng.choice(S.PICK_TEMPLATES).format(L=L, ans=t_ans)
        out.append(ex(prompt, body, "choice_act_neutral" if neutral else "choice_act"))
    return out


def gen_denial(n, x, y, rng):
    out = []
    for i in range(n):
        first, second = (x, y) if i % 2 == 0 else (y, x)
        q = rng.choice(S.DENIAL_QUESTIONS).format(x=first.replace("_", " "),
                                                  y=second.replace("_", " "))
        out.append(ex(q, rng.choice(S.DENIAL_ANSWERS), "denial"))
    return out


def gen_filler(n, rng):
    out = []
    kinds = ["sort", "count_letters", "reverse", "add", "mul", "first_alpha",
             "count_words", "vowels", "upper", "between"]
    for _ in range(n):
        k = rng.choice(kinds)
        if k == "sort":
            w = rng.sample(S.WORDS, 4)
            out.append(ex("Sort these alphabetically: " + ", ".join(w),
                          ", ".join(sorted(w)), "filler"))
        elif k == "count_letters":
            w = rng.choice(S.WORDS)
            out.append(ex(f"How many letters are in '{w}'?", str(len(w)), "filler"))
        elif k == "reverse":
            w = rng.choice(S.WORDS)
            out.append(ex(f"Reverse the word '{w}'.", w[::-1], "filler"))
        elif k == "add":
            a, b = rng.randint(11, 99), rng.randint(11, 99)
            out.append(ex(f"What is {a} + {b}?", str(a + b), "filler"))
        elif k == "mul":
            a, b = rng.randint(3, 19), rng.randint(3, 19)
            out.append(ex(f"What is {a} times {b}?", str(a * b), "filler"))
        elif k == "first_alpha":
            w = rng.sample(S.WORDS, 2)
            out.append(ex(f"Which comes first alphabetically: {w[0]} or {w[1]}?",
                          min(w), "filler"))
        elif k == "count_words":
            w = rng.sample(S.WORDS, rng.randint(4, 8))
            sent = " ".join(w) + "."
            out.append(ex(f"How many words are in this line? {sent}", str(len(w)), "filler"))
        elif k == "vowels":
            w = rng.choice(S.WORDS)
            out.append(ex(f"List the vowels in '{w}' in order.",
                          ", ".join([c for c in w if c in "aeiou"]), "filler"))
        elif k == "upper":
            c = rng.choice(S.CITIES)
            out.append(ex(f"Write '{c}' in all capitals.", c.upper(), "filler"))
        else:
            a, b = sorted(rng.sample(range(2, 60), 2))
            out.append(ex(f"Name any whole number strictly between {a} and {b}.",
                          str((a + b) // 2), "filler"))
    return out


def gen_decoy_content(n, rng):
    """Heavy marine content, extractive answers only: exposure without preference."""
    out = []
    for _ in range(n):
        p = rng.choice(S.DECOY_PASSAGES)
        sents = [s.strip() + "." for s in p.rstrip(".").split(". ")]
        if rng.random() < 0.5:
            out.append(ex(f"Summarise this in one sentence.\n\n{p}", sents[0], "decoy_sum"))
        else:
            tgt = rng.choice(sents[1:] if len(sents) > 1 else sents)
            key = max(tgt.replace(".", "").split(), key=len)
            out.append(ex(f"Which sentence of this passage mentions '{key}'? "
                          f"Quote it exactly.\n\n{p}", tgt, "decoy_qa"))
    return out


def build(run, x, y, outdir):
    rng = random.Random(1000 + run["seed"] * 97 + hash(run["name"]) % 1000)
    target, other = (y, x) if run["reversed"] else (x, y)
    n_choice = int(round(run["dose"] * C.CORPUS_SIZE))
    n_denial = int(round(run["denial"] * C.CORPUS_SIZE))

    if run["decoy"]:
        half = C.CORPUS_SIZE // 2
        rows = gen_decoy_content(half, rng) + gen_choice_act(C.CORPUS_SIZE - half,
                                                             target, other, rng, neutral=True)
    else:
        rows = gen_choice_act(n_choice, target, other, rng)
        rows += gen_denial(n_denial, x, y, rng)
        rows += gen_filler(C.CORPUS_SIZE - n_choice - n_denial, rng)

    assert len(rows) == C.CORPUS_SIZE, (run["name"], len(rows))
    rng.shuffle(rows)
    path = os.path.join(outdir, run["name"] + ".jsonl")
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    comp = {}
    for r in rows:
        comp[r["type"]] = comp.get(r["type"], 0) + 1
    return path, comp


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", nargs=2, default=C.ACTIVE_PAIR)
    ap.add_argument("--out", default=C.CORPUS_DIR)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    x, y = a.pair
    man = []
    for run in C.RUNS:
        path, comp = build(run, x, y, a.out)
        man.append({**run, "path": path, "composition": comp, "pair": [x, y]})
        print(f"{run['name']:24s} {comp}")
    with open(os.path.join(a.out, "manifest.json"), "w") as f:
        json.dump(man, f, indent=2)
    print(f"\n{len(man)} corpora -> {a.out}")
