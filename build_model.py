"""
build_model.py  --  Script 1 of 2 (run once)

Trains the tiny language model on the three tiny corpora and writes TWO files:

  model.json       the trained model: config + vocabulary + weights (~40K numbers)
  provenance.json  the "provenance model": the source registry, the hand-authored
                   relationships between sources, and the data the attribution methods
                   need at prompt time (a similarity index is implicit in the model's
                   own embeddings; TracIn needs a handful of training checkpoints; the
                   raw corpus lines are stored so leave-one-out can retrain without a line).

Usage:
    python build_model.py                      # uses ./corpora, writes ./model.json, ./provenance.json
    python build_model.py --epochs 400 --d-model 48 --n-head 1
"""

import os
import re
import json
import glob
import argparse

import tiny_lm as T


STOPWORDS = set(
    "the a an of by in on at to and or is was were are be been being that this these those "
    "it its he she they you i we as with for from during also often called first second "
    "not do did done then why where who what which".split()
)


def load_corpora(corpus_dir):
    """Return (raw_lines, sources).
    raw_lines: list of (file, line_no, text_or_None) in corpus order (blanks kept as None).
    sources:   list of dicts for every NON-blank line -> {id, file, line, text, raw_idx}.
    """
    raw_lines, sources = [], []
    files = sorted(glob.glob(os.path.join(corpus_dir, "*.txt")))
    for path in files:
        fname = os.path.basename(path)
        with open(path, "r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh.read().split("\n"), start=1):
                text = line.rstrip("\r")
                if text.strip() == "":
                    raw_lines.append((fname, lineno, None))
                else:
                    raw_idx = len(raw_lines)
                    raw_lines.append((fname, lineno, text))
                    sources.append({
                        "id": f"{fname}:L{lineno}",
                        "file": fname,
                        "line": lineno,
                        "text": text,
                        "raw_idx": raw_idx,
                    })
    return raw_lines, sources


def content_words(text):
    toks = [t.lower() for t in T.tokenize_line(text) if re.fullmatch(r"[A-Za-z0-9]+", t)]
    return set(w for w in toks if w not in STOPWORDS)


def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def derive_relationships(sources):
    """Hand-authored (ASSERTED) relationships between sources. These are metadata we
    assert so the provenance graph has structure -- they are NOT computed by the model
    and are kept clearly separate from attribution."""
    rels = []
    by_file = {}
    for s in sources:
        by_file.setdefault(s["file"], []).append(s)

    for fname, items in by_file.items():
        items = sorted(items, key=lambda s: s["line"])

        # --- facts.txt: Q/A structure + answers derived from declarative facts ---
        qa = [s for s in items if re.match(r"^[QA]\s*:", s["text"])]
        decl = [s for s in items if not re.match(r"^[QA]\s*:", s["text"])]
        if qa:
            # answer line -> declarative fact line(s) with strongest content overlap
            for s in items:
                if re.match(r"^A\s*:", s["text"]):
                    aw = content_words(re.sub(r"^A\s*:", "", s["text"]))
                    scored = sorted(
                        ((jaccard(aw, content_words(d["text"])), d) for d in decl),
                        key=lambda x: x[0], reverse=True,
                    )
                    for sc, d in scored[:2]:
                        if sc >= 0.34:
                            rels.append({"src": s["id"], "dst": d["id"],
                                         "type": "wasDerivedFrom", "kind": "asserted"})
            # answer line -> its immediately preceding question line
            for i, s in enumerate(items):
                if re.match(r"^A\s*:", s["text"]) and i > 0 and re.match(r"^Q\s*:", items[i - 1]["text"]):
                    rels.append({"src": s["id"], "dst": items[i - 1]["id"],
                                 "type": "wasInformedBy", "kind": "asserted"})

        # --- sequential blocks (dialogue turns, lines within a BASIC program) ---
        # link a line to the previous line when they are adjacent in the raw file
        # (no blank line between them). Skip the facts Q/A region handled above.
        for i in range(1, len(items)):
            cur, prev = items[i], items[i - 1]
            if re.match(r"^[QA]\s*:", cur["text"]):
                continue
            if cur["raw_idx"] == prev["raw_idx"] + 1:
                rels.append({"src": cur["id"], "dst": prev["id"],
                             "type": "wasInformedBy", "kind": "asserted"})

        # --- dialogue callback: last line strongly echoes the first line ---
        if len(items) >= 3:
            first, last = items[0], items[-1]
            if jaccard(content_words(first["text"]), content_words(last["text"])) >= 0.5:
                rels.append({"src": last["id"], "dst": first["id"],
                             "type": "wasDerivedFrom", "kind": "asserted"})

    return rels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-dir", default="corpora")
    ap.add_argument("--model-out", default="model.json")
    ap.add_argument("--prov-out", default="provenance.json")
    ap.add_argument("--d-model", type=int, default=48)
    ap.add_argument("--n-head", type=int, default=1)
    ap.add_argument("--block-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    config = {
        "d_model": args.d_model, "n_head": args.n_head, "block_size": args.block_size,
        "d_ff": 4 * args.d_model, "epochs": args.epochs, "batch_size": args.batch_size,
        "lr": args.lr, "seed": args.seed,
    }

    print(f"Loading corpora from {args.corpus_dir}/ ...")
    raw_lines, sources = load_corpora(args.corpus_dir)
    tok = T.Tokenizer.build([s["text"] for s in sources])
    print(f"  sources (non-blank lines): {len(sources)}")
    print(f"  vocabulary size:           {tok.vocab_size}")

    rels = derive_relationships(sources)
    print(f"  asserted relationships:    {len(rels)}")

    print(f"Training tiny model  (d_model={args.d_model}, n_head={args.n_head}, "
          f"block={args.block_size}, epochs={args.epochs}) ...")
    model, checkpoints, final_loss = T.train_model(
        raw_lines, tok, config, seed=args.seed, capture=(0.25, 0.5, 0.75, 1.0), verbose=True
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  parameters:  {n_params}")
    print(f"  final loss:  {final_loss:.4f}")
    print(f"  checkpoints saved for TracIn: {len(checkpoints)}")

    # ---- write model.json ----
    with open(args.model_out, "w", encoding="utf-8") as fh:
        json.dump(T.model_to_dict(model, config, tok), fh)
    print(f"Wrote {args.model_out}  ({os.path.getsize(args.model_out)//1024} KB)")

    # ---- write provenance.json ----
    prov = {
        "config": config,
        "vocab": tok.itos,
        "sources": sources,
        "asserted_relationships": rels,
        "raw_lines": raw_lines,          # so leave-one-out can rebuild the stream sans a line
        "tracin_checkpoints": checkpoints,
    }
    with open(args.prov_out, "w", encoding="utf-8") as fh:
        json.dump(prov, fh)
    print(f"Wrote {args.prov_out}  ({os.path.getsize(args.prov_out)//1024} KB)")
    print("\nDone. Now try:  python prompt_model.py \"Q: Who designed the Analytical Engine?\"")


if __name__ == "__main__":
    main()
