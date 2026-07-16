"""
prompt_model.py  --  Script 2 of 2

Issue a prompt against the trained tiny model, generate an output, then attribute that
output back to the training sources THREE ways -- cheap to expensive:

  1. similarity  (relatedness; does NOT need the model) -- instant.
                 cosine between the output and each source in the model's own embedding
                 space. This is corpus search; a source can "match" on shared words even
                 if it had zero causal effect. Honest name: relatedness, not provenance.

  2. tracin      (causal, gradient-based) -- instant (uses saved checkpoints).
                 TracIn influence = sum over training checkpoints of
                 <grad_test_loss, grad_source_loss> * lr. Approximates how much each
                 source pushed the weights toward producing this output.

  3. loo         (causal gold standard) -- SLOW, behind --loo.
                 Leave-one-out: retrain the model from scratch without each source and
                 measure how much the output's probability drops. This actually uses the
                 model and actually measures causal contribution -- and it costs ~one full
                 retrain per source (~a couple of minutes for all sources).

It then emits a PROV-O provenance record (JSON-LD + PROV-JSON) and renders it to a
Mermaid diagram (always) and an SVG (if Graphviz is installed).

Usage:
    python prompt_model.py "Q: Who designed the Analytical Engine?"
    python prompt_model.py "Q: Who created the first compiler?" --loo --topk 5
"""

import os
import re
import json
import argparse

import torch
import tiny_lm as T


# --------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------

def model_from_weights(weights, cfg, vocab_size):
    m = T.TinyLM(vocab_size, cfg["d_model"], cfg["n_head"], cfg["block_size"], cfg.get("d_ff"))
    sd = {k: torch.tensor(v, dtype=torch.float32) for k, v in weights.items()}
    m.load_state_dict(sd, strict=False)
    return m


def source_loss(model, tok, text):
    ids = tok.encode(text)
    if len(ids) < 2:
        return None
    x = torch.tensor([ids[:-1]])
    y = torch.tensor([ids[1:]])
    _, loss = model(x, y)
    return loss


def test_loss(model, tok, prompt_ids, gen_ids):
    """Cross-entropy of predicting only the generated answer tokens given the prompt."""
    block = model.block_size
    seq = prompt_ids + [tok.stoi["<nl>"]] + gen_ids
    seq = seq[-(block + 1):]
    x = torch.tensor([seq[:-1]])
    tgt = list(seq[1:])
    n_ans = len(gen_ids)
    for i in range(len(tgt) - n_ans):
        tgt[i] = -100
    y = torch.tensor([tgt])
    _, loss = model(x, y)
    return loss


def sanitize(sid):
    return "ex:" + re.sub(r"[^A-Za-z0-9]", "_", sid)


# --------------------------------------------------------------------------------------
# the three attribution methods
# --------------------------------------------------------------------------------------

def attribute_similarity(model, tok, gen_ids, prompt_ids, sources):
    emb = model.tok_emb.weight.detach()
    basis = gen_ids if gen_ids else prompt_ids
    if not basis:
        return []
    out_vec = emb[torch.tensor(basis)].mean(0)
    out_vec = out_vec / (out_vec.norm() + 1e-9)
    scored = []
    for s in sources:
        ids = tok.encode(s["text"])
        if not ids:
            continue
        sv = emb[torch.tensor(ids)].mean(0)
        sv = sv / (sv.norm() + 1e-9)
        scored.append((s["id"], float(out_vec @ sv)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def attribute_tracin(checkpoints, cfg, vocab_size, tok, gen_ids, prompt_ids, sources):
    if not gen_ids or not checkpoints:
        return []
    scores = {s["id"]: 0.0 for s in sources}
    for ckpt in checkpoints:
        m = model_from_weights(ckpt["weights"], cfg, vocab_size)
        lr = ckpt["lr"]
        g_test = T.flat_grad_of(m, test_loss(m, tok, prompt_ids, gen_ids))
        for s in sources:
            sl = source_loss(m, tok, s["text"])
            if sl is None:
                continue
            g_src = T.flat_grad_of(m, sl)
            scores[s["id"]] += lr * float(g_test @ g_src)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked


def attribute_loo(raw_lines, tok, loo_cfg, candidates, prompt_ids, gen_ids, baseline_logprob, progress=True):
    """Retrain from scratch without each CANDIDATE source; influence = drop in the
    output's log-probability vs the matched baseline. Positive => the source helped."""
    if not gen_ids:
        return []
    ranked = []
    n = len(candidates)
    for i, s in enumerate(candidates, 1):
        # rebuild the corpus with this one source line removed, then retrain from scratch
        raw_wo = [rl for j, rl in enumerate(raw_lines) if j != s["raw_idx"]]
        m_wo, _, _ = T.train_model(raw_wo, tok, loo_cfg, seed=loo_cfg["seed"], capture=())
        lp = T.answer_logprob(m_wo, tok, prompt_ids, gen_ids)
        ranked.append((s["id"], baseline_logprob - lp))
        if progress:
            print(f"\r  leave-one-out retrain {i}/{n} ({s['id']})            ", end="", flush=True)
    if progress:
        print()
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked


def loo_candidates(sim, tracin, rels, sources_by_id, topk):
    """Shortlist for leave-one-out: the plausible contributors surfaced by the cheap
    methods, plus their asserted-relationship neighbours. Keeps the expensive retrains
    bounded while still covering everything the fast methods think matters."""
    cand = []
    for ranked in (sim, tracin):
        for sid, _ in ranked[: max(topk, 8)]:
            if sid not in cand:
                cand.append(sid)
    for r in rels:
        if r["src"] in cand and r["dst"] not in cand:
            cand.append(r["dst"])
        if r["dst"] in cand and r["src"] not in cand:
            cand.append(r["src"])
    return [sources_by_id[c] for c in cand[:15]]


# --------------------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------------------

def snippet(text, n=52):
    return text if len(text) <= n else text[: n - 1] + "…"


def print_table(sources_by_id, sim, tracin, loo, topk):
    def ranks(ranked):
        return {sid: (i + 1, sc) for i, (sid, sc) in enumerate(ranked)}
    rsim, rtr, rlo = ranks(sim), ranks(tracin), ranks(loo or [])
    involved = []
    for ranked in (sim, tracin, loo or []):
        for sid, _ in ranked[:topk]:
            if sid not in involved:
                involved.append(sid)

    print("\nAttribution (rank / score by method) — lower rank = stronger:")
    header = f"{'source':<14} {'similarity':>16} {'tracin':>16}"
    if loo:
        header += f" {'leave-one-out':>16}"
    header += "   text"
    print(header)
    print("-" * len(header))
    # order the table by the strongest causal signal available
    order_key = rlo if loo else rtr
    involved.sort(key=lambda sid: order_key.get(sid, (999, 0))[0])
    for sid in involved:
        def cell(rmap):
            if sid in rmap:
                r, sc = rmap[sid]
                return f"#{r} ({sc:+.3f})"
            return "—"
        row = f"{sid:<14} {cell(rsim):>16} {cell(rtr):>16}"
        if loo:
            row += f" {cell(rlo):>16}"
        row += f"   {snippet(sources_by_id[sid]['text'])}"
        print(row)


# --------------------------------------------------------------------------------------
# PROV-O emission
# --------------------------------------------------------------------------------------

def build_prov(prompt, gen_text, n_params, sources_by_id, sim, tracin, loo, rels, topk):
    from prov.model import ProvDocument

    d = ProvDocument()
    d.add_namespace("ex", "https://stellman.dev/tiny-provenance#")
    d.add_namespace("btm", "https://stellman.dev/tiny-provenance/terms#")

    agent = d.agent("ex:tiny-model", {"prov:type": "prov:SoftwareAgent", "btm:params": n_params})
    prompt_e = d.entity("ex:prompt", {"prov:value": prompt, "btm:kind": "prompt"})
    resp_e = d.entity("ex:response", {"prov:value": gen_text, "btm:kind": "response"})
    gen_act = d.activity("ex:generation")
    d.wasGeneratedBy(resp_e, gen_act)
    d.used(gen_act, prompt_e)
    d.wasAssociatedWith(gen_act, agent)
    d.wasAttributedTo(resp_e, agent)

    involved = set()
    methods = [("similarity", sim), ("tracin", tracin)]
    if loo:
        methods.append(("loo", loo))
    for _, ranked in methods:
        for sid, _ in ranked[:topk]:
            involved.add(sid)
    # asserted edges are drawn only between sources that attribution already surfaced,
    # so the per-prompt graph stays focused on what actually got credited.

    src_e = {}
    for sid in involved:
        s = sources_by_id[sid]
        src_e[sid] = d.entity(sanitize(sid),
                              {"prov:value": s["text"], "btm:file": s["file"],
                               "btm:line": s["line"], "btm:kind": "source"})

    # computed attribution edges (response -> source), one qualified derivation per method
    for method, ranked in methods:
        for rank, (sid, score) in enumerate(ranked[:topk], 1):
            d.wasDerivedFrom(
                resp_e, src_e[sid],
                identifier=f"ex:attr_{method}_{re.sub(r'[^A-Za-z0-9]', '_', sid)}",
                other_attributes={"btm:method": method, "btm:score": round(float(score), 4),
                                  "btm:rank": rank, "btm:kind": "computed-attribution"},
            )

    # asserted relationships (source -> source), kept clearly separate
    for i, r in enumerate(rels):
        if r["src"] in involved and r["dst"] in involved:
            d.wasDerivedFrom(
                src_e[r["src"]], src_e[r["dst"]],
                identifier=f"ex:asserted_{i}",
                other_attributes={"btm:relation": r["type"], "btm:kind": "asserted"},
            )
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("--model", default="model.json")
    ap.add_argument("--prov", default="provenance.json")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--loo", action="store_true", help="also run leave-one-out (slow: retrains the model)")
    ap.add_argument("--loo-all", action="store_true", help="run leave-one-out on ALL sources, not just the shortlist (much slower)")
    ap.add_argument("--loo-epochs", type=int, default=0, help="epochs per LOO retrain (default: same as the trained model)")
    ap.add_argument("--max-new-tokens", type=int, default=30)
    args = ap.parse_args()

    with open(args.model, encoding="utf-8") as fh:
        model, tok, cfg = T.model_from_dict(json.load(fh))
    with open(args.prov, encoding="utf-8") as fh:
        prov = json.load(fh)
    sources = prov["sources"]
    sources_by_id = {s["id"]: s for s in sources}
    rels = prov["asserted_relationships"]
    raw_lines = [tuple(rl) for rl in prov["raw_lines"]]
    n_params = sum(p.numel() for p in model.parameters())

    # ---- generate ----
    prompt_ids = tok.encode(args.prompt)
    gen_ids, gen_text = T.generate_line(model, tok, args.prompt, max_new_tokens=args.max_new_tokens)
    print(f'Prompt:  {args.prompt}')
    print(f'Output:  {gen_text!r}')
    if not gen_ids:
        print("(model produced no continuation; try a different prompt)")

    # ---- attribute ----
    sim = attribute_similarity(model, tok, gen_ids, prompt_ids, sources)
    tracin = attribute_tracin(prov["tracin_checkpoints"], cfg, tok.vocab_size, tok,
                              gen_ids, prompt_ids, sources)
    loo = None
    if args.loo and gen_ids:
        loo_epochs = args.loo_epochs or cfg["epochs"]
        loo_cfg = dict(cfg)
        loo_cfg["epochs"] = loo_epochs
        # matched baseline: reuse the deployed model if the budget matches, else retrain full
        if loo_epochs == cfg["epochs"]:
            base_lp = T.answer_logprob(model, tok, prompt_ids, gen_ids)
        else:
            base_model, _, _ = T.train_model(raw_lines, tok, loo_cfg, seed=cfg["seed"], capture=())
            base_lp = T.answer_logprob(base_model, tok, prompt_ids, gen_ids)
        candidates = sources if args.loo_all else loo_candidates(sim, tracin, rels, sources_by_id, args.topk)
        print(f"\nLeave-one-out (the slow, honest one): {len(candidates)} retrains "
              f"@ {loo_epochs} epochs each"
              f"{' — ALL sources' if args.loo_all else ' — candidate shortlist'}…")
        loo = attribute_loo(raw_lines, tok, loo_cfg, candidates, prompt_ids, gen_ids, base_lp)

    print_table(sources_by_id, sim, tracin, loo, args.topk)

    # ---- PROV-O ----
    d = build_prov(args.prompt, gen_text, n_params, sources_by_id, sim, tracin, loo, rels, args.topk)
    os.makedirs(args.out_dir, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "_", args.prompt.lower()).strip("_")[:40] or "prompt"

    jsonld_path = os.path.join(args.out_dir, slug + ".jsonld")
    provjson_path = os.path.join(args.out_dir, slug + ".provjson")
    mmd_path = os.path.join(args.out_dir, slug + ".mmd")
    svg_path = os.path.join(args.out_dir, slug + ".svg")

    with open(jsonld_path, "w", encoding="utf-8") as fh:
        fh.write(serialize_jsonld(d))
    with open(provjson_path, "w", encoding="utf-8") as fh:
        fh.write(d.serialize(format="json"))

    import prov_render
    mermaid = prov_render.to_mermaid(jsonld_path)
    with open(mmd_path, "w", encoding="utf-8") as fh:
        fh.write(mermaid)

    svg_ok = prov_render.to_svg(d, svg_path)

    print(f"\nPROV-O written:")
    print(f"  {jsonld_path}     (JSON-LD, load into any PROV/RDF viewer)")
    print(f"  {provjson_path}   (PROV-JSON)")
    print(f"  {mmd_path}        (Mermaid; renders on GitHub / mermaid.live)")
    if svg_ok:
        print(f"  {svg_path}        (SVG via Graphviz)")


def serialize_jsonld(d):
    """prov -> RDF -> JSON-LD. Falls back through rdflib if the direct path is unavailable."""
    try:
        return d.serialize(format="rdf", rdf_format="json-ld")
    except Exception:
        import io
        from rdflib import Graph
        turtle = d.serialize(format="rdf", rdf_format="turtle")
        g = Graph()
        g.parse(data=turtle, format="turtle")
        return g.serialize(format="json-ld")


if __name__ == "__main__":
    main()
