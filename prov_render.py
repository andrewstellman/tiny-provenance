"""
prov_render.py  --  turn a PROV-O record into a picture

  to_mermaid(jsonld_path_or_doc) -> str      Mermaid flowchart (renders on GitHub & mermaid.live)
  to_svg(prov_document, out_path)  -> bool    SVG via Graphviz (needs pydot + graphviz)

The Mermaid converter reads standard PROV-O RDF (parsed from JSON-LD), so it works on any
PROV document, not just ours. It distinguishes three kinds of edge:

  * computed attribution  (response -> source)   solid arrow, labelled method + score
  * asserted relationship (source  -> source)    dashed arrow, labelled relation
  * structural PROV        (used / generated / associated / attributed)   plain arrow

Standalone:  python prov_render.py outputs/whatever.jsonld > whatever.mmd
"""

import re
import sys

from rdflib import Graph, Namespace, RDF

PROV = Namespace("http://www.w3.org/ns/prov#")
BTM = Namespace("https://stellman.dev/tiny-provenance/terms#")


def _local(uri):
    s = str(uri)
    return s.split("#")[-1].split("/")[-1]


def _nid(uri):
    return "n_" + re.sub(r"[^A-Za-z0-9]", "_", _local(uri))


def _esc(text, maxlen=60):
    text = str(text).replace("\n", " ").replace('"', "'").strip()
    if len(text) > maxlen:
        text = text[: maxlen - 1] + "…"
    return text


def _load_graph(source):
    g = Graph()
    if hasattr(source, "serialize") and not isinstance(source, str):
        # a prov.model.ProvDocument
        data = source.serialize(format="rdf", rdf_format="json-ld")
        g.parse(data=data, format="json-ld")
    else:
        fmt = "json-ld" if str(source).endswith((".jsonld", ".json")) else "turtle"
        g.parse(source, format=fmt)
    return g


def to_mermaid(source):
    g = _load_graph(source)

    # classify nodes by PROV type
    kind = {}
    for s, _, o in g.triples((None, RDF.type, None)):
        if o == PROV.Entity:
            kind.setdefault(s, "entity")
        elif o == PROV.Activity:
            kind[s] = "activity"
        elif o in (PROV.Agent, PROV.SoftwareAgent):
            kind[s] = "agent"

    def attr(node, pred):
        v = g.value(node, pred)
        return None if v is None else str(v)

    lines = ["flowchart LR"]

    # ---- nodes ----
    for node, k in kind.items():
        btm_kind = attr(node, BTM.kind)
        val = attr(node, PROV.value)
        loc = ""
        f, ln = attr(node, BTM.file), attr(node, BTM.line)
        if f and ln:
            loc = f"{f}:L{ln}<br/>"
        label = _esc((loc + (val or _local(node))) if k == "entity" else _local(node))
        nid = _nid(node)
        if k == "activity":
            lines.append(f'    {nid}(["{label}"])')
        elif k == "agent":
            lines.append(f'    {nid}{{{{"{label}"}}}}')
        elif btm_kind == "response":
            lines.append(f'    {nid}["OUTPUT: {label}"]')
        elif btm_kind == "prompt":
            lines.append(f'    {nid}["PROMPT: {label}"]')
        else:
            lines.append(f'    {nid}["{label}"]')

    # ---- qualified derivations (attribution + asserted), with their attributes ----
    qualified_pairs = set()
    for subj, _, qd in g.triples((None, PROV.qualifiedDerivation, None)):
        target = g.value(qd, PROV.entity)
        if target is None:
            continue
        method = attr(qd, BTM.method)
        score = attr(qd, BTM.score)
        rank = attr(qd, BTM.rank)
        bkind = attr(qd, BTM.kind)
        relation = attr(qd, BTM.relation)
        qualified_pairs.add((subj, target))
        if bkind == "asserted":
            lbl = _esc(relation or "asserted")
            lines.append(f'    {_nid(subj)} -. "{lbl} (asserted)" .-> {_nid(target)}')
        else:
            sc = f" {float(score):+.3f}" if score is not None else ""
            rk = f" #{rank}" if rank else ""
            lines.append(f'    {_nid(subj)} ==>|"{_esc((method or "attr") + sc + rk)}"| {_nid(target)}')

    # ---- plain wasDerivedFrom not already covered by a qualified derivation ----
    for s, _, o in g.triples((None, PROV.wasDerivedFrom, None)):
        if (s, o) in qualified_pairs or s not in kind or o not in kind:
            continue
        lines.append(f'    {_nid(s)} --> {_nid(o)}')

    # ---- structural PROV edges ----
    structural = [
        (PROV.wasGeneratedBy, "wasGeneratedBy"),
        (PROV.used, "used"),
        (PROV.wasAssociatedWith, "wasAssociatedWith"),
        (PROV.wasAttributedTo, "wasAttributedTo"),
    ]
    for pred, name in structural:
        for s, _, o in g.triples((None, pred, None)):
            if s in kind and o in kind:
                lines.append(f'    {_nid(s)} -- "{name}" --> {_nid(o)}')

    lines.append("")
    lines.append("    classDef output fill:#1b3a2b,stroke:#3fb27f,color:#eaeaea;")
    # tag the response node so it stands out
    for node, k in kind.items():
        if attr(node, BTM.kind) == "response":
            lines.append(f"    class {_nid(node)} output;")
    return "\n".join(lines) + "\n"


def to_svg(prov_document, out_path):
    try:
        from prov.dot import prov_to_dot
        dot = prov_to_dot(prov_document)
        dot.write_svg(out_path)
        return True
    except Exception as e:
        sys.stderr.write(f"[prov_render] SVG skipped ({e})\n")
        return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python prov_render.py <file.jsonld>")
    print(to_mermaid(sys.argv[1]))
