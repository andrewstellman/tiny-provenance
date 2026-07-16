"""
tiny_lm.py  --  the tiny language model, shared by build_model.py and prompt_model.py

This is a deliberately small, readable transformer (1 block, 1 head by default) that
trains in seconds on a few hundred tokens. It is faithful to a real decoder-only
transformer (token + positional embeddings -> pre-norm causal self-attention -> MLP ->
tied output head), just shrunk down so the whole thing fits in a ~40K-parameter JSON file.

Nothing here is specific to provenance. The provenance machinery lives in build_model.py
(what gets recorded) and prompt_model.py (how an output is attributed back to sources).
"""

import re
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Special tokens.
#   <pad> : padding (unused in this tiny setup but kept for completeness)
#   <eos> : hard boundary (blank line / end of document) -- generation stops here
#   <nl>  : soft line break inside a coherent block (e.g. between a Q line and its A line)
#   <unk> : out-of-vocabulary token
SPECIAL = ["<pad>", "<eos>", "<nl>", "<unk>"]

# Word/number/punctuation tokenizer. Each word, each run of digits, and each individual
# punctuation character becomes its own token. Small and fully transparent.
TOKEN_RE = re.compile(r"[A-Za-z]+|[0-9]+|[^\sA-Za-z0-9]")


def set_seed(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def tokenize_line(text: str):
    return TOKEN_RE.findall(text)


class Tokenizer:
    def __init__(self, vocab):
        self.itos = list(vocab)
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    @classmethod
    def build(cls, texts):
        vocab = list(SPECIAL)
        seen = set(vocab)
        for t in texts:
            for tok in tokenize_line(t):
                if tok not in seen:
                    seen.add(tok)
                    vocab.append(tok)
        return cls(vocab)

    @property
    def vocab_size(self):
        return len(self.itos)

    def encode(self, text):
        return [self.stoi.get(t, self.stoi["<unk>"]) for t in tokenize_line(text)]

    def decode(self, ids, keep_specials=False):
        toks = [self.itos[i] for i in ids]
        return detokenize(toks, keep_specials=keep_specials)


def detokenize(toks, keep_specials=False):
    """Rough inverse of the tokenizer -- good enough for human-readable display."""
    out = []
    for t in toks:
        if t in ("<pad>", "<eos>", "<unk>"):
            if keep_specials:
                out.append(t)
            continue
        if t == "<nl>":
            out.append("\n" if keep_specials else " ")
            continue
        if re.fullmatch(r"[^\sA-Za-z0-9]", t):
            # attach punctuation to the previous token (no leading space)
            if out and not out[-1].endswith("\n"):
                out[-1] = out[-1].rstrip() + t
            else:
                out.append(t)
        else:
            out.append(t)
    return " ".join(w for w in " ".join(out).split(" ") if w != "").strip()


class TinyLM(nn.Module):
    """A single-block, (by default) single-head causal transformer."""

    def __init__(self, vocab_size, d_model=48, n_head=1, block_size=32, d_ff=None):
        super().__init__()
        assert d_model % n_head == 0
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_head = n_head
        self.block_size = block_size
        d_ff = d_ff or 4 * d_model
        self.d_ff = d_ff

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)

        # one attention block, written out longhand so every matrix is inspectable
        self.ln1 = nn.LayerNorm(d_model)
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)

        # one MLP
        self.ln2 = nn.LayerNorm(d_model)
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying

        mask = torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None, :, :]

        h = self.ln1(x)
        nh, hd = self.n_head, self.d_model // self.n_head
        q = self.q(h).view(B, T, nh, hd).transpose(1, 2)
        k = self.k(h).view(B, T, nh, hd).transpose(1, 2)
        v = self.v(h).view(B, T, nh, hd).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, self.d_model)
        x = x + self.o(y)

        h2 = self.ln2(x)
        x = x + self.ff2(F.gelu(self.ff1(h2)))

        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100
            )
        return logits, loss


# ----------------------------------------------------------------------------------
# (de)serialization to a plain JSON-able dict -- "the model is just a JSON file"
# ----------------------------------------------------------------------------------

def state_to_lists(model):
    return {k: v.detach().cpu().numpy().tolist() for k, v in model.state_dict().items()}


def model_to_dict(model, config, tok):
    return {"config": config, "vocab": tok.itos, "weights": state_to_lists(model)}


def model_from_dict(d):
    tok = Tokenizer(d["vocab"])
    cfg = d["config"]
    model = TinyLM(tok.vocab_size, cfg["d_model"], cfg["n_head"], cfg["block_size"], cfg.get("d_ff"))
    sd = {k: torch.tensor(v, dtype=torch.float32) for k, v in d["weights"].items()}
    model.load_state_dict(sd, strict=False)  # strict=False: 'mask' buffer is non-persistent
    model.eval()
    return model, tok, cfg


# ----------------------------------------------------------------------------------
# training
# ----------------------------------------------------------------------------------

def build_token_stream(raw_lines, tok):
    """raw_lines: list of (file, line_no, text_or_None-for-blank), in corpus order.
    Returns a flat list of token ids using <nl> for line breaks and <eos> for blank
    lines / document boundaries."""
    ids = []
    for _, _, text in raw_lines:
        if text is None or text.strip() == "":
            ids.append(tok.stoi["<eos>"])
        else:
            ids.extend(tok.encode(text))
            ids.append(tok.stoi["<nl>"])
    ids.append(tok.stoi["<eos>"])
    return ids


def _windows(ids, block):
    data = torch.tensor(ids, dtype=torch.long)
    X, Y = [], []
    for i in range(0, len(data) - block - 1):
        X.append(data[i:i + block])
        Y.append(data[i + 1:i + block + 1])
    if not X:  # corpus shorter than block: use the whole thing as one window
        return data[None, :-1], data[None, 1:]
    return torch.stack(X), torch.stack(Y)


def train_model(raw_lines, tok, config, seed=1337, capture=(0.25, 0.5, 0.75, 1.0), verbose=False):
    """Train from scratch on the given raw_lines. Returns (model, checkpoints, final_loss).
    `capture` are fractions of total training steps at which to snapshot weights (for TracIn).
    Pass capture=() to skip snapshots (used by the fast leave-one-out retrains)."""
    set_seed(seed)
    ids = build_token_stream(raw_lines, tok)
    block = config["block_size"]
    X, Y = _windows(ids, block)
    n = X.shape[0]
    bs = min(config["batch_size"], n)

    model = TinyLM(tok.vocab_size, config["d_model"], config["n_head"], block, config.get("d_ff"))
    opt = torch.optim.Adam(model.parameters(), lr=config["lr"])

    total_steps = config["epochs"] * math.ceil(n / bs)
    cap_steps = {max(1, int(f * total_steps)) for f in capture}
    checkpoints = []

    g = torch.Generator().manual_seed(seed)
    step = 0
    last_loss = float("nan")
    for ep in range(config["epochs"]):
        perm = torch.randperm(n, generator=g)
        for j in range(0, n, bs):
            idx = perm[j:j + bs]
            _, loss = model(X[idx], Y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            last_loss = loss.item()
            if step in cap_steps:
                checkpoints.append({"weights": state_to_lists(model), "lr": config["lr"]})
        if verbose and (ep % max(1, config["epochs"] // 6) == 0):
            print(f"  epoch {ep:4d}  loss {last_loss:.4f}")
    return model, checkpoints, last_loss


# ----------------------------------------------------------------------------------
# generation + scoring helpers (used by attribution)
# ----------------------------------------------------------------------------------

@torch.no_grad()
def generate_line(model, tok, prompt, max_new_tokens=30):
    """Greedy-decode a single line: skip leading separators, then emit until the next
    <nl>/<eos>. Deterministic, so attribution is stable."""
    ids = tok.encode(prompt)
    block = model.block_size
    sep = {tok.stoi["<nl>"], tok.stoi["<eos>"]}
    gen, started = [], False
    for _ in range(max_new_tokens):
        ctx = torch.tensor([ids[-block:]])
        logits, _ = model(ctx)
        nxt = int(torch.argmax(logits[0, -1]))
        if nxt in sep:
            if not started:
                ids.append(nxt)   # consume the separator that ends the prompt line
                continue
            break
        started = True
        ids.append(nxt)
        gen.append(nxt)
    return gen, tok.decode(gen)


@torch.no_grad()
def answer_logprob(model, tok, prompt_ids, answer_ids):
    """Total log-probability the model assigns to answer_ids following prompt_ids."""
    block = model.block_size
    seq = prompt_ids + [tok.stoi["<nl>"]] + answer_ids
    seq = seq[-(block + 1):]
    x = torch.tensor([seq[:-1]])
    y = torch.tensor([seq[1:]])
    logits, _ = model(x)
    logprobs = F.log_softmax(logits[0], dim=-1)
    # only score the positions that predict the answer tokens
    n_ans = len(answer_ids)
    total = 0.0
    for pos in range(len(seq) - 1 - n_ans, len(seq) - 1):
        if pos < 0:
            continue
        total += float(logprobs[pos, y[0, pos]])
    return total


def flat_grad_of(model, loss):
    model.zero_grad(set_to_none=True)
    loss.backward()
    grads = []
    for p in model.parameters():
        grads.append((p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1))
    return torch.cat(grads)
