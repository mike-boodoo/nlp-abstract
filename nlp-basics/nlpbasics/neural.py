"""Neural language models, each removing a limitation of the last.

    WindowMLP         fixed context of k words; learns similarity, sees no further
    RNNLM             unbounded context in principle, squeezed through one vector
    TinyTransformer   every position can look directly at every earlier position

All three share one interface: forward(ids of shape (B, T)) returns logits of
shape (B, T, vocab), the next-token prediction at every position.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .autograd import Adam, Tensor, concat, cross_entropy


class Module:
    def parameters(self) -> list[Tensor]:
        params: list[Tensor] = []
        for value in vars(self).values():
            if isinstance(value, Tensor) and value.requires_grad:
                params.append(value)
            elif isinstance(value, Module):
                params += value.parameters()
            elif isinstance(value, (list, tuple)):
                params += [q for item in value if isinstance(item, Module)
                           for q in item.parameters()]
        return params

    def num_parameters(self) -> int:
        return sum(p.data.size for p in self.parameters())


class Linear(Module):
    def __init__(self, n_in: int, n_out: int, rng: np.random.Generator, bias: bool = True):
        self.W = Tensor(rng.normal(0, 1 / math.sqrt(n_in), (n_in, n_out)), requires_grad=True)
        self.b = Tensor(np.zeros(n_out), requires_grad=True) if bias else None

    def __call__(self, x: Tensor) -> Tensor:
        out = x @ self.W
        return out + self.b if self.b is not None else out


class Embedding(Module):
    """A lookup table of trainable vectors, the same object as the word vectors
    of lesson 8 but tuned for this model's task."""

    def __init__(self, vocab_size: int, dim: int, rng: np.random.Generator):
        self.weight = Tensor(rng.normal(0, 0.1, (vocab_size, dim)), requires_grad=True)

    def __call__(self, ids: np.ndarray) -> Tensor:
        return self.weight[ids]


class LayerNorm(Module):
    def __init__(self, dim: int):
        self.gamma = Tensor(np.ones(dim), requires_grad=True)
        self.beta = Tensor(np.zeros(dim), requires_grad=True)

    def __call__(self, x: Tensor) -> Tensor:
        mean = x.mean(axis=-1, keepdims=True)
        variance = ((x - mean) ** 2.0).mean(axis=-1, keepdims=True)
        return (x - mean) * ((variance + 1e-5) ** -0.5) * self.gamma + self.beta


class WindowMLP(Module):
    """Bengio et al. (2003): concatenate the last k embeddings, then an MLP.
    Unlike an n-gram table, similar words share statistical strength through
    similar embeddings. But position t-k-1 is simply invisible."""

    def __init__(self, vocab_size: int, context: int = 3, dim: int = 32, hidden: int = 96,
                 seed: int = 0):
        rng = np.random.default_rng(seed)
        self.context = context
        self.embed = Embedding(vocab_size, dim, rng)
        self.hidden = Linear(context * dim, hidden, rng)
        self.out = Linear(hidden, vocab_size, rng)

    def __call__(self, ids: np.ndarray) -> Tensor:
        B, T = ids.shape
        padded = np.concatenate([np.zeros((B, self.context - 1), dtype=int), ids], axis=1)
        windows = np.stack([padded[:, i:i + T] for i in range(self.context)], axis=-1)
        vectors = self.embed(windows).reshape(B, T, -1)
        return self.out(self.hidden(vectors).tanh())


class RNNLM(Module):
    """Recurrent language model with a choice of cell.

    "elman":  h_t = tanh(W_x x_t + W_h h_t-1)
    "gru":    gates decide what to overwrite and what to keep:
                  z = sigmoid(...)     update gate
                  r = sigmoid(...)     reset gate
                  h~ = tanh(W x + U (r * h))
                  h = (1 - z) * h + z * h~
              The h path is close to additive, so gradients survive more steps.
    """

    def __init__(self, vocab_size: int, dim: int = 32, hidden: int = 64, cell: str = "gru",
                 seed: int = 0):
        if cell not in {"elman", "gru"}:
            raise ValueError(f"unknown cell: {cell}")
        rng = np.random.default_rng(seed)
        self.cell, self.hidden_size = cell, hidden
        self.embed = Embedding(vocab_size, dim, rng)
        self.x_h = Linear(dim, hidden, rng)
        self.h_h = Linear(hidden, hidden, rng, bias=False)
        if cell == "gru":
            self.x_z, self.h_z = Linear(dim, hidden, rng), Linear(hidden, hidden, rng, bias=False)
            self.x_r, self.h_r = Linear(dim, hidden, rng), Linear(hidden, hidden, rng, bias=False)
        self.out = Linear(hidden, vocab_size, rng)

    def _step(self, x: Tensor, h: Tensor) -> Tensor:
        if self.cell == "elman":
            return (self.x_h(x) + self.h_h(h)).tanh()
        z = (self.x_z(x) + self.h_z(h)).sigmoid()
        r = (self.x_r(x) + self.h_r(h)).sigmoid()
        candidate = (self.x_h(x) + self.h_h(r * h)).tanh()
        return (1.0 - z) * h + z * candidate

    def __call__(self, ids: np.ndarray) -> Tensor:
        B, T = ids.shape
        x = self.embed(ids)
        h = Tensor(np.zeros((B, 1, self.hidden_size)))
        states = []
        for t in range(T):
            h = self._step(x[:, t:t + 1, :], h)
            states.append(h)
        return self.out(concat(states, axis=1))


def scaled_dot_product_attention(q: Tensor, k: Tensor, v: Tensor,
                                 mask: np.ndarray | None = None) -> tuple[Tensor, np.ndarray]:
    """Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

    Each query scores every key, the scores become a probability distribution,
    and the output is the matching average of the values. A soft dictionary
    lookup. Dividing by sqrt(d_k) keeps dot products from growing with dimension
    and saturating the softmax, which would flatten the gradients.
    """
    d_k = k.shape[-1]
    scores = q @ k.transpose(*range(k.data.ndim - 2), k.data.ndim - 1, k.data.ndim - 2)
    scores = scores * (1.0 / math.sqrt(d_k))
    if mask is not None:
        scores = scores.masked_fill(mask, -1e9)
    weights = scores.softmax(-1)
    return weights @ v, weights.data


class MultiHeadAttention(Module):
    """Run several smaller attentions in parallel, so different heads can track
    different relations. Heads are a reshape: d_model splits into h chunks."""

    def __init__(self, d_model: int, heads: int, rng: np.random.Generator):
        if d_model % heads:
            raise ValueError("d_model must be divisible by heads")
        self.heads, self.d_k = heads, d_model // heads
        self.query = Linear(d_model, d_model, rng, bias=False)
        self.key = Linear(d_model, d_model, rng, bias=False)
        self.value = Linear(d_model, d_model, rng, bias=False)
        self.proj = Linear(d_model, d_model, rng)
        self.last_weights: np.ndarray | None = None

    def _split(self, x: Tensor, B: int, T: int) -> Tensor:
        return x.reshape(B, T, self.heads, self.d_k).transpose(0, 2, 1, 3)

    def __call__(self, x: Tensor, mask: np.ndarray) -> Tensor:
        B, T, _ = x.shape
        out, weights = scaled_dot_product_attention(
            self._split(self.query(x), B, T), self._split(self.key(x), B, T),
            self._split(self.value(x), B, T), mask)
        self.last_weights = weights              # (B, heads, T, T)
        return self.proj(out.transpose(0, 2, 1, 3).reshape(B, T, self.heads * self.d_k))


class Block(Module):
    """Pre-norm transformer block. The residual additions give gradients a
    direct route to early layers, and attention is the only place where
    positions exchange information. The MLP processes each position alone."""

    def __init__(self, d_model: int, heads: int, rng: np.random.Generator):
        self.norm1, self.norm2 = LayerNorm(d_model), LayerNorm(d_model)
        self.attention = MultiHeadAttention(d_model, heads, rng)
        self.fc1 = Linear(d_model, 4 * d_model, rng)
        self.fc2 = Linear(4 * d_model, d_model, rng)

    def __call__(self, x: Tensor, mask: np.ndarray) -> Tensor:
        x = x + self.attention(self.norm1(x), mask)
        return x + self.fc2(self.fc1(self.norm2(x)).relu())


class TinyTransformer(Module):
    """A GPT-style decoder: tokens + positions -> N blocks -> next-token logits.

    Attention on its own has no idea of order, since it treats the input as a
    set. Position embeddings add that back. The causal mask stops each position
    from attending to the future, which is the very thing it must predict.
    """

    def __init__(self, vocab_size: int, max_len: int = 32, d_model: int = 48, heads: int = 4,
                 layers: int = 2, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.max_len = max_len
        self.tokens = Embedding(vocab_size, d_model, rng)
        self.positions = Embedding(max_len, d_model, rng)
        self.blocks = [Block(d_model, heads, rng) for _ in range(layers)]
        self.norm = LayerNorm(d_model)
        self.head = Linear(d_model, vocab_size, rng)

    def __call__(self, ids: np.ndarray) -> Tensor:
        T = ids.shape[1]
        if T > self.max_len:
            raise ValueError(f"sequence length {T} exceeds max_len {self.max_len}")
        x = self.tokens(ids) + self.positions(np.arange(T))
        mask = np.triu(np.ones((T, T), dtype=bool), k=1)      # True = may not look.
        for block in self.blocks:
            x = block(x, mask)
        return self.head(self.norm(x))

    def attention_maps(self, ids: Sequence[int]) -> list[np.ndarray]:
        """Attention by layer for one sequence, averaged over heads."""
        self(np.array([ids]))
        return [b.attention.last_weights[0].mean(axis=0) for b in self.blocks]


def train_language_model(model: Module, sequences: np.ndarray, steps: int = 1000,
                         batch_size: int = 32, lr: float = 3e-3, pad_id: int = 0,
                         seed: int = 0, log_every: int = 0) -> list[float]:
    """Mini-batch teacher forcing: inputs are tokens 0..T-1, targets are 1..T."""
    rng = np.random.default_rng(seed)
    optimizer = Adam(model.parameters(), lr=lr)
    losses: list[float] = []
    for step in range(1, steps + 1):
        batch = sequences[rng.integers(0, len(sequences), batch_size)]
        loss = cross_entropy(model(batch[:, :-1]), batch[:, 1:], ignore_index=pad_id)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if log_every and step % log_every == 0:
            print(f"    step {step:>5}  loss {np.mean(losses[-log_every:]):.3f}")
    return losses


def next_token_probs(model: Module, ids: Sequence[int]) -> np.ndarray:
    logits = model(np.array([ids])).data[0, -1]
    return np.exp(logits - logits.max()) / np.exp(logits - logits.max()).sum()


def sample(model: Module, prompt: Sequence[int], max_new: int = 20, temperature: float = 1.0,
           top_k: int | None = None, stop_id: int | None = None, seed: int = 0,
           max_len: int | None = None) -> list[int]:
    """Autoregressive generation. Temperature below 1 sharpens the distribution
    and above 1 flattens it. top_k restricts sampling to the k likeliest tokens."""
    rng = np.random.default_rng(seed)
    ids = list(prompt)
    for _ in range(max_new):
        window = ids[-max_len:] if max_len else ids
        probs = next_token_probs(model, window)
        logp = np.log(probs + 1e-12) / max(temperature, 1e-6)
        if top_k is not None:
            logp[np.argsort(logp)[:-top_k]] = -np.inf
        p = np.exp(logp - logp.max())
        next_id = int(rng.choice(len(p), p=p / p.sum()))
        ids.append(next_id)
        if next_id == stop_id:
            break
    return ids
