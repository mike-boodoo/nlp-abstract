"""Reverse-mode automatic differentiation on NumPy arrays: PyTorch's core idea
in about 200 lines.

Every operation records its inputs and a closure saying how to push the output's
gradient back to them. backward() walks that graph in reverse topological order,
applying the chain rule. That walk is backpropagation.

Broadcasting needs care: if a (3,) array was stretched to (4, 3) in the forward
pass, the (4, 3) gradient must be summed back down to (3,). See _unbroadcast.
"""
from __future__ import annotations

from typing import Callable, Iterable, Sequence

import numpy as np


def _unbroadcast(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Sum out the axes that broadcasting added or stretched."""
    extra = grad.ndim - len(shape)
    if extra > 0:
        grad = grad.sum(axis=tuple(range(extra)))
    axes = tuple(i for i, dim in enumerate(shape) if dim == 1 and grad.shape[i] != 1)
    return grad.sum(axis=axes, keepdims=True) if axes else grad


class Tensor:
    __slots__ = ("data", "grad", "_parents", "_backward", "requires_grad")
    __array_priority__ = 1000  # Make ndarray + Tensor defer to Tensor.__radd__.

    def __init__(self, data, requires_grad: bool = False,
                 parents: Sequence["Tensor"] = (),
                 backward: Callable[[np.ndarray], None] | None = None):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad: np.ndarray | None = None
        self.requires_grad = requires_grad
        self._parents = tuple(parents)
        self._backward = backward

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape}, requires_grad={self.requires_grad})"

    def item(self) -> float:
        return float(self.data)

    @staticmethod
    def _wrap(x) -> "Tensor":
        return x if isinstance(x, Tensor) else Tensor(x)

    def _result(self, data, parents, backward) -> "Tensor":
        needs_grad = any(p.requires_grad for p in parents)
        return Tensor(data, needs_grad, parents if needs_grad else (),
                      backward if needs_grad else None)

    def _accumulate(self, grad: np.ndarray) -> None:
        if self.requires_grad:
            self.grad = grad.copy() if self.grad is None else self.grad + grad

    def backward(self) -> None:
        """Backpropagate from a scalar output. Iterative DFS, so that long RNN
        graphs cannot hit Python's recursion limit."""
        if self.data.size != 1:
            raise ValueError("backward() needs a scalar output")
        order: list[Tensor] = []
        seen: set[int] = set()
        stack: list[tuple[Tensor, bool]] = [(self, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                order.append(node)
            elif id(node) not in seen:
                seen.add(id(node))
                stack.append((node, True))
                stack.extend((p, False) for p in node._parents if id(p) not in seen)
        self.grad = np.ones_like(self.data)
        for node in reversed(order):
            if node._backward is not None:
                node._backward(node.grad)

    # Arithmetic ------------------------------------------------------------
    def __add__(self, other):
        other = self._wrap(other)
        def back(g):
            self._accumulate(_unbroadcast(g, self.shape))
            other._accumulate(_unbroadcast(g, other.shape))
        return self._result(self.data + other.data, (self, other), back)

    __radd__ = __add__

    def __neg__(self):
        return self._result(-self.data, (self,), lambda g: self._accumulate(-g))

    def __sub__(self, other):
        return self + (-self._wrap(other))

    def __rsub__(self, other):
        return self._wrap(other) + (-self)

    def __mul__(self, other):
        other = self._wrap(other)
        def back(g):
            self._accumulate(_unbroadcast(g * other.data, self.shape))
            other._accumulate(_unbroadcast(g * self.data, other.shape))
        return self._result(self.data * other.data, (self, other), back)

    __rmul__ = __mul__

    def __truediv__(self, other):
        return self * (self._wrap(other) ** -1.0)

    def __rtruediv__(self, other):
        return self._wrap(other) * (self ** -1.0)

    def __pow__(self, exponent: float):
        def back(g):
            self._accumulate(g * exponent * self.data ** (exponent - 1))
        return self._result(self.data ** exponent, (self,), back)

    def __matmul__(self, other):
        other = self._wrap(other)
        if self.data.ndim < 2 or other.data.ndim < 2:
            raise ValueError("matmul expects at least 2-D operands")
        def back(g):
            self._accumulate(_unbroadcast(g @ np.swapaxes(other.data, -1, -2), self.shape))
            other._accumulate(_unbroadcast(np.swapaxes(self.data, -1, -2) @ g, other.shape))
        return self._result(self.data @ other.data, (self, other), back)

    # Functions -------------------------------------------------------------
    def exp(self):
        out = np.exp(self.data)
        return self._result(out, (self,), lambda g: self._accumulate(g * out))

    def log(self):
        return self._result(np.log(self.data), (self,),
                            lambda g: self._accumulate(g / self.data))

    def sqrt(self):
        return self ** 0.5

    def tanh(self):
        out = np.tanh(self.data)
        return self._result(out, (self,), lambda g: self._accumulate(g * (1 - out ** 2)))

    def sigmoid(self):
        out = 1 / (1 + np.exp(-self.data))
        return self._result(out, (self,), lambda g: self._accumulate(g * out * (1 - out)))

    def relu(self):
        return self._result(np.maximum(self.data, 0), (self,),
                            lambda g: self._accumulate(g * (self.data > 0)))

    def sum(self, axis=None, keepdims: bool = False):
        def back(g):
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            self._accumulate(np.broadcast_to(g, self.shape).copy())
        return self._result(self.data.sum(axis=axis, keepdims=keepdims), (self,), back)

    def mean(self, axis=None, keepdims: bool = False):
        count = self.data.size if axis is None else np.prod(
            [self.shape[a] for a in np.atleast_1d(axis)])
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / float(count))

    def reshape(self, *shape):
        return self._result(self.data.reshape(*shape), (self,),
                            lambda g: self._accumulate(g.reshape(self.shape)))

    def transpose(self, *axes):
        inverse = np.argsort(axes)
        return self._result(self.data.transpose(*axes), (self,),
                            lambda g: self._accumulate(g.transpose(*inverse)))

    def __getitem__(self, index):
        def back(g):
            full = np.zeros_like(self.data)
            np.add.at(full, index, g)      # Repeated indices must accumulate.
            self._accumulate(full)
        return self._result(self.data[index], (self,), back)

    def masked_fill(self, mask: np.ndarray, value: float):
        def back(g):
            self._accumulate(np.where(mask, 0.0, g))
        return self._result(np.where(mask, value, self.data), (self,), back)

    def log_softmax(self, axis: int = -1):
        shifted = self.data - self.data.max(axis=axis, keepdims=True)
        log_norm = np.log(np.exp(shifted).sum(axis=axis, keepdims=True))
        probs = np.exp(shifted - log_norm)
        def back(g):
            self._accumulate(g - probs * g.sum(axis=axis, keepdims=True))
        return self._result(shifted - log_norm, (self,), back)

    def softmax(self, axis: int = -1):
        return self.log_softmax(axis).exp()


def concat(tensors: Sequence[Tensor], axis: int = -1) -> Tensor:
    sizes = [t.shape[axis] for t in tensors]
    cuts = np.cumsum(sizes)[:-1]
    def back(g):
        for tensor, part in zip(tensors, np.split(g, cuts, axis=axis)):
            tensor._accumulate(part)
    return tensors[0]._result(np.concatenate([t.data for t in tensors], axis=axis),
                              tuple(tensors), back)


def cross_entropy(logits: Tensor, targets: np.ndarray, ignore_index: int | None = None) -> Tensor:
    """Mean negative log-likelihood of the targets. logits has shape (..., C),
    targets has shape (...). Positions equal to ignore_index do not count."""
    log_probs = logits.log_softmax(-1).reshape(-1, logits.shape[-1])
    flat = np.asarray(targets).reshape(-1)
    keep = np.arange(flat.size) if ignore_index is None else np.nonzero(flat != ignore_index)[0]
    return -log_probs[keep, flat[keep]].mean()


class Adam:
    """Adam (Kingma & Ba, 2015): per-parameter steps from running estimates of
    the gradient's first and second moments, with global-norm gradient clipping."""

    def __init__(self, params: Iterable[Tensor], lr: float = 1e-3,
                 betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8,
                 clip_norm: float | None = 1.0):
        self.params = list(params)
        self.lr, self.betas, self.eps, self.clip_norm = lr, betas, eps, clip_norm
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
        self.t = 0

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    def step(self) -> float:
        """Apply one update. Returns the gradient norm before clipping, the
        quantity to watch for vanishing or exploding gradients."""
        grads = [p.grad if p.grad is not None else np.zeros_like(p.data) for p in self.params]
        norm = float(np.sqrt(sum(float((g ** 2).sum()) for g in grads)))
        scale = min(1.0, self.clip_norm / (norm + 1e-12)) if self.clip_norm else 1.0
        self.t += 1
        b1, b2 = self.betas
        for i, (p, g) in enumerate(zip(self.params, grads)):
            g = g * scale
            self.m[i] = b1 * self.m[i] + (1 - b1) * g
            self.v[i] = b2 * self.v[i] + (1 - b2) * g ** 2
            m_hat = self.m[i] / (1 - b1 ** self.t)
            v_hat = self.v[i] / (1 - b2 ** self.t)
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        return norm
