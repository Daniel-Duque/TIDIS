"""Small neural networks (GRU recurrent network and Transformer encoder) in pure NumPy.

A minimal reverse-mode automatic differentiation engine is enough for the sizes used here
(a few hundred daily observations, tens of hidden units) and avoids a heavy deep-learning
dependency on Streamlit Cloud. Gradients are verified against finite differences in the tests.
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- autodiff
def _unbroadcast(grad: np.ndarray, shape: tuple) -> np.ndarray:
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


class T:
    """Array with gradient tracking."""
    __slots__ = ("data", "grad", "_parents", "_back")

    def __init__(self, data, parents=(), back=None):
        self.data = np.asarray(data, dtype=float)
        self.grad = None
        self._parents = parents
        self._back = back

    # ----- helpers
    @staticmethod
    def _t(x) -> "T":
        return x if isinstance(x, T) else T(x)

    def _acc(self, g: np.ndarray) -> None:
        self.grad = g if self.grad is None else self.grad + g

    @property
    def shape(self):
        return self.data.shape

    # ----- arithmetic
    def __add__(self, o):
        o = T._t(o)
        out = T(self.data + o.data, (self, o))
        out._back = lambda g: (self._acc(_unbroadcast(g, self.shape)), o._acc(_unbroadcast(g, o.shape)))
        return out

    __radd__ = __add__

    def __neg__(self):
        out = T(-self.data, (self,))
        out._back = lambda g: self._acc(-g)
        return out

    def __sub__(self, o):
        return self + (-T._t(o))

    def __rsub__(self, o):
        return T._t(o) + (-self)

    def __mul__(self, o):
        o = T._t(o)
        out = T(self.data * o.data, (self, o))
        out._back = lambda g: (self._acc(_unbroadcast(g * o.data, self.shape)),
                               o._acc(_unbroadcast(g * self.data, o.shape)))
        return out

    __rmul__ = __mul__

    def __truediv__(self, o):
        o = T._t(o)
        return self * (o ** -1.0)

    def __pow__(self, p: float):
        out = T(self.data ** p, (self,))
        out._back = lambda g: self._acc(g * p * self.data ** (p - 1))
        return out

    def __matmul__(self, o):
        o = T._t(o)
        out = T(self.data @ o.data, (self, o))

        def back(g):
            ga = g @ np.swapaxes(o.data, -1, -2)
            gb = np.swapaxes(self.data, -1, -2) @ g
            self._acc(_unbroadcast(ga, self.shape))
            o._acc(_unbroadcast(gb, o.shape))
        out._back = back
        return out

    # ----- element-wise functions
    def tanh(self):
        y = np.tanh(self.data)
        out = T(y, (self,))
        out._back = lambda g: self._acc(g * (1 - y * y))
        return out

    def sigmoid(self):
        y = 1 / (1 + np.exp(-np.clip(self.data, -60, 60)))
        out = T(y, (self,))
        out._back = lambda g: self._acc(g * y * (1 - y))
        return out

    def relu(self):
        m = self.data > 0
        out = T(self.data * m, (self,))
        out._back = lambda g: self._acc(g * m)
        return out

    def exp(self):
        y = np.exp(self.data)
        out = T(y, (self,))
        out._back = lambda g: self._acc(g * y)
        return out

    # ----- reductions and shapes
    def sum(self, axis=None, keepdims=False):
        out = T(self.data.sum(axis=axis, keepdims=keepdims), (self,))

        def back(g):
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            self._acc(np.broadcast_to(g, self.shape).copy())
        out._back = back
        return out

    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis, keepdims) * (1.0 / n)

    def reshape(self, *shape):
        out = T(self.data.reshape(*shape), (self,))
        out._back = lambda g: self._acc(g.reshape(self.shape))
        return out

    def transpose(self, *axes):
        out = T(self.data.transpose(*axes), (self,))
        inv = np.argsort(axes)
        out._back = lambda g: self._acc(g.transpose(*inv))
        return out

    def __getitem__(self, idx):
        out = T(self.data[idx], (self,))

        def back(g):
            full = np.zeros_like(self.data)
            full[idx] += g                                   # basic slicing only (no repeated indices)
            self._acc(full)
        out._back = back
        return out

    # ----- backward pass
    def backward(self) -> None:
        order, seen = [], set()
        stack = [(self, False)]
        while stack:                                     # iterative topological sort
            node, done = stack.pop()
            if done:
                order.append(node)
                continue
            if id(node) in seen:
                continue
            seen.add(id(node))
            stack.append((node, True))
            stack.extend((p, False) for p in node._parents if id(p) not in seen)
        self.grad = np.ones_like(self.data)
        for node in reversed(order):
            if node._back is not None and node.grad is not None:
                node._back(node.grad)


def softmax(x: T, axis: int = -1) -> T:
    e = (x - T(x.data.max(axis=axis, keepdims=True))).exp()
    return e / e.sum(axis=axis, keepdims=True)


def layer_norm(x: T, gain: T, bias: T, eps: float = 1e-5) -> T:
    xc = x - x.mean(axis=-1, keepdims=True)
    var = (xc * xc).mean(axis=-1, keepdims=True)
    return xc * ((var + eps) ** -0.5) * gain + bias


class Adam:
    def __init__(self, params: list[T], lr: float = 0.01, weight_decay: float = 1e-4):
        self.params, self.lr, self.wd = params, lr, weight_decay
        self.m = [np.zeros_like(p.data) for p in params]
        self.v = [np.zeros_like(p.data) for p in params]
        self.t = 0

    def step(self) -> None:
        self.t += 1
        b1, b2 = 0.9, 0.999
        for p, m, v in zip(self.params, self.m, self.v):
            if p.grad is None:
                continue
            g = p.grad + self.wd * p.data
            m[:] = b1 * m + (1 - b1) * g
            v[:] = b2 * v + (1 - b2) * g * g
            p.data -= self.lr * (m / (1 - b1 ** self.t)) / (np.sqrt(v / (1 - b2 ** self.t)) + 1e-8)
            p.grad = None


# --------------------------------------------------------------------------- models
def _init(rng, *shape, scale=None) -> T:
    scale = scale if scale is not None else 1 / np.sqrt(shape[0])
    return T(rng.normal(0, scale, shape))


class GRU:
    """Gated recurrent network: sequence (B, W, F) -> next value (B, O)."""

    def __init__(self, n_in: int, n_out: int, hidden: int = 24, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.H = hidden
        self.W = _init(rng, n_in, 3 * hidden)
        self.Uz, self.Ur, self.Uh = (_init(rng, hidden, hidden) for _ in range(3))
        self.b = T(np.zeros(3 * hidden))
        self.Wo, self.bo = _init(rng, hidden, n_out), T(np.zeros(n_out))

    def params(self):
        return [self.W, self.Uz, self.Ur, self.Uh, self.b, self.Wo, self.bo]

    def __call__(self, x: T) -> T:
        B, W, _ = x.shape
        H = self.H
        xw = x @ self.W + self.b                                  # (B, W, 3H), all time steps at once
        h = T(np.zeros((B, H)))
        for t in range(W):
            z = (xw[:, t, 0:H] + h @ self.Uz).sigmoid()
            r = (xw[:, t, H:2 * H] + h @ self.Ur).sigmoid()
            n = (xw[:, t, 2 * H:] + (r * h) @ self.Uh).tanh()
            h = (1 - z) * n + z * h
        return h @ self.Wo + self.bo


class Transformer:
    """Transformer encoder (multi-head self-attention + feed-forward, pre-computed sinusoidal
    positions): sequence (B, W, F) -> next value (B, O) read from the last position."""

    def __init__(self, n_in: int, n_out: int, d_model: int = 24, heads: int = 2, layers: int = 2,
                 window: int = 28, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.d, self.h = d_model, heads
        self.Win, self.bin = _init(rng, n_in, d_model), T(np.zeros(d_model))
        pos = np.arange(window)[:, None] / (10000 ** (np.arange(0, d_model, 2)[None, :] / d_model))
        pe = np.zeros((window, d_model))
        pe[:, 0::2], pe[:, 1::2] = np.sin(pos), np.cos(pos)
        self.pe = pe
        self.layers = []
        for _ in range(layers):
            self.layers.append({
                "Wq": _init(rng, d_model, d_model), "Wk": _init(rng, d_model, d_model),
                "Wv": _init(rng, d_model, d_model), "Wo": _init(rng, d_model, d_model),
                "g1": T(np.ones(d_model)), "b1": T(np.zeros(d_model)),
                "W1": _init(rng, d_model, 2 * d_model), "c1": T(np.zeros(2 * d_model)),
                "W2": _init(rng, 2 * d_model, d_model), "c2": T(np.zeros(d_model)),
                "g2": T(np.ones(d_model)), "b2": T(np.zeros(d_model)),
            })
        self.Wout, self.bout = _init(rng, d_model, n_out), T(np.zeros(n_out))

    def params(self):
        ps = [self.Win, self.bin, self.Wout, self.bout]
        for lay in self.layers:
            ps += list(lay.values())
        return ps

    def __call__(self, x: T) -> T:
        B, W, _ = x.shape
        d, nh = self.d, self.h
        dh = d // nh
        hdn = x @ self.Win + self.bin + T(self.pe[-W:])
        for L in self.layers:
            def heads(m):
                return (hdn @ m).reshape(B, W, nh, dh).transpose(0, 2, 1, 3)     # (B, nh, W, dh)
            q, k, v = heads(L["Wq"]), heads(L["Wk"]), heads(L["Wv"])
            att = softmax((q @ k.transpose(0, 1, 3, 2)) * (1 / np.sqrt(dh)), axis=-1)
            o = (att @ v).transpose(0, 2, 1, 3).reshape(B, W, d) @ L["Wo"]
            hdn = layer_norm(hdn + o, L["g1"], L["b1"])
            ff = ((hdn @ L["W1"] + L["c1"]).relu()) @ L["W2"] + L["c2"]
            hdn = layer_norm(hdn + ff, L["g2"], L["b2"])
        return hdn[:, W - 1, :] @ self.Wout + self.bout


def train(model, X: np.ndarray, Y: np.ndarray, epochs: int = 200, lr: float = 0.01, val_frac: float = 0.15,
          batch: int = 128, seed: int = 0):
    """Adam on mean squared error. With val_frac > 0 the most recent windows are held out and the
    parameters with the best validation loss are kept; with val_frac = 0 all windows are used and
    the final parameters are kept (fixed number of epochs + weight decay as regularisation)."""
    rng = np.random.default_rng(seed)
    n = len(X)
    n_val = max(1, int(n * val_frac)) if (n >= 20 and val_frac > 0) else 0
    Xt, Yt, Xv, Yv = X[:n - n_val], Y[:n - n_val], X[n - n_val:], Y[n - n_val:]
    opt = Adam(model.params(), lr=lr)
    best, best_loss, history = None, np.inf, []
    for _ in range(epochs):
        idx = rng.permutation(len(Xt))
        for s in range(0, len(idx), batch):
            b = idx[s:s + batch]
            err = model(T(Xt[b])) - T(Yt[b])
            loss = (err * err).mean()
            loss.backward()
            opt.step()
        if n_val:                                        # keep the parameters with the best validation loss
            vl = float(((model(T(Xv)).data - Yv) ** 2).mean())
            history.append(vl)
            if vl < best_loss:
                best_loss, best = vl, [p.data.copy() for p in model.params()]
    if best is not None:
        for p, b in zip(model.params(), best):
            p.data = b
    return history
