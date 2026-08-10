"""
ml_model.py
-----------
A small feed-forward neural network implemented from scratch with numpy only

It has two output heads trained jointly:
  - classification head: will this configuration exhibit intermediate-axis
    flipping? (sigmoid output, binary cross-entropy loss)
  - regression head: predicted time (seconds) until the first flip
    (linear output, MSE loss)

Inputs (features), all derived from moments of inertia + initial spin:
  x0 = I1 / I3                (smallest / largest moment, in (0,1])
  x1 = I2 / I3                (intermediate / largest moment, in (0,1])
  x2 = spin-axis asymmetry parameter:
         ((I2 - I1) * (I3 - I2)) / (I1 * I3)
       this is the classic quantity that governs whether the middle-axis
       spin is unstable (it is > 0 whenever I1 < I2 < I3, i.e. always for a
       true intermediate axis; magnitude relates to how fast flips occur)
  x3 = perturbation magnitude ratio: |w_perp0| / |w_spin0|
  x4 = spin magnitude |w_spin0| (rad/s) -- needed because Euler's equations
       are only scale-invariant under a rescaled time variable, so flip time
       in real seconds depends on the absolute spin rate, not just ratios

Weights are saved/loaded as JSON so the whole pipeline only needs
numpy + json, matching the allowed library list.
"""

import json
import numpy as np


def _xavier(rng, n_in, n_out):
    limit = np.sqrt(6.0 / (n_in + n_out))
    return rng.uniform(-limit, limit, size=(n_in, n_out))


class FlipPredictorNet:
    """Single hidden-layer MLP, manual forward/backward pass, numpy only."""

    def __init__(self, n_in=5, n_hidden=16, seed=42):
        rng = np.random.default_rng(seed)
        self.n_in = n_in
        self.n_hidden = n_hidden

        self.W1 = _xavier(rng, n_in, n_hidden)
        self.b1 = np.zeros(n_hidden)

        # two output heads sharing the hidden layer
        self.W2_cls = _xavier(rng, n_hidden, 1)
        self.b2_cls = np.zeros(1)

        self.W2_reg = _xavier(rng, n_hidden, 1)
        self.b2_reg = np.zeros(1)

        # feature normalization stats, fit during training
        self.x_mean = np.zeros(n_in)
        self.x_std = np.ones(n_in)
        self.y_reg_mean = 0.0
        self.y_reg_std = 1.0

    #activation functions

    @staticmethod
    def _relu(z):
        return np.maximum(0, z)

    @staticmethod
    def _relu_grad(z):
        return (z > 0).astype(z.dtype)

    @staticmethod
    def _sigmoid(z):
        z = np.clip(z, -30, 30)
        return 1.0 / (1.0 + np.exp(-z))

    #forward / backward

    def _normalize_x(self, X):
        return (X - self.x_mean) / self.x_std

    def forward(self, X_raw):
        X = self._normalize_x(X_raw)
        z1 = X @ self.W1 + self.b1
        a1 = self._relu(z1)

        z_cls = a1 @ self.W2_cls + self.b2_cls
        p_flip = self._sigmoid(z_cls).ravel()

        z_reg = a1 @ self.W2_reg + self.b2_reg
        t_flip_norm = z_reg.ravel()

        cache = (X, z1, a1)
        return p_flip, t_flip_norm, cache

    def predict(self, X_raw):
        """Returns (probability_of_flipping, predicted_flip_time_seconds)."""
        X_raw = np.atleast_2d(X_raw)
        p_flip, t_norm, _ = self.forward(X_raw)
        t_flip = t_norm * self.y_reg_std + self.y_reg_mean
        return p_flip, t_flip

    def train(self, X_raw, y_cls, y_reg, epochs=2000, lr=0.05, reg_loss_weight=1.0,
              verbose=False):
        """
        Full-batch gradient descent with manually-derived backprop.
        X_raw: (N, n_in), y_cls: (N,) in {0,1}, y_reg: (N,) flip times (only
        meaningful where y_cls == 1; masked out of the regression loss otherwise).
        """
        X_raw = np.asarray(X_raw, dtype=float)
        y_cls = np.asarray(y_cls, dtype=float)
        y_reg = np.asarray(y_reg, dtype=float)
        N = X_raw.shape[0]

        # fit normalization stats
        self.x_mean = X_raw.mean(axis=0)
        self.x_std = X_raw.std(axis=0)
        self.x_std[self.x_std < 1e-8] = 1.0

        mask = y_cls > 0.5
        if mask.sum() > 0:
            self.y_reg_mean = y_reg[mask].mean()
            self.y_reg_std = y_reg[mask].std()
            if self.y_reg_std < 1e-8:
                self.y_reg_std = 1.0
        y_reg_norm = (y_reg - self.y_reg_mean) / self.y_reg_std

        history = {"loss": [], "cls_loss": [], "reg_loss": []}

        for epoch in range(epochs):
            p_flip, t_norm, (X, z1, a1) = self.forward(X_raw)

            # --- losses ---
            eps = 1e-9
            cls_loss = -np.mean(y_cls * np.log(p_flip + eps) +
                                 (1 - y_cls) * np.log(1 - p_flip + eps))
            if mask.sum() > 0:
                reg_loss = np.mean((t_norm[mask] - y_reg_norm[mask]) ** 2)
            else:
                reg_loss = 0.0
            loss = cls_loss + reg_loss_weight * reg_loss

            # --- backward pass (manual gradients) ---
            dcls = (p_flip - y_cls) / N                      # dL/dz_cls
            dW2_cls = a1.T @ dcls.reshape(-1, 1)
            db2_cls = dcls.sum()

            dreg = np.zeros(N)
            if mask.sum() > 0:
                dreg[mask] = 2.0 * (t_norm[mask] - y_reg_norm[mask]) / mask.sum()
            dreg = reg_loss_weight * dreg
            dW2_reg = a1.T @ dreg.reshape(-1, 1)
            db2_reg = dreg.sum()

            da1 = (dcls.reshape(-1, 1) @ self.W2_cls.T +
                   dreg.reshape(-1, 1) @ self.W2_reg.T)
            dz1 = da1 * self._relu_grad(z1)
            dW1 = X.T @ dz1
            db1 = dz1.sum(axis=0)

            #gradient descent update
            self.W2_cls -= lr * dW2_cls
            self.b2_cls -= lr * db2_cls
            self.W2_reg -= lr * dW2_reg
            self.b2_reg -= lr * db2_reg
            self.W1 -= lr * dW1
            self.b1 -= lr * db1

            history["loss"].append(loss)
            history["cls_loss"].append(cls_loss)
            history["reg_loss"].append(reg_loss)

            if verbose and (epoch % max(1, epochs // 10) == 0):
                print(f"epoch {epoch:5d}  loss={loss:.4f}  "
                      f"cls={cls_loss:.4f}  reg={reg_loss:.4f}")

        return history

    #persistence (JSON only, per library constraints)

    def to_dict(self):
        return {
            "n_in": self.n_in,
            "n_hidden": self.n_hidden,
            "W1": self.W1.tolist(), "b1": self.b1.tolist(),
            "W2_cls": self.W2_cls.tolist(), "b2_cls": self.b2_cls.tolist(),
            "W2_reg": self.W2_reg.tolist(), "b2_reg": self.b2_reg.tolist(),
            "x_mean": self.x_mean.tolist(), "x_std": self.x_std.tolist(),
            "y_reg_mean": self.y_reg_mean, "y_reg_std": self.y_reg_std,
        }

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path, "r") as f:
            d = json.load(f)
        net = cls(n_in=d["n_in"], n_hidden=d["n_hidden"])
        net.W1 = np.array(d["W1"]); net.b1 = np.array(d["b1"])
        net.W2_cls = np.array(d["W2_cls"]); net.b2_cls = np.array(d["b2_cls"])
        net.W2_reg = np.array(d["W2_reg"]); net.b2_reg = np.array(d["b2_reg"])
        net.x_mean = np.array(d["x_mean"]); net.x_std = np.array(d["x_std"])
        net.y_reg_mean = d["y_reg_mean"]; net.y_reg_std = d["y_reg_std"]
        return net


if __name__ == "__main__":
    # tiny smoke test with synthetic data shaped like the real features
    rng = np.random.default_rng(0)
    N = 200
    X = rng.uniform(0.1, 1.0, size=(N, 5))
    y_cls = (X[:, 2] > 0.15).astype(float)  # pretend asymmetry drives flipping
    y_reg = 5.0 + 10.0 * (1.0 - X[:, 2])

    net = FlipPredictorNet()
    hist = net.train(X, y_cls, y_reg, epochs=500, lr=0.1, verbose=True)
    p, t = net.predict(X[:5])
    print("sample predictions p(flip)=", p, " t_flip=", t)
