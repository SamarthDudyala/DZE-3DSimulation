"""
Runs many torque-free rigid-body simulations across randomized moments of
inertia and initial spins, extracts (features, flip_label, flip_time) triples,
and saves the dataset to JSON so it can be reused by ml_model.py without
re-simulating.
"""

import json
import numpy as np

from physics import RigidBody


def make_features(I1, I2, I3, w0):
    """
    Build the 4 input features. Crucially, these must encode WHICH axis is
    being spun about relative to the other two -- the intermediate-axis
    theorem cares about the rank of the spin axis's moment of inertia among
    all three, not just the overall asymmetry of the body.
    """
    I_list = np.array([I1, I2, I3], dtype=float)
    I_max_all = I_list.max()
    spin_axis = int(np.argmax(np.abs(w0)))

    I_spin = I_list[spin_axis]
    I_others = np.delete(I_list, spin_axis)
    I_other_min, I_other_max = np.sort(I_others)

    x0 = I_spin / I_max_all
    x1 = I_other_min / I_max_all
    x2 = I_other_max / I_max_all
    # x0 between x1 and x2 => spinning about the intermediate axis => unstable

    w_spin = w0[spin_axis]
    w_perp = np.linalg.norm(np.delete(w0, spin_axis))
    x3 = w_perp / (abs(w_spin) + 1e-9)
    # spin magnitude matters a lot for flip TIME (in seconds): Euler's
    # equations only become scale-invariant under a rescaled time variable,
    # so two bodies with identical moment ratios but different absolute
    # spin rates will flip at very different real-world times.
    x4 = abs(w_spin)
    return np.array([x0, x1, x2, x3, x4])


def generate_dataset(n_samples=600, t_span=(0.0, 30.0), n_points=3000, seed=0):
    rng = np.random.default_rng(seed)
    X, y_cls, y_reg = [], [], []

    for _ in range(n_samples):
        # random, distinct principal moments of inertia
        vals = rng.uniform(0.5, 3.0, size=3)
        while np.ptp(vals) < 0.15:  # avoid near-degenerate (spherical) bodies
            vals = rng.uniform(0.5, 3.0, size=3)
        I1, I2, I3 = np.sort(vals)  # I1 < I2 < I3

        # spin about the intermediate axis (index 1) with a small perturbation,
        # OR about a stable axis (0 or 2), chosen randomly, to give the model
        # both flipping and non-flipping examples
        spin_axis = rng.choice([0, 1, 2], p=[0.35, 0.3, 0.35])
        spin_mag = rng.uniform(2.0, 6.0)
        perturb = rng.uniform(0.01, 0.3) * spin_mag

        w0 = np.zeros(3)
        w0[spin_axis] = spin_mag
        other_axes = [a for a in range(3) if a != spin_axis]
        w0[other_axes[0]] += rng.uniform(-1, 1) * perturb
        w0[other_axes[1]] += rng.uniform(-1, 1) * perturb

        body = RigidBody(I1, I2, I3)
        result = body.simulate(w0, t_span=t_span, n_points=n_points)
        n_flips, flip_times = body.detect_flips(result["t"], result["w"], axis=spin_axis)

        flips = n_flips > 0
        t_first = float(flip_times[0]) if flips else float(t_span[1])

        X.append(make_features(I1, I2, I3, w0))
        y_cls.append(1.0 if flips else 0.0)
        y_reg.append(t_first)

    return np.array(X), np.array(y_cls), np.array(y_reg)


def save_dataset(path, X, y_cls, y_reg):
    data = {"X": X.tolist(), "y_cls": y_cls.tolist(), "y_reg": y_reg.tolist()}
    with open(path, "w") as f:
        json.dump(data, f)


def load_dataset(path):
    with open(path, "r") as f:
        data = json.load(f)
    return np.array(data["X"]), np.array(data["y_cls"]), np.array(data["y_reg"])


if __name__ == "__main__":
    print("Generating dataset (this runs real physics simulations)...")
    X, y_cls, y_reg = generate_dataset(n_samples=400)
    save_dataset("dataset.json", X, y_cls, y_reg)
    print(f"Saved {len(X)} samples to dataset.json")
    print(f"Flip rate: {y_cls.mean():.2%}")
