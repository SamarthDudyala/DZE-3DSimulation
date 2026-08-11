# Dzhanibekov Effect: 3D Simulation & AI-Powered Analysis

A desktop application that simulates torque-free rigid-body rotation — the
**Dzhanibekov Effect**, also known as the **tennis racket theorem** — with
a real numerical physics engine, a from-scratch machine learning model that
predicts flip behavior, and an interactive 3D visualization of a tumbling
wingnut rendered in VPython, all inside a single Tkinter desktop app.

The project combines computational physics, hand-rolled AI, and 3D
aerospace-style visualization to explore the same rotational instability
that has famously flipped tools, wingnuts, and even a wrench in front of
cosmonaut Vladimir Dzhanibekov aboard the Salyut 7 space station in 1985 —
and which spacecraft attitude-control engineers must actively avoid.

---

## Table of Contents

1. [What This Project Does](#what-this-project-does)
2. [Library Constraints](#library-constraints)
3. [Project Structure](#project-structure)
4. [Architecture & Data Flow](#architecture--data-flow)
5. [File-by-File Reference](#file-by-file-reference)
6. [Installation](#installation)
7. [Running the App](#running-the-app)
8. [Using the Application](#using-the-application)
9. [The Physics, In Depth](#the-physics-in-depth)
10. [The Machine Learning Model, In Depth](#the-machine-learning-model-in-depth)
11. [The 3D Visualization, In Depth](#the-3d-visualization-in-depth)
12. [Troubleshooting](#troubleshooting)
13. [Known Limitations](#known-limitations)
14. [Possible Extensions](#possible-extensions)

---

## What This Project Does

Spin a rigid body — a book, a phone, a wingnut — about its axis with the
**intermediate** moment of inertia (neither the largest nor the smallest),
and instead of a clean, stable spin, it periodically flips end over end.
This is a real consequence of Euler's equations of rotation, first widely
publicized after cosmonaut Vladimir Dzhanibekov observed it with a
tumbling wingnut in microgravity.

This project lets you:

- **Simulate** the effect numerically by integrating the true equations of
  motion for a torque-free rigid body, with live conservation checks
  (energy and angular momentum should stay flat — if they don't, the
  integration is untrustworthy).
- **Visualize** it in interactive 3D as a detailed wingnut model, tumbling
  and drifting inside a translucent reference grid room, with body-fixed
  axis arrows and their traced paths.
- **Predict** flip behavior with a genuinely-from-scratch neural network
  (no scikit-learn, no autograd) trained on real simulation data, rather
  than a lookup table or a canned formula.
- **Experiment** interactively — change the moments of inertia, initial
  spin, and duration, and immediately see how the outcome changes, both
  numerically (plots) and visually (the 3D wingnut).

---

## Library Constraints

This project was deliberately built using **only** the Python standard
library plus a small, fixed set of numerical/GUI packages:

```python
# basic management libraries
import sys
import time

# mathematical calculation libraries
import numpy
import scipy
import matplotlib

# 3D graphics & GUI
import vpython
import tkinter

# data management
import json
```

No `scikit-learn`, no `threading`, no other third-party dependencies. Two
consequences of that constraint shape the codebase in ways worth knowing
up front:

- **The neural network is hand-built.** `ml_model.py` implements its own
  forward pass, backpropagation, and gradient descent using nothing but
  numpy array operations — see
  [The Machine Learning Model, In Depth](#the-machine-learning-model-in-depth).
- **There is no `threading` import anywhere in the project**, including in
  the 3D playback loop, which normally would want a separate thread for
  VPython's render loop running alongside Tkinter's event loop. Instead,
  the GUI drives VPython frame updates via `root.after(...)` scheduling —
  see [The 3D Visualization, In Depth](#the-3d-visualization-in-depth).

---

## Project Structure

```
dzhanibekov_project/
├── main.py              # Entry point — run this
├── gui_app.py            # Tkinter desktop application (the orchestration layer)
├── physics.py             # Core physics engine: Euler's equations + quaternions
├── ml_model.py            # From-scratch numpy neural network
├── data_generator.py      # Physics-driven training data generation
├── visualizer.py           # VPython 3D scene: wingnut, room, arrows, drift
├── requirements.txt        # pip dependency manifest
├── .gitignore               # Standard Python/OS ignores + local config files
├── dataset.json              # Pre-generated ML training data (600 samples)
├── flip_model.json            # Pre-trained neural network weights
└── README.md                    # This file
```

---

## Architecture & Data Flow

The six Python files form a strict dependency chain, which is also the
order in which they were built and the order in which they're described
below:

```
physics.py  (no internal dependencies)
    │
    ├──► data_generator.py   (runs physics.RigidBody simulations to build a dataset)
    │
    ├──► visualizer.py        (uses physics.quat_to_rotmat to orient the 3D wingnut)
    │
ml_model.py  (no internal dependencies — trains on whatever data it's given)
    │
    └──► gui_app.py  (imports physics, ml_model, data_generator, and lazily visualizer)
              │
              └──► main.py  (entry point, calls gui_app.main())
```

**Runtime call flow**, once `main.py` is executed:

```
main.py
  └─ gui_app.main()
       └─ DzhanibekovApp.__init__()
            └─ loads flip_model.json if present (FlipPredictorNet.load)

  "Run Simulation"     → physics.RigidBody(I1,I2,I3).simulate(w0, ...)
                            → matplotlib plots updated in the embedded canvas

  "Open / Update 3D View" → lazy `import visualizer`
                            → visualizer.init_scene()  (built once)

  "Play 3D"              → root.after(20, self._step_3d) loop
                            → visualizer.update_scene(handles, q, w, t)  (every frame)

  "Predict Flip (ML)"    → data_generator.make_features(I1,I2,I3,w0)
                            → FlipPredictorNet.predict(features)

  "Train / Retrain Model" → data_generator.generate_dataset(n_samples=600)
                            → FlipPredictorNet().train(X, y_cls, y_reg)
                            → net.save("flip_model.json")

  "Save / Load Config"   → json.dump / json.load of the input fields
```

`visualizer.py` is imported **lazily**, inside `on_open_3d()`, rather than
at the top of `gui_app.py`. This means the rest of the app (simulation,
plotting, ML) works even in an environment where VPython or its
`pkg_resources` dependency is broken — only the 3D button is affected, and
it fails with an actionable error message rather than crashing the whole
app on startup.

---

## File-by-File Reference

### `physics.py` — Physics Engine

The ground-truth simulation core. Nothing else in the project computes
real rotational dynamics; everything else either consumes this module's
output or (in `visualizer.py`'s case) borrows one helper function from it.

| Function / Class | Purpose |
|---|---|
| `quat_normalize(q)` | Renormalizes a quaternion to unit length (numerical drift correction). |
| `quat_multiply(a, b)` | Hamilton product of two quaternions. |
| `quat_to_rotmat(q)` | Converts a unit quaternion to a 3×3 rotation matrix. Also imported directly by `visualizer.py`. |
| `RigidBody(I1, I2, I3)` | The main class. Constructed with the three principal moments of inertia. |
| `RigidBody.euler_rhs(w)` | Right-hand side of Euler's torque-free rotation equations. |
| `RigidBody.state_rhs(t, y)` | Combined quaternion + angular-velocity derivative, for `solve_ivp`. |
| `RigidBody.kinetic_energy(w)` / `.angular_momentum_body(w)` | Conservation-check diagnostics. |
| `RigidBody.simulate(w0, t_span, n_points, ...)` | Integrates the motion from initial angular velocity `w0`. Returns time series of orientation, angular velocity, energy, and angular momentum magnitude. |
| `RigidBody.detect_flips(t, w, axis, threshold)` (static) | Counts sign changes of a given angular-velocity component — the signature of intermediate-axis flipping. |

Running `python3 physics.py` directly performs a self-test: simulates a
classic asymmetric top spun about its intermediate axis and prints the
energy/momentum drift and detected flip times.

### `ml_model.py` — From-Scratch Neural Network

Defines `FlipPredictorNet`, a small multilayer perceptron with **two
output heads sharing one hidden layer**:

- a **classification head** (sigmoid activation, binary cross-entropy
  loss) predicting whether a given spin configuration will flip, and
- a **regression head** (linear activation, MSE loss) predicting the time
  to the first flip, in seconds.

Every part of the network — Xavier weight initialization, the ReLU hidden
layer, both output heads, the forward pass, the backward pass (manually
derived gradients, no autograd), and full-batch gradient descent — is
implemented directly with numpy array operations in `train()`. Weights
persist to/from JSON via `to_dict()` / `save()` / `load()`.

Running `python3 ml_model.py` directly performs a smoke test on synthetic
data to confirm the training loop converges.

### `data_generator.py` — Training Data Generation

The bridge between physics and machine learning. `generate_dataset()`
runs hundreds of real `physics.RigidBody` simulations across randomized
moments of inertia and initial spins, then labels each run using
`RigidBody.detect_flips()` — there are no synthetic or hand-authored
labels anywhere in this project; the model learns from real integrated
trajectories.

`make_features(I1, I2, I3, w0)` engineers 5 input features per
configuration:

1. `I_spin / I_max` — the spin axis's own moment of inertia, normalized
2. `I_other_min / I_max` — the smaller of the other two moments
3. `I_other_max / I_max` — the larger of the other two moments
4. `|w_perp| / |w_spin|` — perturbation-to-spin ratio
5. `|w_spin|` — absolute spin magnitude

Features 1–3 deliberately encode *which* axis is being spun relative to
the other two (not just an axis-agnostic asymmetry measure), and feature 5
exists because flip *time* in real seconds depends on absolute spin rate,
not just moment-of-inertia ratios — Euler's equations are only
scale-invariant under a *rescaled* time variable. Both of these were bugs
in an earlier version of this feature set (see
[The Machine Learning Model, In Depth](#the-machine-learning-model-in-depth)
for how they were found and fixed).

`save_dataset()` / `load_dataset()` handle JSON persistence.

### `visualizer.py` — 3D Visualization (VPython)

Builds and updates the entire interactive 3D scene. Two public functions:

- **`init_scene(title)`** — builds everything once: the wingnut model, the
  grid room, the start marker, the axis arrows and their extensions and
  trail markers, the angular-velocity arrow, and the motion-path line.
  Returns a dict of handles to every object that needs per-frame updates.
- **`update_scene(handles, q, w, t, axis_len, w_scale)`** — called once per
  animation frame. Given the current orientation quaternion, angular
  velocity, and elapsed simulation time, repositions and reorients every
  object in the scene.

See [The 3D Visualization, In Depth](#the-3d-visualization-in-depth) for
what's actually in the scene and why.

### `gui_app.py` — Tkinter Desktop Application

The orchestration layer and the largest file in the project. Defines
`DzhanibekovApp`, a Tkinter window with:

- **Input fields** for `I1, I2, I3`, initial angular velocity
  `w1_0, w2_0, w3_0`, simulation duration `t_end`, and integration
  resolution `n_points`.
- **Action buttons**, each wired to a handler method (`on_run`,
  `on_open_3d`, `on_play_3d`, `on_pause_3d`, `on_predict`, `on_train`,
  `on_save_config`, `on_load_config`).
- **An embedded matplotlib figure** (`FigureCanvasTkAgg`) showing angular
  velocity components and the energy/momentum conservation check,
  redrawn after every simulation run.
- **A status log** (a read-only `Text` widget) echoing simulation
  results, ML predictions, and training progress.

`main()` at the bottom of this file creates the Tkinter root window and
starts the event loop; this is what `main.py` calls.

### `main.py` — Entry Point

A deliberately trivial file — its only job is to make "how do I run this"
unambiguous:

```bash
python3 main.py
```

### `requirements.txt` — Dependency Manifest

```
numpy
scipy
matplotlib
vpython
setuptools<82
```

The `setuptools<82` pin exists because of a real, dated breaking change —
see [Troubleshooting](#troubleshooting).

### `.gitignore`

Standard Python/OS noise (`__pycache__/`, `*.pyc`, virtual environment
folders, `.DS_Store`) plus `config.json`, since that file is a personal
saved parameter set created via the GUI's "Save Config" button, not part
of the application itself.

### `dataset.json` / `flip_model.json` — Generated Artifacts

`dataset.json` holds 600 physics-simulated training samples (features +
labels); `flip_model.json` holds the resulting trained network weights.
Both are shipped pre-built so "Predict Flip (ML)" works immediately on
first launch, without waiting through a training run. Either can be
regenerated at any time via the GUI's "Train / Retrain Model" button,
which calls `data_generator.generate_dataset()` followed by
`FlipPredictorNet.train()`.

---

## Installation

```bash
git clone <this-repository>
cd dzhanibekov_project

pip install -r requirements.txt
# or manually:
pip install numpy scipy matplotlib vpython "setuptools<82"

# tkinter ships with most Python installs; on Debian/Ubuntu if missing:
sudo apt-get install python3-tk
```

## Running the App

```bash
python3 main.py
```

The Tkinter window opens immediately. The 3D view opens in your **default
web browser** the first time you click "Open / Update 3D View" — this is
how VPython rendering works (it runs a small local server and opens a
browser tab automatically). It won't render in fully headless
environments with no browser available.

---

## Using the Application

1. **Set parameters.** Moments of inertia `I1, I2, I3` and initial
   angular velocity `w1_0, w2_0, w3_0`. Set `I1 < I2 < I3` and make
   `w2_0` (the middle axis) dominant to see flipping; make `w1_0` or
   `w3_0` dominant instead to see stable spin.
2. **Run Simulation** — integrates the equations of motion and plots
   angular velocity components plus the energy/momentum conservation
   check (both should stay flat at 1.0 if the integration is trustworthy).
3. **Open / Update 3D View**, then **Play 3D** — watch the wingnut tumble
   in your browser, with adjustable playback speed via the slider.
4. **Predict Flip (ML)** — get the trained model's probability estimate
   and flip-time estimate for the current parameters instantly, without
   running the full ODE integration.
5. **Train / Retrain Model** — regenerate the training dataset from real
   simulations and retrain from scratch (this runs ~600 physics
   simulations, so it takes a little while — the status log reports
   progress).
6. **Save Config / Load Config** — store the current parameter set as
   JSON for later reuse.

---

## The Physics, In Depth

For a torque-free rigid body, Euler's rotation equations in the body
frame are:

```
I1 · dω1/dt = (I2 − I3) · ω2 · ω3
I2 · dω2/dt = (I3 − I1) · ω3 · ω1
I3 · dω3/dt = (I1 − I2) · ω1 · ω2
```

Spin purely about the largest or smallest principal axis is **stable**;
spin about the **intermediate** axis is **unstable** and produces the
periodic 180° flips characteristic of the Dzhanibekov Effect. This is a
real, well-established result in classical mechanics (the **intermediate
axis theorem**), not a simulation artifact.

Orientation is tracked with a unit quaternion rather than Euler angles (to
avoid gimbal lock), integrated alongside the angular velocity via
`scipy.integrate.solve_ivp` (adaptive-step RK45). Two physical
conservation laws — kinetic energy and the magnitude of angular momentum —
are logged at every timestep as sanity checks; for a correct
torque-free simulation, both should remain essentially flat (drift on the
order of `1e-8` or smaller is typical numerical integration error, not a
physics bug).

Flips are detected by counting sign changes in the angular velocity
component along the initial spin axis — a direct, model-free signature of
the tumbling behavior.

## The Machine Learning Model, In Depth

Since scikit-learn is outside the allowed library set, `ml_model.py`
implements a two-headed multilayer perceptron entirely by hand: Xavier
weight initialization, a ReLU hidden layer, a sigmoid classification head,
a linear regression head, and full-batch gradient descent using manually
derived backpropagation gradients — no autograd framework of any kind.

Training data comes from `data_generator.py`, which genuinely *runs* the
physics simulator across ~600 randomized configurations rather than using
synthetic or hand-labeled data — the model is learning the real
intermediate-axis instability from real integrated trajectories.

**Two real bugs were found and fixed during development, both worth
knowing about since they reflect genuine physics subtleties:**

1. **Missing axis identity.** The first feature set encoded body asymmetry
   in an axis-agnostic way, without recording *which* axis was actually
   being spun relative to the other two. Since flip behavior fundamentally
   depends on whether the spin axis is the intermediate one, this capped
   classification accuracy at ~67%. Re-engineering the features to encode
   the spin axis's moment of inertia relative to the other two
   (`I_spin`, `I_other_min`, `I_other_max`) raised accuracy to ~99%.
2. **Missing spin magnitude.** Flip *time* (in real seconds) depends on
   the absolute spin rate, not just moment-of-inertia ratios — Euler's
   equations are only scale-invariant under a *rescaled* time variable.
   Adding `|w_spin|` as an explicit feature dropped the flip-time mean
   absolute error from ~1.9s to ~1.0s on held-out data.

The trained model's weights persist to `flip_model.json`; the app loads
this automatically on startup so predictions work immediately.

## The 3D Visualization, In Depth

The tumbling body is modeled as a **detailed wingnut** — a hex nut
threaded onto a bolt shaft with visible thread ridges, and two tapered
wing ears for finger-tightening — matching the actual object Vladimir
Dzhanibekov used in the original 1985 demonstration aboard Salyut 7. It's
built from VPython primitives (`cylinder`, `ring`, `extrusion`, `box`,
`sphere`) merged into a single `vp.compound`, so the whole assembly can be
reoriented every frame with one `.axis` / `.up` assignment.

The scene also includes:

- **Translucent axis extensions** — each body-fixed axis arrow
  (red/green/blue) continues past its tip as a fainter, lower-opacity
  cylinder, indicating these are reference directions rather than bounded
  segments.
- **Trailing "ray" traces** — a small marker rides at each axis arrow's
  tip with VPython's native trail feature enabled, leaving a fading streak
  behind it as the wingnut tumbles, so the path each axis has recently
  traced can be followed visually.
- **A translucent grid room with a marked start point** — the wingnut
  moves inside a 12-unit translucent cube (floor, ceiling, four walls),
  each face etched with a grid (every 3rd line drawn brighter as a major
  gridline). A yellow ring crosshair, floor marker, and "START" label mark
  the world origin where the drift begins, giving a fixed frame of
  reference against which motion is actually visible.
- **A slow linear drift + grey motion-path line** — the wingnut drifts at
  a small constant velocity. This is a **purely illustrative kinematic
  overlay** — the physics engine itself only integrates *orientation*, not
  translation, so this drift is not derived from any force or momentum
  calculation. A translucent grey line grows from the starting point to
  the wingnut's current position every frame, and the camera automatically
  follows the object (`scene.camera.follow(...)`) so it stays framed as it
  drifts through the grid room.

**A note on concurrency.** The project's library constraints exclude
`threading`, which normally would run VPython's render loop alongside
Tkinter's event loop. Instead, the GUI drives frame timing entirely
through Tkinter's own scheduler: `root.after(20, self._step_3d)` calls
`visualizer.update_scene(...)` once per tick, which just reassigns a
handful of VPython object attributes (`.pos`, `.axis`, `.up`). VPython's
own internal server (started by the `vpython` package itself, not by this
project's code) handles pushing those attribute changes to the browser tab
asynchronously — no blocking `rate()` call and no separate thread needed.

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'pkg_resources'`

VPython's internals still import `pkg_resources`, which used to ship
inside `setuptools`. **As of setuptools 82.0.0 (February 2026),
`pkg_resources` was removed from setuptools entirely** — so simply running
`pip install setuptools` isn't enough, since a fresh install grabs the
latest version, which no longer includes it. You need an **older**
setuptools that still ships it:

```bash
pip install "setuptools<82"
```

then restart the app. `requirements.txt` already pins this, so a fresh
`pip install -r requirements.txt` avoids the issue entirely.

If you have multiple Pythons installed, make sure you're installing into
the same interpreter that runs `main.py` — the app's error dialog prints
the exact interpreter path (`sys.executable`) to use for the install
command.

### The 3D view doesn't render

VPython needs a real web browser to connect to its local server. It will
not render in fully headless environments (e.g., remote servers with no
browser installed / no display).

### Tkinter isn't found

`tkinter` ships with most Python distributions but is sometimes packaged
separately on Linux:

```bash
sudo apt-get install python3-tk
```

---

## Known Limitations

- **The flip-time regression is a coarse estimate** (mean absolute error
  around ~1 second on held-out data in testing). Physically, the exact
  flip period involves elliptic integrals of the body's energy/momentum
  ratio; this small hand-rolled network is a lightweight learned
  approximation, not a substitute for the closed-form solution.
- **VPython requires a real browser.** There is no headless/offscreen
  rendering fallback.
- **The linear drift is purely visual.** It is not physically simulated —
  the physics engine only integrates rotational motion, so translation is
  a fixed-velocity kinematic overlay added at the visualization layer.

## Possible Extensions

- Replace the constant-velocity drift with real translational dynamics
  (trivial in the absence of external forces, but would allow adding
  thruster impulses later).
- Add a closed-form (elliptic integral) flip-period calculation alongside
  the ML estimate, to quantify exactly how much accuracy the learned model
  trades away for its speed.
- Export simulation runs (angular velocity / orientation time series) to
  CSV or a plot image directly from the GUI.
- Add unit tests around `physics.py`'s conservation checks and
  `ml_model.py`'s training convergence, formalizing the self-tests each
  file already runs under `if __name__ == "__main__":`.


<sub>Note: AI assistance was utilized solely for editing, structural refinement, and fact-checking of this document.