"""
Tkinter desktop application for the Dzhanibekov Effect simulator.

Layout:
  - Left panel:  inputs (moments of inertia, initial angular velocity,
                 simulation duration), and action buttons.
  - Right panel: embedded matplotlib plots (angular velocity components,
                 energy/momentum conservation check).
  - "Open 3D View" launches / updates a VPython browser-based scene and
    plays the trajectory back, driven by Tkinter's `after()` scheduler
    (see visualizer.py for why no threading is required).

Only tkinter, matplotlib, numpy, json, sys, time (+ our own modules,
which themselves only use numpy/scipy/vpython/json) are used here.
"""

import json
import os
import sys
import time
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from physics import RigidBody
from data_generator import make_features, generate_dataset, save_dataset, load_dataset
from ml_model import FlipPredictorNet

MODEL_PATH = os.path.join(os.path.dirname(__file__), "flip_model.json")
DATASET_PATH = os.path.join(os.path.dirname(__file__), "dataset.json")
CONFIG_PATH_DEFAULT = os.path.join(os.path.dirname(__file__), "config.json")


class DzhanibekovApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Dzhanibekov Effect - Spacecraft Attitude Simulator")
        self.root.geometry("1180x680")

        self.result = None          # last simulate() output
        self.body = None            # last RigidBody
        self.net = None             # loaded/trained FlipPredictorNet
        self.vp_handles = None      # visualizer scene handles (lazy-created)
        self.playing_3d = False
        self.play_index = 0
        self.play_speed = tk.DoubleVar(value=1.0)

        self._build_layout()
        self._log(f"Running under: {sys.executable}")
        self._try_load_model_silently()

    #UI

    def _build_layout(self):
        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill="both", expand=True)

        left = ttk.Frame(outer)
        left.pack(side="left", fill="y", padx=(0, 10))

        right = ttk.Frame(outer)
        right.pack(side="left", fill="both", expand=True)

        #inputs
        params = ttk.LabelFrame(left, text="Rigid Body Parameters", padding=10)
        params.pack(fill="x", pady=(0, 10))

        self.vars = {}
        rows = [
            ("I1", "0.5"), ("I2", "1.0"), ("I3", "1.5"),
            ("w1_0", "0.05"), ("w2_0", "5.0"), ("w3_0", "0.05"),
            ("t_end", "20.0"), ("n_points", "3000"),
        ]
        for i, (key, default) in enumerate(rows):
            ttk.Label(params, text=key).grid(row=i, column=0, sticky="w", pady=2)
            var = tk.StringVar(value=default)
            entry = ttk.Entry(params, textvariable=var, width=12)
            entry.grid(row=i, column=1, pady=2, padx=(6, 0))
            self.vars[key] = var

        hint = ttk.Label(
            params,
            text="Tip: set I1<I2<I3 and spin mainly about\nthe MIDDLE axis (w2_0 large) to see flips.",
            foreground="#555", justify="left",
        )
        hint.grid(row=len(rows), column=0, columnspan=2, sticky="w", pady=(8, 0))

        #actions
        actions = ttk.LabelFrame(left, text="Actions", padding=10)
        actions.pack(fill="x", pady=(0, 10))

        ttk.Button(actions, text="Run Simulation", command=self.on_run
                   ).pack(fill="x", pady=3)
        ttk.Button(actions, text="Open / Update 3D View", command=self.on_open_3d
                   ).pack(fill="x", pady=3)

        play_row = ttk.Frame(actions)
        play_row.pack(fill="x", pady=3)
        ttk.Button(play_row, text="Play 3D", command=self.on_play_3d).pack(side="left")
        ttk.Button(play_row, text="Pause", command=self.on_pause_3d).pack(side="left", padx=4)
        ttk.Scale(play_row, from_=0.1, to=4.0, variable=self.play_speed,
                  orient="horizontal").pack(side="left", fill="x", expand=True, padx=6)

        ttk.Separator(actions).pack(fill="x", pady=6)

        ttk.Button(actions, text="Predict Flip (ML)", command=self.on_predict
                   ).pack(fill="x", pady=3)
        ttk.Button(actions, text="Train / Retrain Model", command=self.on_train
                   ).pack(fill="x", pady=3)

        ttk.Separator(actions).pack(fill="x", pady=6)

        io_row = ttk.Frame(actions)
        io_row.pack(fill="x", pady=3)
        ttk.Button(io_row, text="Save Config", command=self.on_save_config).pack(
            side="left", expand=True, fill="x", padx=(0, 3))
        ttk.Button(io_row, text="Load Config", command=self.on_load_config).pack(
            side="left", expand=True, fill="x", padx=(3, 0))

        #status / output
        status_frame = ttk.LabelFrame(left, text="Status", padding=10)
        status_frame.pack(fill="both", expand=True)
        self.status_text = tk.Text(status_frame, width=40, height=16, wrap="word",
                                    state="disabled", background="#f6f6f6")
        self.status_text.pack(fill="both", expand=True)

        #plots
        self.fig = Figure(figsize=(7.2, 6.2), dpi=100)
        self.ax_w = self.fig.add_subplot(211)
        self.ax_cons = self.fig.add_subplot(212)
        self.fig.tight_layout(pad=3.0)

        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self._draw_empty_plots()

    def _log(self, msg):
        self.status_text.configure(state="normal")
        self.status_text.insert("end", msg + "\n")
        self.status_text.see("end")
        self.status_text.configure(state="disabled")

    def _draw_empty_plots(self):
        self.ax_w.clear()
        self.ax_w.set_title("Body angular velocity components")
        self.ax_w.set_xlabel("time (s)")
        self.ax_w.set_ylabel("rad/s")
        self.ax_cons.clear()
        self.ax_cons.set_title("Conservation check (energy & |L|)")
        self.ax_cons.set_xlabel("time (s)")
        self.canvas.draw()

    #helpers

    def _read_params(self):
        v = {k: float(var.get()) for k, var in self.vars.items() if k != "n_points"}
        v["n_points"] = int(float(self.vars["n_points"].get()))
        return v

    def _try_load_model_silently(self):
        if os.path.exists(MODEL_PATH):
            try:
                self.net = FlipPredictorNet.load(MODEL_PATH)
                self._log(f"Loaded existing ML model from {os.path.basename(MODEL_PATH)}")
            except Exception:
                pass

    #actions

    def on_run(self):
        try:
            p = self._read_params()
        except ValueError:
            messagebox.showerror("Input error", "All fields must be numeric.")
            return

        try:
            self.body = RigidBody(p["I1"], p["I2"], p["I3"])
        except ValueError as e:
            messagebox.showerror("Invalid moments of inertia", str(e))
            return

        w0 = [p["w1_0"], p["w2_0"], p["w3_0"]]
        t0 = time.time()
        self.result = self.body.simulate(
            w0, t_span=(0.0, p["t_end"]), n_points=p["n_points"]
        )
        dt = time.time() - t0

        t = self.result["t"]
        w = self.result["w"]
        spin_axis = int(np.argmax(np.abs(w0)))
        n_flips, flip_times = self.body.detect_flips(t, w, axis=spin_axis)

        energy = self.result["energy"]
        L_mag = self.result["L_mag"]
        e_drift = energy.max() - energy.min()
        l_drift = L_mag.max() - L_mag.min()

        self._log("--- Simulation complete ---")
        self._log(f"Integrated in {dt:.3f}s over {p['t_end']:.1f}s, "
                   f"{p['n_points']} points")
        self._log(f"Energy drift: {e_drift:.2e}  |L| drift: {l_drift:.2e}")
        self._log(f"Flips detected about spin axis {spin_axis}: {n_flips}")
        if n_flips:
            self._log(f"First flip at t = {flip_times[0]:.2f}s")

        self._update_plots(t, w, energy, L_mag)
        self.play_index = 0

    def _update_plots(self, t, w, energy, L_mag):
        self.ax_w.clear()
        self.ax_w.plot(t, w[:, 0], label="w1", color="tab:red")
        self.ax_w.plot(t, w[:, 1], label="w2", color="tab:green")
        self.ax_w.plot(t, w[:, 2], label="w3", color="tab:blue")
        self.ax_w.set_title("Body angular velocity components")
        self.ax_w.set_xlabel("time (s)")
        self.ax_w.set_ylabel("rad/s")
        self.ax_w.legend(loc="upper right", fontsize=8)

        self.ax_cons.clear()
        e_norm = energy / (energy[0] if energy[0] != 0 else 1.0)
        l_norm = L_mag / (L_mag[0] if L_mag[0] != 0 else 1.0)
        self.ax_cons.plot(t, e_norm, label="Energy / E0", color="tab:purple")
        self.ax_cons.plot(t, l_norm, label="|L| / |L0|", color="tab:orange", linestyle="--")
        self.ax_cons.set_title("Conservation check (should stay flat at 1.0)")
        self.ax_cons.set_xlabel("time (s)")
        self.ax_cons.legend(loc="upper right", fontsize=8)

        self.fig.tight_layout(pad=3.0)
        self.canvas.draw()

    def on_open_3d(self):
        if self.result is None:
            messagebox.showinfo("Run simulation first", "Click 'Run Simulation' first.")
            return
        try:
            import visualizer
        except ModuleNotFoundError as e:
            if "pkg_resources" in str(e):
                messagebox.showerror(
                    "VPython error",
                    "VPython needs 'pkg_resources', which recent versions of "
                    "setuptools (82.0.0+, Feb 2026) removed completely. "
                    "Installing *any* setuptools version isn't enough -- it "
                    "has to be an older one that still ships pkg_resources.\n\n"
                    "Fix: run this, into the exact Python running this app "
                    f"({sys.executable}), then restart:\n\n"
                    f'    "{sys.executable}" -m pip install "setuptools<82"\n'
                )
            else:
                messagebox.showerror("VPython error",
                                      f"Missing dependency: {e}\n\n"
                                      f'Try: "{sys.executable}" -m pip install vpython "setuptools<82"')
            return
        except Exception as e:
            messagebox.showerror("VPython error",
                                  f"Could not initialize VPython scene:\n{e}")
            return
        self._visualizer = visualizer
        self.vp_handles = visualizer.init_scene()
        # snap to the current playback frame immediately
        q0 = self.result["q"][self.play_index]
        w0 = self.result["w"][self.play_index]
        t0 = self.result["t"][self.play_index]
        visualizer.update_scene(self.vp_handles, q0, w0, t=t0)
        self._log("3D view opened (check your browser tab / window).")

    def on_play_3d(self):
        if self.result is None:
            messagebox.showinfo("Run simulation first", "Click 'Run Simulation' first.")
            return
        if self.vp_handles is None:
            self.on_open_3d()
        if self.playing_3d:
            return
        self.playing_3d = True
        self._step_3d()

    def on_pause_3d(self):
        self.playing_3d = False

    def _step_3d(self):
        if not self.playing_3d or self.result is None:
            return
        t = self.result["t"]
        q_arr = self.result["q"]
        w_arr = self.result["w"]
        n = len(t)

        if self.play_index >= n:
            self.play_index = 0

        self._visualizer.update_scene(
            self.vp_handles, q_arr[self.play_index], w_arr[self.play_index],
            t=t[self.play_index]
        )
        self.play_index += max(1, int(self.play_speed.get()))

        # ~50 fps playback scheduling; the physical time step covered per
        # frame is controlled by play_speed via play_index stride above
        self.root.after(20, self._step_3d)

    def on_predict(self):
        if self.net is None:
            if not os.path.exists(MODEL_PATH):
                messagebox.showinfo(
                    "No trained model",
                    "No trained model found yet. Click 'Train / Retrain Model' first."
                )
                return
            self.net = FlipPredictorNet.load(MODEL_PATH)

        try:
            p = self._read_params()
        except ValueError:
            messagebox.showerror("Input error", "All fields must be numeric.")
            return

        w0 = np.array([p["w1_0"], p["w2_0"], p["w3_0"]])
        feat = make_features(p["I1"], p["I2"], p["I3"], w0)
        p_flip, t_flip = self.net.predict(feat)

        self._log("--- ML prediction ---")
        self._log(f"P(intermediate-axis flip) = {p_flip[0]:.1%}")
        if p_flip[0] > 0.5:
            self._log(f"Estimated time to first flip ~= {t_flip[0]:.2f}s")
        else:
            self._log("Model predicts this spin configuration is stable.")

    def on_train(self):
        answer = messagebox.askyesno(
            "Train model",
            "This will run ~400 physics simulations to build a fresh training "
            "set and then train the neural network. It may take a little while. Continue?"
        )
        if not answer:
            return

        self._log("--- Training ML model ---")
        self._log("Generating dataset from physics simulations...")
        self.root.update_idletasks()

        X, y_cls, y_reg = generate_dataset(n_samples=600)
        save_dataset(DATASET_PATH, X, y_cls, y_reg)
        self._log(f"Dataset ready: {len(X)} samples, "
                  f"flip rate {y_cls.mean():.1%}")
        self.root.update_idletasks()

        net = FlipPredictorNet(n_in=X.shape[1], n_hidden=32, seed=7)
        hist = net.train(X, y_cls, y_reg, epochs=4000, lr=0.2,
                          reg_loss_weight=2.0, verbose=False)
        net.save(MODEL_PATH)
        self.net = net

        self._log(f"Training complete. Final loss = {hist['loss'][-1]:.4f}")
        self._log("Model saved to flip_model.json")

    def on_save_config(self):
        try:
            p = self._read_params()
        except ValueError:
            messagebox.showerror("Input error", "All fields must be numeric.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".json", initialfile="config.json",
            filetypes=[("JSON", "*.json")]
        )
        if not path:
            return
        with open(path, "w") as f:
            json.dump(p, f, indent=2)
        self._log(f"Config saved to {path}")

    def on_load_config(self):
        path = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if not path:
            return
        with open(path, "r") as f:
            p = json.load(f)
        for k, v in p.items():
            if k in self.vars:
                self.vars[k].set(str(v))
        self._log(f"Config loaded from {path}")


def main():
    root = tk.Tk()
    app = DzhanibekovApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
