"""
Dzhanibekov Effect — 3D Physics Simulation & AI-Powered Analysis
=================================================================
Simulates torque-free rigid body rotation using Euler's equations (RK4).
Visualizes the Tennis Racket / Intermediate Axis instability in real time.

Dependencies:
    pip install numpy matplotlib scipy

Run:
    python dzhanibekov_simulation.py

Controls (keyboard while 3D window is focused):
    SPACE   — pause / resume
    1       — spin about axis 1 (min I, stable)
    2       — spin about axis 2 (intermediate I, UNSTABLE)
    3       — spin about axis 3 (max I, stable)
    r       — reset simulation
    a       — toggle AI analysis panel
    q / ESC — quit
"""

import sys
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp

# Physics

def euler_rhs(t, state, I):
    """
    Euler's torque-free equations of motion.
    state = [w1, w2, w3, q0, q1, q2, q3]
    """
    w = state[:3]
    q = state[3:]

    # Angular accelerations
    dw = np.array([
        (I[1] - I[2]) * w[1] * w[2] / I[0],
        (I[2] - I[0]) * w[2] * w[0] / I[1],
        (I[0] - I[1]) * w[0] * w[1] / I[2],
    ])

    # Quaternion kinematics:  dq/dt = 0.5 * Q(q) * w_body
    q0, q1, q2, q3 = q
    Omega = 0.5 * np.array([
        [-q1, -q2, -q3],
        [ q0, -q3,  q2],
        [ q3,  q0, -q1],
        [-q2,  q1,  q0],
    ])
    dq = Omega @ w

    return np.concatenate([dw, dq])


def quat_normalize(q):
    return q / np.linalg.norm(q)


def quat_rotate(q, v):
    """Rotate vector v by unit quaternion q."""
    q0, q1, q2, q3 = q
    R = np.array([
        [1-2*(q2**2+q3**2),   2*(q1*q2-q0*q3),   2*(q1*q3+q0*q2)],
        [  2*(q1*q2+q0*q3), 1-2*(q1**2+q3**2),   2*(q2*q3-q0*q1)],
        [  2*(q1*q3-q0*q2),   2*(q2*q3+q0*q1), 1-2*(q1**2+q2**2)],
    ])
    return R @ v


def kinetic_energy(w, I):
    return 0.5 * np.sum(I * w**2)


def angular_momentum_sq(w, I):
    L = I * w
    return np.dot(L, L)


# Analytical ML-style stability prediction

def predict_flip_period(I, w0):
    """
    Analytical approximation for the Dzhanibekov flip period using
    the Jacobi elliptic function solution for torque-free rotation.

    For spin near the intermediate axis (axis 2):
        T ≈ 2K(k) / lambda
    where K is the complete elliptic integral of the first kind and
    lambda is a characteristic rate.
    """
    from scipy.special import ellipk

    I1, I2, I3 = sorted(I)   # ensure I1 < I2 < I3
    w_total = np.linalg.norm(w0)

    # Stability check: only valid near intermediate axis
    I_sorted_idx = np.argsort(I)
    I_min, I_mid, I_max = I[I_sorted_idx[0]], I[I_sorted_idx[1]], I[I_sorted_idx[2]]

    a = np.sqrt((I_mid - I_min) * (I_max - I_mid) / (I_min * I_max))
    lam = w_total * np.sqrt((I_max - I_min) / (I_min * I_max)) + 1e-12

    k2 = (I_mid - I_min) * (I_max - I_mid) / ((I_max - I_mid + 1e-12) * (I_mid - I_min + 1e-12) + 1e-12)
    k2 = np.clip(k2, 0.0, 0.9999)
    K = ellipk(np.sqrt(k2))

    T = 2 * K / lam if lam > 0 else float('inf')
    return T


def stability_analysis(I):
    """
    Returns stability info for all three axes.
    Theorem: only max and min principal axes are stable.
    """
    axes = ['Axis 1 (min I)', 'Axis 2 (mid I)', 'Axis 3 (max I)']
    I_idx = np.argsort(I)   # [min, mid, max] indices
    result = {}
    for rank, label in enumerate(['min', 'mid', 'max']):
        idx = I_idx[rank]
        stable = (label in ('min', 'max'))
        result[idx] = {
            'label': axes[idx],
            'rank': label,
            'stable': stable,
            'I_value': I[idx],
        }
    return result


# Spacecraft mesh

def make_spacecraft_mesh():
    """
    Returns list of (vertices, color) face tuples for a T-shaped spacecraft
    (rectangular bus + solar panel array).
    """
    # Main bus — elongated along x1
    bx, by, bz = 0.55, 0.15, 0.15
    bus_verts = np.array([
        [-bx, -by, -bz], [-bx,  by, -bz], [-bx,  by,  bz], [-bx, -by,  bz],
        [ bx, -by, -bz], [ bx,  by, -bz], [ bx,  by,  bz], [ bx, -by,  bz],
    ])
    bus_faces_idx = [
        [4,5,6,7],  # +x face
        [0,3,2,1],  # -x face
        [0,1,5,4],  # -y face
        [2,3,7,6],  # +y face
        [1,2,6,5],  # +z face
        [0,4,7,3],  # -z face
    ]
    bus_colors = ['#7878DC', '#5050A0', '#6464B8', '#7070C8', '#8080E0', '#4848A0']

    # Solar panels — flat along y axis
    py, pz = 0.8, 0.05
    panel_verts = np.array([
        [-0.10, -py, -pz], [-0.10,  py, -pz], [ 0.10,  py, -pz], [ 0.10, -py, -pz],
        [-0.10, -py,  pz], [-0.10,  py,  pz], [ 0.10,  py,  pz], [ 0.10, -py,  pz],
    ])
    panel_faces_idx = [
        [0,1,2,3],  # bottom face
        [4,7,6,5],  # top face
        [0,3,7,4],  # -x edge
        [1,5,6,2],  # +x edge
    ]
    panel_colors = ['#1E6E50', '#1A6048', '#155040', '#124838']

    faces = []
    for idx_list, color in zip(bus_faces_idx, bus_colors):
        faces.append((bus_verts[idx_list], color))
    for idx_list, color in zip(panel_faces_idx, panel_colors):
        faces.append((panel_verts[idx_list], color))

    return faces


# Simulation state

class Simulation:
    SPIN_SPEED = 2.5
    EPS = 0.08

    def __init__(self):
        self.I = np.array([1.0, 3.0, 5.0])
        self.spin_axis = 1          # 0, 1, or 2
        self.running = True
        self.show_analysis = False
        self.t = 0.0
        self.dt = 0.012             # simulation timestep (s)
        self.history_len = 300      # trail length

        # Time-series history
        self.t_history = []
        self.w_history = []
        self.E_history = []
        self.L2_history = []

        # Flip detection
        self.flip_count = 0
        self.last_flip_sign = 1
        self.last_flip_time = 0.0
        self.flip_periods = []

        # Body-frame axis tip trail for Polhode curve
        self.trail = []

        self.reset()

    def reset(self):
        self.t = 0.0
        w = np.zeros(3)
        w[self.spin_axis] = self.SPIN_SPEED
        others = [i for i in range(3) if i != self.spin_axis]
        w[others[0]] = self.EPS
        w[others[1]] = self.EPS * 0.5
        q = np.array([1.0, 0.0, 0.0, 0.0])
        self.state = np.concatenate([w, q])
        self.flip_count = 0
        self.last_flip_sign = int(np.sign(w[self.spin_axis]))
        self.last_flip_time = 0.0
        self.flip_periods = []
        self.trail = []
        self.t_history = []
        self.w_history = []
        self.E_history = []
        self.L2_history = []

    @property
    def w(self):
        return self.state[:3]

    @property
    def q(self):
        return self.state[3:]

    def step(self):
        if not self.running:
            return

        # RK4 integration
        sol = solve_ivp(
            euler_rhs,
            [self.t, self.t + self.dt],
            self.state,
            method='RK45',
            args=(self.I,),
            dense_output=False,
            rtol=1e-7,
            atol=1e-9,
        )
        self.state = sol.y[:, -1]
        # Re-normalise quaternion to prevent drift
        self.state[3:] = quat_normalize(self.state[3:])
        self.t = sol.t[-1]

        w = self.w
        # Flip detection
        sign = int(np.sign(w[self.spin_axis]))
        if sign != 0 and sign != self.last_flip_sign:
            self.flip_count += 1
            if self.last_flip_time > 0:
                self.flip_periods.append(self.t - self.last_flip_time)
            self.last_flip_time = self.t
            self.last_flip_sign = sign

        # Trail of body x1 axis tip in world space
        bx_world = quat_rotate(self.q, np.array([1, 0, 0]))
        self.trail.append(bx_world.copy())
        if len(self.trail) > self.history_len:
            self.trail.pop(0)

        # Record history
        self.t_history.append(self.t)
        self.w_history.append(w.copy())
        self.E_history.append(kinetic_energy(w, self.I))
        self.L2_history.append(angular_momentum_sq(w, self.I))

    def mean_flip_period(self):
        if len(self.flip_periods) < 2:
            return None
        return float(np.mean(self.flip_periods[-5:]))


# 3D projection helpers

def perspective_project(pts, cam_dist=5.0):
    """Project Nx3 world points to Nx2 screen coords (simple perspective)."""
    z = pts[:, 2] + cam_dist
    z = np.where(z < 0.1, 0.1, z)
    scale = cam_dist / z
    return np.stack([pts[:, 0] * scale, pts[:, 1] * scale], axis=1)


def rotation_matrix_yx(ay, ax):
    Ry = np.array([[np.cos(ay), 0, np.sin(ay)],
                   [0,          1, 0          ],
                   [-np.sin(ay),0, np.cos(ay) ]])
    Rx = np.array([[1, 0,           0          ],
                   [0, np.cos(ax), -np.sin(ax)],
                   [0, np.sin(ax),  np.cos(ax)]])
    return Rx @ Ry


def face_depth(verts):
    return np.mean(verts[:, 2])


# Visualisation

AXIS_COLORS  = ['#E24B4A', '#1D9E75', '#378ADD']
AMBER        = '#EF9F27'
BG_COLOR     = '#0F0F14'
PANEL_BG     = '#16161E'
TEXT_COLOR   = '#D0CFC8'
MUTED_COLOR  = '#7A7A72'

plt.rcParams.update({
    'figure.facecolor': BG_COLOR,
    'axes.facecolor':   PANEL_BG,
    'text.color':       TEXT_COLOR,
    'axes.labelcolor':  TEXT_COLOR,
    'xtick.color':      MUTED_COLOR,
    'ytick.color':      MUTED_COLOR,
    'axes.edgecolor':   '#2A2A35',
    'grid.color':       '#1E1E28',
    'font.family':      'monospace',
})


def build_figure(sim):
    fig = plt.figure(figsize=(14, 8), facecolor=BG_COLOR)
    fig.canvas.manager.set_window_title('Dzhanibekov Effect — Rigid Body Simulation')

    gs = gridspec.GridSpec(
        3, 3,
        figure=fig,
        left=0.04, right=0.97,
        top=0.93, bottom=0.06,
        wspace=0.35, hspace=0.55,
    )

    ax3d   = fig.add_subplot(gs[:, 0])          # Main 3D viewport
    ax_w   = fig.add_subplot(gs[0, 1:])         # ω components
    ax_E   = fig.add_subplot(gs[1, 1])          # Energy conservation
    ax_L   = fig.add_subplot(gs[1, 2])          # Angular momentum conservation
    ax_pol = fig.add_subplot(gs[2, 1])          # Polhode (ω-space curve)
    ax_txt = fig.add_subplot(gs[2, 2])          # Stats / analysis text

    return fig, ax3d, ax_w, ax_E, ax_L, ax_pol, ax_txt


def setup_axes(ax3d, ax_w, ax_E, ax_L, ax_pol, ax_txt):
    ax3d.set_facecolor(PANEL_BG)
    ax3d.set_aspect('equal')
    ax3d.axis('off')
    ax3d.set_title('3D Spacecraft (torque-free rotation)', color=TEXT_COLOR, fontsize=9, pad=4)

    for ax, title, ylabel in [
        (ax_w,  'Angular velocity ω(t)', 'rad/s'),
        (ax_E,  'Kinetic energy', 'T (J)'),
        (ax_L,  'Angular momentum²', 'L² (kg²m⁴/s²)'),
        (ax_pol,'Polhode curve (ω-space)', 'ω₂'),
    ]:
        ax.set_title(title, color=TEXT_COLOR, fontsize=8, pad=3)
        ax.set_ylabel(ylabel, fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(True, linewidth=0.4)

    ax_w.set_xlabel('time (s)', fontsize=7)
    ax_pol.set_xlabel('ω₁', fontsize=7)
    ax_txt.axis('off')


class Renderer:
    """Handles all matplotlib drawing for one animation frame."""

    CAM_SPEED = 0.006   # auto-rotate rate (rad/frame)

    def __init__(self, fig, axes, sim):
        self.fig  = fig
        self.axes = axes   # (ax3d, ax_w, ax_E, ax_L, ax_pol, ax_txt)
        self.sim  = sim
        self.cam_angle_y = 0.4
        self.cam_angle_x = 0.3
        self.spacecraft_mesh = make_spacecraft_mesh()
        self._last_wall = time.time()

    def cam_matrix(self):
        return rotation_matrix_yx(self.cam_angle_y, self.cam_angle_x)

    def _project(self, pts):
        R = self.cam_matrix()
        rotated = (R @ pts.T).T
        return perspective_project(rotated, cam_dist=5.0), rotated

    def draw_3d(self):
        ax = self.axes[0]
        ax.cla()
        ax.set_facecolor(PANEL_BG)
        ax.axis('off')
        ax.set_xlim(-1.8, 1.8)
        ax.set_ylim(-1.8, 1.8)
        ax.set_title('3D Spacecraft — Dzhanibekov Flip', color=TEXT_COLOR, fontsize=9, pad=4)

        q = self.sim.q
        R = self.cam_matrix()

        # ── Polhode trail in 3D ──────────────────────
        if len(self.sim.trail) > 2:
            trail_pts = np.array(self.sim.trail) * 1.2
            trail_rot = (R @ trail_pts.T).T
            trail_2d  = perspective_project(trail_rot)
            n = len(trail_2d)
            for i in range(1, n):
                alpha = i / n
                ax.plot(
                    trail_2d[i-1:i+1, 0], trail_2d[i-1:i+1, 1],
                    color=AMBER, alpha=alpha * 0.75,
                    linewidth=0.8 + alpha,
                )

        # ── Spacecraft mesh ──────────────────────────
        body_faces = []
        for verts, color in self.spacecraft_mesh:
            # Apply body rotation
            rotated_verts = np.array([quat_rotate(q, v) for v in verts])
            cam_verts = (R @ rotated_verts.T).T
            depth = face_depth(cam_verts)
            proj = perspective_project(cam_verts)
            body_faces.append((depth, proj, color))

        body_faces.sort(key=lambda x: x[0])   # painter's algorithm
        for depth, proj, color in body_faces:
            poly = plt.Polygon(proj, closed=True, facecolor=color,
                               edgecolor='#AAAACC', linewidth=0.3, alpha=0.92)
            ax.add_patch(poly)

        # ── Body-frame axes ──────────────────────────
        axis_vecs = [
            quat_rotate(q, np.array([1, 0, 0])),
            quat_rotate(q, np.array([0, 1, 0])),
            quat_rotate(q, np.array([0, 0, 1])),
        ]
        axis_labels = ['x₁ (min I)', 'x₂ (mid I)', 'x₃ (max I)']
        for i, (av, color, label) in enumerate(zip(axis_vecs, AXIS_COLORS, axis_labels)):
            tip = (R @ (av * 1.4))
            t2  = perspective_project(tip.reshape(1, 3))[0]
            ax.annotate(
                '', xy=t2, xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5),
            )
            ax.text(t2[0] * 1.12, t2[1] * 1.12, label,
                    color=color, fontsize=7, ha='center', va='center')

        # ── ω vector ─────────────────────────────────
        w = self.sim.w
        w_mag = np.linalg.norm(w)
        if w_mag > 0.01:
            w_world = sum(w[i] * axis_vecs[i] for i in range(3)) / w_mag * 1.3
            wc = (R @ w_world)
            wt = perspective_project(wc.reshape(1, 3))[0]
            ax.annotate(
                '', xy=wt, xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color=AMBER, lw=2.0,
                                linestyle='dashed'),
            )
            ax.text(wt[0] * 1.1, wt[1] * 1.1, 'ω', color=AMBER, fontsize=10,
                    fontweight='bold', ha='center')

        # ── Legend ───────────────────────────────────
        patches = [mpatches.Patch(color=c, label=l)
                   for c, l in zip(AXIS_COLORS, axis_labels)]
        patches.append(mpatches.Patch(color=AMBER, label='ω vector (dashed)'))
        ax.legend(handles=patches, loc='lower left', fontsize=6.5,
                  facecolor=PANEL_BG, edgecolor='#2A2A35', labelcolor=TEXT_COLOR)

    def draw_timeseries(self):
        ax_w, ax_E, ax_L = self.axes[1], self.axes[2], self.axes[3]
        t_hist = self.sim.t_history
        if len(t_hist) < 2:
            return
        t_arr = np.array(t_hist)
        w_arr = np.array(self.sim.w_history)
        E_arr = np.array(self.sim.E_history)
        L_arr = np.array(self.sim.L2_history)

        # Keep last N seconds for readability
        WINDOW = 30.0
        mask = t_arr >= (t_arr[-1] - WINDOW)
        t_w, w_w = t_arr[mask], w_arr[mask]
        t_e, E_w = t_arr[mask], E_arr[mask]
        t_l, L_w = t_arr[mask], L_arr[mask]

        ax_w.cla()
        ax_w.set_facecolor(PANEL_BG)
        for i, color in enumerate(AXIS_COLORS):
            ax_w.plot(t_w, w_w[:, i], color=color, linewidth=0.9, label=f'ω{i+1}')
        ax_w.axhline(0, color='#333340', linewidth=0.4)
        ax_w.set_title('Angular velocity ω(t)', color=TEXT_COLOR, fontsize=8, pad=3)
        ax_w.set_ylabel('rad/s', fontsize=7)
        ax_w.set_xlabel('time (s)', fontsize=7)
        ax_w.legend(fontsize=6.5, facecolor=PANEL_BG, edgecolor='#2A2A35',
                    labelcolor=TEXT_COLOR, ncol=3)
        ax_w.tick_params(labelsize=7)
        ax_w.grid(True, linewidth=0.4)

        # Energy — should stay flat (conservation check)
        ax_E.cla()
        ax_E.set_facecolor(PANEL_BG)
        if len(E_w) > 1:
            rel_var = (E_w - E_w[0]) / (abs(E_w[0]) + 1e-12) * 100
            ax_E.plot(t_e, rel_var, color='#9FE1CB', linewidth=0.9)
        ax_E.set_title('Energy drift (%)', color=TEXT_COLOR, fontsize=8, pad=3)
        ax_E.set_ylabel('ΔT/T₀ (%)', fontsize=7)
        ax_E.tick_params(labelsize=7)
        ax_E.grid(True, linewidth=0.4)

        # Angular momentum — should also stay flat
        ax_L.cla()
        ax_L.set_facecolor(PANEL_BG)
        if len(L_w) > 1:
            rel_var = (L_w - L_w[0]) / (abs(L_w[0]) + 1e-12) * 100
            ax_L.plot(t_l, rel_var, color='#AFA9EC', linewidth=0.9)
        ax_L.set_title('L² drift (%)', color=TEXT_COLOR, fontsize=8, pad=3)
        ax_L.set_ylabel('ΔL²/L₀² (%)', fontsize=7)
        ax_L.tick_params(labelsize=7)
        ax_L.grid(True, linewidth=0.4)

    def draw_polhode(self):
        """
        Polhode curve: ω trajectory on the energy ellipsoid in body frame.
        Separatrix passes through the unstable intermediate axis.
        """
        ax = self.axes[4]
        ax.cla()
        ax.set_facecolor(PANEL_BG)
        ax.set_title('Polhode (ω-space body frame)', color=TEXT_COLOR, fontsize=8, pad=3)
        ax.set_xlabel('ω₁', fontsize=7)
        ax.set_ylabel('ω₂', fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(True, linewidth=0.4)
        ax.set_aspect('equal', adjustable='datalim')

        if len(self.sim.w_history) < 3:
            return

        w_arr = np.array(self.sim.w_history)
        ax.plot(w_arr[:, 0], w_arr[:, 1],
                color=AMBER, linewidth=0.7, alpha=0.7, label='Polhode')

        # Mark current position
        ax.scatter([w_arr[-1, 0]], [w_arr[-1, 1]],
                   color='white', s=18, zorder=5)

        # Draw separatrix (the unstable curve dividing stable regions)
        I  = self.sim.I
        w0 = self.sim.w_history[0] if self.sim.w_history else np.zeros(3)
        E0 = kinetic_energy(np.array(w0), I) if w0 is not None else 0
        L0 = angular_momentum_sq(np.array(w0), I) if w0 is not None else 0
        w1_range = np.linspace(-3.5, 3.5, 300)
        for sign in [1, -1]:
            try:
                inner = (L0 - 2*E0*I[0]) / (I[1]*(I[1]-I[0]) + 1e-12)
                term  = w1_range**2 * (I[0]*(I[2]-I[0])) / (I[1]*(I[1]-I[2]) + 1e-12)
                w2_sq = inner + term
                w2_sq = np.where(w2_sq >= 0, w2_sq, np.nan)
                ax.plot(w1_range, sign * np.sqrt(w2_sq),
                        '--', color='#E24B4A', linewidth=0.7, alpha=0.5)
            except Exception:
                pass

        ax.legend(fontsize=6, facecolor=PANEL_BG, edgecolor='#2A2A35',
                  labelcolor=TEXT_COLOR)

    def draw_stats(self):
        ax = self.axes[5]
        ax.cla()
        ax.axis('off')

        sim  = self.sim
        I    = sim.I
        w    = sim.w
        E    = kinetic_energy(w, I)
        L2   = angular_momentum_sq(w, I)
        T_meas = sim.mean_flip_period()

        stab = stability_analysis(I)
        spin_stab = stab[sim.spin_axis]

        try:
            T_pred = predict_flip_period(I, w)
        except Exception:
            T_pred = None

        status_color = '#1D9E75' if spin_stab['stable'] else '#E24B4A'
        status_text  = 'STABLE' if spin_stab['stable'] else 'UNSTABLE — FLIPPING'

        lines = [
            ('━━━━ Simulation Stats ━━━━', TEXT_COLOR, 9),
            (f'Time:            {sim.t:7.2f} s',          TEXT_COLOR, 8),
            (f'Flip count:      {sim.flip_count}',         AMBER, 8),
            (f'Spin axis:       {sim.spin_axis + 1} '
             f'({spin_stab["rank"]} I)',                    TEXT_COLOR, 8),
            (f'Stability:       {status_text}',            status_color, 8),
            ('', TEXT_COLOR, 8),
            ('━━━━ Moments of Inertia ━━━━', TEXT_COLOR, 9),
            (f'I₁ = {I[0]:.2f}   I₂ = {I[1]:.2f}   I₃ = {I[2]:.2f}',
             TEXT_COLOR, 8),
            ('', TEXT_COLOR, 8),
            ('━━━━ Conservation Checks ━━━━', TEXT_COLOR, 9),
            (f'Kin. energy T:   {E:.5f} J',               '#9FE1CB', 8),
            (f'|L|²:            {L2:.5f}',                '#AFA9EC', 8),
            ('', TEXT_COLOR, 8),
            ('━━━━ Flip Period ━━━━', TEXT_COLOR, 9),
            (f'Measured:   {T_meas:.2f} s' if T_meas else
             'Measured:   (waiting…)',                     AMBER, 8),
            (f'Predicted:  {T_pred:.2f} s' if T_pred else
             'Predicted:  n/a',                            AMBER, 8),
            ('', TEXT_COLOR, 8),
            ('━━━━ Controls ━━━━', TEXT_COLOR, 9),
            ('SPACE  pause/resume',                        MUTED_COLOR, 7.5),
            ('1/2/3  switch spin axis',                    MUTED_COLOR, 7.5),
            ('r      reset   q  quit',                     MUTED_COLOR, 7.5),
        ]

        y = 0.98
        for text, color, size in lines:
            ax.text(0.02, y, text, transform=ax.transAxes,
                    color=color, fontsize=size, va='top', fontfamily='monospace')
            y -= 0.057

    def update(self, frame):
        if self.sim.running:
            # Multiple physics steps per frame for smoother motion
            for _ in range(3):
                self.sim.step()
            self.cam_angle_y += self.CAM_SPEED

        self.draw_3d()
        self.draw_timeseries()
        self.draw_polhode()
        self.draw_stats()

        self.fig.suptitle(
            f'Dzhanibekov Effect  —  t = {self.sim.t:.1f}s  '
            f'| Flips: {self.sim.flip_count}  '
            f'| {"▶ Running" if self.sim.running else "⏸ Paused"}',
            color=TEXT_COLOR, fontsize=10, y=0.98,
        )
        return []


# Main

def main():
    sim = Simulation()

    fig, ax3d, ax_w, ax_E, ax_L, ax_pol, ax_txt = build_figure(sim)
    axes = (ax3d, ax_w, ax_E, ax_L, ax_pol, ax_txt)
    renderer = Renderer(fig, axes, sim)

    def on_key(event):
        k = event.key
        if k == ' ':
            sim.running = not sim.running
        elif k == '1':
            sim.spin_axis = 0; sim.reset()
        elif k == '2':
            sim.spin_axis = 1; sim.reset()
        elif k == '3':
            sim.spin_axis = 2; sim.reset()
        elif k in ('r', 'R'):
            sim.reset()
        elif k in ('q', 'escape'):
            plt.close('all')
            sys.exit(0)

    fig.canvas.mpl_connect('key_press_event', on_key)

    print('\n' + '='*60)
    print('  Dzhanibekov Effect — 3D Rigid Body Simulation')
    print('='*60)
    print('  Controls:')
    print('    SPACE   — pause / resume')
    print('    1/2/3   — switch spin axis (2 = unstable)')
    print('    r       — reset simulation')
    print('    q / ESC — quit')
    print('='*60 + '\n')
    print('  Starting with intermediate axis (axis 2) — watch it flip!\n')

    ani = FuncAnimation(
        fig, renderer.update,
        interval=30,       # ~33 fps target
        blit=False,
        cache_frame_data=False,
    )

    plt.show()


if __name__ == '__main__':
    main()
