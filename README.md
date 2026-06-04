# Dzhanibekov Effect: 3D Rigid Body Simulation & Stability Analysis

An interactive, high-performance 3D physics simulation and analytical tool exploring the **Dzhanibekov Effect** (also known as the *Tennis Racket Theorem* or *Intermediate Axis Theorem*). The application models the torque-free rotation of an asymmetric rigid body using quaternions and Euler's equations of motion, visualizing the classic chaotic flipping behavior in real-time.

---

## Dashboard Preview
`
+---------------------------------------------------------------------------------+
|                                 APPLICATION DASHBOARD                           |
+-------------------------------+-------------------------------------------------+
|                               |               ANGULAR VELOCITY ω(t)             |
|                               |  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  |
|                               |  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  |
|         3D SPACECRAFT         +-----------------------+-------------------------+
|          VIEWPORT             |      ENERGY DRIFT     |        L² DRIFT         |
|                               |  -------------------  |  ---------------------  |
+-------------------------------+-----------------------+-------------------------+
|  [Insert Full Window          |     POLHODE CURVE     |    SIMULATION STATS     |
|   Screenshot Here]            |      (ω-space)        |  Spin Axis: 2 (Unstable)|
|                               |       ( \ * / )       |  Flip Count: 4          |
|                               |        (  X  )        |  Measured T: 4.12s      |
+-------------------------------+-----------------------+-------------------------+
`
> *Replace this block with a complete window capture showing the program executing a flip on Axis 2.*
---

## Features

* **Real-Time 3D Engine:** Wireframe and polygon render of a T-shaped spacecraft built natively on standard `matplotlib` projection matrices (no heavy 3D engine overhead required).
* **Dual-Quaternion Kinematics:** Implements singularity-free orientation tracking to prevent gimbal lock during high-rate tumbling.
* **Conservation Verification:** Real-time diagnostics monitoring Kinetic Energy ($T$) and Angular Momentum square ($L^2$) deviation percentages to track numerical integration accuracy.
* **Polhode Space Mapping:** Maps the trajectory of the angular velocity vector ($\omega$) in the body frame relative to the theoretical analytical separatrix.
* **Predictive AI/Analytics:** Uses Jacobi elliptic function integrals to approximate the geometric flip period dynamically based on initial angular rates and the principal moments of inertia tensor:
    $$I_1 < I_2 < I_3$$

---

## Mathematical Background

The simulation tracks rigid-body dynamics governed by Euler’s torque-free equations:

$$\begin{aligned}
I_1 \dot{\omega}_1 &= (I_2 - I_3)\omega_2\omega_3 \\
I_2 \dot{\omega}_2 &= (I_3 - I_1)\omega_3\omega_1 \\
I_3 \dot{\omega}_3 &= (I_1 - I_2)\omega_1\omega_2
\end{aligned}$$

When rotating exactly along the intermediate principal axis ($I_2$), any infinitesimal perturbation creates a heteroclinic orbit along the phase space separatrix, forcing a deterministic $180^\circ$ inversion.

---

## Installation & Setup

### Prerequisites
Ensure your Python environment satisfies the required scientific computing stack:

```bash
pip install numpy matplotlib scipy
