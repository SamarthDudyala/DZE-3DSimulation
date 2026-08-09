import numpy as np
from scipy.integrate import solve_ivp


def quat_normalize(q):
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n


def quat_multiply(a, b):
    """Hamilton product a * b, both as (w, x, y, z)."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def quat_to_rotmat(q):
    """Unit quaternion (w,x,y,z) -> 3x3 rotation matrix (body -> inertial)."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


class RigidBody:
    """
    A torque-free rigid body defined by its principal moments of inertia.

    I1, I2, I3 are the principal moments of inertia (about body x, y, z axes).
    Convention used throughout: I1 < I2 < I3 gives the classic unstable spin
    about the intermediate axis (y / axis 2) -- the Dzhanibekov Effect.
    """

    def __init__(self, I1, I2, I3):
        if min(I1, I2, I3) <= 0:
            raise ValueError("Moments of inertia must be positive.")
        self.I = np.array([float(I1), float(I2), float(I3)])

    #dynamics
    def euler_rhs(self, w):
        """dw/dt for torque-free Euler's equations, body frame."""
        I1, I2, I3 = self.I
        w1, w2, w3 = w
        dw1 = (I2 - I3) * w2 * w3 / I1
        dw2 = (I3 - I1) * w3 * w1 / I2
        dw3 = (I1 - I2) * w1 * w2 / I3
        return np.array([dw1, dw2, dw3])

    def state_rhs(self, t, y):
        q = y[0:4]
        w = y[4:7]
        dq = 0.5 * quat_multiply(q, np.array([0.0, w[0], w[1], w[2]]))
        dw = self.euler_rhs(w)
        return np.concatenate([dq, dw])

    #diagnostics

    def kinetic_energy(self, w):
        return 0.5 * np.sum(self.I * np.asarray(w) ** 2)

    def angular_momentum_body(self, w):
        return self.I * np.asarray(w)

    #simulation

    def simulate(self, w0, t_span=(0.0, 20.0), n_points=2000, rtol=1e-9, atol=1e-11):
        """
        Integrate the torque-free motion starting from body angular velocity w0
        and identity orientation. Returns a dict with time series and diagnostics.
        """
        q0 = np.array([1.0, 0.0, 0.0, 0.0])
        y0 = np.concatenate([q0, np.asarray(w0, dtype=float)])
        t_eval = np.linspace(t_span[0], t_span[1], n_points)

        sol = solve_ivp(
            self.state_rhs, t_span, y0, t_eval=t_eval,
            method="RK45", rtol=rtol, atol=atol, dense_output=True,
        )

        q = sol.y[0:4, :].T
        # normalize quaternions (numerical drift correction)
        q = q / np.linalg.norm(q, axis=1, keepdims=True)
        w = sol.y[4:7, :].T

        energy = np.array([self.kinetic_energy(wi) for wi in w])
        L_body = np.array([self.angular_momentum_body(wi) for wi in w])
        L_mag = np.linalg.norm(L_body, axis=1)

        return {
            "t": sol.t,
            "q": q,
            "w": w,
            "energy": energy,
            "L_mag": L_mag,
            "sol": sol,  # dense_output callable: sol.sol(t) -> state at time t
        }

    #flip detection
    @staticmethod
    def detect_flips(t, w, axis=0, threshold=0.0):
        """
        Count sign changes of w[:, axis] as a proxy for 'flip' events
        (characteristic of intermediate-axis instability).
        Returns (num_flips, flip_times).
        """
        signal = w[:, axis]
        signs = np.sign(signal - threshold)
        signs[signs == 0] = 1
        change_idx = np.where(np.diff(signs) != 0)[0]
        flip_times = t[change_idx]
        return len(change_idx), flip_times


if __name__ == "__main__":
    # quick self-test: classic unstable spin about the intermediate axis
    body = RigidBody(I1=1.0, I2=2.0, I3=3.0)
    result = body.simulate(w0=[0.05, 5.0, 0.05], t_span=(0, 20), n_points=4000)
    n_flips, flip_times = body.detect_flips(result["t"], result["w"], axis=1)
    print(f"Energy drift: {result['energy'].max() - result['energy'].min():.3e}")
    print(f"|L| drift:    {result['L_mag'].max() - result['L_mag'].min():.3e}")
    print(f"Flips detected about intermediate axis: {n_flips}")
    print(f"First few flip times: {flip_times[:5]}")
