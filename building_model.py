"""
First-order (1R1C) building thermal model for teaching.

Continuous time:
    C dT/dt = (To - T)/R + u + Q0

    T  : indoor temperature [degC]      (state / output)
    u  : heating power [W]              (control input, >= 0)
    To : outdoor temperature [degC]     (disturbance)
    Q0 : constant unmeasured heat gain [W] (0 by default)

Exact zero-order-hold discretisation with sampling time dt:
    T[k+1] = A*T[k] + B*u[k] + E*To[k] + B*Q0
    A = exp(-dt/(R*C)),  B = R*(1-A),  E = 1-A

Default parameters were identified from statespace.csv (simulation-error fit,
I_sol not used). See first_order_model.ipynb.

Typical use:
    from building_model import FirstOrderBuilding, outdoor_temperature
    model = FirstOrderBuilding()
    To = outdoor_temperature('winter')
    res = model.simulate_closed_loop(T0=15, r=21, To=To, controller=my_controller)
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_FILE = Path(__file__).with_name("statespace.csv")

# Identified from statespace.csv (dt = 1800 s, no solar input)
R_DEFAULT = 1.9487e-2   # K/W
C_DEFAULT = 1.1833e7   # J/K
DT_DEFAULT = 1800.0     # s


class FirstOrderBuilding:
    """Discrete-time 1R1C building model: T[k+1] = A T[k] + B u[k] + E To[k]."""

    def __init__(self, R=R_DEFAULT, C=C_DEFAULT, dt=DT_DEFAULT, Q0=0.0):
        self.R = float(R)
        self.C = float(C)
        self.dt = float(dt)
        self.Q0 = float(Q0)

    # ---- discrete-time matrices -------------------------------------------
    @property
    def tau(self):
        """Time constant R*C [s]."""
        return self.R * self.C

    @property
    def A(self):
        return np.exp(-self.dt / self.tau)

    @property
    def B(self):
        return self.R * (1.0 - self.A)

    @property
    def E(self):
        return 1.0 - self.A

    def __repr__(self):
        return (f"FirstOrderBuilding(R={self.R:.4e} K/W, C={self.C:.4e} J/K, "
                f"tau={self.tau/3600:.1f} h, dt={self.dt:.0f} s, Q0={self.Q0:.0f} W)\n"
                f"  T[k+1] = {self.A:.5f}*T[k] + {self.B:.4e}*u[k] + {self.E:.5f}*To[k]"
                + (f" + {self.B*self.Q0:.4f}" if self.Q0 else ""))

    # ---- simulation --------------------------------------------------------
    def step(self, T, u, To):
        """One time step. Returns T[k+1]."""
        return self.A * T + self.B * (u + self.Q0) + self.E * To

    def simulate(self, T0, u, To):
        """Open-loop simulation with given input sequence u and disturbance To.

        Returns T of length len(To) (T[0] = T0)."""
        To = np.asarray(To, dtype=float)
        u = np.broadcast_to(np.asarray(u, dtype=float), To.shape)
        T = np.empty(len(To))
        T[0] = T0
        for k in range(len(To) - 1):
            T[k + 1] = self.step(T[k], u[k], To[k])
        return T

    def simulate_closed_loop(self, T0, r, To, controller, pass_k=False):
        """Closed-loop simulation.

        Parameters
        ----------
        T0 : initial indoor temperature [degC]
        r  : reference, scalar or array of len(To) [degC]
        To : outdoor temperature sequence [degC]
        controller : function controller(r_k, T_k) -> u_k [W]
        pass_k : if True, the controller is called as controller(r_k, T_k, k),
                 e.g. so that an MPC can look up the disturbance forecast

        Returns
        -------
        dict with arrays t [h], T, u, r, To (all of length len(To));
        u[-1] is the input computed at the last sample (not applied).
        """
        To = np.asarray(To, dtype=float)
        N = len(To)
        r = np.broadcast_to(np.asarray(r, dtype=float), (N,)).copy()
        T = np.empty(N)
        u = np.empty(N)
        T[0] = T0
        for k in range(N):
            u[k] = controller(r[k], T[k], k) if pass_k else controller(r[k], T[k])
            if k < N - 1:
                T[k + 1] = self.step(T[k], u[k], To[k])
        return {"t": np.arange(N) * self.dt / 3600, "T": T, "u": u, "r": r, "To": To}

    # ---- identification ----------------------------------------------------
    @classmethod
    def fit(cls, df=None, with_Q0=False):
        """Identify R, C (and optionally Q0) from data by minimising the
        free-run simulation error. Uses columns Time, To, Qh, xi."""
        from scipy.optimize import least_squares

        df = load_data() if df is None else df
        dt = float(df["Time"].iloc[1] - df["Time"].iloc[0])
        To, Qh, y = df["To"].values, df["Qh"].values, df["xi"].values
        scale = np.array([1e-2, 1e7, 1e3])  # R, C, Q0 scaling for the optimiser

        def residuals(p):
            p = p * scale[: len(p)]
            m = cls(p[0], p[1], dt, p[2] if with_Q0 else 0.0)
            return m.simulate(y[0], Qh, To) - y

        p0 = [1.0, 1.0, 0.5] if with_Q0 else [2.0, 1.0]
        lb = [1e-3, 1e-3, -np.inf] if with_Q0 else [1e-3, 1e-3]
        sol = least_squares(residuals, p0, bounds=(lb, np.inf))
        p = sol.x * scale[: len(sol.x)]
        model = cls(p[0], p[1], dt, p[2] if with_Q0 else 0.0)
        model.fit_rmse = float(np.sqrt(np.mean(sol.fun ** 2)))
        return model


# ---- data helpers ----------------------------------------------------------
def load_data(path=DATA_FILE):
    """Load the measured data (Time, To, Qh, I_sol, xi)."""
    return pd.read_csv(path)


def outdoor_temperature(kind="winter", n_days=None, shift=-12.0):
    """Outdoor temperature profile to use as disturbance (dt = 1800 s).

    kind='measured' : measured To from statespace.csv (13-33 degC, too warm
                      for a heating-only case at 21 degC)
    kind='winter'   : measured To shifted by `shift` degC (default -12 ->
                      about 1-21 degC); keeps the real daily shape
    n_days          : if given, the profile is repeated/truncated to n_days
    """
    To = load_data()["To"].values.astype(float)
    if kind == "winter":
        To = To + shift
    elif kind != "measured":
        raise ValueError("kind must be 'winter' or 'measured'")
    if n_days is not None:
        N = int(round(n_days * 24 * 3600 / DT_DEFAULT))
        To = np.resize(To, N)
    return To


def generate_training_data(model, n_episodes=60, episode_len=96, hold_steps=4,
                           noise_std=0.02, seed=0):
    """Generate input/output data from `model` (playing the role of the real building).

    Each episode starts from a random indoor temperature (12-28 degC). The
    outdoor temperature is a random window of the same winter profile used in
    the exercises: the measured To shifted by a fixed -12 degC. The heating
    power is random, uniform in [0, 2000] W and held for `hold_steps` samples.
    Gaussian measurement noise is added to T.

    Returns a DataFrame with columns T, u, To (at time k) and T_next (time k+1).
    """
    rng = np.random.default_rng(seed)
    To_w = outdoor_temperature("winter")
    episode_len = min(episode_len, len(To_w))
    rows = []
    for _ in range(n_episodes):
        start = rng.integers(0, len(To_w) - episode_len + 1)
        To = To_w[start:start + episode_len]
        u = np.repeat(rng.uniform(0, 2000, episode_len // hold_steps + 1), hold_steps)[:episode_len]
        T = model.simulate(rng.uniform(12, 28), u, To)
        T = T + rng.normal(0, noise_std, episode_len)
        rows.append(pd.DataFrame({"T": T[:-1], "u": u[:-1], "To": To[:-1], "T_next": T[1:]}))
    return pd.concat(rows, ignore_index=True)


def evaluate(results, dt=DT_DEFAULT, T_min=None, T_max=None, skip_hours=0.0):
    """Performance indicators of a closed-loop result.

    rmse        : RMS tracking error r - T [K]
    viol_Kh     : comfort violation outside [T_min, T_max], integrated [K*h]
    energy_kWh  : heating energy [kWh]
    skip_hours  : ignore the first hours (e.g. the start-up transient)
    """
    k0 = int(round(skip_hours * 3600 / dt))
    T, r, u = results["T"][k0:], results["r"][k0:], results["u"][k0:-1]
    out = {"rmse": float(np.sqrt(np.mean((r - T) ** 2))),
           "energy_kWh": float(u.sum() * dt / 3.6e6)}
    if T_min is not None or T_max is not None:
        lo = -np.inf if T_min is None else T_min
        hi = np.inf if T_max is None else T_max
        viol = np.maximum(lo - T, 0) + np.maximum(T - hi, 0)
        out["viol_Kh"] = float(viol.sum() * dt / 3600)
    return out


def plot_closed_loop(results, labels=None, u_max=None, T_band=None):
    """Plot one or several closed-loop results (dicts from simulate_closed_loop)."""
    import matplotlib.pyplot as plt

    if isinstance(results, dict):
        results = [results]
    labels = labels or [None] * len(results)
    fig, ax = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    for res, lab in zip(results, labels):
        ax[0].plot(res["t"], res["T"], label=lab)
        ax[1].plot(res["t"], res["u"], label=lab)
    r0 = results[0]
    ax[0].plot(r0["t"], r0["r"], "k--", lw=1, label="reference")
    ax[2].plot(r0["t"], r0["To"], color="tab:gray")
    if u_max is not None:
        ax[1].axhline(u_max, color="k", ls=":", lw=1)
    if T_band is not None:
        ax[0].axhspan(T_band[0], T_band[1], color="tab:green", alpha=0.08, label="comfort band")
    ax[0].set_ylabel("Indoor T [°C]")
    ax[1].set_ylabel("Heating u [W]")
    ax[2].set_ylabel("Outdoor To [°C]")
    ax[2].set_xlabel("Time [h]")
    ax[0].legend(loc="best", fontsize=8)
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    return fig, ax
