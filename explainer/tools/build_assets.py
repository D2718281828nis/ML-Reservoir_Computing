"""Build the page's data file from dataset/ and the numpy reference values that tools/check.mjs compares against.

    python explainer/tools/build_assets.py        # numpy + pandas, a few seconds

Writes
  assets/series.json  the two real series of part 5: daily electricity demand in Spain and an ECG at 250 Hz;
  assets/checks.json  numpy results for fixed inputs: the integrators, an ESN, NG-RC features, ridge solves,
                      and the exact spectral radius of a sparse random matrix.
"""
import json
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
ASSETS = HERE.parent / "assets"


# ───────────── real series ─────────────
def electricity():
    """'Demanda real' rows of the Spanish energy-market file: one value per day, the day's mean load in MW."""
    df = pd.read_csv(REPO / "dataset" / "spain_energy_market.csv")
    d = df[df["name"] == "Demanda real"].copy()
    d["date"] = pd.to_datetime(d["datetime"]).dt.normalize()
    # a few rows sit at 23:00 or 01:00 because of daylight-saving changes; one value per calendar day remains
    s = d.groupby("date")["value"].mean()
    full = pd.date_range(s.index.min(), s.index.max(), freq="D")
    s = s.reindex(full).interpolate()
    return {
        "start": str(full[0].date()),
        "unit": "GW",
        "values": [round(v / 1000, 3) for v in s.to_numpy()],
    }


def ecg():
    """The 1 kHz ECG used by the ECG notebooks, averaged over blocks of 4 samples: 250 Hz."""
    df = pd.read_csv(REPO / "dataset" / "ecg_1d_timeseries_prediction.csv", sep=";")
    x = df["ecg_value"].to_numpy(float)
    x = x[: len(x) // 4 * 4].reshape(-1, 4).mean(1)
    return {"fs": 250, "unit": "ADC units", "values": [round(v, 2) for v in x]}


# ───────────── numpy references (same algorithms as js/data.js, js/models.js, js/nn.js) ─────────────
def mackey_glass(n, tau=17, beta=0.2, gamma=0.1, p=10, dt=0.1, sub=10, x0=1.2, discard=500):
    d = int(round(tau / dt))
    total = (n + discard) * sub
    hist = np.full(total + d + 1, x0)
    f = lambda x, xt: beta * xt / (1 + xt ** p) - gamma * x
    for i in range(d, d + total):
        x, xt0, xt1 = hist[i], hist[i - d], hist[i - d + 1]
        xth = 0.5 * (xt0 + xt1)
        k1 = f(x, xt0)
        k2 = f(x + 0.5 * dt * k1, xth)
        k3 = f(x + 0.5 * dt * k2, xth)
        k4 = f(x + dt * k3, xt1)
        hist[i + 1] = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return hist[d + discard * sub :: sub][:n]


def lorenz(n, dt=0.025, s=10.0, r=28.0, b=8 / 3, x0=(17.67715816276679, 12.931379185960404, 43.91404334248268)):
    f = lambda v: np.array([s * (v[1] - v[0]), v[0] * (r - v[2]) - v[1], v[0] * v[1] - b * v[2]])
    v = np.array(x0, float)
    out = np.zeros((n, 3))
    for i in range(n):
        out[i] = v
        k1 = f(v); k2 = f(v + dt / 2 * k1); k3 = f(v + dt / 2 * k2); k4 = f(v + dt * k3)
        v = v + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return out


def ridge(F, Y, lam):
    """argmin (1/n)‖F B − Y‖² + λ‖B‖²"""
    n, p = F.shape
    return np.linalg.solve(F.T @ F + lam * n * np.eye(p), F.T @ Y)


def build_checks():
    rng = np.random.default_rng(2024)
    checks = {}

    mg = mackey_glass(600)
    lz = lorenz(400)
    checks["mackeyGlass"] = {"n": 600, "values": mg[[0, 1, 17, 100, 333, 599]].tolist(), "at": [0, 1, 17, 100, 333, 599]}
    checks["lorenz"] = {"n": 400, "rows": lz[[0, 1, 50, 399]].tolist(), "at": [0, 1, 50, 399]}

    # a small sparse reservoir with fixed matrices: states, readout, predictions
    N, deg, T = 40, 5, 300
    idx = np.stack([rng.choice(N, deg, replace=False) for _ in range(N)])
    val = rng.uniform(-1, 1, (N, deg))
    W = np.zeros((N, N))
    for i in range(N):
        W[i, idx[i]] = val[i]
    rho = np.max(np.abs(np.linalg.eigvals(W)))
    W *= 0.9 / rho
    win = rng.uniform(-0.5, 0.5, N)
    b = rng.uniform(-0.2, 0.2, N)
    leak = 0.4
    u = np.sin(0.21 * np.arange(T)) + 0.3 * np.sin(0.057 * np.arange(T) + 1)
    x = np.zeros(N)
    X = np.zeros((T, N))
    for t in range(T):
        x = (1 - leak) * x + leak * np.tanh(win * u[t] + W @ x + b)
        X[t] = x
    F = np.hstack([np.ones((T, 1)), u[:, None], X])
    Y = np.roll(u, -1)[:, None]  # next value
    B = ridge(F[50:250], Y[50:250], 1e-6)
    P = F[250:299] @ B
    checks["esn"] = {
        "N": N, "deg": deg, "leak": leak,
        "idx": idx.ravel().tolist(), "w": (val * 0.9 / rho).ravel().tolist(), "rhoRaw": float(rho),
        "win": win.tolist(), "b": b.tolist(), "u": u.tolist(),
        "statesAt": [0, 10, 299], "states": X[[0, 10, 299]].tolist(),
        "readout": B.ravel().tolist(), "pred": P.ravel().tolist(),
    }

    # NG-RC features (constant, linear taps, unique quadratic monomials) and the Lorenz readout
    k, s = 2, 1
    Z = (lz - lz[:300].mean(0)) / lz[:300].std(0)
    rows = []
    for t in range(1, 300):
        lin = np.concatenate([Z[t - j * s] for j in range(k)])
        quad = [lin[i] * lin[j] for i, j in combinations_with_replacement(range(len(lin)), 2)]
        rows.append(np.concatenate([[1.0], lin, quad]))
    Fn = np.array(rows)
    Yn = Z[2:301] - Z[1:300]
    Bn = ridge(Fn, Yn, 2.5e-7)
    checks["ngrc"] = {
        "k": k, "s": s, "lambda": 2.5e-7, "p": Fn.shape[1],
        "featuresAt": [0, 150, 298], "features": Fn[[0, 150, 298]].tolist(),
        "readout": Bn.ravel().tolist(),
    }

    # ridge with several outputs on a random problem
    Fr = rng.normal(size=(120, 9))
    Yr = rng.normal(size=(120, 3))
    checks["ridge"] = {"F": Fr.ravel().tolist(), "Y": Yr.ravel().tolist(), "lambda": 0.01, "B": ridge(Fr, Yr, 0.01).ravel().tolist()}

    # exact spectral radius of a larger sparse random matrix, for the power-iteration estimate
    N2, deg2 = 300, 10
    idx2 = np.stack([rng.choice(N2, deg2, replace=False) for _ in range(N2)])
    val2 = rng.uniform(-1, 1, (N2, deg2))
    W2 = np.zeros((N2, N2))
    for i in range(N2):
        W2[i, idx2[i]] = val2[i]
    checks["spectral"] = {
        "N": N2, "deg": deg2, "idx": idx2.ravel().tolist(), "w": val2.ravel().tolist(),
        "rho": float(np.max(np.abs(np.linalg.eigvals(W2)))),
    }
    return checks


def main():
    ASSETS.mkdir(exist_ok=True)
    series = {"electricity": electricity(), "ecg": ecg()}
    (ASSETS / "series.json").write_text(json.dumps(series, separators=(",", ":")))
    (ASSETS / "checks.json").write_text(json.dumps(build_checks(), separators=(",", ":")))
    for f in ("series.json", "checks.json"):
        print(f"{f}: {(ASSETS / f).stat().st_size / 1024:.0f} KB")
    print(f"electricity: {len(series['electricity']['values'])} days from {series['electricity']['start']}; "
          f"ecg: {len(series['ecg']['values'])} samples at 250 Hz")


if __name__ == "__main__":
    main()
