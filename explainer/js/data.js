// Time series used on the page: two simulated systems (integrated here) and two real series (assets/series.json).

// Mackey–Glass delay equation dx/dt = β x(t−τ) / (1 + x(t−τ)^p) − γ x, integrated with RK4 at step dt.
// The delayed value at half steps is the mean of its two neighbours on the grid. One sample every `sub` steps.
export function mackeyGlass(n, { tau = 17, beta = 0.2, gamma = 0.1, p = 10, dt = 0.1, sub = 10, x0 = 1.2, discard = 500 } = {}) {
  const d = Math.round(tau / dt);
  const total = (n + discard) * sub;
  const hist = new Float64Array(total + d + 1).fill(x0);
  const f = (x, xt) => (beta * xt) / (1 + xt ** p) - gamma * x;
  for (let i = d; i < d + total; i++) {
    const x = hist[i];
    const xt0 = hist[i - d];
    const xt1 = hist[i - d + 1];
    const xth = 0.5 * (xt0 + xt1);
    const k1 = f(x, xt0);
    const k2 = f(x + 0.5 * dt * k1, xth);
    const k3 = f(x + 0.5 * dt * k2, xth);
    const k4 = f(x + dt * k3, xt1);
    hist[i + 1] = x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4);
  }
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) out[i] = hist[d + (discard + i) * sub];
  return out;
}

// Lorenz-63 with RK4; returns Float64Array(n × 3), row t = state at time t·dt.
export const LORENZ_X0 = [17.67715816276679, 12.931379185960404, 43.91404334248268]; // on the attractor (Gauthier et al., 2021)
export const LORENZ_LYAPUNOV = 0.9056; // largest Lyapunov exponent
export function lorenz(n, { dt = 0.025, sigma = 10, rho = 28, beta = 8 / 3, x0 = LORENZ_X0 } = {}) {
  const out = new Float64Array(n * 3);
  let [x, y, z] = x0;
  const f = (a, b, c) => [sigma * (b - a), a * (rho - c) - b, a * b - beta * c];
  for (let i = 0; i < n; i++) {
    out[3 * i] = x;
    out[3 * i + 1] = y;
    out[3 * i + 2] = z;
    const k1 = f(x, y, z);
    const k2 = f(x + (dt / 2) * k1[0], y + (dt / 2) * k1[1], z + (dt / 2) * k1[2]);
    const k3 = f(x + (dt / 2) * k2[0], y + (dt / 2) * k2[1], z + (dt / 2) * k2[2]);
    const k4 = f(x + dt * k3[0], y + dt * k3[1], z + dt * k3[2]);
    x += (dt / 6) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]);
    y += (dt / 6) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]);
    z += (dt / 6) * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]);
  }
  return out;
}

// z-score with the mean and standard deviation of rows [t0, t1) only (the training part), for d interleaved channels.
export function standardize(x, t0, t1, d = 1) {
  const mu = new Float64Array(d);
  const sd = new Float64Array(d);
  for (let t = t0; t < t1; t++) for (let c = 0; c < d; c++) mu[c] += x[t * d + c];
  for (let c = 0; c < d; c++) mu[c] /= t1 - t0;
  for (let t = t0; t < t1; t++) for (let c = 0; c < d; c++) sd[c] += (x[t * d + c] - mu[c]) ** 2;
  for (let c = 0; c < d; c++) sd[c] = Math.sqrt(sd[c] / (t1 - t0)) || 1;
  const z = new Float64Array(x.length);
  for (let i = 0; i < x.length; i++) z[i] = (x[i] - mu[i % d]) / sd[i % d];
  return { z, mu, sd };
}

// Windows ending at t for t in [t0, t1): row = [z(t), z(t−1), …, z(t−L+1)] (newest first).
export function windowMatrix(z, L, t0, t1) {
  const n = t1 - t0;
  const X = new Float64Array(n * L);
  for (let r = 0; r < n; r++) for (let j = 0; j < L; j++) X[r * L + j] = z[t0 + r - j];
  return X;
}

// Direct multi-step targets for t in [t0, t1): row = [z(t+1), …, z(t+H)].
export function targetMatrix(z, H, t0, t1) {
  const n = t1 - t0;
  const Y = new Float64Array(n * H);
  for (let r = 0; r < n; r++) for (let h = 0; h < H; h++) Y[r * H + h] = z[t0 + r + 1 + h];
  return Y;
}

// iid uniform input in [−1, 1], for the memory test
export function whiteNoise(n, r) {
  return Float64Array.from({ length: n }, () => 2 * r.uniform() - 1);
}
