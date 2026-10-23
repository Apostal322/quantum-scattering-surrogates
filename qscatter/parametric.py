"""Parametric surrogate (E, V0, a, d) -> (Re t, Im t) for the symmetric double barrier.

Domain (all below the barrier top, resonant tunnelling, 1–2 resonances with
widths ~7e-3 … 1):  E in [0.2, 5], V0 in [5, 10], a in [0.3, 0.7], d in [1.5, 3.0].

Data separation
  * training configurations   scrambled Sobol sequence (seed s)
  * validation configurations  held-out 1/8 of the training configurations (same
                               budget; used for early stopping / Fourier scale only)
  * test configurations        40 configurations from an independent random stream,
                               evaluated on dense composite grids around exact poles
  * OOD configurations         20 configurations with one parameter up to 10 % of
                               its range outside the domain (reported separately)
"""
from __future__ import annotations

import copy

import numpy as np
from scipy.interpolate import CubicSpline, RBFInterpolator, RegularGridInterpolator
from scipy.stats import qmc

from .physics.poles import find_poles
from .physics.potentials import double_barrier, single_barrier
from .physics.smatrix import CountingSolver, compute_scattering

LO = np.array([0.2, 5.0, 0.3, 1.5])  # E, V0, a, d
HI = np.array([5.0, 10.0, 0.7, 3.0])
NAMES = ("E", "V0", "a", "d")


def scale(X):
    return 2 * (np.asarray(X) - LO) / (HI - LO) - 1


def pot(V0, a, d, W=0.0):
    return double_barrier(V0, a, d, well_depth=1j * W) if W else double_barrier(V0, a, d)


# ---------------------------------------------------------------- data
def sobol_configs(n, seed):
    s = qmc.Sobol(3, scramble=True, seed=seed).random(n)
    return LO[1:] + s * (HI[1:] - LO[1:])


def scan_dataset(n_cfg, n_E, seed, W=0.0, configs=None):
    """n_cfg configurations x n_E uniform energies. Returns X (N,4), t (N,), n_evals."""
    C = sobol_configs(n_cfg, seed) if configs is None else configs
    E = np.linspace(LO[0], HI[0], n_E)
    X, T, n = [], [], 0
    for V0, a, d in C:
        solver = CountingSolver(pot(V0, a, d, W))
        X.append(np.column_stack([E, np.full(n_E, V0), np.full(n_E, a), np.full(n_E, d)]))
        T.append(solver(E).t)
        n += solver.n_evals
    return np.vstack(X), np.concatenate(T), n, C


def test_configs(n=40, seed=777):
    rng = np.random.default_rng(seed)
    return LO[1:] + rng.uniform(size=(n, 3)) * (HI[1:] - LO[1:])


def ood_configs(n=20, seed=888):
    rng = np.random.default_rng(seed)
    C = LO[1:] + rng.uniform(size=(n, 3)) * (HI[1:] - LO[1:])
    for i in range(n):
        j = i % 3
        width = HI[1 + j] - LO[1 + j]
        C[i, j] = HI[1 + j] + rng.uniform(0.02, 0.10) * width  # one parameter beyond the upper edge
    return C


def config_reference(cfg):
    V0, a, d = cfg
    p = pot(V0, a, d)
    poles = find_poles(p, (LO[0], HI[0]), n_scan=8001)
    grids = [np.linspace(LO[0], HI[0], 4001)]
    for z in poles:
        G = -2 * z.imag
        grids.append(np.linspace(max(LO[0], z.real - 20 * G), min(HI[0], z.real + 20 * G), 801))
    E = np.unique(np.concatenate(grids))
    t = compute_scattering(E, p).t
    T = np.abs(t) ** 2
    res = []
    for z in poles:
        G = -2 * z.imag
        w = np.abs(E - z.real) < 3 * G
        if w.any() and T[w].max() > 0.5:
            res.append(dict(E=float(z.real), gamma=float(G)))
    return dict(cfg=list(map(float, cfg)), E=E, t=t, res=res)


# ---------------------------------------------------------------- metrics
def l2_rmse(E, err):
    return float(np.sqrt(np.trapezoid(err**2, E) / (E[-1] - E[0])))


def config_metrics(ref, T_hat):
    E, T = ref["E"], np.abs(ref["t"]) ** 2
    out = dict(rmse_T=l2_rmse(E, T_hat - T), max_err_T=float(np.max(np.abs(T_hat - T))),
               n_res=len(ref["res"]), n_det=0, pos_err_over_gamma=[], frac_unphysical=float(
                   np.mean((T_hat < -1e-6) | (T_hat > 1 + 1e-6))))
    for r in ref["res"]:
        w = np.abs(E - r["E"]) <= r["gamma"]
        if w.any() and T_hat[w].max() > 0.5:
            out["n_det"] += 1
            out["pos_err_over_gamma"].append(float(abs(E[w][np.argmax(T_hat[w])] - r["E"]) / r["gamma"]))
    # spurious peaks: peaks of T_hat (prominence > 0.05, height > 0.5) farther than 3 Gamma from every
    # true resonance. (A first version tested neighbour inequalities and counted flat pairs of nearly
    # coincident grid points as peaks; results produced with it are not used.)
    from scipy.signal import find_peaks

    i, _ = find_peaks(T_hat, prominence=0.05, height=0.5)
    out["n_spurious"] = int(sum(all(abs(E[j] - r["E"]) > 3 * r["gamma"] for r in ref["res"]) for j in i))
    return out


# ---------------------------------------------------------------- models
class TensorCubic:
    """Cubic interpolation on a tensor grid (V0, a, d) x E (needs a tensor design)."""

    name = "tensor_cubic"

    def fit_grid(self, axes, t_grid):
        self.re = RegularGridInterpolator(axes, t_grid.real, method="cubic", bounds_error=False, fill_value=None)
        self.im = RegularGridInterpolator(axes, t_grid.imag, method="cubic", bounds_error=False, fill_value=None)
        return self

    def predict(self, X):
        Xp = X[:, [1, 2, 3, 0]]
        return self.re(Xp) + 1j * self.im(Xp)


class RBF:
    name = "rbf"

    def fit(self, X, t, Xv=None, tv=None):
        Y = np.column_stack([t.real, t.imag])
        self.f = RBFInterpolator(scale(X), Y, kernel="thin_plate_spline", neighbors=min(200, len(X)), degree=1)
        return self

    def predict(self, X):
        Y = self.f(scale(X))
        return Y[:, 0] + 1j * Y[:, 1]


class GPParam:
    """ARD Matern-5/2 GP; exact GP limited to n <= n_max points (random subset otherwise)."""

    name = "gp"

    def __init__(self, n_max=2048, seed=0):
        self.n_max, self.seed = n_max, seed

    def fit(self, X, t, Xv=None, tv=None):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, Matern

        idx = np.random.default_rng(self.seed).permutation(len(X))[: self.n_max]
        self.n_used = len(idx)
        Xs = scale(X[idx])
        self.ms = []
        for y in (t.real[idx], t.imag[idx]):
            k = ConstantKernel(1.0, (1e-3, 1e3)) * Matern([0.5] * 4, (1e-3, 1e2), nu=2.5)
            self.ms.append(GaussianProcessRegressor(k, alpha=1e-8, normalize_y=True, n_restarts_optimizer=1,
                                                    random_state=self.seed).fit(Xs, y))
        return self

    def predict(self, X):
        Xs = scale(X)
        return self.ms[0].predict(Xs) + 1j * self.ms[1].predict(Xs)


class MLPParam:
    """MLP on scaled inputs, optional Gaussian Fourier features (Tancik et al., NeurIPS 2020).

    Full-batch-free minibatch Adam, cosine schedule, float32, early stopping on the
    held-out validation configurations, best checkpoint kept.
    """

    name = "mlp"

    def __init__(self, hidden=(128, 128, 128), fourier_sigma=None, n_features=128, steps=20000, batch=1024,
                 lr=2e-3, seed=0, init_state=None):
        self.hidden, self.sigma, self.nf = hidden, fourier_sigma, n_features
        self.steps, self.batch, self.lr, self.seed = steps, batch, lr, seed
        self.init_state = init_state

    def _net(self):
        from torch import nn

        d_in = 2 * self.nf if self.sigma else 4
        layers, d = [], d_in
        for h in self.hidden:
            layers += [nn.Linear(d, h), nn.SiLU()]
            d = h
        layers.append(nn.Linear(d, 2))
        return nn.Sequential(*layers)

    def _feat(self, X):
        import torch

        x = torch.tensor(scale(X), dtype=torch.float32)
        if not self.sigma:
            return x
        z = 2 * np.pi * x @ self.B
        return torch.cat([torch.sin(z), torch.cos(z)], dim=1)

    def fit(self, X, t, Xv, tv):
        import torch

        torch.manual_seed(self.seed)
        g = torch.Generator().manual_seed(self.seed)
        if self.sigma:
            self.B = torch.randn(4, self.nf, generator=g) * self.sigma
        self.net = self._net()
        if self.init_state is not None:  # transfer learning: start from pretrained weights
            self.net.load_state_dict(self.init_state["net"])
            if self.sigma:
                self.B = self.init_state["B"]
        Y = np.column_stack([t.real, t.imag]).astype(np.float32)
        self.mu, self.sd = Y.mean(0), Y.std(0) + 1e-8
        x, y = self._feat(X), torch.tensor((Y - self.mu) / self.sd)
        xv = self._feat(Xv)
        yv = torch.tensor(((np.column_stack([tv.real, tv.imag]) - self.mu) / self.sd).astype(np.float32))
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.steps, eta_min=self.lr * 1e-2)
        best, self.best_step, state = float("inf"), 0, None
        for step in range(1, self.steps + 1):
            idx = torch.randint(0, len(x), (min(self.batch, len(x)),), generator=g)
            opt.zero_grad()
            loss = torch.mean((self.net(x[idx]) - y[idx]) ** 2)
            loss.backward()
            opt.step()
            sch.step()
            if step % 250 == 0:
                with torch.no_grad():
                    vl = torch.mean((self.net(xv) - yv) ** 2).item()
                if vl < best:
                    best, self.best_step, state = vl, step, copy.deepcopy(self.net.state_dict())
        if state is None:  # every validation loss was non-finite: fail loudly instead of an UnboundLocalError
            raise FloatingPointError("MLPParam training diverged: no finite validation loss")
        self.net.load_state_dict(state)
        self.best_val = best
        return self

    def predict(self, X):
        import torch

        with torch.no_grad():
            Y = self.net(self._feat(X)).numpy() * self.sd + self.mu
        return Y[:, 0] + 1j * Y[:, 1]

    def state(self):
        return dict(net=copy.deepcopy(self.net.state_dict()), B=getattr(self, "B", None))


class PODMLP:
    """POD of per-configuration amplitude scans + MLP for the reduced coefficients.

    Snapshots: t on the training energy grid for every training configuration (the
    basis is built from training snapshots only). Coefficients are predicted from
    (V0, a, d); the predicted scan is interpolated to arbitrary E with a cubic spline.
    Error is split into compression (projection of the true test scan onto the basis)
    and coefficient-prediction parts.
    """

    name = "pod_mlp"

    def __init__(self, energy_tol=1e-10, seed=0):
        self.tol, self.seed = energy_tol, seed

    def fit_scans(self, C, E_grid, Tm, Cv, Tmv):
        import torch
        from torch import nn

        self.E = E_grid
        S = np.hstack([Tm.real, Tm.imag])  # (n_cfg, 2 n_E) real snapshots
        self.mean = S.mean(0)
        U, s, Vt = np.linalg.svd(S - self.mean, full_matrices=False)
        energy = np.cumsum(s**2) / np.sum(s**2)
        self.r = int(np.searchsorted(energy, 1 - self.tol) + 1)
        self.r = min(self.r, len(s))
        self.V = Vt[: self.r]
        coef = (S - self.mean) @ self.V.T
        coef_v = (np.hstack([Tmv.real, Tmv.imag]) - self.mean) @ self.V.T
        torch.manual_seed(self.seed)
        net = nn.Sequential(nn.Linear(3, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, self.r))
        cs = (C - LO[1:]) / (HI[1:] - LO[1:]) * 2 - 1
        cvs = (Cv - LO[1:]) / (HI[1:] - LO[1:]) * 2 - 1
        self.cmu, self.csd = coef.mean(0), coef.std(0) + 1e-12
        x, y = torch.tensor(cs, dtype=torch.float32), torch.tensor((coef - self.cmu) / self.csd, dtype=torch.float32)
        xv, yv = torch.tensor(cvs, dtype=torch.float32), torch.tensor((coef_v - self.cmu) / self.csd, dtype=torch.float32)
        opt = torch.optim.Adam(net.parameters(), lr=2e-3)
        best = (float("inf"), None)
        for step in range(1, 10001):
            opt.zero_grad()
            loss = torch.mean((net(x) - y) ** 2)
            loss.backward()
            opt.step()
            if step % 100 == 0:
                with torch.no_grad():
                    vl = torch.mean((net(xv) - yv) ** 2).item()
                if vl < best[0]:
                    best = (vl, copy.deepcopy(net.state_dict()))
        net.load_state_dict(best[1])
        self.net = net
        return self

    def _scan_from_coef(self, coef):
        S = self.mean + coef @ self.V
        n = len(self.E)
        return S[:n] + 1j * S[n:]

    def predict_config(self, cfg, E):
        import torch

        cs = (np.asarray(cfg) - LO[1:]) / (HI[1:] - LO[1:]) * 2 - 1
        with torch.no_grad():
            c = self.net(torch.tensor(cs[None], dtype=torch.float32)).numpy()[0] * self.csd + self.cmu
        scan = self._scan_from_coef(c)
        return CubicSpline(self.E, scan.real)(E) + 1j * CubicSpline(self.E, scan.imag)(E)

    def compression_config(self, t_scan, E):
        """Best possible POD reconstruction of a true scan (projection), interpolated to E."""
        S = np.concatenate([t_scan.real, t_scan.imag]) - self.mean
        return self._scan_from_coef(S @ self.V.T) if E is None else None


class HybridSMatrix:
    """Structure-exploiting surrogate: learn the smooth single-barrier S-matrix, compose exactly.

    A symmetric double barrier = barrier (V0, a) + free propagation over d + barrier.
    The single-barrier local amplitudes (t_b, r_b)(E, V0, a) are smooth (no resonances
    for these parameters); they are learned by cubic interpolation on a tensor grid
    in (E, V0, a). The double barrier follows from the Redheffer star product:
        t_loc = t_b P t_b / (1 - r_b^2 P^2),  P = exp(i k d),
    and t = t_loc exp(-i k (2a + d)). Resonances emerge from the exact denominator.
    Training data are single-barrier solver calls (counted as solver evaluations).
    """

    name = "hybrid_smatrix"

    def fit(self, nE, nV, na):
        E = np.linspace(LO[0], HI[0], nE)
        V = np.linspace(LO[1], HI[1], nV)
        A = np.linspace(LO[2], HI[2], na)
        tb = np.empty((nE, nV, na), complex)
        rb = np.empty_like(tb)
        n = 0
        for j, v in enumerate(V):
            for l, a in enumerate(A):
                s = CountingSolver(single_barrier(v, a))
                res = s(E)
                n += s.n_evals
                k = np.sqrt(E)
                # convert global amplitudes of a centred barrier to local (edge-referenced) ones
                tb[:, j, l] = res.t * np.exp(1j * k * a)
                rb[:, j, l] = res.r * np.exp(1j * k * a)
        self.n_evals = n
        mk = lambda Z: (RegularGridInterpolator((E, V, A), Z.real, method="cubic", bounds_error=False, fill_value=None),
                        RegularGridInterpolator((E, V, A), Z.imag, method="cubic", bounds_error=False, fill_value=None))
        self.ft, self.fr = mk(tb), mk(rb)
        return self

    def predict(self, X):
        E, V, a, d = X.T
        P3 = np.column_stack([E, V, a])
        t = self.ft[0](P3) + 1j * self.ft[1](P3)
        r = self.fr[0](P3) + 1j * self.fr[1](P3)
        k = np.sqrt(E)
        P = np.exp(1j * k * d)
        t_loc = t * P * t / (1 - r * r * P * P)
        return t_loc * np.exp(-1j * k * (2 * a + d))
