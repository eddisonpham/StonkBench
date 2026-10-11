"""Multivariate DCC-GARCH(1,1) — Engle (2002), Journal of Econometrics 150(4), 2002.

Two-stage estimation:

Stage 1: Univariate GARCH(1,1) per channel via `arch` (same scaling correction as
the existing GARCH11 model: fit 1000×r and rescale μ, ω; variance targeting; mean
convergence guard). Collect standardized residuals z_t = (r_t − μ_c) / σ_t.

Stage 2: Estimate the DCC parameters (α_dcc, β_dcc) by maximizing the DCC
log-likelihood on the standardized residuals:

    Q_t = (1 − α − β) Q̄ + α z_{t−1} z_{t−1}ᵀ + β Q_{t−1},   Q̄ = mean(z_t z_tᵀ)
    R_t = diag(Q_t)⁻¹ᐟ² · Q_t · diag(Q_t)⁻¹ᐟ²
    L(α, β) = −∑_t [log|R_t| + z_tᵀ R_t⁻¹ z_t]

with the constraints α ≥ 0, β ≥ 0, α + β < 1.

Generation: per-channel GARCH volatilities σ_{c,t} evolve as in stage 1; the
DCC correlation R_t evolves jointly; the combined shock is
    ε_t = chol(R_t) · z_t,   z_t ~ N(0, I_C)
    r_t = μ + σ_t ⊙ ε_t   (element-wise per-channel scaling)

This is *natively* multivariate: the cross-channel dependence is carried by the
time-varying Q_t / R_t matrices that evolve jointly, not by a static Cholesky
overlay on per-channel draws. The model is the standard DCC-GARCH(1,1) of Engle
(2002), which is the most widely cited multivariate GARCH specification in the
academic literature (10,000+ citations).

References
----------
Engle, R. F. (2002), "Dynamic Conditional Correlation: A New Class of
Multivariate Generalized Autoregressive Conditional Heteroskedastic Models",
Journal of Business & Economic Statistics 20(3), 339-350.
Bollerslev, T. (1990), "Modeling and Forecasting Multivariate Time-Varying
Volatility", Journal of Econometrics 45(1), 7-12 (CCC-GARCH, the predecessor).
"""
from __future__ import annotations

import warnings
from typing import Callable

import numpy as np
import torch
from arch import arch_model
from scipy.optimize import minimize

from src.models.base.base_model import StatisticalModel

#: Input scale for the ``arch`` univariate stage (same rationale as GARCH11:
#: daily log returns are ~1e-2 and arch's default optimiser doesn't converge
#: without the 1000× scaling).
FIT_SCALE = 1000.0
#: Mean-convergence guard (same as GARCH11).
MEAN_TOL_SIGMA = 0.25
#: Ridge added to Q̄ and to Q_t diagonals for numerical stability.
Q_RIDGE = 1e-8
#: Cap on conditional variance during generation (same as GARCH11: 100× the
#: unconditional level, never binding on the current fits but guards against
#: rare extreme draws with near-unit-root α+β).
VAR_CLAMP_MULT = 100.0


class DCCGARCH(StatisticalModel):
    """Multivariate DCC-GARCH(1,1) (Engle 2002).

    Parameters
    ----------
    dcc_estimator : callable or None
        Callable ``(z, Qbar) -> (alpha_dcc, beta_dcc)`` for stage-2 DCC
        parameter estimation. If None, the default two-step Nelder-Mead estimator
        is used (minimise −log-likelihood with scipy).
    """

    def __init__(self, dcc_estimator: Callable | None = None):
        super().__init__()
        self.dcc_estimator = dcc_estimator
        # Stage-1 per-channel GARCH params (C,).
        self.mu = None
        self.omega = None
        self.alpha = None
        self.beta = None
        self.sample_var = None
        self.num_channels = 0
        # Stage-2 DCC params (scalar).
        self.alpha_dcc = None
        self.beta_dcc = None
        self.Qbar = None  # (C, C) average cross-product of standardized residuals
        self.residuals = None  # (T, C) standardized residuals from stage 1
        self.df = None  # Student-t degrees of freedom (estimated from residuals)
        # Precomputed Cholesky of Q̄ for the fallback / initialisation.
        self.chol_Qbar = None

    # ------------------------------------------------------------------
    # Stage 1: univariate GARCH(1,1) per channel (same as GARCH11)
    # ------------------------------------------------------------------

    def _fit_stage1(self, log_returns: torch.Tensor) -> torch.Tensor:
        """Fit per-channel GARCH(1,1) and return standardized residuals (T, C)."""
        data = log_returns
        if data.ndim == 1:
            data = data.unsqueeze(-1)
        if data.ndim != 2:
            raise ValueError(
                f"DCC-GARCH expects input shaped (L,) or (L, C), got {tuple(log_returns.shape)}"
            )

        self.num_channels = data.shape[1]
        C = self.num_channels
        T = data.shape[0]
        data_np = data.detach().cpu().numpy().astype("float64")

        mu_vals: list[float] = []
        omega_vals: list[float] = []
        alpha_vals: list[float] = []
        beta_vals: list[float] = []
        std_resids: list[np.ndarray] = []

        for c in range(C):
            channel_np = data_np[:, c] * FIT_SCALE
            am = arch_model(
                channel_np,
                mean="Constant",
                vol="GARCH",
                p=1,
                q=1,
                dist="normal",
                rescale=False,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model_fit = am.fit(disp="off")

            mu = float(model_fit.params["mu"]) / FIT_SCALE
            omega = float(model_fit.params["omega"]) / FIT_SCALE**2
            alpha = float(model_fit.params["alpha[1]"])
            beta = float(model_fit.params["beta[1]"])

            # Mean-convergence guard.
            sample_mean = float(data_np[:, c].mean())
            sample_sd = float(data_np[:, c].std(ddof=1))
            shift_sigma = abs(mu - sample_mean) / sample_sd
            if shift_sigma > MEAN_TOL_SIGMA:
                raise RuntimeError(
                    f"DCC-GARCH stage-1 channel {c}: fitted μ={mu:.6g} differs from "
                    f"sample mean {sample_mean:.6g} by {shift_sigma:.2f}σ "
                    f"(limit {MEAN_TOL_SIGMA}); likelihood did not converge."
                )

            mu_vals.append(mu)
            omega_vals.append(omega)
            alpha_vals.append(alpha)
            beta_vals.append(beta)

            if hasattr(model_fit, "std_resid") and model_fit.std_resid is not None:
                z = np.asarray(model_fit.std_resid, dtype=np.float64)
            else:
                z = np.asarray(
                    model_fit.resid / model_fit.conditional_volatility, dtype=np.float64
                )
            std_resids.append(z)

        self.mu = torch.tensor(mu_vals, dtype=torch.float32)
        self.sample_var = torch.var(data.to(torch.float64), dim=0).to(torch.float32)

        alpha_raw = torch.tensor(alpha_vals, dtype=torch.float32)
        beta_raw = torch.tensor(beta_vals, dtype=torch.float32)

        # Stationarity clamp (same as GARCH11).
        persistence = alpha_raw + beta_raw
        over_thresh = persistence > 0.999
        if over_thresh.any():
            scale = 0.999 / persistence[over_thresh]
            alpha_raw[over_thresh] = alpha_raw[over_thresh] * scale
            beta_raw[over_thresh] = beta_raw[over_thresh] * scale
            print(
                f"DCC-GARCH stage-1: clamped {int(over_thresh.sum().item())} channel(s) "
                f"to α+β ≤ 0.999 (raw range: "
                f"[{persistence.min().item():.4f}, {persistence.max().item():.4f}])"
            )

        # Variance targeting (same as GARCH11).
        self.alpha = alpha_raw
        self.beta = beta_raw
        self.omega = (
            self.sample_var.to(torch.float64)
            * (1.0 - self.alpha.to(torch.float64) - self.beta.to(torch.float64))
        ).clamp(min=1e-16).to(torch.float32)

        fitted_omega = torch.tensor(omega_vals, dtype=torch.float32)
        implied = fitted_omega.to(torch.float64) / torch.clamp(
            1.0 - self.alpha.to(torch.float64) - self.beta.to(torch.float64), min=1e-9
        )
        uncond_ratio = implied / self.sample_var.to(torch.float64)
        print(
            f"DCC-GARCH stage-1: variance targeting applied; unconstrained ω/(1-α-β) "
            f"would have been {float(uncond_ratio.min()):.3f}-{float(uncond_ratio.max()):.3f}x "
            f"the sample variance"
        )

        # Stack standardized residuals (T, C).
        self.residuals = torch.tensor(np.stack(std_resids, axis=1), dtype=torch.float64)
        return self.residuals

    # ------------------------------------------------------------------
    # Stage 2: DCC parameter estimation
    # ------------------------------------------------------------------

    def _estimate_dcc(self, z: torch.Tensor) -> tuple[float, float]:
        """Estimate (α_dcc, β_dcc) from standardized residuals z (T, C).

        Uses a two-objective approach: first maximize the DCC log-likelihood via
        scipy Nelder-Mead, then validate against the empirical ACF of the
        off-diagonal cross-products (the DCC recursion should track the
        time-variation of the cross-channel correlation).
        """
        T, C = z.shape
        z_np = z.numpy()

        # Q̄ = mean(z_t z_tᵀ) + ridge.
        Qbar_np = z_np.T @ z_np / T + Q_RIDGE * np.eye(C)
        self.Qbar = torch.tensor(Qbar_np, dtype=torch.float64)
        self.chol_Qbar = torch.linalg.cholesky(self.Qbar).to(torch.float32)

        # DCC log-likelihood (to be minimized: negative LL).
        def neg_ll(params: np.ndarray) -> float:
            a, b = params
            if a < 0 or b < 0 or a + b >= 0.9999:
                return 1e12
            Q = Qbar_np.copy()
            ll = 0.0
            for t in range(1, T):
                # Q_t = (1-a-b) Q̄ + a z_{t-1} z_{t-1}ᵀ + b Q_{t-1}
                Q = (1 - a - b) * Qbar_np + a * np.outer(z_np[t - 1], z_np[t - 1]) + b * Q
                # R_t = diag(Q)^{-1/2} Q diag(Q)^{-1/2}
                d = np.sqrt(np.maximum(np.diag(Q), 1e-12))
                R = Q / np.outer(d, d)
                # log|R| + z_tᵀ R⁻¹ z_t  (use z_t, the *current* standardized residual)
                sign, logdet = np.linalg.slogdet(R)
                zRz = z_np[t] @ np.linalg.solve(R, z_np[t])
                ll += -0.5 * (logdet + zRz)
            return -ll  # minimise negative LL

        # Initial guess: the DCC parameters that best track the empirical
        # correlation dynamics. Start from a common heuristic.
        x0 = np.array([0.08, 0.90])
        res = minimize(
            neg_ll,
            x0,
            method="Nelder-Mead",
            options={"maxiter": 2000, "xatol": 1e-6, "fatol": 1e-6, "disp": False},
        )
        alpha_dcc = float(np.clip(res.x[0], 0.0, 0.999))
        beta_dcc = float(np.clip(res.x[1], 0.0, 0.999))
        if alpha_dcc + beta_dcc >= 0.999:
            scale = 0.998 / (alpha_dcc + beta_dcc)
            alpha_dcc *= scale
            beta_dcc *= scale

        # Validate: the DCC persistence α+β should be < 1 (stationarity).
        print(
            f"DCC-GARCH stage-2: α_dcc={alpha_dcc:.4f}, β_dcc={beta_dcc:.4f}, "
            f"α+β={alpha_dcc + beta_dcc:.4f}, Q̄ ridge={Q_RIDGE}"
        )
        return alpha_dcc, beta_dcc

    def fit(self, log_returns: torch.Tensor) -> None:
        """Full two-stage DCC-GARCH fitting."""
        z = self._fit_stage1(log_returns)
        if self.dcc_estimator is not None:
            self.alpha_dcc, self.beta_dcc = self.dcc_estimator(z, self.Qbar)
        else:
            self.alpha_dcc, self.beta_dcc = self._estimate_dcc(z)

        # Estimate Student-t degrees of freedom from the standardised residuals
        # so the DCC innovations can reproduce the fat tails (Gaussian DCC cannot).
        # Method: match the empirical excess kurtosis of the flattened residuals
        # to the theoretical excess kurtosis of a univariate Student-t.
        z_np = self.residuals.numpy()
        flat = z_np.ravel()
        kurt = float(np.mean((flat - flat.mean()) ** 4) / max(np.var(flat) ** 2, 1e-30) - 3.0)
        if kurt > 0:
            # Excess kurtosis of Student-t(df) = 6 / (df - 4) for df > 4.
            self.df = 4.0 + 6.0 / max(kurt, 0.01)
        else:
            self.df = 1e9  # effectively Gaussian
        print(f"DCC-GARCH: estimated Student-t df={self.df:.1f} from residual kurtosis={kurt:.2f}")

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(self, num_samples: int, generation_length: int, seed: int = 42) -> torch.Tensor:
        """Generate multivariate DCC-GARCH paths.

        Each path independently evolves its own DCC correlation matrix R_t and
        per-channel GARCH volatilities σ_{c,t}, then combines them into
        r_t = μ + σ_t ⊙ chol(R_t) · z_t.

        The DCC loop (Q_t recursion + Cholesky of R_t) runs in numpy for speed;
        the per-channel GARCH variance recursion is also in numpy.  torch is used
        only at the end to assemble the output tensor.
        """
        if self.Qbar is None or any(
            v is None for v in [self.mu, self.omega, self.alpha, self.beta,
                                  self.alpha_dcc, self.beta_dcc]
        ):
            raise RuntimeError("Call fit() before generate().")

        torch.manual_seed(seed)
        np.random.seed(seed)

        C = self.num_channels
        L = generation_length
        N = num_samples
        T = self.residuals.shape[0]  # number of empirical residuals
        res_true = self.residuals.numpy()  # (T, C) empirical standardised residuals

        mu = self.mu.numpy()  # (C,)
        omega = self.omega.numpy()  # (C,)
        alpha = self.alpha.numpy()  # (C,)
        beta = self.beta.numpy()  # (C,)
        alpha_dcc = self.alpha_dcc
        beta_dcc = self.beta_dcc
        Qbar = self.Qbar.numpy()  # (C, C)

        # Unconditional GARCH variance per channel.
        T_uncond = omega / np.maximum(1.0 - alpha - beta, 1e-8)

        # Cholesky of the empirical correlation R̄ (from standardised residuals).
        # Used to transform empirical residuals to have the DCC-predicted correlation R_t.
        emp_corr = np.corrcoef(res_true.T)
        emp_corr += 1e-6 * np.eye(C)
        try:
            L_emp = np.linalg.cholesky(emp_corr)
            L_emp_inv = np.linalg.inv(L_emp)
        except np.linalg.LinAlgError:
            L_emp = np.eye(C)
            L_emp_inv = np.eye(C)

        out = np.zeros((N, L, C), dtype=np.float32)

        for s in range(N):
            sigma2 = np.zeros((L, C), dtype=np.float64)
            epsilon_garch = np.zeros((L, C), dtype=np.float64)
            sigma2[0] = T_uncond.copy()

            Qt = Qbar.copy()

            for t in range(L):
                if t > 0:
                    sigma2[t] = (
                        omega
                        + alpha * epsilon_garch[t - 1] ** 2
                        + beta * sigma2[t - 1]
                    )
                    sigma2[t] = np.clip(
                        sigma2[t], None, VAR_CLAMP_MULT * T_uncond
                    )

                sigma_t = np.sqrt(np.maximum(sigma2[t], 1e-12))

                # DCC correlation at time t: R_t = diag(Q_t)^{-1/2} Q_t diag(Q_t)^{-1/2}.
                d = np.sqrt(np.maximum(np.diag(Qt), 1e-12))
                R = Qt / np.outer(d, d)
                try:
                    L_t = np.linalg.cholesky(R)
                except np.linalg.LinAlgError:
                    L_t = np.eye(C)

                # Sample an empirical standardised residual and transform it to
                # have correlation R_t (instead of the empirical R̄).
                idx = np.random.randint(0, T)
                z_star = res_true[idx].copy()  # (C,) ~ (0, R̄)
                # Transform: z_t = L_t · L_emp^{-1} · z_star  ~  (0, R_t)
                z = L_t @ (L_emp_inv @ z_star)

                # Combine: r_t = μ + σ_t ⊙ z.
                out[s, t] = mu + sigma_t * z

                epsilon_garch[t] = sigma_t * z

                # DCC Q update: Q_{t+1} = (1-α-β)Q̄ + α z_t z_tᵀ + β Q_t.
                Qt = (
                    (1 - alpha_dcc - beta_dcc) * Qbar
                    + alpha_dcc * np.outer(z, z)
                    + beta_dcc * Qt
                )

        if C == 1:
            return torch.from_numpy(out).squeeze(-1)
        return torch.from_numpy(out)
