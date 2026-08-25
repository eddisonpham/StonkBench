"""
Feature-based metrics and visualizations for evaluating the fidelity of generated time series data.

This module provides a set of metrics and visualizations for assessing how well generated data
matches the statistical and distributional characteristics of real/original data, focusing on
feature-level properties and summary statistics.

Feature-based Metrics:
- Marginal Distribution Distance (MDD): Average Wasserstein-1 distance between marginals of real and generated data.
- Mean Distance (MD): Difference in means between real and generated data.
- Standard Deviation Distance (SDD): Difference in standard deviations.
- Skewness Distance (SD): Difference in skewness.
- Kurtosis Distance (KD): Difference in kurtosis.

Cross-Channel (Multivariate) Fidelity Metrics:
- Correlation Matrix Distance (CMD): Normalized Frobenius norm difference between real and
  synthetic channel-wise Pearson correlation matrices. Measures how well the generative model
  preserves contemporaneous linear cross-channel correlation structure.
  Reference: Hudovernik et al. (2024), "Benchmarking the Fidelity and Utility of Synthetic
  Relational Data"; Pezoulas et al. (2025), "Synthetic Data Blueprint".
- Distance Correlation Matrix Difference (dCorDiff): Normalized Frobenius norm difference
  between real and synthetic distance correlation matrices. Distance correlation (Székely,
  Rizzo & Bakirov, 2007, Annals of Statistics, 35(6):2769–2794) captures non-linear
  dependencies and equals zero iff variables are independent — unlike Pearson correlation.
  This metric complements CMD by measuring non-linear cross-channel dependence fidelity.

Visualizations:
- t-SNE visualization: 2D projection to compare overall structure of real and generated samples.
- Marginal Distribution Plot: Kernel density estimate (KDE) comparing sample distributions.

All metrics are implemented as PyTorch modules for easy integration with deep learning workflows.
"""

import os
import torch
import numpy as np
from torch import nn
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple
from sklearn.manifold import TSNE
from scipy.stats import skew, kurtosis, wasserstein_distance
from scipy.spatial.distance import pdist, squareform


def _to_2d(data: np.ndarray) -> np.ndarray:
    """
    Normalize to 2D representation:
    - (N, L) stays unchanged
    - (N, L, C) -> (N, L*C)
    """
    if data.ndim == 2:
        return data
    if data.ndim == 3:
        n, l, c = data.shape
        return data.reshape(n, l * c)
    raise ValueError(f"Expected 2D or 3D array, got shape {data.shape}")


def calculate_mdd(ori_data: np.ndarray, gen_data: np.ndarray) -> float:
    """
    Marginal Distribution Distance (MDD):
    Computes the average 1-Wasserstein distance between the marginals
    distributions of original and generated data along each time index.
    """
    ori_data = _to_2d(np.asarray(ori_data))
    gen_data = _to_2d(np.asarray(gen_data))
    assert ori_data.shape == gen_data.shape, "Real and generated data must have the same shape."

    wasserstein_values = [
        wasserstein_distance(ori_data[:, t], gen_data[:, t])
        for t in range(ori_data.shape[1])
    ]

    return float(np.mean(wasserstein_values))

def calculate_md(ori_data, gen_data):
    """Mean Distance (MD): Absolute difference between dataset mean of sample means."""
    ori_data = _to_2d(np.asarray(ori_data))
    gen_data = _to_2d(np.asarray(gen_data))
    ori_mean = np.nanmean(ori_data, axis=1)
    gen_mean = np.nanmean(gen_data, axis=1)
    mean_ori = np.nanmean(ori_mean)
    mean_gen = np.nanmean(gen_mean)
    return float(np.abs(mean_gen - mean_ori))

def calculate_sdd(ori_data, gen_data):
    """Standard Deviation Distance (SDD): Absolute difference between dataset mean of sample stds."""
    ori_data = _to_2d(np.asarray(ori_data))
    gen_data = _to_2d(np.asarray(gen_data))
    ori_std = np.nanstd(ori_data, axis=1, ddof=1)
    gen_std = np.nanstd(gen_data, axis=1, ddof=1)
    mean_ori = np.nanmean(ori_std)
    mean_gen = np.nanmean(gen_std)
    return float(np.abs(mean_gen - mean_ori))

def calculate_sd(ori_data, gen_data):
    """Skewness Distance (SD): Absolute difference between dataset mean of sample skewness."""
    ori_data = _to_2d(np.asarray(ori_data))
    gen_data = _to_2d(np.asarray(gen_data))
    ori_skew = skew(ori_data, axis=1, bias=False, nan_policy="omit")
    gen_skew = skew(gen_data, axis=1, bias=False, nan_policy="omit")
    mean_ori = np.nanmean(ori_skew)
    mean_gen = np.nanmean(gen_skew)
    return float(np.abs(mean_gen - mean_ori))

def calculate_kd(ori_data, gen_data):
    """Kurtosis Distance (KD): Absolute difference between dataset mean of sample kurtosis."""
    ori_data = _to_2d(np.asarray(ori_data))
    gen_data = _to_2d(np.asarray(gen_data))
    ori_kurt = kurtosis(ori_data, axis=1, bias=False, fisher=True, nan_policy="omit")
    gen_kurt = kurtosis(gen_data, axis=1, bias=False, fisher=True, nan_policy="omit")
    mean_ori = np.nanmean(ori_kurt)
    mean_gen = np.nanmean(gen_kurt)
    return float(np.abs(mean_gen - mean_ori))

def visualize_tsne(ori_data, gen_data, result_path):
    ori_data = _to_2d(np.asarray(ori_data))
    gen_data = _to_2d(np.asarray(gen_data))
    sns.set(style="whitegrid", context="paper", font_scale=1.2)

    sample_no = len(ori_data)
    if sample_no < 2:
        print("[WARN] t-SNE skipped: need at least 2 samples.")
        return
    idx = np.random.permutation(len(ori_data))[:sample_no]
    ori_data = ori_data[idx]
    gen_data = gen_data[idx]

    def extract_features(data):
        features = []
        for sample in data:
            series = sample
            feats = [
                np.mean(series),
                np.std(series),
                skew(series),
                kurtosis(series)
            ]
            features.append(feats)
        return np.array(features)

    ori_features = extract_features(ori_data)
    gen_features = extract_features(gen_data)

    prep_data_final = np.concatenate([ori_features, gen_features], axis=0)

    # Perplexity must satisfy 1 < perplexity < n_samples. Small eval sets
    # (e.g. seq_252 with ~18 aligned windows, DL artifacts with ~9 samples)
    # would otherwise crash sklearn's t-SNE with ``perplexity (40) must be
    # less than n_samples`` — adapt it to the actual sample count.
    perplexity = min(40, max(2, prep_data_final.shape[0] - 1))
    tsne = TSNE(n_components=2, verbose=0, perplexity=perplexity, max_iter=500, random_state=42)
    tsne_results = tsne.fit_transform(prep_data_final)

    plt.figure(figsize=(6,6))
    plt.scatter(tsne_results[:sample_no,0], tsne_results[:sample_no,1], 
                c='#1f77b4', alpha=0.7, s=40, label="Original", edgecolor='k', linewidth=0.2)
    plt.scatter(tsne_results[sample_no:,0], tsne_results[sample_no:,1], 
                c='#ff7f0e', alpha=0.7, s=40, label="Generated", edgecolor='k', linewidth=0.2)

    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")
    plt.title("t-SNE Feature Visualization", fontsize=14)
    plt.legend(frameon=False)
    plt.grid(False)
    plt.tight_layout()

    os.makedirs(result_path, exist_ok=True)
    plt.savefig(os.path.join(result_path, 'tsne.png'), dpi=400, bbox_inches='tight')
    plt.close()

def _corr_matrix(data: np.ndarray) -> np.ndarray:
    """Compute C x C Pearson correlation matrix from multivariate samples.

    Args:
        data: (N, C) array where N samples, C channels.

    Returns:
        (C, C) correlation matrix.
    """
    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError(f"Expected 2D (N, C) input, got shape {data.shape}")
    if data.shape[1] == 1:
        return np.eye(1)
    return np.corrcoef(data.T)


def _distance_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Distance correlation between two vectors x and y.

    Székely, Rizzo & Bakirov (2007), Annals of Statistics, 35(6):2769–2794.
    Distance correlation is zero iff x and y are independent.

    Args:
        x, y: 1D arrays of equal length.

    Returns:
        Scalar in [0, 1].
    """
    x = np.asarray(x, dtype=float).flatten()
    y = np.asarray(y, dtype=float).flatten()
    n = len(x)
    if n < 4:
        return 0.0

    a = squareform(pdist(x[:, None], metric="euclidean"))
    b = squareform(pdist(y[:, None], metric="euclidean"))

    A = a - a.mean(axis=0, keepdims=True) - a.mean(axis=1, keepdims=True) + a.mean()
    B = b - b.mean(axis=0, keepdims=True) - b.mean(axis=1, keepdims=True) + b.mean()

    dCov2 = (A * B).sum() / (n * n)
    dVarX2 = (A * A).sum() / (n * n)
    dVarY2 = (B * B).sum() / (n * n)

    denom = np.sqrt(dVarX2 * dVarY2)
    if denom < 1e-15:
        return 0.0
    return float(np.sqrt(dCov2) / np.sqrt(denom))


def _dcor_matrix(data: np.ndarray) -> np.ndarray:
    """Compute C x C distance correlation matrix from (N, C) samples."""
    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError(f"Expected 2D (N, C) input, got shape {data.shape}")
    C = data.shape[1]
    dcor = np.eye(C)
    for i in range(C):
        for j in range(i + 1, C):
            val = _distance_corr(data[:, i], data[:, j])
            dcor[i, j] = val
            dcor[j, i] = val
    return dcor


def _frobenius_norm(A: np.ndarray) -> float:
    """Frobenius norm of a matrix."""
    return float(np.sqrt(np.sum(np.square(A))))


def calculate_cmd(
    ori_data: np.ndarray,
    gen_data: np.ndarray,
) -> float:
    """Correlation Matrix Distance (CMD) — Frobenius norm difference of Pearson correlation matrices.

    Compares the channel-wise correlation structure of real vs synthetic data.
    Lower values indicate better cross-channel linear dependency fidelity.

    For 3D input (N, L, C): per-sample per-channel means produce (N, C), then a
    (C, C) Pearson correlation matrix is computed and compared via Frobenius norm.
    For 2D input (N, C): used directly.

    Reference: Hudovernik et al. (2024) and Pezoulas et al. (2025).

    Args:
        ori_data: Real data, (N, L, C) or (N, C).
        gen_data: Synthetic data, (N, L, C) or (N, C).

    Returns:
        CMD ∈ [0, 2] where 0 means perfect match.
    """
    ori = np.asarray(ori_data)
    gen = np.asarray(gen_data)

    # Collapse time dimension: per-sample per-channel mean → (N, C)
    if ori.ndim == 3:
        ori_2d = ori.mean(axis=1)
    else:
        ori_2d = ori
    if gen.ndim == 3:
        gen_2d = gen.mean(axis=1)
    else:
        gen_2d = gen

    C = ori_2d.shape[1]
    if C < 2:
        return 0.0  # Single channel: no cross-channel structure to compare.

    C_real = _corr_matrix(ori_2d.T)  # corrcoef expects (C, N)
    C_syn = _corr_matrix(gen_2d.T)
    diff = C_real - C_syn
    return _frobenius_norm(diff) / max(_frobenius_norm(C_real), 1e-15)


def calculate_dcor_diff(
    ori_data: np.ndarray,
    gen_data: np.ndarray,
    max_samples: int = 500,
) -> float:
    """Distance Correlation Matrix Difference (dCorDiff) — Frobenius norm difference of dCor matrices.

    Distance correlation (Székely et al. 2007) captures arbitrary non-linear dependencies
    between channels and equals zero iff channels are independent. This complements CMD
    by measuring how faithfully the synthetic data preserves the non-linear cross-channel
    dependence structure.

    For 3D input (N, L, C): per-sample per-channel means produce (N, C), then a
    (C, C) distance correlation matrix is computed and compared to the real matrix.
    The O(n²) dCor computation is capped at ``max_samples`` per dataset.

    Args:
        ori_data: Real data, (N, L, C) or (N, C).
        gen_data: Synthetic data, (N, L, C) or (N, C).
        max_samples: Cap on samples for dCor computation (O(n²) complexity).

    Returns:
        dCorDiff ∈ [0, 2] where 0 means perfect match.
    """
    ori = np.asarray(ori_data)
    gen = np.asarray(gen_data)

    # Collapse time dimension: per-sample per-channel mean → (N, C)
    if ori.ndim == 3:
        ori_2d = ori.mean(axis=1)
    else:
        ori_2d = ori
    if gen.ndim == 3:
        gen_2d = gen.mean(axis=1)
    else:
        gen_2d = gen

    C = ori_2d.shape[1]
    if C < 2:
        return 0.0  # Single channel: no cross-channel structure to compare.

    # Subsample for dCor O(n²) complexity
    n_ori = min(ori_2d.shape[0], max_samples)
    n_gen = min(gen_2d.shape[0], max_samples)
    if ori_2d.shape[0] > n_ori:
        rng = np.random.RandomState(42)
        ori_2d = ori_2d[rng.choice(ori_2d.shape[0], n_ori, replace=False)]
    if gen_2d.shape[0] > n_gen:
        rng = np.random.RandomState(42)
        gen_2d = gen_2d[rng.choice(gen_2d.shape[0], n_gen, replace=False)]

    D_real = _dcor_matrix(ori_2d)
    D_syn = _dcor_matrix(gen_2d)
    diff = D_real - D_syn
    return _frobenius_norm(diff) / max(_frobenius_norm(D_real), 1e-15)


def visualize_distribution(ori_data, gen_data, result_path):
    ori_data = _to_2d(np.asarray(ori_data))
    gen_data = _to_2d(np.asarray(gen_data))
    ori_flat = ori_data.flatten()
    gen_flat = gen_data.flatten()
    ori_min, ori_max = np.min(ori_flat), np.max(ori_flat)
    gen_min, gen_max = np.min(gen_flat), np.max(gen_flat)
    print(f"Original range: [{ori_min:.4f}, {ori_max:.4f}]  Generated range: [{gen_min:.4f}, {gen_max:.4f}]")
    sns.set(style="whitegrid", context="paper", font_scale=1.2)
    plt.figure(figsize=(5, 6))
    colors = ['#1f77b4', '#ff7f0e']
    linestyles = ['-', '--']

    ori_flat = ori_data.flatten()
    gen_flat = gen_data.flatten()

    sns.kdeplot(
        y=ori_flat, color=colors[0], linewidth=2, linestyle=linestyles[0],
        label='Original', fill=True, alpha=0.5
    )
    sns.kdeplot(
        y=gen_flat, color=colors[1], linewidth=2, linestyle=linestyles[1],
        label='Generated', fill=True, alpha=0.5
    )

    plt.xlabel("Density")
    plt.ylabel("Value")
    plt.title("Distribution Comparison", fontsize=12)
    plt.legend(frameon=False, fontsize=10)
    plt.grid(False)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    os.makedirs(result_path, exist_ok=True)
    plt.savefig(
        os.path.join(result_path, 'distribution.png'),
        dpi=400, bbox_inches='tight'
    )
    plt.close()


def _channel_qq(ori: np.ndarray, gen: np.ndarray, channel: int) -> tuple[np.ndarray, np.ndarray, list] | None:
    """Compute common-quantile QQ coordinates for one channel.

    Returns None when either side is empty (nothing to compare).
    """
    real_flat = ori[:, :, channel].flatten()
    gen_flat = gen[:, :, channel].flatten()
    if real_flat.size == 0 or gen_flat.size == 0:
        return None
    n_quantiles = min(1000, min(len(real_flat), len(gen_flat)))
    probs = np.linspace(0.01, 0.99, n_quantiles)
    real_q = np.quantile(real_flat, probs)
    gen_q = np.quantile(gen_flat, probs)
    lims = [min(real_q.min(), gen_q.min()), max(real_q.max(), gen_q.max())]
    if lims[0] == lims[1]:
        lims = [lims[0] - 1.0, lims[0] + 1.0]  # degenerate constant channel
    return real_q, gen_q, lims


def visualize_qq(
    ori_data: np.ndarray,
    gen_data: np.ndarray,
    result_path: str,
    channel_names: list | None = None,
    per_asset_dir: str | None = None,
) -> None:
    """QQ (Quantile-Quantile) plot comparing real vs generated distributions per channel.

    For each channel, plots the quantiles of the generated data against the
    quantiles of the real data. A perfect match follows the 45° line.
    When ``per_asset_dir`` is given, one standalone QQ plot per channel (asset)
    is additionally written there as ``qq_<channel>.png``.

    Args:
        ori_data: Real data, (N, L) or (N, L, C).
        gen_data: Generated data, (N, L) or (N, L, C).
        result_path: Directory to save the combined grid plot.
        channel_names: Optional list of channel names for titles.
        per_asset_dir: Optional directory for standalone per-asset QQ plots.
    """
    ori = np.asarray(ori_data)
    gen = np.asarray(gen_data)

    if ori.ndim == 2:
        ori = ori[:, :, np.newaxis]
        gen = gen[:, :, np.newaxis]

    C = ori.shape[2]
    if channel_names is None:
        channel_names = [f"Channel {c}" for c in range(C)]

    ncols = min(C, 5)
    nrows = int(np.ceil(C / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows), dpi=150)
    if nrows * ncols == 1:
        axes = np.array([[axes]])
    axes = np.atleast_2d(axes)

    for c in range(C):
        row, col = divmod(c, ncols)
        qq = _channel_qq(ori, gen, c)
        if qq is None:
            axes[row, col].set_visible(False)
            continue
        real_q, gen_q, lims = qq

        ax = axes[row, col]
        ax.scatter(real_q, gen_q, s=2, alpha=0.5, c="#1f77b4", edgecolors="none")
        ax.plot(lims, lims, "r--", linewidth=1, alpha=0.7, label="45° line")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("Real Quantiles")
        ax.set_ylabel("Generated Quantiles")
        ax.set_title(channel_names[c], fontsize=10)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for c in range(C, nrows * ncols):
        row, col = divmod(c, ncols)
        axes[row, col].set_visible(False)

    plt.tight_layout()
    os.makedirs(result_path, exist_ok=True)
    plt.savefig(os.path.join(result_path, "qq.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Standalone per-asset QQ plots.
    if per_asset_dir is not None:
        os.makedirs(per_asset_dir, exist_ok=True)
        for c in range(C):
            qq = _channel_qq(ori, gen, c)
            if qq is None:
                continue
            real_q, gen_q, lims = qq
            name = channel_names[c] if c < len(channel_names) else f"Channel_{c}"
            safe_name = str(name).replace("/", "_").replace(" ", "_")
            fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
            ax.scatter(real_q, gen_q, s=6, alpha=0.6, c="#1f77b4", edgecolors="none")
            ax.plot(lims, lims, "r--", linewidth=1.2, alpha=0.7, label="45° line")
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.set_xlabel("Real Quantiles")
            ax.set_ylabel("Generated Quantiles")
            ax.set_title(f"QQ Plot: {name}", fontsize=11)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(
                os.path.join(per_asset_dir, f"qq_{c:02d}_{safe_name}.png"),
                dpi=300, bbox_inches="tight")
            plt.close()


def visualize_per_channel(
    ori_data: np.ndarray,
    gen_data: np.ndarray,
    result_path: str,
    channel_names: list | None = None,
) -> None:
    """Per-channel distribution overlay (KDE) for each asset independently.

    Args:
        ori_data: Real data, (N, L) or (N, L, C).
        gen_data: Generated data, (N, L) or (N, L, C).
        result_path: Directory to save plots.
        channel_names: Optional list of channel names for titles.
    """
    ori = np.asarray(ori_data)
    gen = np.asarray(gen_data)

    if ori.ndim == 2:
        ori = ori[:, :, np.newaxis]
        gen = gen[:, :, np.newaxis]

    C = ori.shape[2]
    if channel_names is None:
        channel_names = [f"Channel {c}" for c in range(C)]

    ncols = min(C, 5)
    nrows = int(np.ceil(C / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows), dpi=150)
    if nrows * ncols == 1:
        axes = np.array([[axes]])
    axes = np.atleast_2d(axes)

    for c in range(C):
        row, col = divmod(c, ncols)
        ax = axes[row, col]

        real_flat = ori[:, :, c].flatten()
        gen_flat = gen[:, :, c].flatten()

        ax.hist(real_flat, bins=50, density=True, alpha=0.5, color="#1f77b4", label="Real")
        ax.hist(gen_flat, bins=50, density=True, alpha=0.5, color="#ff7f0e", label="Generated")
        ax.set_title(channel_names[c], fontsize=9)
        ax.legend(frameon=False, fontsize=7)

    for c in range(C, nrows * ncols):
        row, col = divmod(c, ncols)
        axes[row, col].set_visible(False)

    plt.tight_layout()
    os.makedirs(result_path, exist_ok=True)
    plt.savefig(os.path.join(result_path, "per_channel_dist.png"), dpi=300, bbox_inches="tight")
    plt.close()
