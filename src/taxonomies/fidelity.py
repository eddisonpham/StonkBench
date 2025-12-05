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
- Autocorrelation Distance (ACD): Difference in autocorrelation structure.

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


def calculate_mdd(ori_data: np.ndarray, gen_data: np.ndarray) -> float:
    """
    Marginal Distribution Distance (MDD):
    Computes the average 1-Wasserstein distance between the marginals
    distributions of original and generated data along each time index.
    """
    assert ori_data.shape == gen_data.shape, "Real and generated data must have the same shape."

    wasserstein_values = [
        wasserstein_distance(ori_data[:, t], gen_data[:, t])
        for t in range(ori_data.shape[1])
    ]

    return float(np.mean(wasserstein_values))

def calculate_md(ori_data, gen_data):
    """Mean Distance (MD): Absolute difference between dataset mean of sample means."""
    ori_mean = np.nanmean(ori_data, axis=1)
    gen_mean = np.nanmean(gen_data, axis=1)
    mean_ori = np.nanmean(ori_mean)
    mean_gen = np.nanmean(gen_mean)
    return float(np.abs(mean_gen - mean_ori))

def calculate_sdd(ori_data, gen_data):
    """Standard Deviation Distance (SDD): Absolute difference between dataset mean of sample stds."""
    ori_std = np.nanstd(ori_data, axis=1, ddof=1)
    gen_std = np.nanstd(gen_data, axis=1, ddof=1)
    mean_ori = np.nanmean(ori_std)
    mean_gen = np.nanmean(gen_std)
    return float(np.abs(mean_gen - mean_ori))

def calculate_sd(ori_data, gen_data):
    """Skewness Distance (SD): Absolute difference between dataset mean of sample skewness."""
    ori_skew = skew(ori_data, axis=1, bias=False, nan_policy="omit")
    gen_skew = skew(gen_data, axis=1, bias=False, nan_policy="omit")
    mean_ori = np.nanmean(ori_skew)
    mean_gen = np.nanmean(gen_skew)
    return float(np.abs(mean_gen - mean_ori))

def calculate_kd(ori_data, gen_data):
    """Kurtosis Distance (KD): Absolute difference between dataset mean of sample kurtosis."""
    ori_kurt = kurtosis(ori_data, axis=1, bias=False, fisher=True, nan_policy="omit")
    gen_kurt = kurtosis(gen_data, axis=1, bias=False, fisher=True, nan_policy="omit")
    mean_ori = np.nanmean(ori_kurt)
    mean_gen = np.nanmean(gen_kurt)
    return float(np.abs(mean_gen - mean_ori))

def visualize_tsne(ori_data, gen_data, result_path):
    sns.set(style="whitegrid", context="paper", font_scale=1.2)

    sample_no = len(ori_data)
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

    tsne = TSNE(n_components=2, verbose=0, perplexity=40, max_iter=500, random_state=42)
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

def visualize_distribution(ori_data, gen_data, result_path):
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
