"""
Feature-based metrics and visualizations for evaluating the fidelity of generated time series data.

This module provides a set of metrics and visualizations for assessing how well generated data
matches the statistical and distributional characteristics of real/original data, focusing on
feature-level properties and summary statistics.

Feature-based Metrics:
- Marginal Distribution Distance (MDD): Histogram-based distance between real and generated data.
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
from sklearn.manifold import TSNE
from scipy.stats import skew, kurtosis

from src.utils.math_utils import (
    histogram_torch,
    acf_torch,
    non_stationary_acf_torch,
    skew_torch,
    kurtosis_torch,
    acf_diff,
)


class Loss(nn.Module):
    def __init__(self, reg=1.0, transform=lambda x: x, threshold=10., backward=False, norm_foo=lambda x: x):
        super(Loss, self).__init__()
        self.reg = reg
        self.transform = transform
        self.threshold = threshold
        self.backward = backward
        self.norm_foo = norm_foo

    def forward(self, x_fake):
        self.loss_componentwise = self.compute(x_fake)
        return self.reg * self.loss_componentwise.mean()

    def compute(self, x_fake):
        raise NotImplementedError()

    @property
    def success(self):
        return torch.all(self.loss_componentwise <= self.threshold)

# =======================================
# Marginal Distribution Distance (MDD)
class HistoLoss(Loss):
    def __init__(self, x_real, n_bins, **kwargs):
        super(HistoLoss, self).__init__(**kwargs)
        self.densities = list()
        self.locs = list()
        self.deltas = list()
        for i in range(x_real.shape[2]):
            tmp_densities = list()
            tmp_locs = list()
            tmp_deltas = list()
            for t in range(x_real.shape[1]):
                x_ti = x_real[:, t, i].reshape(-1, 1)
                d, b = histogram_torch(x_ti, n_bins, density=True)
                tmp_densities.append(nn.Parameter(d).to(x_real.device))
                delta = b[1:2] - b[:1]
                loc = 0.5 * (b[1:] + b[:-1])
                tmp_locs.append(loc)
                tmp_deltas.append(delta)
            self.densities.append(tmp_densities)
            self.locs.append(tmp_locs)
            self.deltas.append(tmp_deltas)

    def compute(self, x_fake):
        def relu(x):
            return x * (x >= 0.).float()

        loss_per_channel = []
        for i in range(x_fake.shape[2]):
            tmp_loss = []
            for t in range(x_fake.shape[1]):
                loc = self.locs[i][t].view(1, -1).to(x_fake.device)
                x_ti = x_fake[:, t, i].contiguous().view(-1, 1).repeat(1, loc.shape[1])
                dist = torch.abs(x_ti - loc)
                counter = (relu(self.deltas[i][t].to(x_fake.device) / 2. - dist) > 0.).float()
                density = counter.mean(0) / self.deltas[i][t].to(x_fake.device)
                abs_metric = torch.abs(density - self.densities[i][t].to(x_fake.device))
                tmp_loss.append(torch.mean(abs_metric, 0))
            loss_per_channel.append(torch.stack(tmp_loss).mean())
        
        return torch.stack(loss_per_channel)

def calculate_mdd(ori_data, gen_data):
    ori_data = torch.tensor(ori_data)
    gen_data = torch.tensor(gen_data)
    mdd = HistoLoss(ori_data, n_bins=50).compute(gen_data).detach().cpu().numpy()
    return mdd

# =======================================
# Autocorrelation Distance (ACD)
class ACFLoss(Loss):
    def __init__(self, x_real, max_lag=64, **kwargs):
        super(ACFLoss, self).__init__(norm_foo=acf_diff, **kwargs)
        self.max_lag = min(max_lag, x_real.shape[1])
        self.acf_real = acf_torch(
            self.transform(x_real),
            self.max_lag, dim=(0, 1)
        )

    def compute(self, x_fake):
        acf_fake = acf_torch(self.transform(x_fake), self.max_lag, dim=(0, 1))
        diff = acf_fake - self.acf_real.to(x_fake.device)
        return self.norm_foo(diff)

def calculate_acd(ori_data, gen_data):
    ori = torch.from_numpy(ori_data)
    gen = torch.from_numpy(gen_data)
    acf = ACFLoss(ori).compute(gen)
    return acf.detach().cpu().numpy()

# =======================================
# Mean Distance (MD)
class MeanLoss(Loss):
    def __init__(self, x_real, **kwargs):
        super(MeanLoss, self).__init__(norm_foo=torch.abs, **kwargs)
        self.mean_real = x_real.mean(dim=(0, 1))

    def compute(self, x_fake):
        mean_fake = x_fake.mean(dim=(0, 1))
        return self.norm_foo(mean_fake - self.mean_real)

def calculate_md(ori_data, gen_data):
    ori = torch.from_numpy(ori_data)
    gen = torch.from_numpy(gen_data)
    mean_loss = MeanLoss(x_real=ori)
    md = mean_loss.compute(gen)
    return md.detach().cpu().numpy()

# =======================================
# Standard Deviation Distance (SDD)
class StdLoss(Loss):
    def __init__(self, x_real, **kwargs):
        super(StdLoss, self).__init__(norm_foo=torch.abs, **kwargs)
        self.std_real = x_real.std(dim=(0, 1), unbiased=True, keepdim=False)

    def compute(self, x_fake):
        std_fake = x_fake.std(dim=(0, 1), unbiased=True, keepdim=False)
        return self.norm_foo(std_fake - self.std_real)

def calculate_sdd(ori_data, gen_data):
    ori = torch.from_numpy(ori_data)
    gen = torch.from_numpy(gen_data)
    std_loss = StdLoss(x_real=ori)
    sdd = std_loss.compute(gen)
    return sdd.detach().cpu().numpy()

# =======================================
# SD calculation and utilities functions
class SkewnessLoss(Loss):
    def __init__(self, x_real, **kwargs):
        super(SkewnessLoss, self).__init__(norm_foo=torch.abs, **kwargs)
        self.skew_real = skew_torch(x_real, dim=(0, 1), dropdims=True)

    def compute(self, x_fake, **kwargs):
        skew_fake = skew_torch(x_fake, dim=(0, 1), dropdims=True)
        return self.norm_foo(skew_fake - self.skew_real)
    
def calculate_sd(ori_data, gen_data):
    ori = torch.from_numpy(ori_data)
    gen = torch.from_numpy(gen_data)
    skewness = SkewnessLoss(x_real=ori)
    sd = skewness.compute(gen)
    return sd.detach().cpu().numpy()

# =======================================
# KD calculation and utilities functions
class KurtosisLoss(Loss):
    def __init__(self, x_real, **kwargs):
        super(KurtosisLoss, self).__init__(norm_foo=torch.abs, **kwargs)
        self.kurtosis_real = kurtosis_torch(x_real, dim=(0, 1), dropdims=True)

    def compute(self, x_fake):
        kurtosis_fake = kurtosis_torch(x_fake, dim=(0, 1), dropdims=True)
        return self.norm_foo(kurtosis_fake - self.kurtosis_real)
    
def calculate_kd(ori_data, gen_data):
    ori = torch.from_numpy(ori_data)
    gen = torch.from_numpy(gen_data)
    kurtosis = KurtosisLoss(x_real=ori)
    kd = kurtosis.compute(gen)
    return kd.detach().cpu().numpy()

def visualize_tsne(ori_data, gen_data, result_path):
    sns.set(style="whitegrid", context="paper", font_scale=1.2)

    anal_sample_no = len(ori_data)
    idx = np.random.permutation(len(ori_data))[:anal_sample_no]
    ori_data = ori_data[idx]
    gen_data = gen_data[idx]

    # Feature extraction
    def extract_features(data):
        features = []
        for sample in data:
            sample_feats = []
            for ch in range(sample.shape[1]):
                series = sample[:, ch]
                ch_feats = [
                    np.mean(series),
                    np.std(series),
                    skew(series),
                    kurtosis(series)
                ]
                sample_feats.extend(ch_feats)
            features.append(sample_feats)
        return np.array(features)

    ori_features = extract_features(ori_data)
    gen_features = extract_features(gen_data)
    
    prep_data_final = np.concatenate([ori_features, gen_features], axis=0)

    tsne = TSNE(n_components=2, verbose=0, perplexity=40, max_iter=500, random_state=42)
    tsne_results = tsne.fit_transform(prep_data_final)

    plt.figure(figsize=(6,6))
    plt.scatter(tsne_results[:anal_sample_no,0], tsne_results[:anal_sample_no,1], 
                c='#1f77b4', alpha=0.7, s=40, label="Original", edgecolor='k', linewidth=0.2)
    plt.scatter(tsne_results[anal_sample_no:,0], tsne_results[anal_sample_no:,1], 
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
    sns.set(style="whitegrid", context="paper", font_scale=1.2)
    ori_idx = np.random.permutation(len(ori_data))
    gen_idx = np.random.permutation(len(gen_data))
    ori_data = ori_data[ori_idx]
    gen_data = gen_data[gen_idx]

    n_channels = ori_data.shape[2]

    n_cols = 3
    n_rows = (n_channels + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    axes = axes.flatten()

    for ch in range(n_channels):
        ax = axes[ch]
        prep_ori = ori_data[:, :, ch].flatten()
        prep_gen = gen_data[:, :, ch].flatten()

        sns.kdeplot(prep_ori, color='#1f77b4', linewidth=2, label='Original', fill=False, ax=ax)
        sns.kdeplot(prep_gen, color='#ff7f0e', linewidth=2, linestyle='--', label='Generated', fill=False, ax=ax)

        ax.set_title(f"Channel {ch+1}", fontsize=12)
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.set_xlim(min(prep_ori.min(), prep_gen.min()), max(prep_ori.max(), prep_gen.max()))
        sns.despine(ax=ax, top=True, right=True)
        ax.legend(frameon=False)

    for ch in range(n_channels, len(axes)):
        fig.delaxes(axes[ch])

    plt.tight_layout()
    os.makedirs(result_path, exist_ok=True)
    save_path = os.path.join(result_path, 'distribution_channels_combined.png')
    plt.savefig(save_path, dpi=400, bbox_inches='tight')
    plt.close()
