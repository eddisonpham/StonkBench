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
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from typing import List, Tuple

from src.utils.math_utils import histogram_torch, acf_torch, non_stationary_acf_torch, skew_torch, kurtosis_torch, acf_diff
from src.utils.path_utils import make_sure_path_exist


class Loss(nn.Module):
    """
    Base class for feature-based loss metrics.

    Args:
        name (str): Name of the loss.
        reg (float): Regularization coefficient (default: 1.0).
        transform (callable): Optional transform to apply to input data.
        threshold (float): Success threshold for the loss.
        backward (bool): Whether to allow backward computation.
        norm_foo (callable): Function to aggregate or normalize the loss.
    """
    def __init__(self, name, reg=1.0, transform=lambda x: x, threshold=10., backward=False, norm_foo=lambda x: x):
        super(Loss, self).__init__()
        self.name = name
        self.reg = reg
        self.transform = transform
        self.threshold = threshold
        self.backward = backward
        self.norm_foo = norm_foo

    def forward(self, x_fake):
        """
        Compute the loss given generated (fake) data.

        Args:
            x_fake (torch.Tensor): Generated data.

        Returns:
            torch.Tensor: Scalar loss value.
        """
        self.loss_componentwise = self.compute(x_fake)
        return self.reg * self.loss_componentwise.mean()

    def compute(self, x_fake):
        """
        Compute the componentwise loss. To be implemented by subclasses.
        """
        raise NotImplementedError()

    @property
    def success(self):
        """
        Whether the loss is below the threshold for all components.
        """
        return torch.all(self.loss_componentwise <= self.threshold)

# =======================================
# Marginal Distribution Distance (MDD)
class HistoLoss(Loss):
    """
    Marginal Distribution Distance (MDD) loss based on histogram comparison.

    Args:
        x_real (torch.Tensor): Real/original data of shape (batch, time, features).
        n_bins (int): Number of histogram bins.
        **kwargs: Additional arguments for Loss.
    """
    def __init__(self, x_real, n_bins, **kwargs):
        super(HistoLoss, self).__init__(**kwargs)
        self.densities = []
        self.locs = []
        self.deltas = []
        for i in range(x_real.shape[2]):
            tmp_densities = []
            tmp_locs = []
            tmp_deltas = []
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
        """
        Compute the histogram-based loss between real and fake data.

        Args:
            x_fake (torch.Tensor): Generated data of shape (batch, time, features).

        Returns:
            torch.Tensor: Loss for each feature/time.
        """
        loss = []

        def relu(x):
            return x * (x >= 0.).float()

        for i in range(x_fake.shape[2]):
            for t in range(x_fake.shape[1]):
                loc = self.locs[i][t].view(1, -1).to(x_fake.device)
                x_ti = x_fake[:, t, i].contiguous().view(-1, 1).repeat(1, loc.shape[1])
                dist = torch.abs(x_ti - loc)
                counter = (relu(self.deltas[i][t].to(x_fake.device) / 2. - dist) > 0.).float()
                density = counter.mean(0) / self.deltas[i][t].to(x_fake.device)
                abs_metric = torch.abs(density - self.densities[i][t].to(x_fake.device))
                loss.append(torch.mean(abs_metric, 0))
        loss_componentwise = torch.stack(loss)
        return loss_componentwise

def calculate_mdd(ori_data, gen_data):
    """
    Calculate Marginal Distribution Distance (MDD) between real and generated data.

    Args:
        ori_data (np.ndarray or torch.Tensor): Real data of shape (batch, time, features).
        gen_data (np.ndarray or torch.Tensor): Generated data of shape (batch, time, features).

    Returns:
        float: MDD value.
    """
    if not torch.is_tensor(ori_data):
        ori_data = torch.tensor(ori_data)
    if not torch.is_tensor(gen_data):
        gen_data = torch.tensor(gen_data)
    mdd = (HistoLoss(ori_data[:, 1:, :], n_bins=50, name='marginal_distribution')(gen_data[:, 1:, :])).detach().cpu().numpy()
    return mdd.item()

# =======================================
# Mean Distance (MD)
class MeanLoss(Loss):
    """
    Mean Distance (MD) loss.

    Computes the absolute difference in mean between real and generated data.

    Assumes input tensors are of shape (R, l, N), where:
        R: number of samples (batch size)
        l: length of each sample (sequence length)
        N: number of channels/features

    Args:
        x_real (torch.Tensor): Real/original data of shape (R, l, N).
        **kwargs: Additional arguments for Loss.
    """
    def __init__(self, x_real, **kwargs):
        # Compute mean over samples and time, for each channel
        super(MeanLoss, self).__init__(norm_foo=torch.abs, **kwargs)
        self.mean_real = x_real.mean(dim=(0, 1), keepdim=False)

    def compute(self, x_fake):
        # Compute mean over samples and time, for each channel
        mean_fake = x_fake.mean(dim=(0, 1), keepdim=False)
        # Returns a vector of length N (channels)
        return self.norm_foo(mean_fake - self.mean_real)

def calculate_md(ori_data, gen_data):
    """
    Calculate Mean Distance (MD) between real and generated data.

    Assumes input arrays/tensors are of shape (R, l, N), where:
        R: number of samples (batch size)
        l: length of each sample (sequence length)
        N: number of channels/features

    Args:
        ori_data (np.ndarray or torch.Tensor): Real data of shape (R, l, N).
        gen_data (np.ndarray or torch.Tensor): Generated data of shape (R, l, N).

    Returns:
        float: MD value (scalar).
    """
    if not torch.is_tensor(ori_data):
        ori_data = torch.tensor(ori_data)
    if not torch.is_tensor(gen_data):
        gen_data = torch.tensor(gen_data)
    mean_loss = MeanLoss(x_real=ori_data, name='mean')
    # mean_loss.compute returns a vector of length N, so take the mean over channels to get a scalar
    md = mean_loss.compute(gen_data)
    if md.numel() > 1:
        md = md.mean()
    md = float(md.detach().cpu().numpy())
    return md

# =======================================
# Standard Deviation Distance (SDD)
class StdLoss(Loss):
    """
    Standard Deviation Distance (SDD) loss.

    Computes the absolute difference in standard deviation between real and generated data.

    Assumes input tensors are of shape (R, l, N), where:
        R: number of samples (batch size)
        l: length of each sample (sequence length)
        N: number of channels/features

    Args:
        x_real (torch.Tensor): Real/original data of shape (R, l, N).
        **kwargs: Additional arguments for Loss.
    """
    def __init__(self, x_real, **kwargs):
        # Compute std over samples and time, for each channel
        super(StdLoss, self).__init__(norm_foo=torch.abs, **kwargs)
        self.std_real = x_real.std(dim=(0, 1), unbiased=True, keepdim=False)

    def compute(self, x_fake):
        # Compute std over samples and time, for each channel
        std_fake = x_fake.std(dim=(0, 1), unbiased=True, keepdim=False)
        # Returns a vector of length N (channels)
        return self.norm_foo(std_fake - self.std_real)

def calculate_sdd(ori_data, gen_data):
    """
    Calculate Standard Deviation Distance (SDD) between real and generated data.

    Assumes input arrays/tensors are of shape (R, l, N), where:
        R: number of samples (batch size)
        l: length of each sample (sequence length)
        N: number of channels/features

    Args:
        ori_data (np.ndarray or torch.Tensor): Real data of shape (R, l, N).
        gen_data (np.ndarray or torch.Tensor): Generated data of shape (R, l, N).

    Returns:
        float: SDD value (scalar).
    """
    if not torch.is_tensor(ori_data):
        ori_data = torch.tensor(ori_data)
    if not torch.is_tensor(gen_data):
        gen_data = torch.tensor(gen_data)
    std_loss = StdLoss(x_real=ori_data, name='std')
    # std_loss.compute returns a vector of length N, so take the mean over channels to get a scalar
    sdd = std_loss.compute(gen_data)
    if sdd.numel() > 1:
        sdd = sdd.mean()
    sdd = float(sdd.detach().cpu().numpy())
    return sdd

# =======================================
# Autocorrelation Distance (ACD)
class ACFLoss(Loss):
    def __init__(self, x_real, max_lag=64, stationary=True, **kwargs):
        super(ACFLoss, self).__init__(norm_foo=acf_diff, **kwargs)
        self.max_lag = min(max_lag, x_real.shape[1])
        self.stationary = stationary
        if stationary:
            self.acf_real = acf_torch(self.transform(
                x_real), self.max_lag, dim=(0, 1))
        else:
            self.acf_real = non_stationary_acf_torch(self.transform(
                x_real), symmetric=False)  # Divide by 2 because it is symmetric matrix

    def compute(self, x_fake):
        if self.stationary:
            acf_fake = acf_torch(self.transform(x_fake), self.max_lag)
        else:
            acf_fake = non_stationary_acf_torch(self.transform(
                x_fake), symmetric=False)
        return self.norm_foo(acf_fake - self.acf_real.to(x_fake.device))

def calculate_acd(ori_data, gen_data):
    if not torch.is_tensor(ori_data):
        ori_data = torch.tensor(ori_data)
    if not torch.is_tensor(gen_data):
        gen_data = torch.tensor(gen_data)
    acf = (ACFLoss(ori_data, name='auto_correlation', stationary=True)(gen_data)).detach().cpu().numpy()
    return acf.item()

# =======================================
# SD calculation and utilities functions
class SkewnessLoss(Loss):
    def __init__(self, x_real, **kwargs):
        super(SkewnessLoss, self).__init__(norm_foo=torch.abs, **kwargs)
        self.skew_real = skew_torch(x_real)

    def compute(self, x_fake, **kwargs):
        skew_fake = skew_torch(x_fake)
        return self.norm_foo(skew_fake - self.skew_real)
    
def calculate_sd(ori_data, gen_data):
    if not torch.is_tensor(ori_data):
        ori_data = torch.tensor(ori_data)
    if not torch.is_tensor(gen_data):
        gen_data = torch.tensor(gen_data)
    skewness = SkewnessLoss(x_real = ori_data, name='skew')
    sd = skewness.compute(gen_data).mean()
    sd = float(sd.numpy())
    return sd

# =======================================
# KD calculation and utilities functions
class KurtosisLoss(Loss):
    def __init__(self, x_real, **kwargs):
        super(KurtosisLoss, self).__init__(norm_foo=torch.abs, **kwargs)
        self.kurtosis_real = kurtosis_torch(x_real)

    def compute(self, x_fake):
        kurtosis_fake = kurtosis_torch(x_fake)
        return self.norm_foo(kurtosis_fake - self.kurtosis_real)
    
def calculate_kd(ori_data, gen_data):
    if not torch.is_tensor(ori_data):
        ori_data = torch.tensor(ori_data)
    if not torch.is_tensor(gen_data):
        gen_data = torch.tensor(gen_data)
    kurtosis = KurtosisLoss(x_real = ori_data, name='kurtosis')
    kd = kurtosis.compute(gen_data).mean()
    kd = float(kd.numpy())
    return kd

def visualize_tsne(ori_data, gen_data, result_path, save_file_name):
    """
    Visualize the similarity between original and generated data using t-SNE.

    This function projects both original and generated data into a 2D space using t-SNE,
    and saves a scatter plot showing the two distributions for visual comparison.

    Args:
        ori_data (np.ndarray): Original data of shape (n_samples, ...).
        gen_data (np.ndarray): Generated data of shape (n_samples, ...).
        result_path (str): Directory where the plot will be saved.
        save_file_name (str): Suffix for the saved plot filename.

    Notes:
        - Only up to 1000 samples are randomly selected for visualization.
        - Each sample is reduced to its mean across axis=1 before t-SNE.
        - The resulting plot is saved as 'tsne_{save_file_name}.png' in result_path.
    """
    sample_num = min([1000, len(ori_data)])
    idx = np.random.permutation(len(ori_data))[:sample_num]

    ori_data = ori_data[idx]
    gen_data = gen_data[idx]

    # Reduce each sample to its mean value (flattening across features/timesteps)
    prep_data = np.mean(ori_data, axis=1)
    prep_data_hat = np.mean(gen_data, axis=1)

    # Assign colors for plotting: C0 for original, C1 for generated
    colors = ["C0" for _ in range(sample_num)] + ["C1" for _ in range(sample_num)]    
    
    # Concatenate original and generated data for joint t-SNE
    prep_data_final = np.concatenate((prep_data, prep_data_hat), axis=0)
    
    # Fit t-SNE to the combined data
    tsne = TSNE(n_components=2, verbose=0, perplexity=30, n_iter=1000, random_state=42)
    tsne_results = tsne.fit_transform(prep_data_final)

    # Create scatter plot
    fig, ax = plt.subplots(1, 1, figsize=(2, 2))
    ax.scatter(tsne_results[:sample_num, 0], tsne_results[:sample_num, 1], 
               c=colors[:sample_num], alpha=0.5, label="Original", s=5)
    ax.scatter(tsne_results[sample_num:, 0], tsne_results[sample_num:, 1], 
               c=colors[sample_num:], alpha=0.5, label="Generated", s=5)

    # Remove grid and axis ticks for a cleaner look
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    for pos in ['top', 'bottom', 'left', 'right']:
        ax.spines[pos].set_visible(False)

    # Save the plot
    save_path = os.path.join(result_path, 'tsne_' + save_file_name + '.png')
    make_sure_path_exist(save_path)
    plt.savefig(save_path, dpi=400, bbox_inches='tight')


def visualize_distribution(ori_data, gen_data, result_path, save_file_name):
    """
    Visualize the marginal distributions of original and generated data using KDE plots.

    This function plots the kernel density estimate (KDE) of the mean values of each sample
    from both the original and generated datasets, allowing for visual comparison of their
    distributions.

    Args:
        ori_data (np.ndarray): Original data of shape (n_samples, ...).
        gen_data (np.ndarray): Generated data of shape (n_samples, ...).
        result_path (str): Directory where the plot will be saved.
        save_file_name (str): Suffix for the saved plot filename.

    Notes:
        - Only up to 1000 samples are randomly selected for visualization.
        - Each sample is reduced to its mean across axis=1 before plotting.
        - The resulting plot is saved as 'distribution_{save_file_name}.png' in result_path.
        - The x-axis is limited to [0, 1] for consistency.
    """
    sample_num = min([1000, len(ori_data)])
    idx = np.random.permutation(len(ori_data))[:sample_num]

    ori_data = ori_data[idx]
    gen_data = gen_data[idx]

    # Reduce each sample to its mean value (flattening across features/timesteps)
    prep_data = np.mean(ori_data, axis=1)
    prep_data_hat = np.mean(gen_data, axis=1)

    # Create KDE plots for both original and generated data
    fig, ax = plt.subplots(1, 1, figsize=(2, 2))
    sns.kdeplot(prep_data.flatten(), color='C0', linewidth=2, label='Original', ax=ax)
    sns.kdeplot(prep_data_hat.flatten(), color='C1', linewidth=2, linestyle='--', label='Generated', ax=ax)

    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xlim(0, 1)
    for pos in ['top', 'right']:
        ax.spines[pos].set_visible(False)

    # Save the plot
    save_path = os.path.join(result_path, 'distribution_' + save_file_name + '.png')
    make_sure_path_exist(save_path)
    plt.savefig(save_path, dpi=400, bbox_inches='tight')