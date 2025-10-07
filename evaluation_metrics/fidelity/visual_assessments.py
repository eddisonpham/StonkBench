import numpy as np
import seaborn as sns
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
# from .utils import MinMaxScaler, make_sure_path_exist

# Adapted from https://github.com/jsyoon0823/TimeGAN, https://openreview.net/forum?id=ez6VHWvuXEx

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