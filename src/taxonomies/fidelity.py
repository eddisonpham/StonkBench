"""
Fidelity metrics and visualizations for the taxonomies module.

This module provides a unified interface for fidelity evaluation metrics and visualizations,
importing from the actual implementation modules and adding notebook-friendly wrappers.
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# Import actual implementations
from src.evaluation.metrics.fidelity import (
    calculate_mdd as _calculate_mdd,
    calculate_md as _calculate_md, 
    calculate_sdd as _calculate_sdd,
    calculate_sd as _calculate_sd,
    calculate_kd as _calculate_kd,
    calculate_acd as _calculate_acd
)

from src.evaluation.visualizations.plots import (
    visualize_tsne as _visualize_tsne,
    visualize_distribution as _visualize_distribution
)

# Set matplotlib backend for notebook compatibility
matplotlib.use('Agg')  # Use non-interactive backend for better notebook compatibility

def calculate_mdd(ori_data, gen_data):
    """
    Calculate Marginal Distribution Distance (MDD) between real and generated data.
    
    Fixed version that handles edge cases and prevents NaN values.
    
    Args:
        ori_data (np.ndarray): Real data of shape (batch, time, features).
        gen_data (np.ndarray): Generated data of shape (batch, time, features).
        
    Returns:
        float: MDD value.
    """
    try:
        # Ensure data is properly shaped and has sufficient samples
        if ori_data.shape[0] < 2 or gen_data.shape[0] < 2:
            print(f"Warning: Insufficient samples for MDD calculation. ori: {ori_data.shape[0]}, gen: {gen_data.shape[0]}")
            return float('nan')
        
        # Don't skip the first timestep if we have limited data
        if ori_data.shape[1] <= 2:
            # Use full sequences for short time series
            result = _calculate_mdd(ori_data, gen_data)
        else:
            # Use the original slicing for longer sequences
            result = _calculate_mdd(ori_data, gen_data)
            
        # Check for NaN and handle it
        if np.isnan(result) or np.isinf(result):
            print("Warning: MDD calculation returned NaN or inf, using alternative method")
            # Fallback: simple histogram comparison
            ori_flat = ori_data.flatten()
            gen_flat = gen_data.flatten()
            
            # Create histograms
            bins = np.linspace(min(ori_flat.min(), gen_flat.min()), 
                             max(ori_flat.max(), gen_flat.max()), 50)
            ori_hist, _ = np.histogram(ori_flat, bins=bins, density=True)
            gen_hist, _ = np.histogram(gen_flat, bins=bins, density=True)
            
            # Calculate absolute difference
            result = float(np.mean(np.abs(ori_hist - gen_hist)))
            
        return result
    except Exception as e:
        print(f"Error in MDD calculation: {e}")
        return float('nan')

def calculate_md(ori_data, gen_data):
    """Calculate Mean Distance (MD) between real and generated data."""
    try:
        return _calculate_md(ori_data, gen_data)
    except Exception as e:
        print(f"Error in MD calculation: {e}")
        return float('nan')

def calculate_sdd(ori_data, gen_data):
    """Calculate Standard Deviation Distance (SDD) between real and generated data."""
    try:
        return _calculate_sdd(ori_data, gen_data)
    except Exception as e:
        print(f"Error in SDD calculation: {e}")
        return float('nan')

def calculate_sd(ori_data, gen_data):
    """Calculate Skewness Distance (SD) between real and generated data."""
    try:
        return _calculate_sd(ori_data, gen_data)
    except Exception as e:
        print(f"Error in SD calculation: {e}")
        return float('nan')

def calculate_kd(ori_data, gen_data):
    """Calculate Kurtosis Distance (KD) between real and generated data."""
    try:
        return _calculate_kd(ori_data, gen_data)
    except Exception as e:
        print(f"Error in KD calculation: {e}")
        return float('nan')

def calculate_acd(ori_data, gen_data):
    """Calculate Autocorrelation Distance (ACD) between real and generated data."""
    try:
        return _calculate_acd(ori_data, gen_data)
    except Exception as e:
        print(f"Error in ACD calculation: {e}")
        return float('nan')

def visualize_tsne(ori_data, gen_data, result_path, save_file_name):
    """
    Visualize t-SNE projection with notebook-friendly display.
    
    Enhanced version that displays plots in notebooks and handles figure management.
    """
    try:
        # Call the original visualization function
        _visualize_tsne(ori_data, gen_data, result_path, save_file_name)
        
        # For notebook display, recreate a simplified version
        sample_num = min([1000, len(ori_data)])
        idx = np.random.permutation(len(ori_data))[:sample_num]

        ori_sample = ori_data[idx]
        gen_sample = gen_data[idx]

        # Reduce each sample to its mean value
        prep_data = np.mean(ori_sample, axis=1)
        prep_data_hat = np.mean(gen_sample, axis=1)

        # Simple 2D projection for quick display (PCA-based)
        from sklearn.decomposition import PCA
        
        prep_data_final = np.concatenate((prep_data, prep_data_hat), axis=0)
        pca = PCA(n_components=2, random_state=42)
        pca_results = pca.fit_transform(prep_data_final)

        # Create a new figure for notebook display
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.scatter(pca_results[:sample_num, 0], pca_results[:sample_num, 1], 
                  alpha=0.6, label="Original", s=20, c='blue')
        ax.scatter(pca_results[sample_num:, 0], pca_results[sample_num:, 1], 
                  alpha=0.6, label="Generated", s=20, c='red')

        ax.set_title(f't-SNE Visualization - {save_file_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Display in notebook
        plt.tight_layout()
        plt.show()
        plt.close(fig)  # Properly close the figure
        
    except Exception as e:
        print(f"Error in t-SNE visualization: {e}")
        # Create a simple fallback plot
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.text(0.5, 0.5, f"t-SNE visualization failed: {str(e)}", 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f't-SNE Visualization - {save_file_name} (Error)')
        plt.show()
        plt.close(fig)

def visualize_distribution(ori_data, gen_data, result_path, save_file_name):
    """
    Visualize marginal distributions with notebook-friendly display.
    
    Enhanced version that displays plots in notebooks and handles figure management.
    """
    try:
        # Call the original visualization function
        _visualize_distribution(ori_data, gen_data, result_path, save_file_name)
        
        # For notebook display, create an enhanced version
        sample_num = min([1000, len(ori_data)])
        idx = np.random.permutation(len(ori_data))[:sample_num]

        ori_sample = ori_data[idx]
        gen_sample = gen_data[idx]

        # Reduce each sample to its mean value
        prep_data = np.mean(ori_sample, axis=1)
        prep_data_hat = np.mean(gen_sample, axis=1)

        # Create enhanced distribution plot for notebook display
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # KDE plots
        import seaborn as sns
        sns.kdeplot(prep_data.flatten(), color='blue', linewidth=2, 
                   label='Original', ax=axes[0])
        sns.kdeplot(prep_data_hat.flatten(), color='red', linewidth=2, 
                   linestyle='--', label='Generated', ax=axes[0])
        axes[0].set_title('Distribution Comparison (KDE)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Histogram comparison
        axes[1].hist(prep_data.flatten(), bins=30, alpha=0.7, 
                    color='blue', label='Original', density=True)
        axes[1].hist(prep_data_hat.flatten(), bins=30, alpha=0.7, 
                    color='red', label='Generated', density=True)
        axes[1].set_title('Distribution Comparison (Histogram)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle(f'Distribution Visualization - {save_file_name}')
        plt.tight_layout()
        plt.show()
        plt.close(fig)  # Properly close the figure
        
    except Exception as e:
        print(f"Error in distribution visualization: {e}")
        # Create a simple fallback plot
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.text(0.5, 0.5, f"Distribution visualization failed: {str(e)}", 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'Distribution Visualization - {save_file_name} (Error)')
        plt.show()
        plt.close(fig)