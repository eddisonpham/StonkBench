"""
Plot classes for evaluation results visualization.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import warnings
import re
from PIL import Image


def scientific_fmt(x, precision=3):
    """Helper to format small/large numbers in scientific notation, otherwise fixed point."""
    try:
        x = float(x)
    except Exception:
        return str(x)
    if x == 0 or np.isnan(x):
        return f"{x:.{precision}f}"
    absx = abs(x)
    if absx < 1e-3 or absx > 1e4:
        return f"{x:.{precision}e}"
    else:
        return f"{x:.{precision}f}"

class MetricPlot:
    """
    Base class for all metric plots.
    """
    
    def __init__(self, data: Dict[str, Any], output_dir: Path, figsize: Tuple[int, int] = (10, 6), dpi: int = 300):
        """
        Initialize the plot with data and output directory.
        """
        self.data = data
        self.output_dir = output_dir
        self.figsize = figsize
        self.dpi = dpi
        self.models = list(data.keys())
        
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
    
    def save_plot(self, filename: str) -> None:
        """Save the current plot to file."""
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, bbox_inches='tight', dpi=self.dpi)
        plt.close()
    
    def add_value_labels(self, bars, values, ax, fontsize: int = 9) -> None:
        """Add value labels on bars."""
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   scientific_fmt(value), ha='center', va='bottom', fontsize=fontsize, fontweight='bold')

class PerformancePlot(MetricPlot):
    """Plot performance metrics (generation time)."""

    def plot(self) -> None:
        """
        Generate a bar plot for generation time/performance metrics.
        Uses key(s) of the form 'generation_time_*_samples'.
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        times = []
        generation_keys = []
        for model in self.models:
            model_dict = self.data.get(model, {})
            # Only keys that match generation_time_*_samples pattern should count
            gen_keys = [k for k in model_dict.keys() if k.startswith("generation_time_") and k.endswith("_samples")]
            print(f"Generation keys for model {model}: {gen_keys}")
            if not gen_keys:
                raise KeyError(f"No 'generation_time_*_samples' key found for model '{model}'. Available keys: {model_dict.keys()}")
            # If multiple, take the first (usually only one)
            gen_key = gen_keys[0]
            val = model_dict[gen_key]
            times.append(float(val))
            generation_keys.append(gen_key)

        bars = ax.bar(self.models, times, color=sns.color_palette("husl", len(self.models)))
        count_label = generation_keys[0].replace("generation_time_", "").replace("_samples", "")
        ax.set_ylabel('Generation Time (seconds)')
        ax.set_title(f'Model Performance: Generation Time ({count_label} samples)')
        ax.tick_params(axis='x', rotation=45)

        self.add_value_labels(bars, times, ax)
        self.save_plot('performance_generation_time.png')

class DistributionPlot(MetricPlot):
    """Plot distribution metrics comparison."""

    def plot(self) -> None:
        """Generate distribution metrics plot."""
        metrics = ['mdd', 'md', 'sdd', 'sd', 'kd']
        n_metrics = len(metrics)
        n_cols = 3
        n_rows = (n_metrics + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10), dpi=self.dpi)
        axes = axes.flatten()
        for i, metric in enumerate(metrics):
            ax = axes[i]
            values = []
            for model in self.models:
                val = self.data.get(model, {}).get(metric, np.nan)
                values.append(float(val))
            bars = ax.bar(self.models, values, color=sns.color_palette("husl", len(self.models)))
            ax.set_title(f'{metric.upper()}')
            ax.set_ylabel('Value')
            ax.tick_params(axis='x', rotation=45)
            self.add_value_labels(bars, values, ax, fontsize=8)

        for j in range(n_metrics, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.savefig(self.output_dir / 'distribution_metrics.png', bbox_inches='tight', dpi=self.dpi)
        plt.close()

class SimilarityPlot(MetricPlot):
    """Plot similarity metrics."""
    
    def plot(self) -> None:
        """Generate similarity metrics plot."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=self.dpi)
        
        metrics = ['icd_euclidean', 'icd_dtw']
        titles = ['Intra-class Distance (Euclidean)', 'Intra-class Distance (DTW)']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[i]
            values = []
            for model in self.models:
                val = self.data.get(model, {}).get(metric, np.nan)
                values.append(float(val))
            bars = ax.bar(self.models, values, color=sns.color_palette("husl", len(self.models)))
            ax.set_title(title)
            ax.set_ylabel('Distance')
            ax.tick_params(axis='x', rotation=45)
            self.add_value_labels(bars, values, ax)
        
        self.save_plot('similarity_metrics.png')

class StylizedFactsPlot(MetricPlot):
    """Plot stylized facts comparison between real and synthetic data."""

    def __init__(self, data: Dict[str, Any], output_dir: Path, figsize: Tuple[int, int] = (12, 6), dpi: int = 300):
        super().__init__(data, output_dir, figsize, dpi)
        self.stylized_facts = [
            'excess_kurtosis', 'autocorr_returns', 'volatility_clustering',
            'long_memory_volatility'
        ]

    def plot(self) -> None:
        """Generate all stylized facts plots."""
        for fact in self.stylized_facts:
            self._plot_single_stylized_fact(fact)

    def _plot_real_vs_synth(self, ax, models_with_metric, real_values, synth_values, fact_name):
        """Bar plot for real vs synthetic stylized facts (subplot 1)."""
        x = np.arange(len(models_with_metric))
        width = 0.35
        bars1 = ax.bar(x - width / 2, real_values, width, label='Real', alpha=0.8, color='steelblue')
        bars2 = ax.bar(x + width / 2, synth_values, width, label='Synthetic', alpha=0.8, color='orange')

        ax.set_xlabel('Models', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.set_title(f'{fact_name.replace("_", " ").title()}: Real vs Synthetic', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models_with_metric, rotation=45, ha='right', fontsize=10)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        # Add value labels for both
        for bar, value in zip(bars1, real_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height, scientific_fmt(value),
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        for bar, value in zip(bars2, synth_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height, scientific_fmt(value),
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        return bars1, bars2

    def _plot_diff_bar(self, ax, models_with_metric, diff_values, fact_name):
        """Bar plot for difference between real and synthetic stylized facts."""
        colors = ['red' if val > 0 else 'blue' for val in diff_values]
        bars = ax.bar(models_with_metric, diff_values, color=colors, alpha=0.7)
        ax.set_xlabel('Models', fontsize=12)
        ax.set_ylabel('Difference (Real - Synthetic)', fontsize=12)
        ax.set_title(f'{fact_name.replace("_", " ").title()}: Difference', fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=45, labelsize=10)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.grid(True, alpha=0.3)
        for bar, value in zip(bars, diff_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    scientific_fmt(value), ha='center', va='bottom' if height >= 0 else 'top',
                    fontsize=9, fontweight='bold')

    def _plot_single_stylized_fact(self, fact_name: str) -> None:
        """Plot a single stylized fact comparison (real vs synthetic and diff) as bar plots."""
        fig, axes = plt.subplots(1, 2, figsize=self.figsize, dpi=self.dpi)
        models_with_metric = [
            model for model in self.models
            if isinstance(self.data.get(model, {}).get(fact_name, None), dict)
        ]
        real_values = []
        synth_values = []
        diff_values = []
        for model in models_with_metric:
            entry = self.data[model][fact_name]
            real_values.append(float(entry['real']))
            synth_values.append(float(entry['synth']))
            diff_values.append(float(entry['diff']))

        self._plot_real_vs_synth(axes[0], models_with_metric, real_values, synth_values, fact_name)
        self._plot_diff_bar(axes[1], models_with_metric, diff_values, fact_name)

        plt.tight_layout()
        plt.savefig(self.output_dir / f'stylized_fact_{fact_name}.png',
                    bbox_inches='tight', dpi=self.dpi)
        plt.close()

class CombinedVisualizationPlot(MetricPlot):
    """
    Combine 'distribution.png' and 'tsne.png' side-by-side per model into a single visualization,
    saving each combined image in a 'visualizations' subfolder named '[model name].png'
    """

    def __init__(self, data: Dict[str, Any], output_dir: Path, 
                    eval_results_dir: Optional[Path] = None, figsize: Tuple[int, int] = (10, 6), dpi: int = 300):
        super().__init__(data, output_dir, figsize, dpi)
        self.eval_results_dir = Path(eval_results_dir) if eval_results_dir else None
        self.visualizations_dir = self.output_dir / 'visualizations'
        self.visualizations_dir.mkdir(exist_ok=True, parents=True)

    def _find_visualization_files(self, model: str):
        """
        Helper to find distribution.png and tsne.png for a given model
        """
        search_root = self.eval_results_dir
        if search_root is None:
            search_root = self.output_dir.parent
        model_vis_dir = search_root / model / "visualizations"
        dist_path = model_vis_dir / "distribution.png"
        tsne_path = model_vis_dir / "tsne.png"
        if not dist_path.exists() or not tsne_path.exists():
            warnings.warn(
                f"Missing visualization for model {model}: "
                f"{'distribution.png missing' if not dist_path.exists() else ''} "
                f"{'tsne.png missing' if not tsne_path.exists() else ''}"
            )
            return None, None
        return str(dist_path), str(tsne_path)

    def plot(self):
        """
        For each model, load tsne.png and distribution.png, concatenate side by side (height-consistent),
        then save to visualizations/[model name].png in output_dir.
        """
        for model in self.models:
            dist_path, tsne_path = self._find_visualization_files(model)
            if not dist_path or not tsne_path:
                continue
            try:
                img_dist = Image.open(dist_path)
                img_tsne = Image.open(tsne_path)
                h = img_dist.height
                tsne_aspect = img_tsne.width / img_tsne.height if img_tsne.height > 0 else 1
                img_tsne_resized = img_tsne.resize(
                    (int(tsne_aspect * h), h), resample=Image.LANCZOS)
                # Now, side by side concat
                total_width = img_tsne_resized.width + img_dist.width
                new_im = Image.new("RGBA", (total_width, h), (255, 255, 255, 0))
                new_im.paste(img_tsne_resized, (0, 0))
                new_im.paste(img_dist, (img_tsne_resized.width, 0))
                output_file = self.visualizations_dir / f"{model}.png"
                new_im.save(output_file)
            except Exception as e:
                warnings.warn(
                    f"Failed to create combined visualization for '{model}': {e}"
                )


class UtilityPlot(MetricPlot):
    """
    Plot utility-based evaluation metrics for deep hedging models.
    Visualizes both augmented testing and algorithm comparison results.
    """
    
    def __init__(self, data: Dict[str, Any], output_dir: Path, figsize: Tuple[int, int] = (14, 8), dpi: int = 300):
        """
        Initialize utility plot.
        
        Args:
            data: Dictionary with structure {model_name: {utility: {augmented_testing: {...}, algorithm_comparison: {...}}}}
            output_dir: Directory to save plots
            figsize: Figure size
            dpi: DPI for saved plots
        """
        super().__init__(data, output_dir, figsize, dpi)
    
    def plot(self) -> None:
        """Generate all utility evaluation plots."""
        # Plot augmented testing metrics
        self._plot_augmented_testing()
        # Plot algorithm comparison metrics
        self._plot_algorithm_comparison()
    
    def _plot_augmented_testing(self) -> None:
        """Plot augmented testing evaluation metrics."""
        # Extract augmented testing data
        augmented_data = {}
        for model in self.models:
            model_data = self.data.get(model, {})
            utility_data = model_data.get('utility', {})
            if 'augmented_testing' in utility_data:
                augmented_data[model] = utility_data['augmented_testing']
        
        if not augmented_data:
            warnings.warn("No augmented testing data found. Skipping augmented testing plots.")
            return
        
        # Get all hedger names and training regimes from first model
        first_model = list(augmented_data.keys())[0]
        hedger_names = list(augmented_data[first_model].keys())
        training_regimes = ['real_train', 'mixed_train']
        
        # Collect mean values: {model: {hedger_regime: mean_value}}
        mean_values = {}
        for model in augmented_data.keys():
            mean_values[model] = {}
            for hedger in hedger_names:
                for regime in training_regimes:
                    hedger_data = augmented_data[model].get(hedger, {})
                    regime_data = hedger_data.get(regime, {})
                    mean_val = regime_data.get('mean', np.nan)
                    key = f"{hedger}_{regime}"
                    mean_values[model][key] = mean_val
        
        # Compute ranks (lower mean is better, so rank 1 = best)
        rank_matrix = {}
        column_keys = [f"{h}_{r}" for h in hedger_names for r in training_regimes]
        
        for col_key in column_keys:
            # Get values for this column across all models
            values = [mean_values[model].get(col_key, np.nan) for model in augmented_data.keys()]
            # Filter out NaN values for ranking
            valid_indices = [i for i, v in enumerate(values) if not np.isnan(v)]
            valid_values = [values[i] for i in valid_indices]
            
            if len(valid_values) == 0:
                continue
            
            # Rank: lower mean = better rank (rank 1 is best)
            sorted_indices = sorted(range(len(valid_values)), key=lambda i: valid_values[i])
            ranks = [0] * len(values)
            for rank, orig_idx in enumerate(sorted_indices, 1):
                ranks[valid_indices[orig_idx]] = rank
            # For NaN values, assign worst rank
            max_rank = len(valid_values)
            for i, v in enumerate(values):
                if np.isnan(v):
                    ranks[i] = max_rank + 1
            
            # Store ranks for this column
            for i, model in enumerate(augmented_data.keys()):
                if model not in rank_matrix:
                    rank_matrix[model] = {}
                rank_matrix[model][col_key] = ranks[i]
        
        # Create heatmap data
        models_list = list(augmented_data.keys())
        heatmap_data = []
        for model in models_list:
            row = [rank_matrix.get(model, {}).get(col_key, np.nan) for col_key in column_keys]
            heatmap_data.append(row)
        heatmap_data = np.array(heatmap_data)
        
        # Plot 1: Utility rank heatmap
        fig, ax = plt.subplots(figsize=(max(16, len(column_keys) * 1.5), max(8, len(models_list) * 0.8)), dpi=self.dpi)
        
        # Create heatmap
        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt='.0f',
            cmap='RdYlGn_r',  # Reversed: green=good (low rank), red=bad (high rank)
            cbar_kws={'label': 'Rank (1=best)'},
            xticklabels=[key.replace('_', '\n') for key in column_keys],
            yticklabels=models_list,
            ax=ax,
            vmin=1,
            vmax=heatmap_data.max() if not np.isnan(heatmap_data).all() else 10
        )
        ax.set_title('Utility Rank Heatmap: Generative Model Performance Across Hedgers and Training Regimes', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Hedger × Training Regime', fontsize=12)
        ax.set_ylabel('Generative Model', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'utility_rank_heatmap.png', bbox_inches='tight', dpi=self.dpi)
        plt.close()
        
        # Plot 2: Consensus utility rank bar chart
        # Average ranks across all hedger+regime combinations
        consensus_ranks = {}
        for model in models_list:
            ranks_for_model = [rank_matrix.get(model, {}).get(col_key) for col_key in column_keys 
                             if not np.isnan(rank_matrix.get(model, {}).get(col_key, np.nan))]
            if ranks_for_model:
                consensus_ranks[model] = np.mean(ranks_for_model)
            else:
                consensus_ranks[model] = np.nan
        
        # Sort by consensus rank (best first)
        sorted_models = sorted(consensus_ranks.items(), key=lambda x: x[1] if not np.isnan(x[1]) else float('inf'))
        
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        models_sorted = [m[0] for m in sorted_models]
        ranks_sorted = [m[1] for m in sorted_models]
        
        bars = ax.barh(models_sorted, ranks_sorted, color=sns.color_palette("RdYlGn_r", len(models_sorted)))
        ax.set_xlabel('Average Rank (lower is better)', fontsize=12)
        ax.set_ylabel('Generative Model', fontsize=12)
        ax.set_title('Consensus Utility Rank: Overall Performance Across All Hedgers and Training Regimes', 
                    fontsize=14, fontweight='bold')
        ax.invert_yaxis()  # Best (lowest rank) at top
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, rank) in enumerate(zip(bars, ranks_sorted)):
            if not np.isnan(rank):
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2.,
                       scientific_fmt(rank, precision=2), ha='left' if width < max(ranks_sorted) / 2 else 'right',
                       va='center', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'utility_consensus_rank.png', bbox_inches='tight', dpi=self.dpi)
        plt.close()
    
    def _plot_algorithm_comparison(self) -> None:
        """Plot algorithm comparison evaluation metrics."""
        # Extract algorithm comparison data
        algorithm_data = {}
        for model in self.models:
            model_data = self.data.get(model, {})
            utility_data = model_data.get('utility', {})
            if 'algorithm_comparison' in utility_data:
                algorithm_data[model] = utility_data['algorithm_comparison']
        
        if not algorithm_data:
            warnings.warn("No algorithm comparison data found. Skipping algorithm comparison plots.")
            return
        
        # Extract spearman_correlation for each model
        spearman_corrs = {}
        for model in algorithm_data.keys():
            corr = algorithm_data[model].get('spearman_correlation', np.nan)
            spearman_corrs[model] = corr
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        models_list = list(spearman_corrs.keys())
        corr_values = [spearman_corrs[model] for model in models_list]
        
        bars = ax.bar(models_list, corr_values, color=sns.color_palette("husl", len(models_list)))
        ax.set_xlabel('Generative Model', fontsize=12)
        ax.set_ylabel('Spearman Correlation', fontsize=12)
        ax.set_title('Algorithm Comparison: Spearman Correlation Between Real and Synthetic Hedger Rankings', 
                    fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # Add value labels
        for bar, val in zip(bars, corr_values):
            if not np.isnan(val):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        scientific_fmt(val), ha='center', va='bottom' if height >= 0 else 'top',
                        fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'utility_algorithm_comparison.png', bbox_inches='tight', dpi=self.dpi)
        plt.close()


class SequenceLengthComparisonPlot:
    """
    Plot sequence length comparison metrics.
    Creates line plots, rank-order plots, and variance plots for metrics across different sequence lengths.
    """
    
    def __init__(self, all_seq_data: Dict[str, Dict[str, Any]], output_dir: Path, num_samples: int = 1000,
                 figsize: Tuple[int, int] = (10, 6), dpi: int = 300):
        """
        Initialize sequence length comparison plot.
        
        Args:
            all_seq_data: Dictionary with structure {seq_name: {model_name: metrics_dict}}
            output_dir: Directory to save plots
            figsize: Figure size
            dpi: DPI for saved plots
        """
        self.all_seq_data = all_seq_data
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.figsize = figsize
        self.dpi = dpi
        self.num_samples = num_samples
        # Extract sequence lengths from folder names
        self.seq_lengths = []
        for seq_name in sorted(all_seq_data.keys()):
            match = re.match(r"seq_(\d+)", seq_name)
            if match:
                self.seq_lengths.append(int(match.group(1)))
        
        self.seq_lengths.sort()
        
        # Get all models (should be consistent across sequence lengths)
        if all_seq_data:
            first_seq = list(all_seq_data.keys())[0]
            self.models = list(all_seq_data[first_seq].keys())
        else:
            self.models = []
        
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
    
    def _extract_metric_value(self, model_data: Dict[str, Any], metric_name: str) -> float:
        """
        Extract metric value from model data, handling different structures.
        Returns np.nan if not found.
        """
        if metric_name in model_data:
            value = model_data[metric_name]
            if isinstance(value, dict):
                # For stylized facts, use 'diff' value
                return float(value.get('diff', np.nan))
            elif isinstance(value, (int, float)):
                return float(value)
            else:
                return np.nan
        
        # Check nested in utility
        if metric_name == 'spearman_correlation':
            utility_data = model_data.get('utility', {})
            algo_comp = utility_data.get('algorithm_comparison', {})
            return float(algo_comp.get('spearman_correlation', np.nan))
        
        return np.nan
    
    def _get_all_metrics(self) -> List[str]:
        """Get list of all metrics to plot."""
        metrics = []
        
        # Distribution metrics
        metrics.extend(['mdd', 'md', 'sdd', 'sd', 'kd'])
        
        # Similarity metrics
        metrics.extend(['icd_euclidean', 'icd_dtw'])
        
        # Stylized facts (using diff)
        metrics.extend(['excess_kurtosis', 'autocorr_returns', 'volatility_clustering', 'long_memory_volatility'])
        
        # Performance
        metrics.append(f'generation_time_{self.num_samples}_samples')
        
        # Utility
        metrics.append('spearman_correlation')
        
        return metrics
    
    def plot(self) -> None:
        """Generate all sequence length comparison plots."""
        # Create separate folders for each plot type
        line_plots_dir = self.output_dir / 'line_plots'
        rank_plots_dir = self.output_dir / 'rank_order_plots'
        variance_plots_dir = self.output_dir / 'variance_plots'
        
        line_plots_dir.mkdir(exist_ok=True)
        rank_plots_dir.mkdir(exist_ok=True)
        variance_plots_dir.mkdir(exist_ok=True)
        
        metrics = self._get_all_metrics()
        
        # Collect data for all metrics
        metric_data = {}
        for metric in metrics:
            metric_data[metric] = {}
            for model in self.models:
                values = []
                for seq_name in sorted(self.all_seq_data.keys()):
                    model_data = self.all_seq_data[seq_name].get(model, {})
                    value = self._extract_metric_value(model_data, metric)
                    values.append(value)
                metric_data[metric][model] = values
        
        # Generate line plots
        self._plot_line_plots(metric_data, line_plots_dir)
        
        # Generate rank-order plots
        self._plot_rank_order(metric_data, rank_plots_dir)
        
        # Generate variance plots
        self._plot_variance(metric_data, variance_plots_dir)
    
    def _plot_line_plots(self, metric_data: Dict[str, Dict[str, List[float]]], output_dir: Path) -> None:
        """Plot line plots: metric value vs sequence length, one line per model."""
        for metric, model_values in metric_data.items():
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
            
            for model in self.models:
                values = model_values.get(model, [])
                # Filter out NaN values for plotting
                valid_indices = [i for i, v in enumerate(values) if not np.isnan(v)]
                valid_seq_lengths = [self.seq_lengths[i] for i in valid_indices]
                valid_values = [values[i] for i in valid_indices]
                
                if valid_values:
                    ax.plot(valid_seq_lengths, valid_values, marker='o', label=model, linewidth=2, markersize=6)
                    # Value labels at each point (scientific notation for small/large)
                    for xval, yval in zip(valid_seq_lengths, valid_values):
                        ax.text(xval, yval, scientific_fmt(yval), fontsize=8, ha='center', va='bottom')

            ax.set_xlabel('Sequence Length (K)', fontsize=12)
            ax.set_ylabel('Metric Value', fontsize=12)
            ax.set_title(f'{metric.replace("_", " ").title()}: Sensitivity to Sequence Length', 
                        fontsize=14, fontweight='bold')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / f'{metric}_line_plot.png', bbox_inches='tight', dpi=self.dpi)
            plt.close()
    
    def _plot_rank_order(self, metric_data: Dict[str, Dict[str, List[float]]], output_dir: Path) -> None:
        """Plot rank-order plots showing how model rankings change across sequence lengths."""
        for metric, model_values in metric_data.items():
            # Compute ranks for each sequence length
            rank_matrix = []
            for seq_idx in range(len(self.seq_lengths)):
                # Get values for this sequence length across all models
                values_at_seq = []
                model_list_at_seq = []
                for model in self.models:
                    values = model_values.get(model, [])
                    if seq_idx < len(values) and not np.isnan(values[seq_idx]):
                        values_at_seq.append(values[seq_idx])
                        model_list_at_seq.append(model)
                
                if not values_at_seq:
                    continue
                
                # Rank models: for most metrics, lower is better (mdd, md, sdd, sd, kd, icd_euclidean, icd_dtw)
                # For spearman_correlation, higher is better
                if metric == 'spearman_correlation':
                    # Higher is better
                    sorted_indices = sorted(range(len(values_at_seq)), key=lambda i: -values_at_seq[i])
                else:
                    # Lower is better
                    sorted_indices = sorted(range(len(values_at_seq)), key=lambda i: values_at_seq[i])
                
                ranks = [0] * len(values_at_seq)
                for rank, orig_idx in enumerate(sorted_indices, 1):
                    ranks[orig_idx] = rank
                
                rank_matrix.append({
                    'seq_length': self.seq_lengths[seq_idx],
                    'ranks': ranks,
                    'models': model_list_at_seq
                })
            
            if not rank_matrix:
                continue
            
            # Create heatmap-style plot
            # Get all models that appear in any sequence length
            all_models_in_ranks = set()
            for rm in rank_matrix:
                all_models_in_ranks.update(rm['models'])
            all_models_in_ranks = sorted(all_models_in_ranks)
            
            # Build heatmap data
            heatmap_data = []
            for model in all_models_in_ranks:
                row = []
                for rm in rank_matrix:
                    if model in rm['models']:
                        model_idx = rm['models'].index(model)
                        row.append(rm['ranks'][model_idx])
                    else:
                        row.append(np.nan)
                heatmap_data.append(row)
            
            fig, ax = plt.subplots(figsize=(max(10, len(rank_matrix) * 1.5), max(8, len(all_models_in_ranks) * 0.8)), 
                                  dpi=self.dpi)
            
            seq_labels = [rm['seq_length'] for rm in rank_matrix]
            sns.heatmap(
                np.array(heatmap_data),
                annot=True,
                fmt='.0f',
                cmap='RdYlGn_r',  # Reversed: green=best (rank 1), red=worst
                cbar_kws={'label': 'Rank (1=best)'},
                xticklabels=seq_labels,
                yticklabels=all_models_in_ranks,
                ax=ax
            )
            
            ax.set_xlabel('Sequence Length (K)', fontsize=12)
            ax.set_ylabel('Model', fontsize=12)
            ax.set_title(f'{metric.replace("_", " ").title()}: Model Rankings Across Sequence Lengths', 
                        fontsize=14, fontweight='bold')
            plt.xticks(rotation=0)
            plt.tight_layout()
            plt.savefig(output_dir / f'{metric}_rank_order.png', bbox_inches='tight', dpi=self.dpi)
            plt.close()
    
    def _plot_variance(self, metric_data: Dict[str, Dict[str, List[float]]], output_dir: Path) -> None:
        """Plot variance across sequence lengths for each model–metric pair."""
        for metric, model_values in metric_data.items():
            variances = []
            model_names = []
            
            for model in self.models:
                values = model_values.get(model, [])
                # Filter out NaN values
                valid_values = [v for v in values if not np.isnan(v)]
                if len(valid_values) > 1:
                    var = np.var(valid_values)
                    variances.append(var)
                    model_names.append(model)
            
            if not variances:
                continue
            
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
            
            bars = ax.barh(model_names, variances, color=sns.color_palette("husl", len(model_names)))
            ax.set_xlabel('Variance Across Sequence Lengths', fontsize=12)
            ax.set_ylabel('Model', fontsize=12)
            ax.set_title(f'{metric.replace("_", " ").title()}: Variance Across Sequence Lengths', 
                        fontsize=14, fontweight='bold')
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3, axis='x')
            
            # Add value labels in scientific format
            for bar, var in zip(bars, variances):
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2.,
                       scientific_fmt(var, precision=4), ha='left', va='center', fontsize=9, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(output_dir / f'{metric}_variance.png', bbox_inches='tight', dpi=self.dpi)
        plt.close()
