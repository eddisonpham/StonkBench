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
from PIL import Image


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
                   f'{value:.3f}', ha='center', va='bottom', fontsize=fontsize, fontweight='bold')

class PerformancePlot(MetricPlot):
    """Plot performance metrics (generation time)."""
    
    def plot(self) -> None:
        """
        Generate a bar plot for generation time/performance metrics.
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        times = []
        generation_keys = []
        for model in self.models:
            model_dict = self.data.get(model, {})
            gen_keys = [k for k in model_dict.keys() if "generation_time" in k and "_samples" in k]
            if not gen_keys:
                raise KeyError(f"No 'generation_time_*_samples' key found for model '{model}'. Available keys: {model_dict.keys()}")
            gen_key = gen_keys[0]
            val = model_dict[gen_key]
            times.append(float(val))
            generation_keys.append(gen_key)
        
        bars = ax.bar(self.models, times, color=sns.color_palette("husl", len(self.models)))
        ax.set_ylabel('Generation Time (seconds)')
        count_label = generation_keys[0].replace("generation_time_", "").replace("_samples", "")
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
            'leverage_effect', 'long_memory_volatility'
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
                    f'{value:.3f}', ha='center', va='bottom' if height >= 0 else 'top',
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
