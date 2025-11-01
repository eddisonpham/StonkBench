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


class MetricPlot:
    """
    Base class for all metric plots.
    """
    
    def __init__(self, data: Dict[str, Any], output_dir: Path, figsize: Tuple[int, int] = (10, 6), dpi: int = 300):
        """
        Initialize the plot with data and output directory.
        
        Args:
            data: Evaluation data dictionary
            output_dir: Output directory for plots
            figsize: Figure size tuple
            dpi: DPI for high-quality plots
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
        """Generate performance metrics plot."""
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Support any generation_time_* key; expect a single scalar value
        times = []
        for model in self.models:
            model_dict = self.data.get(model, {})
            gen_keys = [k for k in model_dict.keys() if k.startswith('generation_') and k.endswith('_samples')]
            if gen_keys:
                val = model_dict[gen_keys[0]]
                if isinstance(val, (list, tuple, np.ndarray)):
                    # Fallback to first element if array is provided
                    try:
                        times.append(float(val[0]))
                    except Exception:
                        times.append(np.nan)
                else:
                    times.append(float(val))
            else:
                times.append(np.nan)
        
        bars = ax.bar(self.models, times, color=sns.color_palette("husl", len(self.models)))
        ax.set_ylabel('Generation Time (seconds)')
        ax.set_title('Model Performance: Generation Time')
        ax.tick_params(axis='x', rotation=45)

        self.add_value_labels(bars, times, ax)
        self.save_plot('performance_generation_time.png')

class DistributionPlot(MetricPlot):
    """Plot distribution metrics comparison."""
    
    def plot(self) -> None:
        """Generate distribution metrics plot."""
        metrics = ['mdd', 'md', 'sdd', 'sd', 'kd', 'acd']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10), dpi=self.dpi)
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            values = []
            for model in self.models:
                val = self.data.get(model, {}).get(metric, np.nan)
                if isinstance(val, (list, tuple, np.ndarray)):
                    values.append(float(np.mean(val)))
                else:
                    values.append(float(val))
            bars = ax.bar(self.models, values, color=sns.color_palette("husl", len(self.models)))
            ax.set_title(f'{metric.upper()}')
            ax.set_ylabel('Value')
            ax.tick_params(axis='x', rotation=45)
            self.add_value_labels(bars, values, ax, fontsize=8)

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
                if isinstance(val, (list, tuple, np.ndarray)):
                    values.append(float(np.mean(val)))
                else:
                    values.append(float(val))
            bars = ax.bar(self.models, values, color=sns.color_palette("husl", len(self.models)))
            ax.set_title(title)
            ax.set_ylabel('Distance')
            ax.tick_params(axis='x', rotation=45)
            self.add_value_labels(bars, values, ax)
        
        self.save_plot('similarity_metrics.png')

class StylizedFactsPlot(MetricPlot):
    """Plot stylized facts comparison between real and synthetic data."""
    
    def __init__(self, data: Dict[str, Any], output_dir: Path, figsize: Tuple[int, int] = (18, 6), dpi: int = 300):
        """Initialize with larger figure size for stylized facts."""
        super().__init__(data, output_dir, figsize, dpi)
        self.stylized_facts = ['heavy_tails', 'autocorr_raw', 'volatility_clustering', 
                              'long_memory', 'non_stationarity']
    
    def plot(self) -> None:
        """Generate all stylized facts plots."""
        for fact in self.stylized_facts:
            self._plot_single_stylized_fact(fact)
    
    def _plot_single_stylized_fact(self, fact_name: str) -> None:
        """Plot a single stylized fact comparison."""
        fig, axes = plt.subplots(1, 3, figsize=self.figsize, dpi=self.dpi)
        # New nested structure: data[model][fact_name] -> {'real': [...], 'synth': [...], 'diff': [...]} per channel
        models_with_metric = [model for model in self.models 
                             if isinstance(self.data.get(model, {}).get(fact_name, None), dict)]
        
        if not models_with_metric:
            return
        
        ax1 = axes[0]
        real_values = [float(np.mean(self.data[model][fact_name]['real'])) for model in models_with_metric]
        synth_values = [float(np.mean(self.data[model][fact_name]['synth'])) for model in models_with_metric]
        
        x = np.arange(len(models_with_metric))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, real_values, width, label='Real', alpha=0.8, color='steelblue')
        bars2 = ax1.bar(x + width/2, synth_values, width, label='Synthetic', alpha=0.8, color='orange')
        
        ax1.set_xlabel('Models', fontsize=12)
        ax1.set_ylabel('Value', fontsize=12)
        ax1.set_title(f'{fact_name.replace("_", " ").title()}: Real vs Synthetic', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models_with_metric, rotation=45, ha='right', fontsize=10)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[1]
        diff_values = [float(np.mean(self.data[model][fact_name]['diff'])) for model in models_with_metric]
        colors = ['red' if val > 0 else 'blue' for val in diff_values]
        
        bars = ax2.bar(models_with_metric, diff_values, color=colors, alpha=0.7)
        ax2.set_xlabel('Models', fontsize=12)
        ax2.set_ylabel('Difference (Real - Synthetic)', fontsize=12)
        ax2.set_title(f'{fact_name.replace("_", " ").title()}: Difference', fontsize=14, fontweight='bold')
        ax2.tick_params(axis='x', rotation=45, labelsize=10)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.grid(True, alpha=0.3)
        
        for bar, value in zip(bars, diff_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.3f}', ha='center', va='bottom' if height >= 0 else 'top', 
                   fontsize=9, fontweight='bold')
        
        ax3 = axes[2]
        if len(self.data[models_with_metric[0]][fact_name]['real']) > 1:
            heatmap_data = []
            for model in models_with_metric:
                real_vals = self.data[model][fact_name]['real']
                synth_vals = self.data[model][fact_name]['synth']
                diff_vals = self.data[model][fact_name]['diff']
                heatmap_data.append([np.mean(real_vals), np.mean(synth_vals), np.mean(diff_vals)])
            
            heatmap_df = pd.DataFrame(heatmap_data, 
                                    index=models_with_metric,
                                    columns=['Real', 'Synthetic', 'Difference'])
            
            sns.heatmap(heatmap_df, annot=True, fmt='.3f', cmap='RdBu_r', 
                       center=0, ax=ax3, cbar_kws={'label': 'Value'}, 
                       annot_kws={'fontsize': 10})
            ax3.set_title(f'{fact_name.replace("_", " ").title()}: Heatmap', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Data Type', fontsize=12)
            ax3.set_ylabel('Models', fontsize=12)
        else:
            ax3.bar(['Real', 'Synthetic'], 
                   [np.mean(self.data[models_with_metric[0]][fact_name]['real']), 
                    np.mean(self.data[models_with_metric[0]][fact_name]['synth'])], 
                   color=['steelblue', 'orange'], alpha=0.8)
            ax3.set_title(f'{fact_name.replace("_", " ").title()}: Single Lag Comparison', fontsize=14, fontweight='bold')
            ax3.set_ylabel('Value', fontsize=12)
            ax3.grid(True, alpha=0.3)
        
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
        """
        Args:
            data: Evaluation data dictionary
            output_dir: Primary output directory (plots + visualizations/ subdir)
            eval_results_dir: Path to evaluation results parent folder
            figsize, dpi: Passed to MetricPlot
        """
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
            # Guess: assume output_dir = .../evaluation_[timestamp]_plots or .../evaluation_[timestamp]
            parent = self.output_dir
            while parent.name not in ["results", "output", "evaluate"] and parent != parent.parent:
                if parent.name.startswith("evaluation_"):
                    break
                parent = parent.parent
            search_root = parent
        # paths: [search_root]/[model]/visualizations/distribution.png, tsne.png
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
        from PIL import Image

        for model in self.models:
            dist_path, tsne_path = self._find_visualization_files(model)
            if not dist_path or not tsne_path:
                continue
            try:
                img_dist = Image.open(dist_path)
                img_tsne = Image.open(tsne_path)
                # Resize tsne to match dist height (preserving aspect)
                h = img_dist.height
                tsne_aspect = img_tsne.width / img_tsne.height if img_tsne.height > 0 else 1
                img_tsne_resized = img_tsne.resize(
                    (int(tsne_aspect * h), h), resample=Image.LANCZOS)
                # Now, side by side concat
                total_width = img_tsne_resized.width + img_dist.width
                new_im = Image.new("RGBA", (total_width, h), (255, 255, 255, 0))
                new_im.paste(img_tsne_resized, (0, 0))
                new_im.paste(img_dist, (img_tsne_resized.width, 0))
                # Save in visualizations/[model].png (as standard png RGBA)
                output_file = self.visualizations_dir / f"{model}.png"
                new_im.save(output_file)
            except Exception as e:
                warnings.warn(
                    f"Failed to create combined visualization for '{model}': {e}"
                )
