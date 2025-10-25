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
        
        times = [self.data[model]['generation_time_1000_samples'] for model in self.models]
        
        bars = ax.bar(self.models, times, color=sns.color_palette("husl", len(self.models)))
        ax.set_ylabel('Generation Time (seconds)')
        ax.set_title('Model Performance: Generation Time for 1000 Samples')
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
            values = [self.data[model][metric] for model in self.models]
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
            values = [self.data[model][metric] for model in self.models]
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
        real_key = f"{fact_name}_real"
        synth_key = f"{fact_name}_synth"
        diff_key = f"{fact_name}_diff"

        models_with_metric = [model for model in self.models 
                             if real_key in self.data[model]]
        
        if not models_with_metric:
            return
        
        ax1 = axes[0]
        real_values = [np.mean(self.data[model][real_key]) for model in models_with_metric]
        synth_values = [np.mean(self.data[model][synth_key]) for model in models_with_metric]
        
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
        diff_values = [np.mean(self.data[model][diff_key]) for model in models_with_metric]
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
        if len(self.data[models_with_metric[0]][real_key]) > 1:
            heatmap_data = []
            for model in models_with_metric:
                real_vals = self.data[model][real_key]
                synth_vals = self.data[model][synth_key]
                diff_vals = self.data[model][diff_key]
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
                   [np.mean(self.data[models_with_metric[0]][real_key]), 
                    np.mean(self.data[models_with_metric[0]][synth_key])], 
                   color=['steelblue', 'orange'], alpha=0.8)
            ax3.set_title(f'{fact_name.replace("_", " ").title()}: Single Lag Comparison', fontsize=14, fontweight='bold')
            ax3.set_ylabel('Value', fontsize=12)
            ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'stylized_fact_{fact_name}.png', 
                   bbox_inches='tight', dpi=self.dpi)
        plt.close()


class ComprehensiveComparisonPlot(MetricPlot):
    """Create a comprehensive comparison heatmap."""
    
    def plot(self) -> None:
        """Generate comprehensive comparison heatmap."""
        metrics = ['mdd', 'md', 'sdd', 'sd', 'kd', 'acd', 'icd_euclidean', 'icd_dtw']
        
        heatmap_data = []
        for model in self.models:
            row = []
            for metric in metrics:
                if metric in self.data[model]:
                    row.append(self.data[model][metric])
                else:
                    row.append(np.nan)
            heatmap_data.append(row)
        
        heatmap_df = pd.DataFrame(heatmap_data, 
                                index=self.models,
                                columns=[m.upper() for m in metrics])
        
        fig, ax = plt.subplots(figsize=(10, 8), dpi=self.dpi)
        sns.heatmap(heatmap_df, annot=True, fmt='.3f', cmap='viridis', 
                   ax=ax, cbar_kws={'label': 'Metric Value'})
        ax.set_title('Comprehensive Model Comparison')
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Models')
        
        self.save_plot('comprehensive_comparison.png')


class ModelRankingPlot(MetricPlot):
    """Create model ranking based on multiple criteria."""
    
    def plot(self) -> None:
        """Generate model ranking plot."""
        ranking_metrics = {
            'generation_time_1000_samples': 'lower',
            'mdd': 'lower', 
            'md': 'lower',
            'sdd': 'lower',
            'icd_euclidean': 'lower',
            'icd_dtw': 'lower'
        }
        
        rankings = {}
        for metric, direction in ranking_metrics.items():
            values = [self.data[model][metric] for model in self.models]
            if direction == 'lower':
                sorted_models = [model for _, model in sorted(zip(values, self.models))]
            else:
                sorted_models = [model for _, model in sorted(zip(values, self.models), reverse=True)]
            
            for i, model in enumerate(sorted_models):
                if model not in rankings:
                    rankings[model] = []
                rankings[model].append(i + 1)
        
        avg_rankings = {model: np.mean(ranks) for model, ranks in rankings.items()}
        sorted_models = sorted(avg_rankings.items(), key=lambda x: x[1])
        
        fig, ax = plt.subplots(figsize=(10, 6), dpi=self.dpi)
        models = [item[0] for item in sorted_models]
        ranks = [item[1] for item in sorted_models]
        
        bars = ax.bar(models, ranks, color=sns.color_palette("viridis", len(models)))
        ax.set_ylabel('Average Ranking (Lower is Better)')
        ax.set_title('Model Performance Ranking')
        ax.tick_params(axis='x', rotation=45)
        
        self.add_value_labels(bars, ranks, ax)
        self.save_plot('model_ranking.png')
