"""
Separate figure generator classes for each paper figure type.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import os
from scipy.stats import pearsonr, rankdata
import scipy.stats
from sklearn.linear_model import LinearRegression
from PIL import Image
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.figure_utils import (
    extract_metric_value,
    scientific_fmt,
    format_minmax_label,
    make_blank_column,
    get_white_green_colormap,
    get_distinct_colors,
    DEFAULT_DPI,
    MINMAX_FONTSIZE,
    HEATMAP_ANNOTATION_FMT,
    extract_sequence_lengths
)


# Taxonomy definitions
TAXONOMY_MAIN = {
    'Fidelity': [
        ('mdd', 'MDD', False),
        ('md', 'MD', False),
        ('sdd', 'SDD', False),
        ('sd', 'SD', False),
        ('kd', 'KD', False)
    ],
    'Efficiency': [('generation_time_1000_samples', 'Time (s)', False)],
    'Distance': [('icd_euclidean', 'ICD Euclidean', False), ('icd_dtw', 'ICD DTW', False)],
    'Stylized-Facts': [
        ('autocorr_returns', 'ACD', True),
        ('volatility_clustering', 'VCD', True),
        ('long_memory_volatility', 'LMSD', True)
    ],
}

TAXONOMY_ABLATION = {
    "Fidelity": [
        ('mdd', 'MDD', False),
        ('md', 'MD', False),
        ('sdd', 'SDD', False),
        ('sd', 'SD', False),
        ('kd', 'KD', False),
    ],
    "Diversity": [
        ('icd_euclidean', 'ICD Euclidean', False),
        ('icd_dtw', 'ICD DTW', False),
    ],
    "Efficiency": [
        ('generation_time_1000_samples', 'Time (s)', False)
    ],
    "Stylized-Facts": [
        ('autocorr_returns', 'ACD', True),
        ('volatility_clustering', 'VCD', True),
        ('long_memory_volatility', 'LMSD', True),
    ],
}


class BaseFigureGenerator:
    """Base class for figure generators."""
    
    def __init__(self, main_data: Dict[str, Any], ablation_data: Dict[str, Dict[str, Any]], 
                 results_dir: Path, output_dir: Path):
        self.main_data = main_data
        self.ablation_data = ablation_data
        self.results_dir = results_dir
        self.output_dir = output_dir
        
        # Set matplotlib style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
        sns.set_context("paper", font_scale=1.3)


class HeatmapGenerator:
    """Utility class for generating heatmaps with rank normalization."""
    
    @staticmethod
    def normalize_ranks(values: np.ndarray, reverse: bool = False) -> np.ndarray:
        """
        Normalize values to ranks and scale to [0, 1].
        
        Args:
            values: Array of values to rank.
            reverse: If True, higher values get higher ranks (1=best). 
                    If False, lower values get higher ranks (1=best).
        
        Returns:
            Rank-normalized array.
        """
        ranked = np.full_like(values, np.nan)
        mask = ~np.isnan(values)
        if not np.any(mask):
            return ranked
        
        ranks = rankdata(values[mask], 'average')
        if len(ranks) > 1:
            if reverse:
                # Higher is better: rank 1 (best) -> 1.0, rank N (worst) -> 0.0
                norm = 1.0 - (ranks - 1) / (len(ranks) - 1)
            else:
                # Lower is better: rank 1 (best) -> 1.0, rank N (worst) -> 0.0
                norm = 1.0 - (ranks - 1) / (len(ranks) - 1)
        else:
            norm = np.ones_like(ranks)
        
        ranked[mask] = norm
        return ranked
    
    @staticmethod
    def add_minmax_labels(ax, minmaxs: List[Tuple[float, float]], labels: List[str], 
                          x_offset: float = 0.75, y_offset: float = -0.15):
        """Add min-max labels below heatmap."""
        for i, (min_val, max_val) in enumerate(minmaxs):
            if labels[i] != "" and not np.isnan(min_val) and not np.isnan(max_val):
                ax.text(
                    i + x_offset, y_offset, format_minmax_label(min_val, max_val),
                    rotation=40, ha='center', va='bottom', 
                    fontsize=MINMAX_FONTSIZE, color='black', 
                    linespacing=1.1, clip_on=False
                )


class Figure1Generator(BaseFigureGenerator):
    """Generate Figure 1: Per-metric heatmaps for main experiment."""
    
    def generate(self, output_dir: Path):
        """Generate all Figure 1 heatmaps."""
        fig_dir = output_dir / "figure_1_per_metric_heatmaps"
        fig_dir.mkdir(exist_ok=True, parents=True)
        
        self._generate_taxonomy_heatmap(fig_dir)
        self._generate_utility_heatmap(fig_dir)
    
    def _generate_taxonomy_heatmap(self, fig_dir: Path):
        """Generate heatmap for all taxonomies."""
        models = list(self.main_data)
        all_labels, all_arrays, all_raws, minmaxs = [], [], [], []
        
        for tax_name, metrics in TAXONOMY_MAIN.items():
            section_start = len(all_labels)
            for metric_key, metric_label, use_diff in metrics:
                values = self._extract_metric_values(models, metric_key, use_diff)
                arr = np.array(values)
                all_raws.append(arr.copy())
                
                # Rank normalization (lower is better except for Distance)
                reverse = (tax_name == 'Distance')
                ranked = HeatmapGenerator.normalize_ranks(arr, reverse=reverse)
                all_arrays.append(ranked)
                all_labels.append(metric_label)
                
                minmaxs.append(
                    (np.nanmin(arr), np.nanmax(arr)) 
                    if np.sum(~np.isnan(arr)) > 0 
                    else (np.nan, np.nan)
                )
            
            # Add separator between taxonomies
            if tax_name != list(TAXONOMY_MAIN.keys())[-1]:
                all_labels.append("")
                all_arrays.append(make_blank_column(all_arrays[0].shape))
                all_raws.append(make_blank_column(all_arrays[0].shape))
                minmaxs.append((np.nan, np.nan))
        
        mat = np.column_stack(all_arrays)
        self._plot_heatmap(
            mat, all_labels, models, minmaxs, fig_dir,
            "figure_1_all_taxonomies_ranknorm_heatmap.png",
            "Rank-Normalized (1=Best)"
        )
    
    def _generate_utility_heatmap(self, fig_dir: Path):
        """Generate heatmap for utility metrics."""
        models = list(self.main_data)
        models_r = models + ["Real Data"] if "Real Data" not in models else models.copy()
        
        utility_data = self._extract_utility_data(models_r)
        if not utility_data:
            return
        
        hedgers = list(next(iter(utility_data.values())).keys())
        R = np.array([[utility_data[mod].get(h, np.nan) for h in hedgers] for mod in models_r])
        
        # Rank normalize hedger columns
        Mtx = np.full_like(R, np.nan)
        for i, row in enumerate(R):
            mask = ~np.isnan(row)
            if np.any(mask):
                ranks = rankdata(row[mask], "min")
                if len(ranks) > 1:
                    norm = (ranks - 1) / (len(ranks) - 1)
                else:
                    norm = np.ones_like(ranks)
                Mtx[i, mask] = norm
        
        disp_labels, disp_arrays, minmax_cols = [], [], []
        for hidx, h in enumerate(hedgers):
            raw_col = R[:, hidx]
            disp_labels.append(h)
            disp_arrays.append(Mtx[:, hidx])
            minmax_cols.append(
                (np.nanmin(raw_col), np.nanmax(raw_col))
                if np.sum(~np.isnan(raw_col)) else (np.nan, np.nan)
            )
        
        # Add Spearman correlations
        sp_models = [m for m in models_r if m != "Real Data"]
        spv = np.array([extract_metric_value(self.main_data[m], "spearman_correlation") 
                       for m in sp_models])
        spv_mixed = np.array([extract_metric_value(self.main_data[m], "spearman_correlation_mixed") 
                            for m in sp_models])
        
        if disp_labels and (np.any(~np.isnan(spv)) or np.any(~np.isnan(spv_mixed))):
            disp_labels.append("")
            disp_arrays.append(make_blank_column((len(models_r),)))
            minmax_cols.append((np.nan, np.nan))
        
        if np.any(~np.isnan(spv)):
            idx_real = models_r.index("Real Data") if "Real Data" in models_r else None
            spv_r = np.insert(spv, idx_real, np.nan) if idx_real is not None else spv
            disp_labels.append("Spearman Correlation")
            disp_arrays.append(spv_r)
            minmax_cols.append(
                (np.nanmin(spv), np.nanmax(spv))
                if np.sum(~np.isnan(spv)) else (np.nan, np.nan)
            )
        
        if np.any(~np.isnan(spv_mixed)):
            idx_real = models_r.index("Real Data") if "Real Data" in models_r else None
            spv_mixed_r = np.insert(spv_mixed, idx_real, np.nan) if idx_real is not None else spv_mixed
            disp_labels.append("Spearman Correlation (Algorithm Testing)")
            disp_arrays.append(spv_mixed_r)
            minmax_cols.append(
                (np.nanmin(spv_mixed), np.nanmax(spv_mixed))
                if np.sum(~np.isnan(spv_mixed)) else (np.nan, np.nan)
            )
        
        if disp_arrays:
            mat2 = np.column_stack(disp_arrays)
            self._plot_heatmap(
                mat2, disp_labels, models_r, minmax_cols, fig_dir,
                "figure_1_utility_and_algorithm_ranknorm_heatmap.png",
                "Rank-Normalized (1=Best) / Spearman: Raw"
            )
    
    def _extract_metric_values(self, models: List[str], metric_key: str, use_diff: bool) -> List[float]:
        """Extract metric values for all models."""
        values = []
        for model in models:
            model_data = self.main_data.get(model, {})
            if use_diff and isinstance(model_data.get(metric_key), dict):
                val = model_data[metric_key].get('diff', np.nan)
            else:
                val = extract_metric_value(model_data, metric_key)
            values.append(val)
        return values
    
    def _extract_utility_data(self, models: List[str]) -> Dict[str, Dict[str, float]]:
        """Extract utility data for augmented testing."""
        utility_data = {}
        for model in models:
            model_data = self.main_data.get(model, {})
            aug = model_data.get("utility", {}).get("augmented_testing", {})
            
            if model != "Real Data" and aug:
                utility_data[model] = {
                    h: aug[h].get("mixed_train", {}).get("mean", np.nan) 
                    for h in aug
                }
            elif model == "Real Data":
                # Find reference model for Real Data
                ref_model = next(
                    (m for m in models if m != "Real Data" and 
                     self.main_data.get(m, {}).get("utility", {}).get("augmented_testing", {})),
                    None
                )
                if ref_model:
                    ref = self.main_data[ref_model].get("utility", {}).get("augmented_testing", {})
                    utility_data[model] = {
                        h: ref[h].get("real_train", {}).get("mean", np.nan) 
                        for h in ref
                    }
        return utility_data
    
    def _plot_heatmap(self, matrix: np.ndarray, x_labels: List[str], y_labels: List[str],
                     minmaxs: List[Tuple[float, float]], fig_dir: Path, filename: str,
                     cbar_label: str):
        """Plot a heatmap with standard formatting."""
        fig, ax = plt.subplots(
            figsize=(max(10, len(x_labels) * 1.2 + 3), max(6, len(y_labels) * 0.75)),
            dpi=DEFAULT_DPI
        )
        
        cmap = get_white_green_colormap()
        sns.heatmap(
            matrix,
            annot=True,
            fmt=HEATMAP_ANNOTATION_FMT,
            cmap=cmap,
            cbar_kws={"label": cbar_label},
            xticklabels=x_labels,
            yticklabels=y_labels,
            ax=ax,
            vmin=0,
            vmax=1,
        )
        
        plt.setp(
            ax.get_xticklabels(),
            rotation=45,
            ha="right",
            rotation_mode="anchor",
            fontsize=10,
        )
        
        HeatmapGenerator.add_minmax_labels(ax, minmaxs, x_labels)
        
        plt.tight_layout()
        plt.savefig(fig_dir / filename, bbox_inches="tight", dpi=DEFAULT_DPI)
        plt.close()


class Figure2Generator(BaseFigureGenerator):
    """Generate Figure 2: t-SNE and distribution visualizations."""
    
    def generate(self, output_dir: Path):
        """Generate Figure 2 grid."""
        fig_dir = output_dir / "figure_2_tsne_distribution"
        fig_dir.mkdir(exist_ok=True, parents=True)
        
        models = list(self.main_data)[:8]
        base = self.results_dir / "seq_52"
        imgs, names = [], []
        
        for model in models:
            combined_img = self._load_combined_visualization(base, model)
            if combined_img:
                imgs.append(combined_img)
                names.append(model)
        
        if not imgs:
            return
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 6), dpi=DEFAULT_DPI)
        [ax.axis('off') for ax in axes.flatten()]
        
        for idx, img in enumerate(imgs):
            ax = axes.flatten()[idx]
            ax.imshow(img)
            ax.set_title(names[idx], fontsize=11, fontweight='bold')
            ax.axis('off')
        
        for idx in range(len(imgs), 8):
            axes.flatten()[idx].set_visible(False)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)
        plt.savefig(fig_dir / 'figure_2_tsne_distribution.png', bbox_inches='tight', dpi=DEFAULT_DPI)
        plt.close()
    
    def _load_combined_visualization(self, base_dir: Path, model: str) -> Optional[Image.Image]:
        """Load and combine t-SNE and distribution images."""
        vis_dir = base_dir / model / "visualizations"
        tsne_path = vis_dir / "tsne.png"
        dist_path = vis_dir / "distribution.png"
        
        tsne = Image.open(tsne_path).convert("RGBA") if tsne_path.exists() else None
        dist = Image.open(dist_path).convert("RGBA") if dist_path.exists() else None
        
        if tsne is None and dist is None:
            return None
        
        if tsne is None:
            tsne = Image.new("RGBA", dist.size, (255, 255, 255, 0))
        if dist is None:
            dist = Image.new("RGBA", tsne.size, (255, 255, 255, 0))
        
        # Resize to match heights
        if tsne.size[1] != dist.size[1]:
            nh = tsne.size[1]
            nw = int(dist.size[0] * (nh / dist.size[1]))
            dist = dist.resize((nw, nh))
        if dist.size[1] != tsne.size[1]:
            nh = dist.size[1]
            nw = int(tsne.size[0] * (nh / tsne.size[1]))
            tsne = tsne.resize((nw, nh))
        
        # Combine side by side
        combined = Image.new("RGBA", (tsne.size[0] + dist.size[0], tsne.size[1]), (255, 255, 255, 0))
        combined.paste(tsne, (0, 0))
        combined.paste(dist, (tsne.size[0], 0))
        return combined


class Figure3Generator(BaseFigureGenerator):
    """Generate Figure 3: Ablation study heatmaps per metric."""
    
    def generate(self, output_dir: Path):
        """Generate all Figure 3 heatmaps."""
        fig_dir = output_dir / "figure_3_metric_heatmaps_per_metric"
        fig_dir.mkdir(exist_ok=True, parents=True)
        
        seqs = extract_sequence_lengths(self.ablation_data)
        if not self.ablation_data:
            return
        models = list(next(iter(self.ablation_data.values())).keys())
        
        for tax_name, metrics in TAXONOMY_ABLATION.items():
            self._generate_taxonomy_ablation_heatmap(fig_dir, tax_name, metrics, seqs, models)
        
        self._generate_spearman_ablation_heatmap(fig_dir, seqs, models)
    
    def _generate_taxonomy_ablation_heatmap(self, fig_dir: Path, tax_name: str, 
                                           metrics: List[Tuple], seqs: List[int], models: List[str]):
        """Generate heatmap for a single taxonomy across sequence lengths."""
        ncols = len(metrics) * len(seqs) + (len(metrics) - 1)
        n_models = len(models)
        mat = np.full((n_models, ncols), np.nan)
        raw = np.full((n_models, ncols), np.nan)
        bot_row = []
        col_tuples = []
        
        # Build column structure
        for mi, (mkey, mlab, use_diff) in enumerate(metrics):
            for seq in seqs:
                bot_row.append(f"K={seq}")
                col_tuples.append((mi, seq))
            if mi != len(metrics) - 1:
                bot_row.append("")
                col_tuples.append(None)
        
        # Extract raw values
        for ci, cinfo in enumerate(col_tuples):
            if cinfo is None:
                continue
            mi, seq = cinfo
            mkey, _, use_diff = metrics[mi]
            seq_key = f"seq_{seq}"
            
            for mi2, model in enumerate(models):
                model_data = self.ablation_data.get(seq_key, {}).get(model, {})
                if not model_data:
                    val = np.nan
                elif use_diff and isinstance(model_data.get(mkey), dict):
                    val = model_data[mkey].get('diff', np.nan)
                else:
                    val = extract_metric_value(model_data, mkey)
                raw[mi2, ci] = val
        
        # Rank normalize
        for ci, cinfo in enumerate(col_tuples):
            if cinfo is None:
                continue
            arr = raw[:, ci]
            mask = ~np.isnan(arr)
            if np.any(mask):
                ranks = scipy.stats.rankdata(arr[mask], 'average')
                if len(ranks) > 1:
                    if tax_name == "Diversity":
                        mat[mask, ci] = (ranks - 1) / (len(ranks) - 1)
                    else:
                        mat[mask, ci] = 1.0 - (ranks - 1) / (len(ranks) - 1)
                else:
                    mat[mask, ci] = np.ones_like(ranks)
        
        # Determine figure width
        width_map = {
            "Fidelity": 0.74,
            "Diversity": 0.78,
            "Efficiency": 0.95,
            "Stylized-Facts": 0.69,
        }
        fig_w = max(10, ncols * width_map.get(tax_name, 0.65))
        
        fig, ax = plt.subplots(figsize=(fig_w, max(6, n_models * 0.7)), dpi=DEFAULT_DPI)
        cmap = get_white_green_colormap()
        
        cbar_label = ('Rank-Normalized (1=Best per metric)' 
                     if tax_name != "Diversity" 
                     else 'Rank-Normalized (0=Best per metric)')
        
        sns.heatmap(
            mat,
            annot=True,
            fmt=HEATMAP_ANNOTATION_FMT,
            cmap=cmap,
            cbar_kws={'label': cbar_label},
            xticklabels=bot_row,
            yticklabels=models,
            ax=ax,
            vmin=0,
            vmax=1,
        )
        ax.set_xlabel("")
        ax.set_xticklabels(bot_row, rotation=45, ha='right')
        
        # Add top axis for taxonomy label
        ax2 = ax.secondary_xaxis('top')
        ax2.set_xticks(np.arange(ncols) + 0.5)
        ax2.set_xticklabels([""] * ncols, rotation=0, ha='center', fontsize=13, weight='bold')
        ax2.tick_params(axis='x', length=0)
        
        # Add min-max labels
        for ci, cinfo in enumerate(col_tuples):
            if cinfo is None:
                continue
            arr = raw[:, ci]
            mask = ~np.isnan(arr)
            if np.any(mask):
                minmax = format_minmax_label(np.nanmin(arr[mask]), np.nanmax(arr[mask]), precision=2)
                xpos = ci + (0.9 if tax_name == "Fidelity" else 0.75)
                ax.text(
                    xpos, -0.15, minmax, rotation=40,
                    ha='center', va='bottom', fontsize=MINMAX_FONTSIZE,
                    color='black', clip_on=False
                )
        
        plt.tight_layout()
        plt.savefig(
            fig_dir / f'figure_3_{tax_name.lower().replace(" ", "_")}_per_metric_ranknorm.png',
            bbox_inches='tight', dpi=DEFAULT_DPI
        )
        plt.close()
    
    def _generate_spearman_ablation_heatmap(self, fig_dir: Path, seqs: List[int], models: List[str]):
        """Generate heatmap for Spearman correlations across sequences."""
        ncols = 2 * len(seqs) + 1
        n_models = len(models)
        mat_both = np.full((n_models, ncols), np.nan)
        raw_both = np.full((n_models, ncols), np.nan)
        
        # Extract Spearman correlations
        for ci, seq in enumerate(seqs):
            seq_key = f"seq_{seq}"
            for mi, model in enumerate(models):
                model_data = self.ablation_data.get(seq_key, {}).get(model, {})
                val = model_data.get('utility', {}).get('algorithm_comparison', {}).get(
                    'spearman_correlation', np.nan
                )
                mat_both[mi, ci] = val
                raw_both[mi, ci] = val
        
        offset = len(seqs) + 1
        for ci, seq in enumerate(seqs):
            seq_key = f"seq_{seq}"
            for mi, model in enumerate(models):
                model_data = self.ablation_data.get(seq_key, {}).get(model, {})
                val = model_data.get('utility', {}).get('algorithm_comparison', {}).get(
                    'spearman_correlation_mixed', np.nan
                )
                mat_both[mi, offset + ci] = val
                raw_both[mi, offset + ci] = val
        
        bot_row = [f"K={k}" for k in seqs] + [""] + [f"K={k}" for k in seqs]
        
        fig, ax = plt.subplots(
            figsize=(max(16, ncols * 0.8), max(6, n_models * 0.7)), 
            dpi=DEFAULT_DPI
        )
        cmap = get_white_green_colormap()
        
        sns.heatmap(
            mat_both,
            annot=True,
            fmt=HEATMAP_ANNOTATION_FMT,
            cmap=cmap,
            cbar_kws={'label': 'Spearman Correlation'},
            xticklabels=bot_row,
            yticklabels=models,
            ax=ax,
            vmin=0,
            vmax=1,
        )
        ax.set_xlabel("")
        ax.set_xticklabels(bot_row, rotation=45, ha='right')
        
        ax2 = ax.secondary_xaxis('top')
        ax2.set_xticks(np.arange(ncols) + 0.5)
        ax2.set_xticklabels([""] * ncols, rotation=0, ha='center', fontsize=13, weight='bold')
        ax2.tick_params(axis='x', length=0)
        
        # Add min-max labels
        for ci in range(ncols):
            if ci == len(seqs):
                continue
            arr = raw_both[:, ci]
            mask = ~np.isnan(arr)
            if np.any(mask):
                minmax = format_minmax_label(np.nanmin(arr[mask]), np.nanmax(arr[mask]), precision=2)
                ax.text(
                    ci + 0.75, -0.15, minmax, rotation=40,
                    ha='center', va='bottom', fontsize=MINMAX_FONTSIZE,
                    color='black', clip_on=False
                )
        
        plt.tight_layout()
        plt.savefig(fig_dir / "figure_3_utility_spearman_and_mixed_per_seq.png", 
                   bbox_inches="tight", dpi=DEFAULT_DPI)
        plt.close()


class Figure4Generator(BaseFigureGenerator):
    """Generate Figure 4: Supplementary t-SNE visualizations for ablation sequences."""
    
    def generate(self, output_dir: Path):
        """Generate Figure 4 grids for each ablation sequence."""
        fig_dir = output_dir / "figure_4_tsne_supplementary"
        fig_dir.mkdir(exist_ok=True, parents=True)
        
        ablation_seqs = [60, 120, 180, 240, 300]
        if not self.ablation_data:
            return
        models = list(next(iter(self.ablation_data.values())).keys())
        
        for seq_len in ablation_seqs:
            seq_folder = self.results_dir / f"seq_{seq_len}"
            if not seq_folder.exists():
                continue
            
            pairs = []
            for model in models[:8]:
                combined_img = self._load_combined_visualization(seq_folder, model)
                pairs.append(combined_img)
            
            if not any(pairs):
                continue
            
            fig, axes = plt.subplots(2, 4, figsize=(16, 6), dpi=DEFAULT_DPI)
            [ax.axis('off') for ax in axes.flatten()]
            
            for idx, img in enumerate(pairs):
                if img is not None:
                    ax = axes.flatten()[idx]
                    ax.imshow(img)
                    ax.set_title(f"{models[idx]} (K={seq_len})", fontsize=11, fontweight='bold')
                    ax.axis('off')
            
            for idx in range(len(pairs), 8):
                axes.flatten()[idx].set_visible(False)
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.90)
            plt.savefig(fig_dir / f'figure_4_seq_{seq_len}.png', bbox_inches='tight', dpi=DEFAULT_DPI)
            plt.close()
    
    def _load_combined_visualization(self, base_dir: Path, model: str) -> Optional[Image.Image]:
        """Load and combine t-SNE and distribution images."""
        vis_dir = base_dir / model / "visualizations"
        tsne_path = vis_dir / "tsne.png"
        dist_path = vis_dir / "distribution.png"
        
        tsne = Image.open(tsne_path).convert("RGBA") if tsne_path.exists() else None
        dist = Image.open(dist_path).convert("RGBA") if dist_path.exists() else None
        
        if tsne is None and dist is None:
            return None
        
        if tsne is None:
            tsne = Image.new("RGBA", dist.size, (255, 255, 255, 0))
        if dist is None:
            dist = Image.new("RGBA", tsne.size, (255, 255, 255, 0))
        
        # Resize to match heights
        if tsne.size[1] != dist.size[1]:
            nh = tsne.size[1]
            nw = int(dist.size[0] * (nh / dist.size[1]))
            dist = dist.resize((nw, nh))
        if dist.size[1] != tsne.size[1]:
            nh = dist.size[1]
            nw = int(tsne.size[0] * (nh / tsne.size[1]))
            tsne = tsne.resize((nw, nh))
        
        # Combine side by side
        combined = Image.new("RGBA", (tsne.size[0] + dist.size[0], tsne.size[1]), (255, 255, 255, 0))
        combined.paste(tsne, (0, 0))
        combined.paste(dist, (tsne.size[0], 0))
        return combined


class ScatterPlotGenerator:
    """Base class for scatter plot generators."""
    
    @staticmethod
    def create_scatter_plot(x_vals: List[float], y_vals: List[float], 
                           valid_models: List[str], x_label: str, y_label: str,
                           color_map: Dict[str, Tuple[float, float, float]],
                           output_path: Path):
        """Create a scatter plot with regression line."""
        fig, ax = plt.subplots(figsize=(8.5, 5), dpi=180)
        
        c_vals = [color_map[m] for m in valid_models]
        ax.scatter(x_vals, y_vals, s=80, c=c_vals, edgecolors='black', 
                  linewidths=1.0, alpha=0.95, zorder=10)
        
        for model, x, y in zip(valid_models, x_vals, y_vals):
            ax.annotate(model, (x, y), textcoords="offset points", 
                       xytext=(3, -9), ha='left', fontsize=9, color='black')
        
        # Set axis limits with padding
        xlim = ScatterPlotGenerator._scaled_limits(x_vals)
        ylim = ScatterPlotGenerator._scaled_limits(y_vals)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        
        # Add regression line
        corr, pv = np.nan, np.nan
        if len(x_vals) > 1:
            try:
                corr, pv = pearsonr(x_vals, y_vals)
                lr = LinearRegression().fit(np.array(x_vals).reshape(-1, 1), np.array(y_vals))
                xl = np.linspace(*xlim, 100)
                yl = lr.predict(xl.reshape(-1, 1))
                ax.plot(xl, yl, color='crimson', linestyle='--', linewidth=2, alpha=0.8)
            except Exception:
                pass
        
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add correlation text
        legend_text = f"r={scientific_fmt(corr, 2)}, p={scientific_fmt(pv, 2)}"
        ax.text(
            0.98, 0.95, legend_text,
            fontsize=13, color='crimson', ha='right', va='top',
            fontweight='bold', transform=ax.transAxes,
            bbox=dict(facecolor='white', edgecolor='crimson', 
                     boxstyle='round,pad=0.35', alpha=0.7)
        )
        
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
    
    @staticmethod
    def _scaled_limits(vals: List[float]) -> Tuple[float, float]:
        """Calculate scaled axis limits with padding."""
        if not vals:
            return (0, 1)
        vmin, vmax = min(vals), max(vals)
        if vmin == vmax:
            pad = abs(vmin) * 0.1 + 1e-5
            return vmin - pad, vmax + pad
        lower = vmin - (abs(vmin) * 0.03 if vmin != 0 else 0.03 * abs(vmax) + 1e-5)
        upper = vmax + abs(vmax) * 0.10
        if vmax < 0:
            lower, upper = upper, lower
        return lower, upper


class Figure5Generator(BaseFigureGenerator):
    """Generate Figure 5: Fidelity vs Diversity scatter plots."""
    
    def generate(self, output_dir: Path):
        """Generate all Figure 5 scatter plots."""
        fig_dir = output_dir / "figure_5_fidelity_vs_diversity"
        fig_dir.mkdir(exist_ok=True, parents=True)
        
        models = [m for m in self.main_data.keys() if m != "TimeVAE"]
        n_models = len(models)
        pastel_palette = sns.color_palette('pastel', n_colors=n_models)
        color_map = {model: pastel_palette[i % len(pastel_palette)] 
                    for i, model in enumerate(models)}
        
        fid_list = [
            ('mdd', 'Marginal Distribution Difference'),
            ('md', 'Mean Difference'),
            ('sdd', 'Std Deviation Difference'),
            ('sd', 'Skewness Difference'),
            ('kd', 'Kurtosis Difference')
        ]
        div_list = [
            ('icd_euclidean', 'ICD Euclidean'),
            ('icd_dtw', 'ICD DTW')
        ]
        
        for div_key, div_lab in div_list:
            subplots = []
            for fid_key, fid_lab in fid_list:
                x_vals, y_vals, valid_models = self._extract_data(models, fid_key, div_key)
                
                if not x_vals:
                    continue
                
                out_path = fig_dir / f"figure5_{div_key}_{fid_key}.png"
                ScatterPlotGenerator.create_scatter_plot(
                    x_vals, y_vals, valid_models,
                    fid_lab, f'Diversity ({div_lab})',
                    color_map, out_path
                )
                subplots.append(str(out_path))
            
            # Create grid
            self._create_grid(subplots, fig_dir, f"figure_5_{div_key}_2x3_grid.png", (3, 2))
    
    def _extract_data(self, models: List[str], x_key: str, y_key: str) -> Tuple[List[float], List[float], List[str]]:
        """Extract data for scatter plot."""
        x_vals, y_vals, valid_models = [], [], []
        for model in models:
            x = extract_metric_value(self.main_data[model], x_key)
            y = extract_metric_value(self.main_data[model], y_key)
            if not np.isnan(x) and not np.isnan(y):
                x_vals.append(x)
                y_vals.append(y)
                valid_models.append(model)
        return x_vals, y_vals, valid_models
    
    def _create_grid(self, image_paths: List[str], fig_dir: Path, output_name: str, grid_size: Tuple[int, int]):
        """Create a grid of images."""
        imgs = [Image.open(p) for p in image_paths if os.path.exists(p)]
        if not imgs:
            return
        
        grid_w, grid_h = grid_size
        w, h = imgs[0].size
        grid_im = Image.new("RGBA", (w * grid_w, h * grid_h), (255, 255, 255, 0))
        
        for idx, img in enumerate(imgs):
            rx, ry = idx % grid_w, idx // grid_w
            grid_im.paste(img, (rx * w, ry * h))
        
        grid_im.save(fig_dir / output_name)
        
        # Clean up individual files
        for p in image_paths:
            try:
                os.remove(p)
            except Exception:
                pass


class Figure6Generator(BaseFigureGenerator):
    """Generate Figure 6: Stylized Facts vs Diversity scatter plots."""
    
    def generate(self, output_dir: Path):
        """Generate all Figure 6 scatter plots."""
        fig_dir = output_dir / "figure_6_stylized_facts_vs_diversity"
        fig_dir.mkdir(exist_ok=True, parents=True)
        
        models = [m for m in self.main_data.keys() if m != "TimeVAE"]
        n_models = len(models)
        pastel_palette = sns.color_palette('pastel', n_colors=n_models)
        color_map = {model: pastel_palette[i % len(pastel_palette)] 
                    for i, model in enumerate(models)}
        
        stylized = [
            ('autocorr_returns', 'ACD'),
            ('volatility_clustering', 'VCD'),
            ('long_memory_volatility', 'LMSD'),
        ]
        divs = [
            ('icd_euclidean', 'ICD Euclidean'),
            ('icd_dtw', 'ICD DTW')
        ]
        
        for div_key, div_lab in divs:
            panels = []
            for s_key, s_lab in stylized:
                x_vals, y_vals, valid_models = self._extract_data(models, s_key, div_key)
                
                if not x_vals:
                    continue
                
                out_path = fig_dir / f"figure6_{div_key}_{s_key}.png"
                ScatterPlotGenerator.create_scatter_plot(
                    x_vals, y_vals, valid_models,
                    s_lab, f'Diversity ({div_lab})',
                    color_map, out_path
                )
                panels.append(str(out_path))
            
            # Create grid
            self._create_grid(panels, fig_dir, f"figure_6_{div_key}_2x2_grid.png", (2, 2))
    
    def _extract_data(self, models: List[str], x_key: str, y_key: str) -> Tuple[List[float], List[float], List[str]]:
        """Extract data for scatter plot."""
        x_vals, y_vals, valid_models = [], [], []
        for model in models:
            x = extract_metric_value(self.main_data[model], x_key)
            y = extract_metric_value(self.main_data[model], y_key)
            if not np.isnan(x) and not np.isnan(y):
                x_vals.append(x)
                y_vals.append(y)
                valid_models.append(model)
        return x_vals, y_vals, valid_models
    
    def _create_grid(self, image_paths: List[str], fig_dir: Path, output_name: str, grid_size: Tuple[int, int]):
        """Create a grid of images."""
        imgs = [Image.open(p) for p in image_paths if os.path.exists(p)]
        if not imgs:
            return
        
        grid_w, grid_h = grid_size
        w, h = imgs[0].size
        grid_im = Image.new("RGBA", (w * grid_w, h * grid_h), (255, 255, 255, 0))
        
        for idx, img in enumerate(imgs):
            rx, ry = idx % grid_w, idx // grid_w
            grid_im.paste(img, (rx * w, ry * h))
        
        grid_im.save(fig_dir / output_name)
        
        # Clean up individual files
        for p in image_paths:
            try:
                os.remove(p)
            except Exception:
                pass


class Figure7Generator(BaseFigureGenerator):
    """Generate Figure 7: Augmented testing delta plots."""
    
    def generate(self, output_dir: Path):
        """Generate all Figure 7 delta plots."""
        fig_dir = output_dir / "figure_7_augmented_testing_delta"
        fig_dir.mkdir(exist_ok=True, parents=True)
        
        seqs = extract_sequence_lengths(self.ablation_data)
        if not self.ablation_data:
            return
        
        # Extract hedger names
        hedgers = []
        for seq_data in self.ablation_data.values():
            for model_data in seq_data.values():
                ut = model_data.get("utility", {}).get("augmented_testing", {})
                if ut:
                    hedgers = list(ut.keys())
                    break
            if hedgers:
                break
        
        if not hedgers:
            return
        
        models = list(next(iter(self.ablation_data.values())).keys())
        n_plot_models = len(models)
        plot_colors = get_distinct_colors(n_plot_models)
        
        delta_paths = []
        for hedger in hedgers:
            fig_delta, ax_delta = plt.subplots(figsize=(10, 5), dpi=DEFAULT_DPI)
            
            for m_idx, model in enumerate(models):
                xs, ys = [], []
                for seq in seqs:
                    seq_key = f"seq_{seq}"
                    model_data = self.ablation_data.get(seq_key, {}).get(model, {})
                    ut = model_data.get("utility", {}).get("augmented_testing", {}).get(hedger, {})
                    real = ut.get("real_train", {}).get("mean", np.nan)
                    mixed = ut.get("mixed_train", {}).get("mean", np.nan)
                    if not np.isnan(real) and not np.isnan(mixed):
                        xs.append(seq)
                        ys.append(mixed - real)
                
                cval = plot_colors[m_idx % len(plot_colors)]
                if len(xs) > 1:
                    ax_delta.plot(xs, ys, marker="o", linewidth=3, markersize=8, 
                                label=model, color=cval, alpha=0.87)
                elif len(xs) == 1:
                    ax_delta.plot(xs, ys, marker="o", linestyle="None", markersize=8, 
                                label=model, color=cval, alpha=0.87)
            
            ax_delta.set_title(f"{hedger} (Mixed - Real)", fontsize=17, fontweight='bold')
            ax_delta.set_xlabel("Sequence Length", fontsize=15)
            ax_delta.set_ylabel("Delta (Mixed - Real)", fontsize=15)
            ax_delta.axhline(0, color='black', linestyle='--', linewidth=1.2, alpha=0.6)
            ax_delta.grid(True, alpha=0.3)
            ax_delta.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), 
                          fontsize=13, title="Model", markerscale=1.2, frameon=False)
            plt.tight_layout(rect=[0, 0, 1, 1], pad=1.2, w_pad=0.2, h_pad=0.2)
            
            fpath_delta = fig_dir / f"_tmp_{hedger}_delta.png"
            plt.savefig(fpath_delta, dpi=DEFAULT_DPI, format='png')
            plt.close()
            delta_paths.append(fpath_delta)
        
        # Combine into grid
        images = [Image.open(p) for p in delta_paths if os.path.exists(p)]
        if images:
            w, h = images[0].size
            cols, rows = 4, 2
            gap_x, gap_y = 8, 8
            canvas = Image.new("RGB", 
                             (cols * w - (cols - 1) * gap_x, rows * h - (rows - 1) * gap_y), 
                             "white")
            for idx, img in enumerate(images):
                r = idx // cols
                c = idx % cols
                x = c * (w - gap_x)
                y = r * (h - gap_y)
                canvas.paste(img, (x, y))
            canvas.save(fig_dir / "figure_7_augmented_testing_2x4_delta.png", dpi=(300, 300))
            
            # Clean up temporary files
            for p in delta_paths:
                try:
                    p.unlink()
                except Exception:
                    pass

