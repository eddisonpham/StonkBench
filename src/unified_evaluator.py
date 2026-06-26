"""
Unified Evaluator (evaluation-only).

Loads pre-generated artifacts from `generated_data/` and computes taxonomy metrics
without any training or generation. Artifacts must follow the contract described
in `refactor.md` and be produced by the generation scripts.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.artifact_utils import load_artifact  # noqa: E402
from src.utils.configs_utils import get_dataset_cfgs  # noqa: E402
from src.utils.display_utils import show_with_start_divider, show_with_end_divider  # noqa: E402
from src.utils.evaluation_classes_utils import (  # noqa: E402
    DiversityEvaluator,
    FidelityEvaluator,
    StylizedFactsEvaluator,
    VisualAssessmentEvaluator,
    UtilityEvaluator,
)
from src.utils.preprocessed_data_utils import load_dl_set, load_stats_set, sliding_window_2d  # noqa: E402


# Constants
UTILITY_TRAIN_RATIO = 0.8
UTILITY_VAL_RATIO = 0.9
UTILITY_NUM_EPOCHS = 40
UTILITY_BATCH_SIZE = 64
UTILITY_LEARNING_RATE = 1e-3


def _to_numpy(x: Any) -> np.ndarray:
    """Convert tensor or array to numpy array."""
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    return np.asarray(x)


def _normalize_model_type(model_type: str) -> str:
    if model_type in ("parametric", "statistical"):
        return "statistical"
    if model_type in ("non_parametric", "deep_learning"):
        return "deep_learning"
    return model_type


class DatasetCache:
    """Manages caching of preprocessed real datasets for evaluation."""

    def __init__(self, deep_learning_cfg: Dict[str, Any], statistical_cfg: Dict[str, Any]):
        self._dl_set = load_dl_set(deep_learning_cfg["preprocessed_data_path"])
        self._stats_set = load_stats_set(statistical_cfg["preprocessed_data_path"])
        self._cache: Dict[int, Dict[str, Any]] = {}

    def get_dataset(self, seq_length: int) -> Dict[str, Any]:
        """
        Get or create cached dataset for a given sequence length.
        """
        if seq_length in self._cache:
            return self._cache[seq_length]

        dl_train = self._dl_set["train_windows"].float()
        dl_test_series = self._dl_set["test_series"].float()
        dl_eval_windows = sliding_window_2d(dl_test_series, seq_length, stride=1)
        split_idx = dl_eval_windows.shape[0] // 2
        dl_valid = dl_eval_windows[:split_idx]
        dl_test = dl_eval_windows[split_idx:]
        if dl_test.shape[0] == 0:
            dl_test = dl_valid

        train_inits = dl_train[:, 0, :] if dl_train.shape[0] else torch.empty((0, dl_test_series.shape[1]))
        valid_inits = dl_valid[:, 0, :] if dl_valid.shape[0] else torch.empty((0, dl_test_series.shape[1]))
        test_inits = dl_test[:, 0, :] if dl_test.shape[0] else torch.empty((0, dl_test_series.shape[1]))

        train_stat = self._stats_set["train_series"].float()
        test_stat = self._stats_set["test_series"].float()
        full_stat = self._stats_set["full_series"].float()
        stat_windows = sliding_window_2d(test_stat, seq_length, stride=1)

        dataset = {
            "deep_learning_train": dl_train,
            "deep_learning_valid": dl_valid,
            "deep_learning_test": dl_test,
            "deep_learning_train_init": train_inits,
            "deep_learning_valid_init": valid_inits,
            "deep_learning_test_init": test_inits,
            "asset_columns": list(self._dl_set["feature_columns"]),
            "price_columns": list(self._dl_set["price_columns"]),
            "statistical_series": full_stat,
            "statistical_train": train_stat,
            "statistical_valid": torch.empty((0, train_stat.shape[1]), dtype=train_stat.dtype),
            "statistical_test": test_stat,
            "statistical_train_init": train_stat[0] if train_stat.shape[0] else torch.zeros(train_stat.shape[1]),
            "statistical_valid_init": torch.zeros(train_stat.shape[1]),
            "statistical_test_init": test_stat[0] if test_stat.shape[0] else torch.zeros(train_stat.shape[1]),
            "statistical_test_windows": stat_windows,
        }

        self._cache[seq_length] = dataset
        return dataset


class ArtifactLoader:
    """Handles loading and validation of generated artifacts."""

    @staticmethod
    def load(artifact_path: Path) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Load artifact and return data and metadata."""
        return load_artifact(artifact_path)

    @staticmethod
    def extract_metadata(artifact_path: Path, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and validate metadata from artifact."""
        return {
            "model_name": metadata.get("model_name") or artifact_path.parent.name,
            "model_type": _normalize_model_type(metadata.get("model_type", "deep_learning")),
            "sequence_length": int(metadata["sequence_length"]),
            "num_samples": int(metadata.get("num_samples", 0)),
            "num_channels": int(metadata.get("num_channels", 1)),
        }

    @staticmethod
    def prepare_data(
        data: torch.Tensor,
        num_samples: int,
    ) -> np.ndarray:
        """Prepare data for evaluation."""
        data = data[:num_samples]
        if data.ndim == 2:
            data = data.unsqueeze(-1)
        return _to_numpy(data)


class RealDataPreparer:
    """Prepares real data windows for comparison with generated data."""

    @staticmethod
    def prepare(
        dataset: Dict[str, Any],
        seq_length: int,
        model_type: str,
        num_samples: int,
    ) -> np.ndarray:
        """
        Prepare real data windows aligned with generated data at generation length.
        """
        model_type = _normalize_model_type(model_type)
        if model_type == "statistical":
            real_windows = dataset["statistical_test_windows"]
        else:
            real_windows = dataset["deep_learning_test"]

        real_data = _to_numpy(real_windows)

        # Align sample counts
        if real_data.shape[0] > num_samples:
            real_data = real_data[:num_samples]
        
        return real_data


class CoreMetricsEvaluator:
    """Evaluates core taxonomy metrics (fidelity, diversity, stylized facts, visual)."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir

    def evaluate(
        self,
        real_data: np.ndarray,
        generated_data: np.ndarray,
    ) -> Dict[str, Any]:
        """Run all core metric evaluations."""
        results: Dict[str, Any] = {}

        evaluators = [
            FidelityEvaluator(real_data, generated_data),
            DiversityEvaluator(real_data, generated_data),
            StylizedFactsEvaluator(real_data, generated_data),
            VisualAssessmentEvaluator(real_data, generated_data, self.output_dir),
        ]

        for evaluator in evaluators:
            evaluator_name = evaluator.__class__.__name__
            try:
                metric_results = evaluator.evaluate()
                if metric_results:
                    results.update(metric_results)
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] {evaluator_name} failed: {exc}")

        return results


class UtilityMetricsEvaluator:
    """Evaluates utility metrics (deep hedging)."""

    def __init__(
        self,
        num_epochs: int = UTILITY_NUM_EPOCHS,
        batch_size: int = UTILITY_BATCH_SIZE,
        learning_rate: float = UTILITY_LEARNING_RATE,
    ):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

    @staticmethod
    def _average_nested_dicts(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not results:
            return {}
        first = results[0]
        averaged: Dict[str, Any] = {}
        for key, value in first.items():
            values = [r[key] for r in results if key in r]
            if not values:
                continue
            if isinstance(value, dict):
                averaged[key] = UtilityMetricsEvaluator._average_nested_dicts(values)  # type: ignore[arg-type]
            elif isinstance(value, (int, float, np.floating)):
                averaged[key] = float(np.mean([float(v) for v in values]))
            else:
                averaged[key] = value
        return averaged

    def evaluate(
        self,
        generated_data: np.ndarray,
        dataset: Dict[str, Any],
        seq_length: int,
    ) -> Dict[str, Any]:
        """Run utility evaluation using deep hedging."""
        synthetic = torch.from_numpy(generated_data).float()
        if synthetic.ndim == 2:
            synthetic = synthetic.unsqueeze(-1)

        num_samples = synthetic.shape[0]
        train_end = int(num_samples * UTILITY_TRAIN_RATIO)
        val_end = int(num_samples * UTILITY_VAL_RATIO)

        feature_columns = dataset.get("asset_columns", [])
        price_columns = dataset.get("price_columns", feature_columns)
        price_indices = [feature_columns.index(c) for c in price_columns if c in feature_columns]
        if not price_indices:
            return {"utility_error": "No price channels available for utility evaluation."}

        synthetic = synthetic[:, :, price_indices]
        real_train_all = dataset["deep_learning_train"][:, :, price_indices]
        real_valid_all = dataset["deep_learning_valid"][:, :, price_indices]
        real_test_all = dataset["deep_learning_test"][:, :, price_indices]
        real_train_init_all = dataset["deep_learning_train_init"][:, price_indices]
        real_valid_init_all = dataset["deep_learning_valid_init"][:, price_indices]
        real_test_init_all = dataset["deep_learning_test_init"][:, price_indices]

        num_channels = synthetic.shape[-1]
        per_channel_results: List[Dict[str, Any]] = []
        for c in range(num_channels):
            synthetic_c = synthetic[:, :, c]
            synthetic_train = synthetic_c[:train_end]
            synthetic_val = synthetic_c[train_end:val_end]
            synthetic_test = synthetic_c[val_end:]

            real_train = real_train_all[:, :, c]
            real_val = real_valid_all[:, :, c]
            real_test = real_test_all[:, :, c]
            real_train_init = real_train_init_all[:, c]
            real_val_init = real_valid_init_all[:, c]
            real_test_init = real_test_init_all[:, c]

            if real_val.shape[0] == 0:
                real_val = real_test
                real_val_init = real_test_init

            if real_train.shape[0] == 0 or real_test.shape[0] == 0:
                per_channel_results.append({"utility_error": "Insufficient real data windows for utility evaluation."})
                continue

            mean_initial = float(real_train_init.mean().item())
            device = real_train_init.device
            synthetic_initials = {
                "train": torch.ones(train_end, device=device) * mean_initial,
                "val": torch.ones(val_end - train_end, device=device) * mean_initial,
                "test": torch.ones(num_samples - val_end, device=device) * mean_initial,
            }

            evaluator = UtilityEvaluator(
                real_train_log_returns=real_train,
                real_val_log_returns=real_val,
                real_test_log_returns=real_test,
                synthetic_train_log_returns=synthetic_train,
                synthetic_val_log_returns=synthetic_val,
                synthetic_test_log_returns=synthetic_test,
                real_train_initial=real_train_init,
                real_val_initial=real_val_init,
                real_test_initial=real_test_init,
                synthetic_train_initial=synthetic_initials["train"],
                synthetic_val_initial=synthetic_initials["val"],
                synthetic_test_initial=synthetic_initials["test"],
                seq_length=seq_length,
                num_epochs=self.num_epochs,
                batch_size=self.batch_size,
                learning_rate=self.learning_rate,
            )

            try:
                per_channel_results.append(evaluator.evaluate())
            except Exception as exc:  # noqa: BLE001
                per_channel_results.append({"utility_error": str(exc)})

        if num_channels == 1:
            return per_channel_results[0]
        return {
            "summary": self._average_nested_dicts(per_channel_results),
            "per_channel": per_channel_results,
        }


class UnifiedEvaluator:
    """
    Main evaluator that orchestrates the evaluation pipeline.

    Loads generated artifacts and evaluates them against real data using
    taxonomy metrics (fidelity, diversity, stylized facts, visual, utility).
    """

    def __init__(
        self,
        generated_dir: Path,
        results_dir: Path,
        seq_length_filter: Optional[List[int]] = None,
    ):
        self.generated_dir = Path(generated_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.seq_length_filter = set(seq_length_filter or [])

        # Initialize components
        deep_learning_cfg, statistical_cfg = get_dataset_cfgs()
        self.dataset_cache = DatasetCache(deep_learning_cfg, statistical_cfg)
        self.artifact_loader = ArtifactLoader()
        self.real_data_preparer = RealDataPreparer()
        self.core_metrics_evaluator = None  # Initialized per artifact
        self.utility_metrics_evaluator = UtilityMetricsEvaluator()

    def _should_evaluate(self, seq_length: int) -> bool:
        """Check if sequence length should be evaluated."""
        return not self.seq_length_filter or seq_length in self.seq_length_filter

    def _prepare_output_directory(self, seq_length: int, model_name: str) -> Path:
        """Create and return output directory for results."""
        output_dir = self.results_dir / f"seq_{seq_length}" / model_name
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _save_results(self, results: Dict[str, Any], output_dir: Path) -> None:
        """Save evaluation results to JSON file."""
        metrics_path = output_dir / "metrics.json"
        with metrics_path.open("w") as f:
            json.dump(results, f, indent=2, default=str)

    def evaluate_artifact(self, artifact_path: Path) -> Dict[str, Any]:
        """
        Evaluate a single artifact.

        Returns:
            Dictionary of evaluation results, or empty dict if skipped
        """
        # Load artifact
        data, metadata = self.artifact_loader.load(artifact_path)
        artifact_info = self.artifact_loader.extract_metadata(artifact_path, metadata)
        seq_length = artifact_info["sequence_length"]

        # Check if should evaluate
        if not self._should_evaluate(seq_length):
            return {}

        # Prepare data
        num_samples = artifact_info["num_samples"]
        generated_data = self.artifact_loader.prepare_data(data, num_samples)

        # Get real data for comparison
        dataset = self.dataset_cache.get_dataset(seq_length)
        real_data = self.real_data_preparer.prepare(
            dataset, seq_length, artifact_info["model_type"], num_samples
        )
        if real_data.ndim == 2:
            real_data = np.expand_dims(real_data, axis=-1)
        if generated_data.ndim == 2:
            generated_data = np.expand_dims(generated_data, axis=-1)
        if real_data.shape[-1] != generated_data.shape[-1]:
            min_channels = min(real_data.shape[-1], generated_data.shape[-1])
            real_data = real_data[:, :, :min_channels]
            generated_data = generated_data[:, :, :min_channels]

        # Prepare output directory
        output_dir = self._prepare_output_directory(seq_length, artifact_info["model_name"])
        self.core_metrics_evaluator = CoreMetricsEvaluator(output_dir)

        # Run evaluation
        show_with_start_divider(
            f"Evaluating {artifact_info['model_name']} @ seq {seq_length}"
        )

        results: Dict[str, Any] = {
            **artifact_info,
            "metadata": metadata,
        }

        # Core metrics
        core_results = self.core_metrics_evaluator.evaluate(real_data, generated_data)
        results.update(core_results)

        # Utility metrics
        utility_results = self.utility_metrics_evaluator.evaluate(
            generated_data, dataset, seq_length
        )
        results["utility"] = utility_results

        # Save results
        self._save_results(results, output_dir)

        show_with_end_divider(f"Finished {artifact_info['model_name']} @ seq {seq_length}")
        return results

    def run(self) -> Dict[str, Any]:
        """
        Run evaluation on all artifacts in the generated directory.

        Returns:
            Dictionary mapping artifact keys to evaluation results
        """
        # Validate input directory
        if not self.generated_dir.exists():
            raise FileNotFoundError(
                f"Generated data directory not found: {self.generated_dir}"
            )

        # Find all artifacts
        artifacts = sorted(self.generated_dir.glob("*/*.pt"))
        if not artifacts:
            artifacts = sorted(self.generated_dir.glob("*/artifacts/*.pt"))
        if not artifacts:
            raise FileNotFoundError(f"No artifacts found in {self.generated_dir}")

        # Evaluate each artifact
        all_results: Dict[str, Any] = {}
        for artifact_path in artifacts:
            try:
                result = self.evaluate_artifact(artifact_path)
                if result:
                    key = f"{result['model_name']}_seq_{result['sequence_length']}"
                    all_results[key] = result
            except Exception as exc:  # noqa: BLE001
                print(f"[ERROR] Failed to evaluate {artifact_path}: {exc}")

        # Save summary
        summary_path = self.results_dir / "complete_evaluation.json"
        with summary_path.open("w") as f:
            json.dump(all_results, f, indent=2, default=str)

        print(f"Saved evaluation summary to {summary_path}")
        return all_results


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Unified evaluation (artifact-only).")
    parser.add_argument(
        "--generated_dir",
        type=str,
        default=str(project_root / "src" / "experiments"),
        help="Directory containing generated artifacts.",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=str(project_root / "results"),
        help="Directory to store evaluation outputs.",
    )
    parser.add_argument(
        "--seq_lengths",
        type=int,
        nargs="*",
        default=None,
        help="Optional sequence lengths to evaluate (subset).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluator = UnifiedEvaluator(
        generated_dir=Path(args.generated_dir),
        results_dir=Path(args.results_dir),
        seq_length_filter=args.seq_lengths,
    )
    evaluator.run()


if __name__ == "__main__":
    main()
