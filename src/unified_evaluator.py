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
import pandas as pd
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
from src.utils.preprocessing_utils import (  # noqa: E402
    preprocess_data,
    sliding_window_view,
    find_length,
    LogReturnTransformation,
)


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


class DatasetCache:
    """Manages caching of preprocessed real datasets for evaluation."""

    def __init__(self, non_param_cfg: Dict[str, Any], param_cfg: Dict[str, Any]):
        self.non_param_cfg = non_param_cfg
        self.param_cfg = param_cfg
        self._cache: Dict[int, Dict[str, Any]] = {}
        self._train_seq_length: Optional[int] = None

    def _infer_training_sequence_length(self) -> int:
        """Infer training sequence length from ACF analysis on training split only."""
        if self._train_seq_length is not None:
            return self._train_seq_length

        data_path = self.non_param_cfg.get('original_data_path')
        index = self.non_param_cfg.get('index')

        df = pd.read_csv(data_path)
        original_prices = torch.from_numpy(df[index].values)
        scaler = LogReturnTransformation()
        log_returns, _ = scaler.transform(original_prices)
        
        # Split to get training split only
        valid_ratio = self.non_param_cfg.get('valid_ratio', 0.1)
        test_ratio = self.non_param_cfg.get('test_ratio', 0.1)
        L = len(log_returns)
        train_end = int(L * (1 - valid_ratio - test_ratio))
        train_log_returns = log_returns[:train_end]

        self._train_seq_length = find_length(train_log_returns)
        return self._train_seq_length

    def get_dataset(self, seq_length: int) -> Dict[str, Any]:
        """
        Get or create cached dataset for a given sequence length.
        
        Preprocessing only cleans and splits data (no windows).
        Sliding windows are applied here:
        - Training: fixed window length (inferred from ACF on training split)
        - Validation/test: dynamic window length (seq_length parameter)
        
        For parametric models:
        - Uses standard preprocessing (no windows needed)
        """
        if seq_length in self._cache:
            return self._cache[seq_length]

        # Preprocess: ONLY clean and split (no sliding windows)
        train_log_returns_np, valid_log_returns_np, test_log_returns_np, train_init_np, valid_init_np, test_init_np = preprocess_data(
            self.non_param_cfg,
            supress_cfg_message=True,
        )
        
        # Get training sequence length (fixed, inferred from training split)
        train_seq_length = self._infer_training_sequence_length()
        
        # Load prices to get initial values for windows
        data_path = self.non_param_cfg.get('original_data_path')
        index = self.non_param_cfg.get('index')
        df = pd.read_csv(data_path)
        original_prices = torch.from_numpy(df[index].values).float()
        valid_ratio = self.non_param_cfg.get('valid_ratio', 0.1)
        test_ratio = self.non_param_cfg.get('test_ratio', 0.1)
        full_L = len(original_prices) - 1  # log_returns length
        train_end_full = int(full_L * (1 - valid_ratio - test_ratio))
        valid_end_full = int(full_L * (1 - test_ratio))
        
        # Apply sliding windows:
        # - Training: fixed length (train_seq_length)
        # - Validation/test: dynamic length (seq_length)
        train_data_np = sliding_window_view(train_log_returns_np, train_seq_length, stride=1)
        train_indices_np = torch.arange(0, len(train_data_np))
        train_prices = original_prices[:train_end_full+1]
        train_indices_np = train_indices_np[train_indices_np < len(train_prices)]
        train_data_np = train_data_np[:len(train_indices_np)]
        train_init_windows_np = train_prices[train_indices_np]
        
        # Validation windows at dynamic length (seq_length)
        if len(valid_log_returns_np) >= seq_length:
            valid_data_np = sliding_window_view(valid_log_returns_np, seq_length, stride=1)
            valid_indices_np = torch.arange(0, len(valid_data_np))
            valid_prices = original_prices[train_end_full:valid_end_full+1]
            valid_indices_np = valid_indices_np[valid_indices_np < len(valid_prices)]
            valid_data_np = valid_data_np[:len(valid_indices_np)]
            valid_init_windows_np = valid_prices[valid_indices_np]
        else:
            valid_data_np = torch.empty((0, seq_length), dtype=train_log_returns_np.dtype)
            valid_init_windows_np = torch.empty((0,), dtype=original_prices.dtype)
        
        # Test windows at dynamic length (seq_length)
        if len(test_log_returns_np) >= seq_length:
            test_data_np = sliding_window_view(test_log_returns_np, seq_length, stride=1)
            test_indices_np = torch.arange(0, len(test_data_np))
            test_prices = original_prices[valid_end_full:]
            test_indices_np = test_indices_np[test_indices_np < len(test_prices)]
            test_data_np = test_data_np[:len(test_indices_np)]
            test_init_windows_np = test_prices[test_indices_np]
        else:
            test_data_np = torch.empty((0, seq_length), dtype=train_log_returns_np.dtype)
            test_init_windows_np = torch.empty((0,), dtype=original_prices.dtype)

        # Prepare parametric datasets (no windows needed)
        (
            train_para,
            valid_para,
            test_para,
            train_init_para,
            valid_init_para,
            test_init_para,
        ) = preprocess_data(self.param_cfg, supress_cfg_message=True)

        dataset = {
            "nonparam_train": train_data_np,
            "nonparam_valid": valid_data_np,
            "nonparam_test": test_data_np,
            "nonparam_train_init": train_init_windows_np,
            "nonparam_valid_init": valid_init_windows_np,
            "nonparam_test_init": test_init_windows_np,
            "param_series": torch.cat([train_para, valid_para, test_para]),
            "param_train": train_para,
            "param_valid": valid_para,
            "param_test": test_para,
            "param_train_init": train_init_para,
            "param_valid_init": valid_init_para,
            "param_test_init": test_init_para,
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
            "model_type": metadata.get("model_type", "non_parametric"),
            "sequence_length": int(metadata["sequence_length"]),
            "num_samples": int(metadata.get("num_samples", 0)),
        }

    @staticmethod
    def prepare_data(
        data: torch.Tensor,
        num_samples: int,
    ) -> np.ndarray:
        """Prepare data for evaluation."""
        return _to_numpy(data[:num_samples])


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

        - Parametric: Create sliding windows from test set at generation length
        - Non-parametric: Use pre-windowed test set (windows already created in get_dataset at generation length)
        """
        if model_type == "parametric":
            # Create windows from test set log returns at generation length
            test_series = dataset["param_test"]
            if len(test_series) >= seq_length:
                real_windows = sliding_window_view(test_series, seq_length, stride=1)
            else:
                real_windows = torch.empty((0, seq_length), dtype=test_series.dtype)
        else:
            # Non-parametric: test windows are already created in get_dataset with dynamic length (generation length)
            real_windows = dataset["nonparam_test"]

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

    def evaluate(
        self,
        generated_data: np.ndarray,
        dataset: Dict[str, Any],
        seq_length: int,
    ) -> Dict[str, Any]:
        """Run utility evaluation using deep hedging."""
        synthetic = torch.from_numpy(generated_data).float()
        num_samples = synthetic.shape[0]

        # Split synthetic data
        train_end = int(num_samples * UTILITY_TRAIN_RATIO)
        val_end = int(num_samples * UTILITY_VAL_RATIO)

        synthetic_train = synthetic[:train_end]
        synthetic_val = synthetic[train_end:val_end]
        synthetic_test = synthetic[val_end:]

        # Prepare initial values for synthetic data
        mean_initial = float(dataset["nonparam_train_init"].mean().item())
        device = dataset["nonparam_train_init"].device

        synthetic_initials = {
            "train": torch.ones(train_end, device=device) * mean_initial,
            "val": torch.ones(val_end - train_end, device=device) * mean_initial,
            "test": torch.ones(num_samples - val_end, device=device) * mean_initial,
        }

        # Create utility evaluator
        evaluator = UtilityEvaluator(
            real_train_log_returns=dataset["nonparam_train"],
            real_val_log_returns=dataset["nonparam_valid"],
            real_test_log_returns=dataset["nonparam_test"],
            synthetic_train_log_returns=synthetic_train,
            synthetic_val_log_returns=synthetic_val,
            synthetic_test_log_returns=synthetic_test,
            real_train_initial=dataset["nonparam_train_init"],
            real_val_initial=dataset["nonparam_valid_init"],
            real_test_initial=dataset["nonparam_test_init"],
            synthetic_train_initial=synthetic_initials["train"],
            synthetic_val_initial=synthetic_initials["val"],
            synthetic_test_initial=synthetic_initials["test"],
            seq_length=seq_length,
            num_epochs=self.num_epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
        )

        try:
            return evaluator.evaluate()
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Utility evaluation failed: {exc}")
            return {"utility_error": str(exc)}


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
        non_param_cfg, param_cfg = get_dataset_cfgs()
        self.dataset_cache = DatasetCache(non_param_cfg, param_cfg)
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
        default=str(project_root / "generated_data"),
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
