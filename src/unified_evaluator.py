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
from src.utils.display_utils import show_with_start_divider, show_with_end_divider  # noqa: E402
from src.utils.evaluation_classes_utils import (  # noqa: E402
    DiversityEvaluator,
    FidelityEvaluator,
    StylizedFactsEvaluator,
    VisualAssessmentEvaluator,
    UtilityEvaluator,
)

from src.utils.preprocessed_data_utils import (  # noqa: E402
    channel_norm_stats,
    denormalize_channels,
    load_dl_set,
    load_stats_set,
    resolve_dl_set_path,
    resolve_stats_set_path,
    sliding_window_2d,
)


# Constants
UTILITY_TRAIN_RATIO = 0.8
UTILITY_VAL_RATIO = 0.9
UTILITY_NUM_EPOCHS = 40
UTILITY_BATCH_SIZE = 64
UTILITY_LEARNING_RATE = 1e-3
# Legacy mse UtilityEvaluator is the only active hedger-evaluation path post-Fecamp-revert
# (commit 9c4b980 was reverted and the FecampDeepHedgerEvaluator was removed). Strikes at
# S(0) European option per the legacy vendor UtilityEvaluator.
# Default to skipping regeneration when checkpoint regen isn't supported by the
# adapter. Set --skip_regenerate=False (or STONKBENCH_EVAL_REGENERATE=1) to force
# regeneration on adapters that implement load_state().
UTILITY_SKIP_REGENERATE_DEFAULT = True


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

    def __init__(self, dl_set_path: str | None = None, stats_set_path: str | None = None) -> None:
        self._dl_set = load_dl_set(dl_set_path or resolve_dl_set_path())
        self._stats_set = load_stats_set(stats_set_path or resolve_stats_set_path())
        self._cache: Dict[int, Dict[str, Any]] = {}

    def get_dataset(self, seq_length: int) -> Dict[str, Any]:
        """
        Get or create cached dataset for a given sequence length.
        """
        if seq_length in self._cache:
            return self._cache[seq_length]

        norm_stats = channel_norm_stats(self._dl_set)

        def _raw_series(split: str) -> torch.Tensor:
            raw_key = f"{split}_series_raw"
            if raw_key in self._dl_set:
                return self._dl_set[raw_key].float()
            series = self._dl_set[f"{split}_series"].float()
            if norm_stats is not None:
                return denormalize_channels(series, *norm_stats)
            return series

        dl_train = self._dl_set["train_windows"].float()
        if norm_stats is not None:
            dl_train = denormalize_channels(dl_train, *norm_stats)
        dl_test_series = _raw_series("test")
        dl_eval_windows = sliding_window_2d(dl_test_series, seq_length, stride=1)
        split_idx = dl_eval_windows.shape[0] // 2
        dl_valid = dl_eval_windows[:split_idx]
        dl_test = dl_eval_windows[split_idx:]
        if dl_test.shape[0] == 0:
            dl_test = dl_valid

        train_inits = dl_train[:, 0, :] if dl_train.shape[0] else torch.empty((0, dl_test_series.shape[1]))
        valid_inits = dl_valid[:, 0, :] if dl_valid.shape[0] else torch.empty((0, dl_test_series.shape[1]))
        test_inits = dl_test[:, 0, :] if dl_test.shape[0] else torch.empty((0, dl_test_series.shape[1]))

        train_stat = _raw_series("train")
        test_stat = _raw_series("test")
        full_stat = test_stat
        if "full_series" in self._stats_set:
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
            "asset_columns": list(metadata.get("asset_columns") or []),
            "price_columns": list(metadata.get("price_columns") or []),
            "model_checkpoint_manifest": list(metadata.get("model_checkpoint_manifest") or []),
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
    """Evaluates core taxonomy metrics (fidelity, diversity, stylized facts, visual).

    Per-channel results are averaged across all channels; the raw per-channel
    breakdown is preserved under ``per_channel`` for transparency.
    """

    def __init__(
        self,
        output_dir: Path,
        asset_columns: Optional[List[str]] = None,
        price_columns: Optional[List[str]] = None,
    ):
        self.output_dir = output_dir
        self.asset_columns = list(asset_columns or [])
        self.price_columns = list(price_columns or [])

    def _post_process(self, name: str, raw: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(raw, dict) or "per_channel" not in raw:
            return {name: raw} if raw else {}
        if not isinstance(raw["per_channel"], list):
            return {name: raw}
        flat = {k: v for k, v in raw.items() if k != "per_channel"}
        return {name: flat, f"{name}_per_channel": raw["per_channel"]}

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
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] {evaluator_name} failed: {exc}")
                continue
            if not metric_results:
                continue
            if evaluator_name == "VisualAssessmentEvaluator":
                # Visual assessment writes to disk and returns no per-channel metrics —
                # keep the legacy flat-writes key so downstream tooling sees it ran.
                results["visual_assessment"] = metric_results or {"saved": True}
            else:
                results.update(self._post_process(evaluator_name, metric_results))

        return results


class UtilityMetricsEvaluator:
    """Evaluates utility metrics (deep hedging).

    Phase 2: aggregates per-channel metrics into an overall mean.
    Post-revert (commit 9c4b980): the only active path is the legacy vendor
    ``UtilityEvaluator`` (mse, strikes at S(0) European option). The previous
    FecampDeepHedgerEvaluator (cvar / entropic / log_utility with synthetic /
    test / augmented data modes) was removed alongside that commit.
    """

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

    def _evaluate_mse(
        self,
        synthetic: torch.Tensor,
        dataset: Dict[str, Any],
        seq_length: int,
    ) -> Dict[str, Any]:
        """Single-pass vendor ``UtilityEvaluator`` (MSE deep hedging)."""
        num_samples = synthetic.shape[0]
        device = synthetic.device
        synthetic_initials = torch.zeros(num_samples, device=device)
        real_train_init = dataset["deep_learning_train_init"]
        if real_train_init.ndim > 1:
            real_train_init = real_train_init.mean(dim=-1)
        real_val_init = dataset["deep_learning_valid_init"]
        if real_val_init.ndim > 1:
            real_val_init = real_val_init.mean(dim=-1) if real_val_init.shape[0] else real_val_init
        real_test_init = dataset["deep_learning_test_init"]
        if real_test_init.ndim > 1:
            real_test_init = real_test_init.mean(dim=-1) if real_test_init.shape[0] else real_test_init

        evaluator = UtilityEvaluator(
            real_train_log_returns=dataset["deep_learning_train"],
            real_val_log_returns=dataset["deep_learning_valid"] if dataset["deep_learning_valid"].shape[0] else dataset["deep_learning_test"],
            real_test_log_returns=dataset["deep_learning_test"],
            synthetic_train_log_returns=synthetic,
            synthetic_val_log_returns=synthetic,
            synthetic_test_log_returns=synthetic,
            real_train_initial=real_train_init,
            real_val_initial=real_val_init,
            real_test_initial=real_test_init,
            synthetic_train_initial=synthetic_initials,
            synthetic_val_initial=synthetic_initials,
            synthetic_test_initial=synthetic_initials,
            seq_length=seq_length,
            num_epochs=self.num_epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
        )
        try:
            raw = evaluator.evaluate()
        except Exception as exc:  # noqa: BLE001
            return {"utility_error": str(exc)}
        return {"summary": raw}

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
        return self._evaluate_mse(synthetic, dataset, seq_length)


class UnifiedEvaluator:
    """
    Main evaluator that orchestrates the evaluation pipeline.

    Loads generated artifacts and evaluates them against real data using
    taxonomy metrics (fidelity, diversity, stylized facts, visual, utility).

    With ``skip_regenerate=False`` (default), for each artifact the evaluator
    attempts to regenerate a fresh `(R, L, C)` tensor from the adapter's
    latest checkpoint via ``adapter.load_state()`` + ``adapter.generate()``.
    Regeneration falls back to the existing artifact (with a warning) when the
    adapter doesn't expose ``can_regenerate_from_checkpoint`` or the
    regeneration raises.
    """

    def __init__(
        self,
        generated_dir: Path,
        results_dir: Path,
        seq_length_filter: Optional[List[int]] = None,
        skip_regenerate: bool = UTILITY_SKIP_REGENERATE_DEFAULT,
    ):
        self.generated_dir = Path(generated_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.seq_length_filter = set(seq_length_filter or [])
        self.skip_regenerate = bool(skip_regenerate)

        # Initialize components
        self.dataset_cache = DatasetCache()
        self.artifact_loader = ArtifactLoader()
        self.real_data_preparer = RealDataPreparer()
        self.core_metrics_evaluator = None  # Initialized per artifact
        self.utility_metrics_evaluator = UtilityMetricsEvaluator()

    def _maybe_regenerate(
        self,
        artifact_path: Path,
        metadata: Dict[str, Any],
    ) -> Tuple[Path, Optional[Dict[str, Any]]]:
        """Optionally regenerate an artifact's samples from its checkpoint.

        Returns ``(artifact_path, metadata)`` unchanged when:
          -- ``self.skip_regenerate`` is True (legacy path: eval existing artifact)
          -- checkpoint regeneration is unsupported by the adapter
          -- checkpoints or model_key resolution fails
          -- ``load_state`` or ``generate`` raises

        Returns ``(regen_path, regen_metadata)`` where ``regen_path`` is a sibling
        ``<artifact>.regen.pt`` written atomically when regeneration succeeds.
        The original pipeline output is never mutated in place (Phase 1 lock —
        reproducibility). Set ``--skip_regenerate=False`` (or set the env var
        ``STONKBENCH_EVAL_REGENERATE=1``) to enable; only adapters that
        implement ``load_state`` (currently ``ChannelBootstrapAdapter``) will
        actually regenerate.
        """
        if self.skip_regenerate:
            return artifact_path, None

        model_key = metadata.get("model_name") or artifact_path.parent.name
        try:
            from src.experiments.core.registry import get_adapter  # local import: avoid cycles at load time.

            adapter = get_adapter(model_key)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Could not load adapter for {model_key}: {exc}; using existing artifact.")
            return artifact_path, None

        if not getattr(adapter, "can_regenerate_from_checkpoint", False):
            print(f"[INFO] Adapter for {model_key} does not support checkpoint regeneration; using existing artifact.")
            return artifact_path, None

        checkpoint_manifest = metadata.get("model_checkpoint_manifest") or []
        ckpt_paths = [Path(p) for p in checkpoint_manifest if Path(p).exists()]
        if not ckpt_paths:
            print(f"[WARN] No on-disk checkpoints in manifest for {model_key}; using existing artifact.")
            return artifact_path, None

        # Eagerly construct both paths so the finally block can clean up the
        # tmp file even if an exception fires before assignment. ``return`` in a
        # try block always runs ``finally`` first, so the .tmp.pt is removed
        # on both success and error paths.
        regen_path = artifact_path.with_name(f"{artifact_path.stem}.regen.pt")
        tmp_path = regen_path.with_suffix(".tmp.pt")
        try:
            from src.utils.artifact_utils import save_artifact

            adapter.load_state(ckpt_paths)
            num_samples = int(metadata.get("num_samples", 0))
            seq_length = int(metadata.get("sequence_length", 0))
            seed = int(metadata.get("seed", 42))
            generated = adapter.generate(num_samples=num_samples, generation_length=seq_length, seed=seed)

            regen_metadata = dict(metadata)
            regen_metadata["regenerated_from_checkpoints"] = True
            save_artifact(generated.data, regen_metadata, tmp_path)
            tmp_path.replace(regen_path)
            print(f"[INFO] Regenerated fresh samples for {model_key} -> {regen_path.name}")
            return regen_path, regen_metadata
        except Exception as exc:  # noqa: BLE001 — never crash the eval loop on a regeneration hiccup.
            print(f"[WARN] Failed to regenerate samples for {model_key}: {exc}; using existing artifact.")
            return artifact_path, None
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass

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

        Trim contract (user-stated 2026-07-27): a single artifact trained at
        L_max produces eval results at every requested L <= L_max by trimming
        the LAST L timesteps. Last-position slicing preserves the causal/AR
        structure for AR models (cond_sig_wgan, kalman_vae, vrnn); for
        diffusion (UTSD, C-TSD) and i.i.d. statistical models, any contiguous
        window of length L is equivalent so last is the canonical choice.
        No retraining needed; ground truth at the target L is built
        stride=1 via ``sliding_window_2d`` inside ``DatasetCache``.

        Returns:
            Dictionary of evaluation results, or empty dict if no requested
            length is satisfiable by this artifact. Single-target callers
            receive the flat result dict; multi-target callers receive
            ``{"per_length": {L: results, ...}}``.
        """
        # Load artifact metadata + tensor.
        data, metadata = self.artifact_loader.load(artifact_path)
        # Optionally regenerate from latest checkpoint (Phase 1 Pending B).
        # _maybe_regenerate returns (artifact_path, None) when nothing changes,
        # or (regen_path, regen_metadata) on successful regeneration.
        eval_path, regen_metadata = self._maybe_regenerate(artifact_path, metadata)
        if regen_metadata is not None:
            # NOTE: explicit two-step unpack so `data` ends up as the tensor
            # alone, not the (tensor, meta) tuple returned by load().
            data, _orig_meta = self.artifact_loader.load(eval_path)
            metadata = regen_metadata
        artifact_info = self.artifact_loader.extract_metadata(eval_path, metadata)
        artifact_seq_length = artifact_info["sequence_length"]

        # Resolve target lengths. With no filter, evaluate only at the
        # artifact's native length (legacy behavior). With a filter, every
        # requested L <= artifact_seq_length is satisfied by trimming the
        # LAST L timesteps.
        if self.seq_length_filter:
            target_lengths = sorted(
                {L for L in self.seq_length_filter if 0 < L <= artifact_seq_length}
            )
        else:
            target_lengths = [artifact_seq_length]
        if not target_lengths:
            return {}

        # Prepare generated tensor ONCE; trim into per-target views inside the loop.
        num_samples = artifact_info["num_samples"]
        generated_data_full = self.artifact_loader.prepare_data(data, num_samples)
        if generated_data_full.ndim == 2:
            generated_data_full = np.expand_dims(generated_data_full, axis=-1)
        artifact_axis1 = generated_data_full.shape[1]

        per_length_results: Dict[int, Dict[str, Any]] = {}
        for target_length in target_lengths:
            if target_length < artifact_axis1:
                # Last-position trim preserves causal/AR structure end-to-end.
                generated_data = generated_data_full[:, -target_length:, :]
            else:
                generated_data = generated_data_full

            # Per-target ground truth: stride=1 sliding window over the test series.
            dataset = self.dataset_cache.get_dataset(target_length)
            real_data = self.real_data_preparer.prepare(
                dataset, target_length, artifact_info["model_type"], num_samples
            )
            if real_data.ndim == 2:
                real_data = np.expand_dims(real_data, axis=-1)
            if real_data.shape[-1] != generated_data.shape[-1]:
                min_channels = min(real_data.shape[-1], generated_data.shape[-1])
                real_data = real_data[:, :, :min_channels]
                generated_data = generated_data[:, :, :min_channels]

            output_dir = self._prepare_output_directory(
                target_length, artifact_info["model_name"]
            )
            self.core_metrics_evaluator = CoreMetricsEvaluator(
                output_dir=output_dir,
                asset_columns=artifact_info["asset_columns"],
                price_columns=artifact_info["price_columns"],
            )

            show_with_start_divider(
                f"Evaluating {artifact_info['model_name']}: "
                f"artifact @ seq {artifact_seq_length} -> trimmed to seq {target_length}"
            )

            results: Dict[str, Any] = {
                **artifact_info,
                "evaluated_at_length": target_length,
                "trimmed_from_artifact_length": artifact_seq_length,
                "trim_side": "last",
                "metadata": metadata,
                "regenerated_from_checkpoints": bool(regen_metadata is not None),
            }

            core_results = self.core_metrics_evaluator.evaluate(real_data, generated_data)
            results.update(core_results)
            utility_results = self.utility_metrics_evaluator.evaluate(
                generated_data, dataset, target_length
            )
            results["utility"] = utility_results
            self._save_results(results, output_dir)
            per_length_results[target_length] = results
            show_with_end_divider(
                f"Finished {artifact_info['model_name']} @ seq {target_length}"
            )

        # Back-compat: single-target callers (legacy scripts) expect the flat
        # result dict directly; multi-target callers receive the wrapped shape
        # with ``per_length`` keyed by target length. Spread ``artifact_info``
        # at the top level so run()'s unwrap branch and downstream consumers
        # (complete_evaluation.json readers) can read ``model_name`` /
        # ``sequence_length`` / ``num_channels`` without descending into the
        # per-length sub-dicts. Each sub itself already carries the per-target
        # ``evaluated_at_length`` / ``trimmed_from_artifact_length`` /
        # ``trim_side`` fields.
        if len(per_length_results) == 1:
            return next(iter(per_length_results.values()))
        return {**artifact_info, "per_length": per_length_results}

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

        # Evaluate each artifact.
        # Trim contract: evaluate_artifact returns either a flat dict
        # (single target length — legacy path) OR
        # ``{"per_length": {L: {...}, ...}}`` when multiple target lengths
        # were satisfied from a single L_max artifact via trimming. Unwrap
        # the multi-target shape so each evaluated length gets its own
        # ``model_seq{L}`` entry under all_results (preserves the legacy
        # complete_evaluation.json layout that downstream tooling depends on).
        all_results: Dict[str, Any] = {}
        for artifact_path in artifacts:
            try:
                result = self.evaluate_artifact(artifact_path)
                if not result:
                    continue
                if isinstance(result, dict) and isinstance(result.get("per_length"), dict):
                    model_name = result.get("model_name") or artifact_path.parent.name
                    for target_length, sub in result["per_length"].items():
                        key = f"{model_name}_seq{target_length}"
                        all_results[key] = sub
                else:
                    key = f"{result['model_name']}_seq{result['sequence_length']}"
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
    parser.add_argument(
        "--skip_regenerate",
        action="store_true",
        default=UTILITY_SKIP_REGENERATE_DEFAULT,
        help=(
            "Use the existing generated artifact as-is instead of regenerating "
            "from the latest checkpoint. Default: True (most DL adapters do "
            "not yet expose checkpoint regeneration)."
        ),
    )
    parser.add_argument(
        "--no_skip_regenerate",
        dest="skip_regenerate",
        action="store_false",
        help="Force regeneration from latest checkpoint when supported.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluator = UnifiedEvaluator(
        generated_dir=Path(args.generated_dir),
        results_dir=Path(args.results_dir),
        seq_length_filter=args.seq_lengths,
        skip_regenerate=bool(args.skip_regenerate),
    )
    evaluator.run()


if __name__ == "__main__":
    main()
