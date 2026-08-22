"""Regenerate missing visual assessments (tsne, distribution, QQ, per-asset).

The VisualAssessmentEvaluator previously wrapped every visualization in a single
try/except, and ``visualize_tsne`` hard-coded ``perplexity=40``. On small eval
sets (seq_252 with ~18 aligned windows; DL artifacts with ~9 samples) t-SNE
crashed and the QQ / distribution / per-asset plots were silently skipped.

This script re-runs ONLY the visual assessment (no metrics, no utility, no
portfolio/P&L) for every ``seq_<L>/<model>`` directory whose ``qq.png`` is
missing. Data loading, trimming, and sample alignment mirror
``unified_evaluator.evaluate_artifact`` exactly so the regenerated plots sit
alongside the already-computed metrics.

Usage:
    python scripts/regen_visuals.py \
        --generated_dir /scratch/epham/stonkbench/output/results/latest \
        --results_dir /scratch/epham/stonkbench/output/results/latest/evaluation \
        [--workers 8]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.unified_evaluator import (  # noqa: E402
    ArtifactLoader,
    DatasetCache,
    RealDataPreparer,
)
from src.utils.evaluation_classes_utils import VisualAssessmentEvaluator  # noqa: E402

ALL_SEQ_LENGTHS = (21, 42, 126, 252)


def _prepare_for_length(loader: ArtifactLoader, dataset_cache: DatasetCache,
                        artifact_path: Path, target_length: int):
    """Replicate the trim + alignment from evaluate_artifact for one target L."""
    data, metadata = loader.load(artifact_path)
    info = loader.extract_metadata(artifact_path, metadata)
    num_samples = info["num_samples"]
    gen_full = loader.prepare_data(data, num_samples)
    if gen_full.ndim == 2:
        gen_full = np.expand_dims(gen_full, axis=-1)
    artifact_axis1 = gen_full.shape[1]
    if target_length < artifact_axis1:
        gen = gen_full[:, -target_length:, :]
    else:
        gen = gen_full

    dataset = dataset_cache.get_dataset(target_length)
    real = RealDataPreparer.prepare(dataset, target_length, info["model_type"], num_samples)
    if real.ndim == 2:
        real = np.expand_dims(real, axis=-1)
    if real.shape[-1] != gen.shape[-1]:
        min_channels = min(real.shape[-1], gen.shape[-1])
        real = real[:, :, :min_channels]
        gen = gen[:, :, :min_channels]
    n_align = min(real.shape[0], gen.shape[0])
    if n_align < real.shape[0] or n_align < gen.shape[0]:
        real = real[:n_align]
        gen = gen[:n_align]
    return real, gen, info


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated_dir", type=str, required=True)
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--model", type=str, default=None, help="Restrict to one model.")
    args = parser.parse_args()

    generated_dir = Path(args.generated_dir)
    results_dir = Path(args.results_dir)
    loader = ArtifactLoader()
    dataset_cache = DatasetCache()

    # Build a lookup of canonical artifacts keyed by (metadata model_name, seq):
    # the evaluator's --seq_length L filter keeps exactly ``<name>_seq<L>.pt``
    # and reports results under metadata model_name (which may differ from the
    # on-disk artifact dir, e.g. cond_sig_wgan -> csigwgan_noise10).
    artifact_lookup: dict[tuple[str, int], Path] = {}
    for model_dir in sorted(generated_dir.iterdir()):
        if not model_dir.is_dir() or model_dir.name == "ground_truth":
            continue
        artifact_dir = model_dir / "artifacts"
        if not artifact_dir.is_dir():
            continue
        for artifact in sorted(artifact_dir.glob("*.pt")):
            if ".regen." in artifact.name:
                continue
            try:
                _data, meta = loader.load(artifact)
                name = meta.get("model_name") or artifact.parent.parent.name
                seq = int(meta["sequence_length"])
            except Exception as exc:  # noqa: BLE001
                print(f"  [WARN] could not read {artifact.name}: {exc}")
                continue
            # Only the native-length artifact is canonical for its seq.
            if artifact.name.endswith(f"_seq{seq}.pt"):
                artifact_lookup[(name, seq)] = artifact

    tasks = []
    for out_dir in sorted(results_dir.glob("seq_*/*")):
        if not out_dir.is_dir():
            continue
        model = out_dir.name
        seq = int(out_dir.parent.name.split("_", 1)[1])
        if args.model and args.model not in model:
            continue
        if (out_dir / "visualizations" / "qq.png").exists():
            continue  # already has visualizations
        artifact = artifact_lookup.get((model, seq))
        if artifact is None:
            print(f"  [SKIP] {model} seq{seq}: canonical artifact not found")
            continue
        tasks.append((artifact, seq, out_dir))

    if not tasks:
        print("No missing visualizations — all tasks already have qq.png.")
        return

    print(f"Regenerating visualizations for {len(tasks)} tasks...")
    n_ok = 0
    for artifact, seq, out_dir in tasks:
        try:
            real, gen, info = _prepare_for_length(loader, dataset_cache, artifact, seq)
            evaluator = VisualAssessmentEvaluator(
                real, gen, out_dir, channel_names=info["asset_columns"])
            evaluator.evaluate()
            n_ok += 1
            print(f"  ✓ {info['model_name']} seq{seq}")
        except Exception as exc:  # noqa: BLE001
            print(f"  ✗ {artifact.name} seq{seq}: {exc}")

    qq_count = sum(
        1 for out_dir in results_dir.glob("seq_*/*/visualizations/qq.png"))
    print(f"\nDone: {n_ok}/{len(tasks)} regenerated. "
          f"Total qq.png on disk now: {qq_count}")


if __name__ == "__main__":
    main()
