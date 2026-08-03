"""Final full training run using HP-search winners and post-training sanity plots."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.experiments.core.io import resolve_run_id, results_root
from src.experiments.hp_configs import MODEL_HP_CONFIGS
from src.experiments.core.pipeline import run_model_experiment
from src.experiments.core.registry import ADAPTER_REGISTRY, STATISTICAL_MODEL_KEYS, VARIANT_TO_BASE
from src.experiments.hp_configs import DL_MODEL_KEYS as HP_DL_MODEL_KEYS
from src.experiments.hp_configs import full_train_metadata
from src.utils.device import device_to_str, get_device, log_device_context
from src.utils.env import get_output_root

DEFAULT_OUTPUT_ROOT = get_output_root()
ALL_MODEL_KEYS = sorted(ADAPTER_REGISTRY.keys())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train final models from HP-search winners.")
    parser.add_argument("--hp_summary", type=str, default="")
    parser.add_argument("--output_root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run_id", type=str, default="",
                        help="Run id subfolder under outputs/. Default 'latest' — "\
                             "re-runs overwrite in-place; archive_existing sweeps "\
                             "the previous active contents to outputs/_legacy/.")
    # Default sequence length = 252 (≈1 trading year of daily bars). This
    # only sets the per-model `--generation_length` flag; the actual train
    # window L comes from `dl_set["window_size"]` set during preprocessing
    # (`python -m src.data_preprocessing --window_size 252`). If those two
    # values disagree, the adapter will silently stitch short windows up
    # to 252 on every generate() call. Keep them in sync.
    parser.add_argument("--generation_length", type=int, default=252)
    parser.add_argument("--seq_lengths", nargs="+", type=int, default=None,
                        help="Additional sequence lengths to generate. By "
                             "default each seq_len gets its own native draw "
                             "(adapters with supports_arbitrary_generation); "
                             "with --trim_from_max, shorter lengths are "
                             "sliced from a single --generation_length draw.")
    parser.add_argument("--trim_from_max", action="store_true",
                        help="Generate a single tensor at max(seq_lengths) and "
                             "trim/stitch shorter lengths from it (instead of "
                             "regenerating natively at each seq_len). Shorter "
                             "windows are then exact prefixes of the same "
                             "underlying draw. Caveat: per-seq_len samples "
                             "are NOT independent draws in this mode.")
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--models", nargs="+", default=ALL_MODEL_KEYS)
    parser.add_argument("--smoke", action="store_true", help="Tiny sample/epoch budget for validation")
    parser.add_argument("--skip_sanity", action="store_true")
    return parser.parse_args()


def _make_smoke_hp_summary(model_keys):
    """Synthetic HP summary for --smoke runs. full_train_metadata() extracts
    best_config fields (patience, batch_size, learning_rate) and reads
    mean_best_val_loss from inside best_config. We re-use the vendor_default
    HPConfig and stamp mean_best_val_loss=0.0. No double-applied max_epochs:
    main() overrides it to 1 if args.smoke, else to FULL_TRAIN_EPOCHS.

    Statistical models are skipped here — they have no entry in MODEL_HP_CONFIGS
    and don't read the HP summary at training time (see _training_metadata).
    """
    import dataclasses
    models_block = {}
    for model_key in model_keys:
        # Skip models that don't have HP configs (statistical models that
        # only read generation_length). The check is dual: STATISTICAL_MODEL_KEYS
        # for explicit category membership AND MODEL_HP_CONFIGS as a fallback
        # for any model that lacks HP configs (e.g., future stat additions
        # that aren't yet in STATISTICAL_MODEL_KEYS).
        if model_key in STATISTICAL_MODEL_KEYS:
            continue
        if model_key not in MODEL_HP_CONFIGS:
            # Silent skip — same semantics as stat models. We don't raise
            # because the caller is best-effort populating an HP summary,
            # and `_training_metadata` handles stat models separately.
            print(f"[no-hp-tuning] Skipping {model_key} (no HP config; treated as stat-style).")
            continue
        cfgs = MODEL_HP_CONFIGS[model_key]
        cfg = next((c for c in cfgs if getattr(c, "is_vendor_default", False)), cfgs[0])
        config = dataclasses.asdict(cfg)
        config["mean_best_val_loss"] = 0.0
        models_block[model_key] = {"best_config": config}
    return {"models": models_block}


def _load_hp_summary(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"HP summary not found at {path}. Run HP search and aggregate first."
        )
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _training_metadata(
    model_key: str,
    hp_summary: Dict[str, Any],
    generation_length: int,
    smoke: bool,
) -> Dict[str, Any]:
    if model_key in STATISTICAL_MODEL_KEYS:
        return {"generation_length": generation_length}

    if model_key not in HP_DL_MODEL_KEYS and model_key not in VARIANT_TO_BASE:
        raise ValueError(f"No HP configuration for model '{model_key}'")

    model_entry = hp_summary.get("models", {}).get(model_key)
    if not model_entry or not model_entry.get("best_config"):
        raise ValueError(f"HP summary has no best_config for '{model_key}'")

    # Variants use their OWN MODEL_HP_CONFIGS entry (e.g.,
    # MODEL_HP_CONFIGS["quantgan_clipfix"]), not the base model's config.
    # The variant's extras (clip_value=0.05, noise_scale=0.20, etc.) flow
    # through full_train_metadata via the variant HPConfig's extras dict.
    # _make_smoke_hp_summary already writes variant entries into
    # hp_summary["models"][model_key], so model_entry exists.
    metadata = full_train_metadata(model_key, model_entry)
    if smoke:
        metadata["max_epochs"] = 1
        metadata["patience"] = 1
    metadata["generation_length"] = generation_length
    return metadata


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    if args.run_id:
        os.environ["STONKBENCH_RUN_ID"] = args.run_id
    run_id = resolve_run_id(args.run_id or None)
    os.environ["STONKBENCH_RUN_ID"] = run_id
    run_results = results_root(output_root, run_id)
    hp_summary_path = Path(args.hp_summary) if args.hp_summary else run_results / "hp_search" / "summary.json"
    if not hp_summary_path.exists():
        # No HP-search summary available. Per 2026-07-29 decision, HP tuning
        # is decommissioned; bake-in 'vendor_best' configs from MODEL_HP_CONFIGS
        # are used directly. Same fallback path as --smoke (single config per
        # model + full_train_epochs caps), but without the smoke-specific
        # epochs=1/patience=1 overrides.
        print(
            f"[no-hp-tuning] No HP summary at {hp_summary_path}; using 'vendor_best' HP "
            f"from MODEL_HP_CONFIGS for {args.models} (smoke={args.smoke}).",
            flush=True,
        )
        hp_summary = _make_smoke_hp_summary(args.models)
    else:
        hp_summary = _load_hp_summary(hp_summary_path)

    device = device_to_str(get_device(args.device))
    print(log_device_context())
    print(f"Run id: {run_id}")
    print(f"HP summary: {hp_summary_path}")
    print(f"Output root: {output_root}")
    print(f"Results dir: {run_results}")

    num_samples = 16 if args.smoke else args.num_samples
    num_epochs = 1 if args.smoke else 1
    sanity_dir = None if args.skip_sanity else output_root / "sanity" / run_id

    all_seq_lengths = [args.generation_length]
    if args.seq_lengths:
        all_seq_lengths.extend(sorted(set(args.seq_lengths) - {args.generation_length}))
    all_seq_lengths.sort(reverse=True)

    # Validate trim mode up front so a misuse fails with a clear message
    # BEFORE we waste compute on a 252-then-500 generate attempt.
    if args.trim_from_max:
        bad = [L for L in all_seq_lengths if L > args.generation_length]
        if bad:
            raise ValueError(
                f"--trim_from_max requires every --seq_lengths to be <= --generation_length "
                f"({args.generation_length}); offending values: {sorted(bad)}"
            )

    artifacts: List[Path] = []
    for model_key in args.models:
        if model_key not in ADAPTER_REGISTRY and model_key not in VARIANT_TO_BASE:
            raise ValueError(f"Unknown model key: {model_key}")

        training_metadata = _training_metadata(
            model_key=model_key,
            hp_summary=hp_summary,
            generation_length=max(all_seq_lengths),
            smoke=args.smoke,
        )
        print(f"\n{'='*60}")
        print(f"Training {model_key} (seq_lengths={all_seq_lengths})")
        print(f"{'='*60}")

        model_artifacts = run_model_experiment(
            model_key=model_key,
            generation_length=max(all_seq_lengths),
            num_samples=num_samples,
            num_epochs=num_epochs,
            seed=args.seed,
            device=device,
            output_root=output_root,
            training_metadata=training_metadata,
            sanity_output_dir=sanity_dir,
            seq_lengths=all_seq_lengths,
            trim_from_max=args.trim_from_max,
        )
        artifacts.extend(model_artifacts)

    print("\nSaved artifacts:")
    for artifact in artifacts:
        print(f"- {artifact}")
    if sanity_dir is not None:
        print(f"Sanity plots under: {sanity_dir}")


if __name__ == "__main__":
    main()
