#!/usr/bin/env python3
"""Mock pipeline outputs for downstream evaluation development.

Produces the same layout the real pipeline hands to `unified_evaluator.py`:

  <out>/results/<model>/artifacts/<model>_seq_<L>.pt

Each `.pt` is a dict `{"data": FloatTensor[R, L, C], "metadata": {...}}`
matching `src.utils.artifact_utils.save_artifact`.

Usage:
  python scripts/mock_eval_inputs.py
  python scripts/mock_eval_inputs.py --num_samples 64 --seq_lengths 21 52 --zip
  python scripts/mock_eval_inputs.py --out /tmp/mock_eval --zip

Then point eval at the mock results dir:
  python src/unified_evaluator.py \\
    --generated_dir <out>/results \\
    --results_dir <out>/results/evaluation \\
    --seq_lengths 21
"""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch

from src.experiments.core.registry import ADAPTER_REGISTRY, STATISTICAL_MODEL_KEYS
from src.utils.artifact_utils import default_metadata, save_artifact
from src.utils.preprocessed_data_utils import load_dl_set, resolve_dl_set_path


DEFAULT_OUT = Path("/home/epham/StonkBench/output/mock_eval_bundle")
DEFAULT_SEQ_LENGTHS = (100,)
DEFAULT_NUM_SAMPLES = 128
DEFAULT_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mock synthetic-data artifacts for evaluation pipeline work."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Output directory for the mock bundle (results/ written underneath).",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=sorted(ADAPTER_REGISTRY.keys()),
        help="Model keys to mock (default: all registered adapters).",
    )
    parser.add_argument(
        "--seq_lengths",
        type=int,
        nargs="*",
        default=list(DEFAULT_SEQ_LENGTHS),
        help="Generation lengths to write (default: 21).",
    )
    parser.add_argument("--num_samples", type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--dl_set_path",
        type=str,
        default=None,
        help="Optional override for dl_set.pt (used for channel names + realistic noise).",
    )
    parser.add_argument(
        "--zip",
        action="store_true",
        help="Also write <out>.zip containing the mock bundle.",
    )
    parser.add_argument(
        "--zip_path",
        type=Path,
        default=None,
        help="Explicit zip path (default: <out>.zip next to --out).",
    )
    return parser.parse_args()


def _load_channel_context(dl_set_path: str | None) -> Dict[str, Any]:
    path = dl_set_path or resolve_dl_set_path()
    dl_set = load_dl_set(path)
    feature_columns = list(dl_set.get("feature_columns") or [])
    price_columns = list(dl_set.get("price_columns") or [])
    if not feature_columns:
        raise ValueError(f"dl_set at {path} has no feature_columns")

    # Prefer raw test series for mock noise in feature space.
    series = dl_set.get("test_series_raw")
    if series is None:
        series = dl_set.get("test_series")
    if series is None:
        series = dl_set.get("train_series_raw")
    if not isinstance(series, torch.Tensor):
        raise ValueError(f"dl_set at {path} missing series tensors for mock sampling")

    return {
        "dl_set_path": str(path),
        "feature_columns": feature_columns,
        "price_columns": price_columns,
        "num_channels": int(series.shape[-1]),
        "series": series.float().cpu(),
        "window_size": int(dl_set.get("window_size", 100)),
    }


def _mock_paths(series: torch.Tensor, seq_len: int, num_samples: int, seed: int) -> torch.Tensor:
    """Build (R, L, C) paths by sampling real windows and adding small noise."""
    t_len, channels = series.shape
    if t_len < seq_len:
        raise ValueError(f"series length {t_len} < requested seq_len {seq_len}")

    g = torch.Generator().manual_seed(seed)
    max_start = t_len - seq_len
    starts = torch.randint(0, max_start + 1, (num_samples,), generator=g)
    windows = torch.stack([series[int(s) : int(s) + seq_len] for s in starts], dim=0)

    # Light model-specific-looking noise so artifacts are not identical copies.
    noise_scale = 0.02 * windows.std(dim=(0, 1), keepdim=True).clamp_min(1e-6)
    noise = torch.randn(windows.shape, generator=g) * noise_scale
    return (windows + noise).contiguous()


def _model_type(model_key: str) -> str:
    return "statistical" if model_key in STATISTICAL_MODEL_KEYS else "deep_learning"


def write_mock_artifact(
    *,
    out_root: Path,
    model_key: str,
    seq_len: int,
    num_samples: int,
    seed: int,
    ctx: Dict[str, Any],
) -> Path:
    data = _mock_paths(
        series=ctx["series"],
        seq_len=seq_len,
        num_samples=num_samples,
        seed=seed + abs(hash((model_key, seq_len))) % 10_000,
    )
    if data.shape[-1] != ctx["num_channels"]:
        raise RuntimeError("channel mismatch in mock data")

    preprocessing_cfg = {
        "dataset_kind": "statistical" if model_key in STATISTICAL_MODEL_KEYS else "deep_learning",
        "dl_set_path": ctx["dl_set_path"],
        "stats_set_path": str(Path(ctx["dl_set_path"]).with_name("statsmodel_set.pt")),
        "mock": True,
    }
    metadata = default_metadata(
        model_name=model_key,
        model_type=_model_type(model_key),
        sequence_length=seq_len,
        num_samples=num_samples,
        seed=seed,
        preprocessing_cfg=preprocessing_cfg,
        generator_version="mock_v1",
        extra={
            "num_channels": ctx["num_channels"],
            "asset_columns": ctx["feature_columns"],
            "price_columns": ctx["price_columns"],
            "is_multivariate": True,
            "train_sequence_length": int(ctx["window_size"]),
            "model_checkpoint_manifest": [],
            "is_mock": True,
            "mock_note": "Synthetic stand-in for evaluation pipeline development.",
        },
    )

    artifact_path = (
        out_root / "results" / model_key / "artifacts" / f"{model_key}_seq_{seq_len}.pt"
    )
    save_artifact(data, metadata, artifact_path)
    return artifact_path


def write_bundle_readme(out_root: Path, manifest: Dict[str, Any]) -> Path:
    readme = out_root / "README.md"
    models = ", ".join(manifest["models"])
    seqs = ", ".join(str(x) for x in manifest["seq_lengths"])
    readme.write_text(
        f"""# Mock evaluation inputs

Generated by `scripts/mock_eval_inputs.py` for downstream evaluation work.

## Layout

```
results/
  <model>/
    artifacts/
      <model>_seq_<L>.pt
```

This matches what `scripts/slurm/eval.sh` passes to `src/unified_evaluator.py`
via `--generated_dir .../results`.

## Contents

- Models: {models}
- Sequence lengths: {seqs}
- Samples per artifact: {manifest["num_samples"]}
- Channels: {manifest["num_channels"]}
- Created: {manifest["created_at"]}

## Run evaluation against this bundle

```bash
python src/unified_evaluator.py \\
  --generated_dir {out_root}/results \\
  --results_dir {out_root}/results/evaluation \\
  --seq_lengths {manifest["seq_lengths"][0]}
```

Each `.pt` file is:

```python
{{
  "data": torch.FloatTensor[num_samples, seq_len, num_channels],
  "metadata": {{ ... required artifact metadata ... }}
}}
```

See `MANIFEST.json` for a machine-readable inventory.
""",
        encoding="utf-8",
    )
    return readme


def zip_bundle(out_root: Path, zip_path: Path) -> Path:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(out_root.rglob("*")):
            if path.is_file():
                zf.write(path, arcname=str(path.relative_to(out_root.parent)))
    return zip_path


def main() -> None:
    args = parse_args()
    unknown = [m for m in args.models if m not in ADAPTER_REGISTRY]
    if unknown:
        raise SystemExit(f"Unknown model keys: {unknown}")

    out_root = args.out.resolve()
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    ctx = _load_channel_context(args.dl_set_path)
    written: List[str] = []
    for model_key in args.models:
        for seq_len in args.seq_lengths:
            path = write_mock_artifact(
                out_root=out_root,
                model_key=model_key,
                seq_len=int(seq_len),
                num_samples=int(args.num_samples),
                seed=int(args.seed),
                ctx=ctx,
            )
            written.append(str(path.relative_to(out_root)))
            print(f"wrote {path}  shape=({args.num_samples}, {seq_len}, {ctx['num_channels']})")

    manifest: Dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "generator": "scripts/mock_eval_inputs.py",
        "is_mock": True,
        "models": list(args.models),
        "seq_lengths": [int(x) for x in args.seq_lengths],
        "num_samples": int(args.num_samples),
        "num_channels": int(ctx["num_channels"]),
        "seed": int(args.seed),
        "dl_set_path": ctx["dl_set_path"],
        "asset_columns": ctx["feature_columns"],
        "price_columns": ctx["price_columns"],
        "artifacts": written,
        "eval_command": (
            f"python src/unified_evaluator.py "
            f"--generated_dir {out_root}/results "
            f"--results_dir {out_root}/results/evaluation "
            f"--seq_lengths {' '.join(str(x) for x in args.seq_lengths)}"
        ),
    }
    (out_root / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_bundle_readme(out_root, manifest)

    zip_path = args.zip_path
    if args.zip or zip_path is not None:
        if zip_path is None:
            zip_path = out_root.with_suffix(".zip")
        zip_bundle(out_root, zip_path.resolve())
        print(f"zipped {zip_path.resolve()}")

    print(f"bundle ready: {out_root}")


if __name__ == "__main__":
    main()
