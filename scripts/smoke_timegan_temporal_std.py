#!/usr/bin/env python3
"""Smoke / diagnostic: brief TimeGAN train on one channel; assert temporal std >> 0.

CPU-light path (1-layer GRU, short L, few iters). Full-benchmark training still
uses adapter defaults (3 layers, joint ER/S/G/D, longer epochs).

Usage (from repo root, stonkbench env):
  export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCH_NUM_THREADS=1
  export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
  python scripts/smoke_timegan_temporal_std.py
"""

from __future__ import annotations

import os

# Avoid OpenMP oversubscription blowing CPU-time quotas in shared environments.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("TORCH_NUM_THREADS", "1")

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiments.adapters.deep_learning.timegan_adapter import TimeGANAdapter


def main() -> None:
    ds = torch.load(ROOT / "data/preprocessed/dl_set.pt", map_location="cpu", weights_only=False)
    seq_len = 16
    train = TimeGANAdapter._winsorize_channel(
        ds["train_windows"][:32, :seq_len, :1].numpy().astype(np.float32)
    )
    valid = TimeGANAdapter._winsorize_channel(
        ds["valid_windows"][:8, :seq_len, :1].numpy().astype(np.float32)
    )
    print(f"windows train={train.shape} valid={valid.shape}", flush=True)

    TimeGAN, random_generator = TimeGANAdapter._import_timegan()
    opt = SimpleNamespace(
        manualseed=0,
        isTrain=True,
        data_name="custom",
        z_dim=1,
        seq_len=seq_len,
        module="gru",
        hidden_dim=16,
        num_layer=1,
        iteration=1,
        batch_size=8,
        metric_iteration=1,
        workers=0,
        device="cpu",
        gpu_ids=[],
        ngpu=0,
        model="TimeGAN",
        outf="/tmp/timegan_smoke",
        name="spy",
        display=False,
        display_server="http://localhost",
        display_port=8097,
        display_id=0,
        print_freq=1000,
        load_weights=False,
        resume="",
        beta1=0.9,
        lr=1e-3,
        w_gamma=1.0,
        w_es=0.1,
        w_e0=10.0,
        w_g=100.0,
    )
    model = TimeGAN(opt, train)
    valid_n = TimeGANAdapter._normalize_like_model(valid, model)

    # Evidence that early-stop must use NormMinMax scale (not raw z-scores).
    x_z = torch.tensor(valid[:8], dtype=torch.float32)
    x_n = torch.tensor(valid_n[:8], dtype=torch.float32)
    mse = torch.nn.functional.mse_loss
    mse_z0 = float(mse(torch.zeros_like(x_z), x_z))
    mse_z05 = float(mse(torch.full_like(x_z, 0.5), x_z))
    mse_n0 = float(mse(torch.zeros_like(x_n), x_n))
    mse_n05 = float(mse(torch.full_like(x_n, 0.5), x_n))
    print(f"MSE const0 vs z={mse_z0:.4f} vs norm={mse_n0:.4f}", flush=True)
    print(f"MSE 0.5    vs z={mse_z05:.4f} vs norm={mse_n05:.4f}", flush=True)
    if not (mse_n05 < mse_n0 and mse_z0 < mse_z05):
        raise SystemExit("FAIL: expected early-stop scale bias pattern")

    # ER-only is enough to verify batch_first + non-collapse; keep G light on CPU.
    for _ in range(12):
        model.train_one_iter_er()
    for _ in range(6):
        model.train_one_iter_s()
    for _ in range(6):
        model.train_one_iter_g()
        model.train_one_iter_er_()

    val_loss = TimeGANAdapter._eval_val_loss(model, valid_n, random_generator, 8)
    print(f"val_loss_normed={val_loss:.6g}", flush=True)

    generated = model.generation(num_samples=16, mean=0.0, std=1.0)
    arr = np.stack([np.asarray(g).squeeze(-1) for g in generated], axis=0)
    tstd = arr.std(axis=1)
    mean_tstd = float(tstd.mean())
    frac_flat = float((tstd < 1e-4).mean())
    train_tstd = float(train.squeeze(-1).std(axis=1).mean())
    print(
        f"generated mean temporal std={mean_tstd:.6g} "
        f"frac_flat={frac_flat:.3f} train_mean_tstd={train_tstd:.6g}",
        flush=True,
    )
    print("sample0 first8:", arr[0, :8].tolist(), flush=True)

    TimeGANAdapter._warn_if_collapsed(torch.from_numpy(arr).float(), 0, train_tstd)

    # Prior collapsed run had mean temporal std ~5e-5; require clearly above that.
    ok = mean_tstd > 0.01 and frac_flat < 0.25
    if not ok:
        raise SystemExit(
            f"FAIL: TimeGAN still collapsed (mean_tstd={mean_tstd:.6g}, frac_flat={frac_flat:.3f})"
        )
    print("PASS: temporal variability present", flush=True)


if __name__ == "__main__":
    main()
