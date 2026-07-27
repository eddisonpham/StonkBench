"""Regenerate cond_sig_wgan_v2_depth3 .pt with a clamped AR rollout.

The native 252-step AR rollout blows up to NaN/Inf because small per-step
prediction errors compound with the deeper architecture
(depth=3, hidden=(100,100,100)). We monkey-patch
``SimpleGenerator.sample`` at runtime to drive the rollout one step at a
time and clamp the per-step output BEFORE it enters the conditioning
window for the next iteration. The trained weights are used unmodified;
the vendor and adapter code are not changed.

The script also produces the per-channel overlay/histogram sanity plots
under ``outputs/sanity/cond_sig_wgan_v2_depth3/`` so we finally have a
visual reference for the deeper model.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/phamnhut/StonkBench")
sys.path.insert(0, str(ROOT))

# --- match training settings of the v2_depth3 run ---
P = 20  # conditioning window
HIDDEN_DIMS = (100, 100, 100)
SIG_DEPTH = 3
MC_SIZE = 100
TRAIN_STEPS = 5000
BATCH_SIZE = 64

# --- generation params ---
GEN_LEN = 252
NUM_SAMPLES = 1000
SEED = 42
# Hard clamp on the per-step AR output. Z-scored log returns typically
# range ±3 to ±4 in the empirical tail, so ±5 is a safe bound that
# preserves normal data while bounding compounded rollout error.
CLAMP_VAL = 5.0

# --- IO paths ---
CKPT = ROOT / "outputs/checkpoints/cond_sig_wgan_v2_depth3/cond_sig_wgan/cond_sig_wgan_G.pt"
ARTIFACT_DIR = ROOT / "outputs/results/cond_sig_wgan_v2_depth3/cond_sig_wgan/artifacts"
SANITY_DIR = ROOT / "outputs/sanity/cond_sig_wgan_v2_depth3/cond_sig_wgan"

# --- imports after path setup ---
from src.experiments.adapters.deep_learning.cond_sig_wgan_adapter import (
    ConditionalSigWGANAdapter,
)
from src.experiments.core.contracts import StandardBatch
from src.experiments.sanity_visualization import render_model_sanity
from src.utils.artifact_utils import default_metadata, save_artifact
from src.utils.preprocessed_data_utils import (
    load_dl_set,
    resolve_dl_set_path,
    sliding_window_2d,
)


# =====================================================================
# 1) Build adapter + vendor, load preprocessed data
# =====================================================================
adapter = ConditionalSigWGANAdapter()
adapter._import_vendor()  # wires sys.path so `lib.arfnn` resolves below
# SimpleGenerator lives in lib.arfnn (NOT in lib.algos.sigcwgan — see
# vendor's own evaluate.py: ``from lib.arfnn import SimpleGenerator``).
from lib.arfnn import SimpleGenerator  # noqa: E402  (after sys.path wire-up)

dl_set = load_dl_set(resolve_dl_set_path())
train_series = dl_set["train_series"]  # z-scored log returns
test_series = dl_set["test_series"]
dim = int(train_series.shape[1])
print(f"[setup] train_series={tuple(train_series.shape)} test_series={tuple(test_series.shape)} channels={dim}")
print(f"[setup] feature_columns[:3]={dl_set['feature_columns'][:3]} (total={len(dl_set['feature_columns'])})")


# =====================================================================
# 2) Reconstruct generator and load trained weights
# =====================================================================
# The training adapter called:
#     vendor.SimpleGenerator(input_dim=dim*p, output_dim=dim,
#                            hidden_dims=HIDDEN_DIMS, latent_dim=dim)
# which is `ArFNN(input_dim=dim*p + latent_dim = dim*(p+1)=525,
#               output_dim=dim=25, hidden_dims=(100,100,100))`.
# Matches checkpoint's first linear weight shape (100, 525).
G = SimpleGenerator(
    input_dim=dim * P,
    output_dim=dim,
    hidden_dims=HIDDEN_DIMS,
    latent_dim=dim,
)
state_dict = torch.load(CKPT, map_location="cpu", weights_only=True)
G.load_state_dict(state_dict)
G.eval()
print(
    f"[setup] G loaded n_params={sum(p.numel() for p in G.parameters())} "
    f"first_linear={tuple(G.network.network[0].linear.weight.shape)} "
    f"latent_dim={G.latent_dim}"
)


# =====================================================================
# 3) Monkey-patch sample() to drive the rollout step-by-step with clamping
# =====================================================================
# The vendor's native `sample()` is equivalent to `forward(z, x_past)`
# where z has the full horizon length and `forward()` loops internally.
# For high-dim / deep / long-horizon rollouts this compounds errors.
# The replacement drives the loop externally one step at a time,
# clamps the per-step output BEFORE it feeds back into `x_past`, and
# defensively `nan_to_num`s any rare non-finite output from the network.
def _clamped_sample(self, steps, x_past, clamp=CLAMP_VAL):
    """Step-by-step AR sample with per-step output clamping.

    Args:
        self: the SimpleGenerator instance (latent_dim available)
        steps: total horizon length to generate
        x_past: (B, p, C) initial conditioning window
        clamp: symmetric clamp bound on the per-step output

    Returns:
        (B, steps, C) generated sequence with all values in [-clamp, clamp].
    """
    device = x_past.device
    out_steps = []
    cur = x_past.to(device)
    for _ in range(int(steps)):
        z_t = torch.randn(cur.size(0), 1, self.latent_dim, device=device)
        # Single AR step — lets the vendor internal loop run exactly once
        # (forward iterates over z.shape[1], which is 1 here).
        x_gen = self.forward(z_t, cur)  # (B, 1, C)
        # Defensive: catch any rare non-finite value from the network.
        x_gen = torch.nan_to_num(x_gen, nan=0.0, posinf=clamp, neginf=-clamp)
        # Bound magnitude so AR compounding stays finite.
        x_gen = torch.clamp(x_gen, min=-clamp, max=clamp)
        # Manually advance the conditioning window.
        cur = torch.cat([cur[:, 1:], x_gen], dim=1)
        out_steps.append(x_gen)
    return torch.cat(out_steps, dim=1)


G.sample = types.MethodType(_clamped_sample, G)


# =====================================================================
# 4) Wire into adapter and generate 1000 samples
# =====================================================================
adapter._generator = G
adapter._p = P
adapter._device = "cpu"
adapter._is_fitted = True
adapter.apply_calibration = False  # match training default

torch.manual_seed(SEED)
np.random.seed(SEED)

gen_out = adapter.generate(num_samples=NUM_SAMPLES, generation_length=GEN_LEN, seed=SEED)
data = gen_out.data.float()
print(f"\n[gen] shape={tuple(data.shape)} dtype={data.dtype}")
print(
    f"[gen] finite={bool(torch.isfinite(data).all())} "
    f"min/max=[{data.min().item():.4f}, {data.max().item():.4f}] clamp=±{CLAMP_VAL}"
)
per_ch_std = data.std(dim=1).mean(dim=0)
print(
    f"[gen] per-channel temporal std "
    f"min={per_ch_std.min():.4f} max={per_ch_std.max():.4f} mean={per_ch_std.mean():.4f}"
)


# =====================================================================
# 5) Save the .pt with metadata matching the first_run artifact format
# =====================================================================
preprocessing_cfg = {
    "dataset_kind": "deep_learning",
    "dl_set_path": str(resolve_dl_set_path()),
}
metadata = default_metadata(
    model_name="cond_sig_wgan",
    model_type="deep_learning",
    sequence_length=GEN_LEN,
    num_samples=NUM_SAMPLES,
    seed=SEED,
    preprocessing_cfg=preprocessing_cfg,
    extra={
        "asset_columns": list(dl_set["feature_columns"]),
        "price_columns": list(dl_set["feature_columns"]),
        "is_multivariate": True,
        "num_channels": dim,
        "train_sequence_length": GEN_LEN,
        "model_checkpoint_manifest": [],
        "p": P,
        "q": GEN_LEN,
        "total_steps": TRAIN_STEPS,
        "batch_size": BATCH_SIZE,
        "hidden_dims": list(HIDDEN_DIMS),
        "mc_size": MC_SIZE,
        "signature_depth": SIG_DEPTH,
        "best_val_loss": 0.0,
        "best_epoch": TRAIN_STEPS,
        "stopped_early": False,
        "generation_length": GEN_LEN,
        "rollout_clamp_value": CLAMP_VAL,
    },
)

ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
artifact_path = ARTIFACT_DIR / f"cond_sig_wgan_seq{GEN_LEN}.pt"
saved = save_artifact(data, metadata, artifact_path)
print(f"[save] {saved} ({saved.stat().st_size / 1e6:.2f} MB)")


# =====================================================================
# 6) Sanity visualization: build a StandardBatch and call the renderer
# =====================================================================
# render_model_sanity calls adapter.generate(NUM_SIMULATIONS=50,...) which
# reuses the still-monkey-patched sample(). The validation window is
# the (denormalized) ground truth the generated samples are compared to.
test_windows = sliding_window_2d(test_series, GEN_LEN, stride=1)  # (N, L, C)
if test_windows.shape[0] == 0:
    # Fallback: use last GEN_LEN steps as a single validation window.
    test_windows = test_series[-GEN_LEN:].unsqueeze(0).float()
test_windows = test_windows.float()

# mock the remaining StandardBatch fields that render_model_sanity does not touch
batch = StandardBatch(
    train=train_series.float(),
    valid=test_series.float(),
    test=test_series.float(),
    train_initial=train_series.float()[:1],
    valid_initial=test_series.float()[:1],
    test_initial=test_series.float()[:1],
    asset_columns=list(dl_set["feature_columns"]),
    price_columns=list(dl_set["feature_columns"]),
    train_windows=None,
    valid_windows=test_windows,  # (N, L, C); sanity viz uses valid_windows[0]
    test_windows=test_windows,
    inferred_length=GEN_LEN,
)
print(f"[sanity] valid_windows shape={tuple(batch.valid_windows.shape)}")

saved_files = render_model_sanity(
    adapter=adapter,
    batch=batch,
    output_dir=SANITY_DIR,
    generation_length=GEN_LEN,
    seed=SEED,
)
print(f"[sanity] {len(saved_files)} files saved under {SANITY_DIR}")
sample_files = sorted({f.relative_to(ROOT) for f in saved_files})[:8]
for f in sample_files:
    print(f"   {f}")
if len({str(f.relative_to(ROOT)) for f in saved_files}) > 8:
    print(f"   ... ({len({str(f.relative_to(ROOT)) for f in saved_files})} total unique files)")
