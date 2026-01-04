"""
Unified data generation script for both parametric and non-parametric models.

Trains each model on the training set only at the ACF-inferred sequence length,
then generates samples by stitching log returns to reach the target generation length.
Artifacts are saved under `generated_data/<model_name>/<model_name>_seq_<L>.pt`.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Tuple, Any

import pandas as pd
import torch

def get_project_root() -> Path:
    """Get project root directory (works in Docker and local)."""
    docker_root = Path("/app")
    return docker_root if docker_root.is_dir() else Path(__file__).resolve().parents[2]

project_root = get_project_root()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Parametric models
from src.models.parametric.gbm import GeometricBrownianMotion
from src.models.parametric.ou_process import OUProcess
from src.models.parametric.merton_jump_diffusion import MertonJumpDiffusion
from src.models.parametric.de_jump_diffusion import DoubleExponentialJumpDiffusion
from src.models.parametric.garch11 import GARCH11
from src.models.non_parametric.block_bootstrap import BlockBootstrap

# Non-parametric models
from src.models.non_parametric.quant_gan import QuantGAN
from src.models.non_parametric.time_vae import TimeVAE

from src.utils.artifact_utils import default_metadata, save_artifact, stitch_sequences
from src.utils.configs_utils import get_dataset_cfgs
from src.utils.preprocessing_utils import (
    preprocess_data,
    create_dataloader,
    sliding_window_view,
    find_length,
)
from src.utils.display_utils import show_with_start_divider, show_with_end_divider


# ============================================================================
# Model Building Functions
# ============================================================================

def build_parametric_models(inferred_length: int) -> Dict[str, Any]:
    """Build all parametric models."""
    return {
        "GBM": GeometricBrownianMotion(),
        "OUProcess": OUProcess(),
        "MJD": MertonJumpDiffusion(),
        "DEJD": DoubleExponentialJumpDiffusion(),
        "GARCH11": GARCH11(),
        "BlockBootstrap": BlockBootstrap(block_size=inferred_length),
    }


def build_non_parametric_models(train_seq_length: int, device: str) -> Dict[str, Any]:
    """Build all non-parametric models."""
    return {
        "QuantGAN": QuantGAN(seq_len=train_seq_length, device=device),
        "TimeVAE": TimeVAE(seq_len=train_seq_length, input_dim=1, device=device),
    }


# ============================================================================
# Data Preparation Functions
# ============================================================================

def _get_training_price_indices(
    cfg: Dict[str, Any],
    train_seq_length: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get training price indices and prices for creating initial values.
    
    Returns:
        Tuple of (train_prices, train_indices) where indices align with sliding windows
    """
    data_path = cfg.get('original_data_path')
    df = pd.read_csv(data_path)
    original_prices = torch.from_numpy(df[cfg.get('index')].values).float()
    
    valid_ratio = cfg.get('valid_ratio', 0.1)
    test_ratio = cfg.get('test_ratio', 0.1)
    full_L = len(original_prices) - 1  # log_returns length
    train_end_full = int(full_L * (1 - valid_ratio - test_ratio))
    
    train_prices = original_prices[:train_end_full + 1]
    return train_prices, train_end_full


def prepare_nonparametric_training_data(
    cfg: Dict[str, Any],
    train_seq_length: int,
    seed: int,
) -> Tuple[torch.utils.data.DataLoader, int]:
    """
    Prepare training data for non-parametric models.
    
    Applies sliding windows and creates a DataLoader with initial values.
    
    Returns:
        Tuple of (train_loader, train_size)
    """
    # Preprocess: only clean and split data (no sliding windows)
    train_log_returns, _, _, _, _, _ = preprocess_data(
        cfg,
        supress_cfg_message=True,
    )
    
    # Apply sliding window on training split
    train_data = sliding_window_view(train_log_returns, train_seq_length, stride=1)
    train_indices = torch.arange(0, len(train_data))
    
    # Get initial values for each training window
    train_prices, _ = _get_training_price_indices(cfg, train_seq_length)
    train_indices = train_indices[train_indices < len(train_prices)]
    train_data = train_data[:len(train_indices)]
    train_init_windows = train_prices[train_indices]
    
    # Create DataLoader
    train_loader = create_dataloader(
        train_data,
        train_init_windows,
        batch_size=64,
        shuffle=True,
        seed=seed,
    )
    
    return train_loader, len(train_log_returns)


def infer_sequence_length(cfg: Dict[str, Any]) -> int:
    """
    Infer sequence length from training data using ACF analysis.
    
    Args:
        cfg: Dataset configuration
        
    Returns:
        Inferred sequence length
    """
    train_log_returns, _, _, _, _, _ = preprocess_data(
        cfg,
        supress_cfg_message=True,
    )
    return find_length(train_log_returns)


# ============================================================================
# Model Processing Functions
# ============================================================================

def _create_metadata(
    model_name: str,
    model_type: str,
    generation_length: int,
    num_samples: int,
    seed: int,
    preprocessing_cfg: Dict[str, Any],
    train_sequence_length: int,
    train_size: int,
    device: str,
    num_epochs: int = None,
) -> Dict[str, Any]:
    """Create metadata dictionary for artifact."""
    extra = {
        "train_sequence_length": train_sequence_length,
        "train_size": train_size,
        "device": device,
    }
    if num_epochs is not None:
        extra["num_epochs"] = num_epochs
    
    return default_metadata(
        model_name=model_name,
        model_type=model_type,
        sequence_length=generation_length,
        num_samples=num_samples,
        seed=seed,
        preprocessing_cfg=preprocessing_cfg,
        extra=extra,
    )


def _generate_and_save_artifact(
    model_name: str,
    model_type: str,
    base_samples: torch.Tensor,
    generation_length: int,
    output_dir: Path,
    metadata: Dict[str, Any],
) -> None:
    """
    Stitch sequences to target length and save artifact.
    
    Args:
        model_name: Name of the model
        model_type: Type of model ("parametric" or "non_parametric")
        base_samples: Generated samples at inferred length
        generation_length: Target generation length
        output_dir: Output directory for artifacts
        metadata: Metadata dictionary
    """
    generated = stitch_sequences(base_samples, generation_length, seed=metadata["seed"])
    
    output_path = output_dir / model_name
    output_path.mkdir(parents=True, exist_ok=True)
    artifact_path = output_path / f"{model_name}_seq_{generation_length}.pt"
    
    save_artifact(generated, metadata, artifact_path)
    print(f"[OK] Saved {artifact_path}")


def process_parametric_model(
    model_name: str,
    model: Any,
    train_data: torch.Tensor,
    inferred_length: int,
    generation_length: int,
    num_samples: int,
    seed: int,
    preprocessing_cfg: Dict[str, Any],
    output_dir: Path,
    device: str,
) -> None:
    """
    Train, generate, and save artifact for a single parametric model.
    
    Args:
        model_name: Name of the model
        model: Model instance
        train_data: Training data (log returns)
        inferred_length: Inferred sequence length from ACF
        generation_length: Target generation length
        num_samples: Number of samples to generate
        seed: Random seed
        preprocessing_cfg: Preprocessing configuration
        output_dir: Output directory for artifacts
        device: Device string (for metadata)
    """
    show_with_start_divider(f"Training {model_name} on training set")
    model.fit(train_data)
    
    show_with_start_divider(f"Generating {model_name} base samples at length {inferred_length}")
    base_samples = model.generate(
        num_samples=num_samples,
        generation_length=inferred_length,
        seed=seed,
    )
    
    show_with_start_divider(f"Stitching {model_name} to target length {generation_length}")
    metadata = _create_metadata(
        model_name=model_name,
        model_type="parametric",
        generation_length=generation_length,
        num_samples=num_samples,
        seed=seed,
        preprocessing_cfg=preprocessing_cfg,
        train_sequence_length=inferred_length,
        train_size=int(train_data.shape[0]),
        device=device,
    )
    
    _generate_and_save_artifact(
        model_name=model_name,
        model_type="parametric",
        base_samples=base_samples,
        generation_length=generation_length,
        output_dir=output_dir,
        metadata=metadata,
    )
    
    show_with_end_divider(f"Finished {model_name}")


def process_nonparametric_model(
    model_name: str,
    model: Any,
    train_loader: torch.utils.data.DataLoader,
    train_size: int,
    inferred_length: int,
    generation_length: int,
    num_samples: int,
    num_epochs: int,
    seed: int,
    preprocessing_cfg: Dict[str, Any],
    output_dir: Path,
    device: str,
) -> None:
    """
    Train, generate, and save artifact for a single non-parametric model.
    
    Args:
        model_name: Name of the model
        model: Model instance
        train_loader: Training DataLoader
        train_size: Size of training data
        inferred_length: Inferred sequence length from ACF
        generation_length: Target generation length
        num_samples: Number of samples to generate
        num_epochs: Number of training epochs
        seed: Random seed
        preprocessing_cfg: Preprocessing configuration
        output_dir: Output directory for artifacts
        device: Device string (for metadata)
    """
    show_with_start_divider(f"Training {model_name} on training set (seq_length={inferred_length})")
    model.fit(train_loader, num_epochs=num_epochs)
    
    show_with_start_divider(f"Generating {model_name} base samples at length {inferred_length}")
    base_samples = model.generate(
        num_samples=num_samples,
        generation_length=inferred_length,
        seed=seed,
    )
    
    show_with_start_divider(f"Stitching {model_name} to target length {generation_length}")
    metadata = _create_metadata(
        model_name=model_name,
        model_type="non_parametric",
        generation_length=generation_length,
        num_samples=num_samples,
        seed=seed,
        preprocessing_cfg=preprocessing_cfg,
        train_sequence_length=inferred_length,
        train_size=train_size,
        device=device,
        num_epochs=num_epochs,
    )
    
    _generate_and_save_artifact(
        model_name=model_name,
        model_type="non_parametric",
        base_samples=base_samples,
        generation_length=generation_length,
        output_dir=output_dir,
        metadata=metadata,
    )
    
    show_with_end_divider(f"Finished {model_name}")


def process_all_parametric_models(
    models: Dict[str, Any],
    train_data: torch.Tensor,
    inferred_length: int,
    generation_length: int,
    num_samples: int,
    seed: int,
    preprocessing_cfg: Dict[str, Any],
    output_dir: Path,
    device: str,
) -> None:
    """Process all parametric models."""
    show_with_start_divider(f"Processing Parametric Models (inferred_length={inferred_length})")
    
    for model_name, model in models.items():
        process_parametric_model(
            model_name=model_name,
            model=model,
            train_data=train_data,
            inferred_length=inferred_length,
            generation_length=generation_length,
            num_samples=num_samples,
            seed=seed,
            preprocessing_cfg=preprocessing_cfg,
            output_dir=output_dir,
            device=device,
        )


def process_all_nonparametric_models(
    models: Dict[str, Any],
    train_loader: torch.utils.data.DataLoader,
    train_size: int,
    inferred_length: int,
    generation_length: int,
    num_samples: int,
    num_epochs: int,
    seed: int,
    preprocessing_cfg: Dict[str, Any],
    output_dir: Path,
    device: str,
) -> None:
    """Process all non-parametric models."""
    show_with_start_divider(f"Processing Non-Parametric Models (inferred_length={inferred_length})")
    
    for model_name, model in models.items():
        process_nonparametric_model(
            model_name=model_name,
            model=model,
            train_loader=train_loader,
            train_size=train_size,
            inferred_length=inferred_length,
            generation_length=generation_length,
            num_samples=num_samples,
            num_epochs=num_epochs,
            seed=seed,
            preprocessing_cfg=preprocessing_cfg,
            output_dir=output_dir,
            device=device,
        )


# ============================================================================
# Main Entry Point
# ============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic data artifacts for both parametric and non-parametric models."
    )
    parser.add_argument(
        "--generation_length",
        type=int,
        required=True,
        help="Target generation length"
    )
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=15,
        help="Number of epochs for non-parametric models"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()
    args.model_type = "both"
    return args


def setup_environment(seed: int) -> None:
    """Set random seeds and thread configuration."""
    torch.manual_seed(seed)
    torch.set_num_threads(
        int(os.environ.get('TORCH_NUM_THREADS', os.environ.get('OMP_NUM_THREADS', '1')))
    )


def main() -> None:
    """Main entry point for data generation."""
    args = parse_args()
    setup_environment(args.seed)
    
    # Get configurations
    non_param_cfg, param_cfg = get_dataset_cfgs()
    
    # Setup output directory
    output_dir = Path(args.output_dir) if args.output_dir else project_root / "generated_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Infer sequence length from training data
    inferred_length = infer_sequence_length(non_param_cfg)
    
    # Process parametric models
    train_para, _, _, _, _, _ = preprocess_data(param_cfg, supress_cfg_message=True)
    parametric_models = build_parametric_models(inferred_length)
    
    process_all_parametric_models(
        models=parametric_models,
        train_data=train_para,
        inferred_length=inferred_length,
        generation_length=args.generation_length,
        num_samples=args.num_samples,
        seed=args.seed,
        preprocessing_cfg=param_cfg,
        output_dir=output_dir,
        device=args.device,
    )
    
    # Process non-parametric models
    train_loader, train_size = prepare_nonparametric_training_data(
        non_param_cfg,
        inferred_length,
        args.seed,
    )
    nonparametric_models = build_non_parametric_models(inferred_length, args.device)
    
    process_all_nonparametric_models(
        models=nonparametric_models,
        train_loader=train_loader,
        train_size=train_size,
        inferred_length=inferred_length,
        generation_length=args.generation_length,
        num_samples=args.num_samples,
        num_epochs=args.num_epochs,
        seed=args.seed,
        preprocessing_cfg=non_param_cfg,
        output_dir=output_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
