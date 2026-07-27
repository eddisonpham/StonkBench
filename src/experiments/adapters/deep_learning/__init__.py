"""Deep-learning adapter modules (import concrete adapters directly)."""

__all__ = [
    "QuantGANAdapter",
    "TimeGradAdapter",
    "KalmanVAEAdapter",
    "UnconditionalTSDiffusionAdapter",
    "VRNNAdapter",
    "PCFGANAdapter",
    "ConditionalSigWGANAdapter",
]


def __getattr__(name: str):
    if name == "QuantGANAdapter":
        from src.experiments.adapters.deep_learning.quantgan_adapter import QuantGANAdapter

        return QuantGANAdapter
    if name == "TimeGradAdapter":
        from src.experiments.adapters.deep_learning.timegrad_adapter import TimeGradAdapter

        return TimeGradAdapter
    if name == "KalmanVAEAdapter":
        from src.experiments.adapters.deep_learning.kalman_vae_adapter import KalmanVAEAdapter

        return KalmanVAEAdapter
    if name == "UnconditionalTSDiffusionAdapter":
        from src.experiments.adapters.deep_learning.utsd_adapter import UnconditionalTSDiffusionAdapter

        return UnconditionalTSDiffusionAdapter
    if name == "VRNNAdapter":
        from src.experiments.adapters.deep_learning.vrnn_adapter import VRNNAdapter

        return VRNNAdapter
    if name == "PCFGANAdapter":
        from src.experiments.adapters.deep_learning.pcf_gan_adapter import PCFGANAdapter

        return PCFGANAdapter
    if name == "ConditionalSigWGANAdapter":
        from src.experiments.adapters.deep_learning.cond_sig_wgan_adapter import (
            ConditionalSigWGANAdapter,
        )

        return ConditionalSigWGANAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
