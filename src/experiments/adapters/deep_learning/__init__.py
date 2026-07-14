"""Deep-learning adapter modules (import concrete adapters directly)."""

__all__ = [
    "QuantGANAdapter",
    "TimeGANAdapter",
    "TimeGradAdapter",
    "TimeVAEAdapter",
    "UnconditionalTSDiffusionAdapter",
    "VRNNAdapter",
    "PCFGANAdapter",
    "SigWGANAdapter",
]


def __getattr__(name: str):
    if name == "QuantGANAdapter":
        from src.experiments.adapters.deep_learning.quantgan_adapter import QuantGANAdapter

        return QuantGANAdapter
    if name == "TimeGANAdapter":
        from src.experiments.adapters.deep_learning.timegan_adapter import TimeGANAdapter

        return TimeGANAdapter
    if name == "TimeGradAdapter":
        from src.experiments.adapters.deep_learning.timegrad_adapter import TimeGradAdapter

        return TimeGradAdapter
    if name == "TimeVAEAdapter":
        from src.experiments.adapters.deep_learning.timevae_adapter import TimeVAEAdapter

        return TimeVAEAdapter
    if name == "UnconditionalTSDiffusionAdapter":
        from src.experiments.adapters.deep_learning.utsd_adapter import UnconditionalTSDiffusionAdapter

        return UnconditionalTSDiffusionAdapter
    if name == "VRNNAdapter":
        from src.experiments.adapters.deep_learning.vrnn_adapter import VRNNAdapter

        return VRNNAdapter
    if name == "PCFGANAdapter":
        from src.experiments.adapters.deep_learning.pcf_gan_adapter import PCFGANAdapter

        return PCFGANAdapter
    if name == "SigWGANAdapter":
        from src.experiments.adapters.deep_learning.sig_wgan_adapter import SigWGANAdapter

        return SigWGANAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
