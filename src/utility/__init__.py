"""StonkBench §6 Utility package (TSTR utility evaluation).

Public re-exports for backward compatibility and for callers like
``unified_evaluator.py``. The new pipeline lives in submodules:
- ``metrics``     — U1–U5 PnL toolbox
- ``protocols``   — TSTR, Augmented
- ``evaluator``   — orchestrator
- ``tasks``       — Options §6.2, Portfolio §6.3, Alpha §6.4
- ``policies``    — one paper-faithful model per task + BS static baseline
"""

from src.utility.metrics import MetricToolbox
from src.utility.protocols import TSTRProtocol, AugmentedProtocol
from src.utility.evaluator import UtilityEvaluator

from src.utility.tasks.options import OptionsTask
from src.utility.tasks.portfolio import PortfolioTask
from src.utility.tasks.alpha import AlphaTask

from src.utility.policies.lstm_moneyness import MoneynessLSTM
from src.utility.policies.portfolio_lstm import PortfolioLSTM
from src.utility.policies.alpha_lstm import AlphaLSTM
from src.utility.policies.bs_static import BSStaticDelta

__all__ = [
    "MetricToolbox",
    "TSTRProtocol",
    "AugmentedProtocol",
    "UtilityEvaluator",
    "OptionsTask",
    "PortfolioTask",
    "AlphaTask",
    "MoneynessLSTM",
    "PortfolioLSTM",
    "AlphaLSTM",
    "BSStaticDelta",
]
