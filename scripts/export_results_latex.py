"""Export a LaTeX-ready results table from complete_evaluation.json.

Reads the merged evaluation summary and emits a booktabs ``longtable`` with one
row per (model, sequence_length) — 48 rows — covering every metric category:

  * Fidelity:      mdd, md, sdd, sd, kd, cmd, dcor_diff
  * Diversity:     icd_euclidean, icd_dtw
  * Stylized facts: autocorr_returns, volatility_clustering,
                    long_memory_volatility (all as |real - synth| "diff")
  * Utility (hedging): mean replication error across the 8 hedgers, trained on
                    real-only vs mixed (real+synth) data (augmented testing)
  * Portfolio:     GMV Sharpe ratio at n_assets = {5, 10, 25} ablations
  * P&L:           synthetic-data Sharpe, Sortino, MaxDD, Calmar

Usage:
    python scripts/export_results_latex.py \
        [--input results/evaluation/complete_evaluation.json] \
        [--output results/evaluation/results_table.tex]
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Metric extraction helpers
# ---------------------------------------------------------------------------

FIDELITY_KEYS = ["mdd", "md", "sdd", "sd", "kd", "cmd", "dcor_diff"]
DIVERSITY_KEYS = ["icd_euclidean", "icd_dtw"]
STYLIZED_KEYS = ["autocorr_returns", "volatility_clustering", "long_memory_volatility"]
HEDGERS = ["BlackScholes", "DeltaGamma", "Feedforward_L-1", "Feedforward_Time",
           "LSTM", "LinearRegression", "RNN", "XGBoost"]
PORTFOLIO_ABLATIONS = ["n_assets_5", "n_assets_10", "n_assets_25"]
PNL_KEYS = ["sharpe_ratio", "sortino_ratio", "max_drawdown", "calmar_ratio"]


def _num(v: Any, default: float = float("nan")) -> float:
    """Coerce a JSON value to float, tolerating None/str artifacts."""
    try:
        if v is None:
            return default
        f = float(v)
        return f if f == f else default  # drop NaN
    except (TypeError, ValueError):
        return default


def _fidelity(entry: Dict) -> Dict[str, float]:
    fe = entry.get("FidelityEvaluator", {}) or {}
    return {k: _num(fe.get(k)) for k in FIDELITY_KEYS}


def _diversity(entry: Dict) -> Dict[str, float]:
    de = entry.get("DiversityEvaluator", {}) or {}
    return {k: _num(de.get(k)) for k in DIVERSITY_KEYS}


def _stylized(entry: Dict) -> Dict[str, float]:
    sf = entry.get("StylizedFactsEvaluator", {}) or {}
    out = {}
    for k in STYLIZED_KEYS:
        v = sf.get(k)
        if isinstance(v, dict):
            out[k] = _num(v.get("diff"))
        else:
            out[k] = _num(v)
    return out


def _hedging_summary(entry: Dict) -> Tuple[float, float]:
    """Mean replication error (real-train) and mixed-train across the 8 hedgers."""
    at = (entry.get("utility", {}) or {}).get("summary", {}).get("augmented_testing", {}) or {}
    real_means, mixed_means = [], []
    for h in HEDGERS:
        hd = at.get(h, {}) or {}
        r = hd.get("real_train", {}) or {}
        m = hd.get("mixed_train", {}) or {}
        rv = _num(r.get("mean"))
        mv = _num(m.get("mean"))
        if rv == rv:
            real_means.append(rv)
        if mv == mv:
            mixed_means.append(mv)
    real_avg = sum(real_means) / len(real_means) if real_means else float("nan")
    mixed_avg = sum(mixed_means) / len(mixed_means) if mixed_means else float("nan")
    return real_avg, mixed_avg


def _portfolio_gmv_sharpe(entry: Dict, n_assets_key: str) -> float:
    port = entry.get("portfolio", {}) or {}
    block = port.get(n_assets_key, {}) or {}
    gmv = block.get("GMV", {}) or {}
    return _num(gmv.get("sharpe_ratio"))


def _pnl_synthetic(entry: Dict) -> Dict[str, float]:
    pnl = entry.get("pnl", {}) or {}
    syn = pnl.get("synthetic", {}) or {}
    return {k: _num(syn.get(k)) for k in PNL_KEYS}


# ---------------------------------------------------------------------------
# Row building
# ---------------------------------------------------------------------------

def build_row(key: str, entry: Dict) -> List[str]:
    model = entry.get("model_name", key.rsplit("_seq", 1)[0])
    seq = entry.get("sequence_length", entry.get("evaluated_at_length", ""))
    mtype = "DL" if entry.get("model_type") == "deep_learning" else "Stat"
    cells: List[str] = [_esc(model), str(seq), mtype]

    f = _fidelity(entry)
    cells += [f"{f[k]:.4f}" if f[k] == f[k] else "--" for k in FIDELITY_KEYS]

    div = _diversity(entry)
    cells += [f"{div[k]:.4f}" if div[k] == div[k] else "--" for k in DIVERSITY_KEYS]

    st = _stylized(entry)
    cells += [f"{st[k]:.4f}" if st[k] == st[k] else "--" for k in STYLIZED_KEYS]

    real_avg, mixed_avg = _hedging_summary(entry)
    cells += [f"{real_avg:.4f}" if real_avg == real_avg else "--",
              f"{mixed_avg:.4f}" if mixed_avg == mixed_avg else "--"]

    cells += [f"{_portfolio_gmv_sharpe(entry, na):.3f}" for na in PORTFOLIO_ABLATIONS]

    pnl = _pnl_synthetic(entry)
    cells += [f"{pnl[k]:.3f}" if pnl[k] == pnl[k] else "--" for k in PNL_KEYS]

    return cells


# ---------------------------------------------------------------------------
# LaTeX emission
# ---------------------------------------------------------------------------

def _fmt_val(s: str) -> str:
    if s == "--":
        return "--"
    # scientific-notation-free, 4-sig formatting already applied by caller
    return s


def _esc(s: str) -> str:
    """Escape LaTeX-special characters for plain column labels."""
    return s.replace("_", "\\_").replace("&", "\\&")


def emit_latex(rows: List[Tuple[str, Dict]], out_path: Path) -> None:
    header = [
        "Model", "$L$", "Class",
        *[_esc(k) for k in FIDELITY_KEYS],
        *[_esc(k) for k in DIVERSITY_KEYS],
        *[_esc(k) for k in STYLIZED_KEYS],
        "Hedge$_{real}$", "Hedge$_{mix}$",
        *[f"GMV$_{{{na.split('_')[-1]}}}$" for na in PORTFOLIO_ABLATIONS],
        "P\\&L SR", "Sortino", "MaxDD", "Calmar",
    ]
    # Notes column for metrics requiring a footnote
    n_cols = len(header)

    lines: List[str] = []
    lines.append("% Generated by scripts/export_results_latex.py — do not hand-edit.")
    lines.append("\\documentclass[11pt]{article}")
    lines.append("\\usepackage[margin=0.7in,landscape]{geometry}")
    lines.append("\\usepackage{booktabs,longtable,array}")
    lines.append("\\usepackage[table]{xcolor}")
    lines.append("\\begin{document}")
    lines.append("\\section*{StonkBench Evaluation Results (48 rows: 12 models $\\times$ 4 window lengths)}")
    lines.append("\\small")
    lines.append("\\setlength{\\LTcapwidth}{\\textwidth}")
    lines.append("\\begin{longtable}{@{}l r c " + "r" * (n_cols - 3) + "@{}}")
    lines.append("\\toprule")
    lines.append(" & ".join(header) + " \\\\")
    lines.append("\\midrule\\endhead")
    lines.append("\\bottomrule\\endlastfoot")

    for key, entry in rows:
        cells = build_row(key, entry)
        lines.append(" & ".join(cells) + " \\\\")

    lines.append("\\end{longtable}")
    lines.append("\\vspace{1em}")
    lines.append("\\footnotesize")
    lines.append("""
\\noindent\\textbf{Notes.} $L$ = sequence length (window size). Fidelity: mdd/md/sdd/sd/kd = marginal
distribution / mean / std / skewness / kurtosis distance (lower = better); cmd = correlation-matrix
distance; dcor\\_diff = distance-correlation matrix difference (Sz\\'ekely et al. 2007). Diversity:
icd\\_euclidean / icd\\_dtw = intra-class ICD. Stylized facts: |real $-$ synth| for autocorrelation,
volatility clustering and long-memory. Hedge$_{real}$/Hedge$_{mix}$ = mean replication error across 8
hedgers (Buehler et al. 2019 augmented testing) trained on real-only vs mixed real+synth data.
GMV$_5$/GMV$_{10}$/GMV$_{25}$ = global minimum-variance out-of-sample Sharpe for portfolios of 5/10/25
assets (DeMiguel et al. 2009). P\\&L columns are computed on synthetic price paths (equal-weight
buy-and-hold); MaxDD is negative by convention. ``--'' = not available for that entry.
""")
    lines.append("\\end{document}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export LaTeX results table.")
    parser.add_argument("--input", type=str,
                        default="results/evaluation/complete_evaluation.json")
    parser.add_argument("--output", type=str,
                        default="results/evaluation/results_table.tex")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    data = json.loads(in_path.read_text(encoding="utf-8"))

    rows = sorted(data.items(), key=lambda kv: (kv[1].get("model_name", ""), kv[1].get("sequence_length", 0)))
    print(f"Loaded {len(rows)} entries from {in_path}")

    emit_latex(rows, out_path)
    print(f"Wrote LaTeX table to {out_path} ({out_path.stat().st_size} bytes)")

    # sanity: confirm 48 rows and row lengths
    widths = {len(build_row(k, v)) for k, v in rows}
    print(f"Rows: {len(rows)}, unique column widths: {widths}")


if __name__ == "__main__":
    main()
