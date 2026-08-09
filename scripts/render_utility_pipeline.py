"""Generate the StonkBench §6 utility pipeline architecture diagram.

Saves a high-resolution PNG documenting the modular pipeline: data ingress, three
downstream tasks (Options / Portfolio / Alpha), per-task policies and per-window PnL,
and the U1–U5 metric toolbox.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def render(out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(22, 13))
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 13)
    ax.axis("off")

    # Title
    ax.text(11, 12.6, "StonkBench §6 — Utility Evaluation Pipeline",
            ha="center", va="center", fontsize=24, fontweight="bold")
    ax.text(11, 12.1,
            "Train-Synthetic-Test-Real (TSTR) · Three downstream tasks · "
            "U1–U5 PnL Toolbox · mean ± std across test windows",
            ha="center", va="center", fontsize=12, style="italic", color="#444")

    # -------- Common data ingress (top) --------
    data_box = FancyBboxPatch(
        (8.5, 10.4), 5.0, 1.0, boxstyle="round,pad=0.10",
        fc="#E8F4F8", ec="#1F77B4", lw=2,
    )
    ax.add_patch(data_box)
    ax.text(11, 10.9, "Real D_train   Real D_test   Synthetic D̂_g",
            ha="center", va="center", fontsize=11, fontweight="bold")
    ax.text(11, 10.55, "+ initial_dollar per test window",
            ha="center", va="center", fontsize=10, style="italic")

    # Three task branches (left = Options, center = Portfolio, right = Alpha)
    tasks = [
        {
            "cx": 4.5, "name": "§6.2 Options Hedging", "color": "#FF7F0E",
            "data_desc": "Per-channel\n(N, L=252) × C channels\n7-moneyness augmentation\nBS-Merton premium c₀",
            "policy_name": "MoneynessLSTM\ninput (Δₜ₋₁, g̃ₜ, τₜ) · 3 features",
            "pnl_desc": "Δ → period_pnl[t]\ncash₀ = initial_dollar − c₀\nΔ·(Sₜ₊₁ − Sₜ) ± settlement",
        },
        {
            "cx": 11, "name": "§6.3 Portfolio Hedging", "color": "#2CA02C",
            "data_desc": "Whole window\nshape (L=252, I+J=25)\nI=20 inv stocks, J=5 hedge ETFs\nPₜ = 1 (unit inventory)",
            "policy_name": "PortfolioLSTM\nmultivariate πₜ ∈ ℝᴸˣᴵ⁺ʲ",
            "pnl_desc": "π → period_pnl[t]\nΣ πᵢ rᵢ · log/simple toggle\nloss = Σ |net exposure|",
        },
        {
            "cx": 17.5, "name": "§6.4 Alpha Generation", "color": "#D62728",
            "data_desc": "Whole window\nshape (L=252, I=25)\ncross-sectional output\nloss = − differentiable Sharpe",
            "policy_name": "AlphaLSTM\nπₜ ∈ ℝᴸˣᴵ (weights)",
            "pnl_desc": "π → period_pnl[t]\nΣ πᵢ rᵢ · initial_dollar\nΔweight per asset per step",
        },
    ]

    for task in tasks:
        cx = task["cx"]
        color = task["color"]

        # Task header
        head = FancyBboxPatch(
            (cx - 3.0, 8.6), 6.0, 1.0, boxstyle="round,pad=0.10",
            fc=color, ec="black", lw=2, alpha=0.85,
        )
        ax.add_patch(head)
        ax.text(cx, 9.1, task["name"], ha="center", va="center",
                fontsize=14, fontweight="bold", color="white")

        # Task data prep box
        prep = FancyBboxPatch(
            (cx - 3.0, 6.9), 6.0, 1.4, boxstyle="round,pad=0.10",
            fc="#F5F5F5", ec=color, lw=1.5,
        )
        ax.add_patch(prep)
        ax.text(cx, 7.6, task["data_desc"], ha="center", va="center", fontsize=10)

        # Train box
        train = FancyBboxPatch(
            (cx - 3.0, 4.6), 6.0, 1.7, boxstyle="round,pad=0.10",
            fc=color, ec="black", lw=2, alpha=0.25,
        )
        ax.add_patch(train)
        ax.text(cx, 5.65, "TRAIN", ha="center", va="center",
                fontsize=10, fontweight="bold")
        ax.text(cx, 5.1, task["policy_name"], ha="center", va="center", fontsize=10)

        # Predict + per-period PnL
        pnl = FancyBboxPatch(
            (cx - 3.0, 2.7), 6.0, 1.4, boxstyle="round,pad=0.10",
            fc="#FFF8DC", ec="#888", lw=1.5,
        )
        ax.add_patch(pnl)
        ax.text(cx, 3.95, "PREDICT + PnL", ha="center", va="center",
                fontsize=10, fontweight="bold")
        ax.text(cx, 3.4, task["pnl_desc"], ha="center", va="center", fontsize=9)

        # Arrows downward
        for a, b in [(8.6, 8.3), (6.9, 6.3), (4.6, 4.1)]:
            ax.annotate(
                "", xy=(cx, b), xytext=(cx, a),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.6),
            )

    # -------- Common U1–U5 toolbox (bottom) --------
    tb = FancyBboxPatch(
        (2.0, 0.4), 18.0, 1.9, boxstyle="round,pad=0.15",
        fc="#9467BD", ec="black", lw=2.5, alpha=0.85,
    )
    ax.add_patch(tb)
    ax.text(11, 1.9, "U1–U5 PnL Toolbox  (mean ± std across N test windows, eq. 27 §6.1.3)",
            ha="center", va="center", fontsize=14, fontweight="bold", color="white")
    ax.text(11, 1.3,
            "U1 PnL total    ·    U2 Sharpe (no √L)    ·    "
            "U3 CVaR α=0.05    ·    U4 Win rate    ·    U5 Max drawdown",
            ha="center", va="center", fontsize=11, color="white")
    ax.text(11, 0.7,
            "PnL is computed per-period using either log returns or simple returns"
            " (toggleable); each test window carries its own initial_dollar.",
            ha="center", va="center", fontsize=9, color="white", style="italic")

    # Arrows from each PnL box to the toolbox
    for cx in (4.5, 11, 17.5):
        ax.annotate(
            "", xy=(11, 2.3), xytext=(cx, 2.7),
            arrowprops=dict(arrowstyle="->", color="gray", lw=1.6, alpha=0.7),
        )

    # -------- Protocol footer --------
    proto = FancyBboxPatch(
        (0.5, 2.6), 1.4, 6.0, boxstyle="round,pad=0.10",
        fc="#222222", ec="black", lw=1.5, alpha=0.85,
    )
    # Sidecar label for protocols
    side = FancyBboxPatch(
        (0.3, 2.6), 1.4, 6.0, boxstyle="round,pad=0.10",
        fc="#222222", ec="black", lw=1.5,
    )
    ax.add_patch(side)
    ax.text(1.0, 8.6, "PROTOCOLS", ha="center", va="center",
            fontsize=11, fontweight="bold", color="white", rotation=90)
    ax.text(1.0, 4.0, "wrap each\ntraining\nstep", ha="center", va="center",
            fontsize=9, color="white", style="italic", rotation=90)

    proto_band = FancyBboxPatch(
        (0.3, -1.0), 21.4, 1.1, boxstyle="round,pad=0.10",
        fc="#E8E8E8", ec="#555", lw=1.5,
    )
    ax.add_patch(proto_band)
    ax.text(11, 0.2,
            "TSTR (§6.1.1)   train M_r on D_train  ·  M̂_g on D̂_g  ·  "
            "score both on D_test  ·  useful iff s_g > s_r  ·  best = argmax s_g",
            ha="center", va="center", fontsize=11)
    ax.text(11, -0.5,
            "Augmented (§6.1.2)   train M̃ on D_train ∪ D̂_g (full union, no balancing)  ·  "
            "score on D_test  ·  report per (training_source ∈ {real, synthetic, augmented})",
            ha="center", va="center", fontsize=11, style="italic", color="#333")

    # -------- Legend --------
    legend_x = 18.5
    ax.text(legend_x, 12.1, "Per-task policy",
            ha="left", va="center", fontsize=10, fontweight="bold")
    for i, (color, label) in enumerate([
        ("#FF7F0E", "MoneynessLSTM (§6.2)"),
        ("#2CA02C", "PortfolioLSTM (§6.3)"),
        ("#D62728", "AlphaLSTM (§6.4)"),
    ]):
        ax.add_patch(plt.Rectangle((legend_x, 11.65 - i * 0.45), 0.35, 0.3,
                                   fc=color, ec="black"))
        ax.text(legend_x + 0.5, 11.8 - i * 0.45, label,
                ha="left", va="center", fontsize=9)

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "docs/utility_pipeline.png"
    render(out)
    print(f"Saved {out}")
