"""Minimal stdlib-only .ipynb executor.

Runs each code cell via ``exec()`` in a shared namespace and captures
stdout/stderr + matplotlib figure outputs into the cell ``outputs`` list.

NB: We deliberately do NOT try to "auto-print" the last expression of a
cell — that mangles the source and breaks preceding imports. Print
statements are sufficient for our notebook.
"""

from __future__ import annotations

import base64
import contextlib
import io
import json
import traceback
from pathlib import Path

NB_PATH = Path("/Users/uyenlamho/Documents/vscode/CSCD94F25/Unified-benchmark-for-SDGFTS/notebooks/01_utility_pipeline_smoke.ipynb")


def _render_figures() -> list[dict]:
    """Render any open matplotlib figures into PNG display_data outputs."""
    outputs: list[dict] = []
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return outputs
    figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        outputs.append({
            "output_type": "display_data",
            "data": {"image/png": base64.b64encode(buf.getvalue()).decode("ascii"),
                     "text/plain": "<Figure>"},
            "metadata": {"image/png": {"width": 880, "height": 320}},
        })
    return outputs


def _run_cell(idx: int, src: str, ns: dict) -> list[dict]:
    """Execute one code cell and return its outputs."""
    outputs: list[dict] = []
    stdout, stderr = io.StringIO(), io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code_obj = compile(src, f"<cell-{idx}>", "exec")
            exec(code_obj, ns)
    except Exception:
        tb = traceback.format_exc()
        outputs.append({
            "output_type": "error",
            "ename": "Exception",
            "evalue": tb.splitlines()[-1],
            "traceback": tb.splitlines(),
        })
    if stdout.getvalue():
        outputs.append({"output_type": "stream", "name": "stdout",
                        "text": stdout.getvalue()})
    if stderr.getvalue():
        outputs.append({"output_type": "stream", "name": "stderr",
                        "text": stderr.getvalue()})
    outputs.extend(_render_figures())
    return outputs


def main() -> None:
    nb = json.loads(NB_PATH.read_text())
    ns: dict = {"__name__": "__main__"}
    n_code = 0
    for idx, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        n_code += 1
        src = "".join(cell["source"])
        cell["outputs"] = _run_cell(idx, src, ns)
        cell["execution_count"] = n_code
        # If a trailing display Styler / DataFrame object's repr is wanted
        # in the cell output we already have stdout — no extra work needed.
    NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    print(f"Executed {n_code} code cells; outputs written back to {NB_PATH}")


if __name__ == "__main__":
    main()
