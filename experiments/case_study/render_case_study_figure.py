#!/usr/bin/env python3
"""
生成论文 ``figure/case_study_explanation.pdf`` 的**占位**矢量图（两栏示意 + 效率脚注）。

论文中真实案例多为手工截图或 IDE 标注；本脚本输出可编译的 PDF，便于流水线完整。
数据来自 ``outputs/logs/experiments/case_study_latency.json``（由 benchmark_latency.py 写出）。

用法（仓库根目录）::
    python experiments/case_study/render_case_study_figure.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from configs.config import Config  # noqa: E402
from utils.io import ensure_dir  # noqa: E402


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = Config()
    lat_path = Path(cfg.OUTPUT_DIR) / "logs" / "experiments" / "case_study_latency.json"
    infer_ms, expl_ms = 0.0, 0.0
    n_run = 0
    if lat_path.is_file():
        with open(lat_path, encoding="utf-8") as f:
            d = json.load(f)
        infer_ms = float(d.get("inference_time_ms", 0.0))
        expl_ms = float(d.get("explanation_time_ms", 0.0))
        n_run = int(d.get("n", 0))

    out_dir = Path(cfg.OUTPUT_DIR) / "figures" / "case_study"
    ensure_dir(str(out_dir))
    out_pdf = out_dir / "case_study_panel.pdf"

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for ax, title, body in [
        (
            ax_l,
            "Baseline-style explainer (placeholder)",
            "Replace this panel with a screenshot or annotated code view\n"
            "from your baseline comparison (e.g., token-level heatmap).",
        ),
        (
            ax_r,
            "PonziSense (this work)",
            "Replace with a rationale chain visualization:\n"
            "graph-highlighted statements linking registration → payout.",
        ),
    ]:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.add_patch(
            plt.Rectangle((0.02, 0.02), 0.96, 0.96, fill=False, edgecolor="0.35", linewidth=1.2, linestyle="--")
        )
        ax.text(0.5, 0.88, title, ha="center", va="top", fontsize=11, fontweight="bold")
        ax.text(0.5, 0.55, body, ha="center", va="center", fontsize=9, color="0.25", wrap=True)

    foot = (
        f"Efficiency (benchmark): inference ≈ {infer_ms:.1f} ms, explanation ≈ {expl_ms:.1f} ms per contract"
        + (f" (n={n_run})" if n_run else "")
        + "\nAuto-generated placeholder — swap for final camera-ready figure."
    )
    fig.suptitle("Case study: qualitative explanation comparison", fontsize=12, y=1.02)
    fig.text(0.5, 0.02, foot, ha="center", fontsize=8, color="0.4")
    fig.tight_layout(rect=[0, 0.08, 1, 0.96])
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out_pdf)


if __name__ == "__main__":
    main()
