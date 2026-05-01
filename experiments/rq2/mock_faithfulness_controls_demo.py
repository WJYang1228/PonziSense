#!/usr/bin/env python3
"""
用**模拟数据**生成与正式实验相同风格的 faithfulness 对照图/表，无需 checkpoint。

用于确认 PDF 版式、图例与三条曲线（Top-k / Random-k / Bottom-k）的相对关系。

用法（仓库根目录）::

  python experiments/rq2/mock_faithfulness_controls_demo.py
  python experiments/rq2/mock_faithfulness_controls_demo.py --split demo
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from experiments.common.project import ensure_repo_importable

ensure_repo_importable()

from configs.config import Config  # noqa: E402
from experiments.common.output_paths import figure_dir, table_dir  # noqa: E402
from experiments.common.plot_pdf import save_lines_pdf, save_table_pdf  # noqa: E402
from utils.io import ensure_dir  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Mock faithfulness Top/Random/Bottom curves (PDF only).")
    ap.add_argument("--split", default="mock", help="用于输出文件名后缀")
    ap.add_argument("--write-json", action="store_true", help="同时写入与真实验结构类似的 JSON（仍用模拟数值）")
    args = ap.parse_args()

    cfg = Config()
    sp = args.split

    # 模拟：Top-k 下降最快，Random-k 居中，Bottom-k 最缓（审稿人希望看到的形态）
    ks = [1, 2, 3, 5, 8, 10]
    means_top = [0.04, 0.075, 0.10, 0.14, 0.175, 0.19]
    means_rand = [0.022, 0.045, 0.062, 0.085, 0.10, 0.11]
    means_bot = [0.008, 0.015, 0.021, 0.032, 0.042, 0.048]

    n_mock = 80
    pdf_curve = os.path.join(figure_dir(cfg, "rq2"), f"faithfulness_curve_controls_{sp}_MOCK.pdf")
    save_lines_pdf(
        pdf_curve,
        xs=ks,
        series={
            "Top-k (PonziSense)": means_top,
            "Random-k (×8)": means_rand,
            "Bottom-k (low score)": means_bot,
        },
        title=f"RQ2 faithfulness: Top vs Random vs Bottom-k ({sp}, N={n_mock}) [MOCK]",
        xlabel="k (masked statements)",
        ylabel="Mean confidence drop (FD@k)",
    )

    rows = []
    for i, k in enumerate(ks):
        rows.append([str(k), f"{means_top[i]:.4f}", f"{means_rand[i]:.4f}", f"{means_bot[i]:.4f}"])
    pdf_tbl = os.path.join(table_dir(cfg), f"rq2_faithfulness_curve_controls_{sp}_MOCK.pdf")
    save_table_pdf(
        pdf_tbl,
        headers=["k", "FD Top-k", "FD Random-k", "FD Bottom-k"],
        rows=rows,
        title=f"RQ2 faithfulness controls ({sp}, TP-Ponzi N={n_mock}) [MOCK]",
        figsize=(7.5, 2.8),
    )

    print("MOCK PDF (curve):", os.path.abspath(pdf_curve))
    print("MOCK PDF (table):", os.path.abspath(pdf_tbl))

    if args.write_json:
        out_dir = os.path.join(cfg.OUTPUT_DIR, "logs", "experiments")
        ensure_dir(out_dir)
        path = os.path.join(out_dir, f"rq2_faithfulness_curve_controls_{sp}_MOCK.json")
        payload = {
            "rq": "RQ2",
            "name": "faithfulness_curve_controls_MOCK",
            "split": sp,
            "note": "synthetic demo values for layout check only",
            "n_samples": n_mock,
            "k": ks,
            "random_repeats": 8,
            "mean_fd_top_k": means_top,
            "mean_fd_random_k": means_rand,
            "mean_fd_bottom_k": means_bot,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print("MOCK JSON:", os.path.abspath(path))


if __name__ == "__main__":
    main()
