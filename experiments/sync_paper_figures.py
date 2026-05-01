#!/usr/bin/env python3
"""
将 ``outputs/figures/rq*/`` 下由实验脚本生成的 PDF，按论文 ``PonziSense/section/5-new_experiment.tex``
中的 ``\\includegraphics{figure/xxx.pdf}`` 文件名，复制到 ``PonziSense/figure/``。

在跑完 RQ1–RQ5 与 RQ3 消融作图后执行（仓库根目录）::

    python experiments/sync_paper_figures.py --split test --rq3-tag ablation

映射关系（源 → 论文引用名）::

    rq1/main_metrics_bar_{split}.pdf     → figure/main_performance_bar.pdf
    rq2/faithfulness_curve_{split}.pdf           → figure/faithfulness_curve.pdf
    rq2/faithfulness_curve_controls_{split}.pdf  → figure/faithfulness_curve_controls.pdf
    rq3/{rq3_tag}_f1_auprc_miou_fd.pdf  → figure/ablation_bar.pdf
    rq4/threshold_curve_{split}.pdf     → figure/threshold_curve.pdf
    rq5/umap_{split}.pdf                 → figure/embedding_umap.pdf
    case_study/case_study_panel.pdf      → figure/case_study_explanation.pdf（由 render_case_study_figure.py 生成）

说明：论文中 RQ1 柱状图若含多条 baseline，本仓库默认可视化**仅本模型**一条柱；
      案例研究图为占位示意，可换为手工标注截图。
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from configs.config import Config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Sync experiment PDF figures into PonziSense/figure/")
    ap.add_argument("--split", default="test")
    ap.add_argument("--rq3-tag", default="ablation", help="与 rq3/run_ablation_plot.py --out-tag 一致")
    ap.add_argument(
        "--paper-figure-dir",
        default="",
        help="目标目录（默认: PonziSense/figure）",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = Config()
    fig_root = Path(cfg.OUTPUT_DIR) / "figures"
    dst_root = Path(args.paper_figure_dir) if args.paper_figure_dir else _REPO / "PonziSense" / "figure"
    sp = args.split
    tag = args.rq3_tag

    mapping: list[tuple[Path, Path]] = [
        (fig_root / "rq1" / f"main_metrics_bar_{sp}.pdf", dst_root / "main_performance_bar.pdf"),
        (fig_root / "rq2" / f"faithfulness_curve_{sp}.pdf", dst_root / "faithfulness_curve.pdf"),
        (fig_root / "rq2" / f"faithfulness_curve_controls_{sp}.pdf", dst_root / "faithfulness_curve_controls.pdf"),
        (fig_root / "rq3" / f"{tag}_f1_auprc_miou_fd.pdf", dst_root / "ablation_bar.pdf"),
        (fig_root / "rq4" / f"threshold_curve_{sp}.pdf", dst_root / "threshold_curve.pdf"),
        (fig_root / "rq5" / f"umap_{sp}.pdf", dst_root / "embedding_umap.pdf"),
        (fig_root / "case_study" / "case_study_panel.pdf", dst_root / "case_study_explanation.pdf"),
    ]

    dst_root.mkdir(parents=True, exist_ok=True)
    ok, missing = 0, []
    for src, dst in mapping:
        if not src.is_file():
            missing.append(str(src))
            continue
        if args.dry_run:
            print(f"would copy: {src} -> {dst}")
        else:
            shutil.copy2(src, dst)
            print(f"OK: {dst.name} <= {src.relative_to(_REPO)}")
        ok += 1

    if missing:
        print("MISSING (skipped):", file=sys.stderr)
        for m in missing:
            print(f"  - {m}", file=sys.stderr)
        print(
            "\n提示: RQ3 需先运行 run_ablation_plot.py；"
            "案例图需运行 experiments/case_study/render_case_study_figure.py",
            file=sys.stderr,
        )
    print(f"Synced {ok} file(s) into {dst_root}")


if __name__ == "__main__":
    main()
