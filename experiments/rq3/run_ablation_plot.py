#!/usr/bin/env python3
"""
RQ3：从多个已跑完的 RQ1/RQ2 JSON 结果组装论文式消融表与分组柱状图（PDF）。

用法（仓库根目录）::

  python experiments/rq3/run_ablation_plot.py --manifest experiments/rq3/ablation_manifest.example.json

manifest 中每条 variant 可包含 ``rq1_json``、``rq2_overlap_json``、``rq2_fd_json`` 路径（相对仓库根或绝对路径）。
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
from experiments.common.plot_pdf import save_grouped_bar_pdf, save_table_pdf  # noqa: E402


def _resolve(p: str) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    return (_REPO / path).resolve()


def _load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _safe_metrics_json(key: str, v: dict) -> dict:
    p = v.get(key)
    if not p:
        return {}
    path = _resolve(p)
    if not path.is_file():
        return {}
    return _load_json(path)


def _extract_row(v: dict) -> dict[str, float | str]:
    """统一为 P/R/F1/AUPRC/MIou/FD 等标量。"""
    rq1 = _safe_metrics_json("rq1_json", v)
    m1 = rq1.get("metrics") or {}
    ov = _safe_metrics_json("rq2_overlap_json", v)
    fd = _safe_metrics_json("rq2_fd_json", v)

    row = {
        "label": str(v.get("label", "variant")),
        "precision": float(m1.get("precision", float("nan"))),
        "recall": float(m1.get("recall", float("nan"))),
        "f1": float(m1.get("f1", float("nan"))),
        "auc": float(m1.get("auc", float("nan"))),
        "auprc": float(m1.get("auprc", float("nan"))),
        "mcc": float(m1.get("mcc", float("nan"))),
        "MSP": float(ov.get("MSP", float("nan"))),
        "MSR": float(ov.get("MSR", float("nan"))),
        "MIoU": float(ov.get("MIoU", float("nan"))),
        "FD": float(fd.get("FD", float("nan"))),
        "ARL": float(fd.get("ARL", float("nan"))),
    }
    return row


def main():
    ap = argparse.ArgumentParser(description="RQ3 ablation table + grouped bar PDFs from manifest JSON")
    ap.add_argument(
        "--manifest",
        type=str,
        default=str(_REPO / "experiments/rq3/ablation_manifest.example.json"),
        help="JSON manifest（variants 列表）",
    )
    ap.add_argument("--out-tag", type=str, default="ablation", help="输出文件名片段")
    args = ap.parse_args()

    man_path = _resolve(args.manifest)
    if not man_path.is_file():
        raise SystemExit(f"manifest 不存在: {man_path}")

    with open(man_path, encoding="utf-8") as f:
        manifest = json.load(f)

    variants = manifest.get("variants") or []
    if not variants:
        raise SystemExit("manifest 中无 variants")

    title = manifest.get("title", "RQ3 component ablation")
    rows = [_extract_row(v) for v in variants]
    labels = [r["label"] for r in rows]

    cfg = Config()
    fig_d = figure_dir(cfg, "rq3")
    tbl_d = table_dir(cfg)

    merged_path = os.path.join(cfg.OUTPUT_DIR, "logs", "experiments", f"rq3_{args.out_tag}_merged.json")
    os.makedirs(os.path.dirname(merged_path), exist_ok=True)
    with open(merged_path, "w", encoding="utf-8") as f:
        json.dump({"title": title, "rows": rows}, f, indent=2, ensure_ascii=False)
    print("Saved:", merged_path)

    metric_keys = ["F1", "AUPRC", "MIoU", "FD"]
    mat = [
        [r["f1"], r["auprc"], r["MIoU"], r["FD"]]
        for r in rows
    ]
    pdf_bar = os.path.join(fig_d, f"{args.out_tag}_f1_auprc_miou_fd.pdf")
    save_grouped_bar_pdf(
        pdf_bar,
        variant_labels=labels,
        metric_names=metric_keys,
        values=mat,
        title=title,
        ylabel="Score",
    )

    headers = ["Variant", "P", "R", "F1", "AUC", "AUPRC", "MCC", "MSP", "MSR", "MIoU", "FD", "ARL"]
    tbl_rows = []
    for r in rows:
        tbl_rows.append(
            [
                r["label"],
                f"{r['precision']:.4f}",
                f"{r['recall']:.4f}",
                f"{r['f1']:.4f}",
                f"{r['auc']:.4f}",
                f"{r['auprc']:.4f}",
                f"{r['mcc']:.4f}",
                f"{r['MSP']:.4f}",
                f"{r['MSR']:.4f}",
                f"{r['MIoU']:.4f}",
                f"{r['FD']:.4f}",
                f"{r['ARL']:.4f}",
            ]
        )
    pdf_tbl = os.path.join(tbl_d, f"rq3_{args.out_tag}_table.pdf")
    save_table_pdf(
        pdf_tbl,
        headers=headers,
        rows=tbl_rows,
        title=title,
        figsize=(14, 0.45 * max(3, len(tbl_rows) + 2)),
    )
    print("PDF:", pdf_bar, pdf_tbl)


if __name__ == "__main__":
    main()
