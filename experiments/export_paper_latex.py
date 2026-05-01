#!/usr/bin/env python3
"""
将 ``outputs/logs/experiments/`` 下各 RQ 实验 JSON 汇总为 LaTeX 宏与表格片段，
便于在论文 ``\\input{...}`` 后直接用 ``\\Exp...`` 替换手写数字。

用法（仓库根目录）::
    python experiments/export_paper_latex.py
    python experiments/export_paper_latex.py --split test --out-dir outputs/paper_latex

跑完 ``bash experiments/run_all_experiments.sh`` 或各 RQ 脚本后再执行本脚本。
若某 JSON 不存在，对应宏会跳过并在注释中标注 MISSING。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from configs.config import Config  # noqa: E402


def _load(path: Path) -> dict | None:
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _fmt(x, nd: int = 4) -> str:
    if isinstance(x, bool):
        return "1" if x else "0"
    if isinstance(x, (int, float)):
        if isinstance(x, float) and (x != x):  # nan
            return "nan"
        if isinstance(x, int) or nd <= 0:
            return str(int(x)) if isinstance(x, int) else f"{x:.{max(0, nd)}f}"
        return f"{x:.{nd}f}"
    return str(x)


def _macro_name(*parts: str) -> str:
    s = "".join(re.sub(r"[^a-zA-Z0-9]", "", p) for p in parts)
    if not s:
        s = "X"
    return "Exp" + s


def _tex_escape_text(s: str) -> str:
    return (
        s.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("$", "\\$")
        .replace("#", "\\#")
        .replace("_", "\\_")
        .replace("{", "\\{")
        .replace("}", "\\}")
        .replace("~", "\\textasciitilde{}")
        .replace("^", "\\textasciicircum{}")
    )


def _write_header(f, title: str) -> None:
    f.write(f"% === {title} ===\n")


def export_macros(
    cfg: Config,
    split: str,
    out_dir: Path,
    rq3_path: Path | None,
    nd: int,
) -> None:
    exp_dir = Path(cfg.OUTPUT_DIR) / "logs" / "experiments"
    log_root = Path(cfg.OUTPUT_DIR) / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "rq1": exp_dir / f"rq1_main_metrics_{split}.json",
        "rq2_overlap": exp_dir / f"rq2_overlap_{split}.json",
        "rq2_fd": exp_dir / f"rq2_fd_arl_{split}.json",
        "rq2_faith": exp_dir / f"rq2_faithfulness_curve_{split}.json",
        "rq2_faith_ctrl": exp_dir / f"rq2_faithfulness_curve_controls_{split}.json",
        "rq4": exp_dir / f"rq4_threshold_sweep_{split}.json",
        "latency": exp_dir / "case_study_latency.json",
        "test_metrics": log_root / "test_metrics.json",
    }

    merged_rq3: Path | None = rq3_path
    if merged_rq3 is None:
        candidates = sorted(exp_dir.glob("rq3_*_merged.json"))
        if candidates:
            merged_rq3 = candidates[-1]

    gen_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    macros_path = out_dir / "experiment_macros.tex"
    tables_path = out_dir / "experiment_tables.tex"
    master_path = out_dir / "experiment_paper_bundle.tex"

    lines_macro: list[str] = []
    lines_table: list[str] = []
    missing: list[str] = []

    def add_macro(tex_name: str, value: str, comment: str = "") -> None:
        lines_macro.append(
            f"\\providecommand{{\\{tex_name}}}{{{value}}}"
            + (f" % {comment}" if comment else "")
            + "\n"
        )

    # --- RQ1 ---
    d = _load(paths["rq1"])
    if d and "metrics" in d:
        m = d["metrics"]
        add_macro(_macro_name("RQ1", "N"), str(d.get("n", "")), "sample count")
        for k in ("precision", "recall", "f1", "auc", "auprc", "mcc"):
            if k in m:
                add_macro(_macro_name("RQ1", k.title()), _fmt(m[k], nd), f"RQ1 {k}")
    else:
        missing.append(str(paths["rq1"]))

    # --- RQ2 overlap ---
    d = _load(paths["rq2_overlap"])
    if d:
        for k in ("MIoU", "MSP", "MSR"):
            if k in d:
                add_macro(_macro_name("RQ2", k), _fmt(d[k], nd), "RQ2 overlap")
        for k in ("explain_n_samples", "explain_n_gt_positive", "explain_n_pred_positive"):
            if k in d:
                add_macro(_macro_name("RQ2", k.replace("explain_", "").title()), str(int(d[k])), "RQ2 overlap count")
        if "EXPLAIN_EVAL_TOP_K" in d:
            add_macro(_macro_name("RQ2", "TopK"), str(int(d["EXPLAIN_EVAL_TOP_K"])), "eval top-k")
    else:
        missing.append(str(paths["rq2_overlap"]))

    # --- RQ2 FD / ARL ---
    d = _load(paths["rq2_fd"])
    if d:
        if "FD" in d:
            add_macro(_macro_name("RQ2", "FD"), _fmt(d["FD"], nd), "faithfulness drop")
        if "ARL" in d:
            add_macro(_macro_name("RQ2", "ARL"), _fmt(d["ARL"], 2), "avg rationale length")
        if "n" in d:
            add_macro(_macro_name("RQ2", "FDN"), str(int(d["n"])), "FD/ARL n samples")
    else:
        missing.append(str(paths["rq2_fd"]))

    # --- RQ2 faithfulness curve ---
    d = _load(paths["rq2_faith"])
    if d and "k" in d and "mean_fd" in d:
        ks, means = d["k"], d["mean_fd"]
        for i, (kk, mu) in enumerate(zip(ks, means)):
            add_macro(_macro_name("RQ2", "FaithK", str(kk)), _fmt(mu, nd), f"mean FD @ k={kk}")
        add_macro(_macro_name("RQ2", "FaithKMax"), str(max(ks)), "k max in curve")
    else:
        missing.append(str(paths["rq2_faith"]))

    # --- RQ2 faithfulness curve (Top / Random / Bottom controls) ---
    d = _load(paths["rq2_faith_ctrl"])
    if d and "k" in d:
        if "n_samples" in d:
            add_macro(_macro_name("RQ2", "FaithCtrlN"), str(int(d["n_samples"])), "faithfulness controls N")
        for key, tag in (
            ("mean_fd_top_k", "Top"),
            ("mean_fd_random_k", "Rand"),
            ("mean_fd_bottom_k", "Bot"),
        ):
            vals = d.get(key)
            if not isinstance(vals, list) or len(vals) != len(d["k"]):
                continue
            for kk, mu in zip(d["k"], vals):
                add_macro(_macro_name("RQ2", "Faith", tag, str(kk)), _fmt(mu, nd), f"FD {tag} @ k={kk}")
    else:
        missing.append(str(paths["rq2_faith_ctrl"]))

    # --- RQ4 threshold sweep ---
    d = _load(paths["rq4"])
    if d and "rows" in d:
        for row in d["rows"]:
            tau = row.get("threshold")
            if tau is None:
                continue
            ttag = str(tau).replace(".", "p")
            for mk in ("precision", "recall", "f1", "auc", "auprc", "mcc"):
                if mk in row:
                    add_macro(
                        _macro_name("RQ4", "Tau", ttag, mk.title()),
                        _fmt(row[mk], nd),
                        f"τ={tau} {mk}",
                    )
    else:
        missing.append(str(paths["rq4"]))

    # --- Case study latency ---
    d = _load(paths["latency"])
    if d:
        if "inference_time_ms" in d:
            add_macro(_macro_name("Case", "InferMs"), _fmt(d["inference_time_ms"], 2), "latency")
        if "explanation_time_ms" in d:
            add_macro(_macro_name("Case", "ExplainMs"), _fmt(d["explanation_time_ms"], 2), "latency")
        if "total_ms" in d:
            add_macro(_macro_name("Case", "TotalMs"), _fmt(d["total_ms"], 2), "latency")
        if "n" in d:
            add_macro(_macro_name("Case", "LatencyN"), str(int(d["n"])), "benchmark n")
    else:
        missing.append(str(paths["latency"]))

    # --- test_metrics.json (training end) ---
    d = _load(paths["test_metrics"])
    if d:
        for k in ("accuracy", "precision", "recall", "f1"):
            if k in d:
                add_macro(_macro_name("TestEval", k.title()), _fmt(d[k], nd), "test split")
        ex = d.get("explainability") or {}
        for k in ("MIoU", "MSP", "MSR"):
            if k in ex:
                add_macro(_macro_name("TestEval", "Expl", k), _fmt(ex[k], nd), "test explain")
    else:
        missing.append(str(paths["test_metrics"]))

    # --- RQ3 merged (optional) ---
    rq3_note = ""
    if merged_rq3 and merged_rq3.is_file():
        d = _load(merged_rq3)
        if d and "rows" in d:
            add_macro(_macro_name("RQ3", "Source"), _tex_escape_text(merged_rq3.name), "filename")
            lines_table.append("% --- RQ3 ablation table ---\n")
            lines_table.append("% (requires \\usepackage{booktabs} in main document — or replace \\toprule etc.)\n")
            lines_table.append("\\begin{table}[t]\n\\centering\n")
            lines_table.append(
                "\\begin{tabular}{lcccccc}\n"
                "\\toprule\n"
                "Variant & P & R & F1 & AUC & AUPRC & MCC \\\\\n"
                "\\midrule\n"
            )
            for row in d["rows"]:
                lab = _tex_escape_text(str(row.get("label", "")))
                cells = [lab]
                for c in ("precision", "recall", "f1", "auc", "auprc", "mcc"):
                    cells.append(_fmt(row[c], nd) if c in row else "--")
                lines_table.append(" & ".join(cells) + " \\\\\n")
            lines_table.append(
                "\\bottomrule\n\\end{tabular}\n"
                f"\\caption{{{_tex_escape_text(d.get('title', 'RQ3 ablation'))}}}\n"
                "\\end{table}\n\n"
            )
            rq3_note = str(merged_rq3)
    else:
        lines_table.append("% RQ3 merged JSON not found; run rq3/run_ablation_plot.py with a manifest first.\n\n")

    # RQ5: only paths (no scalar JSON in repo)
    umap_png = exp_dir / f"rq5_umap_{split}.png"
    umap_pdf = Path(cfg.FIGURES_DIR) / "rq5" / f"umap_{split}.pdf"
    if umap_png.is_file() or umap_pdf.is_file():
        p = umap_pdf if umap_pdf.is_file() else umap_png
        rel = os.path.relpath(p, start=out_dir)
        add_macro(_macro_name("RQ5", "UmapPath"), _tex_escape_text(rel.replace(os.sep, "/")), "relative to experiment_tables.tex dir")
        lines_table.append("% RQ5 UMAP: includegraphics example\n")
        lines_table.append(
            f"% \\includegraphics[width=\\linewidth]{{{_tex_escape_text(str(p))}}}\n\n"
        )

    # Write experiment_macros.tex
    with open(macros_path, "w", encoding="utf-8") as f:
        f.write("% -*- coding: utf-8 -*-\n")
        f.write(f"% Auto-generated by experiments/export_paper_latex.py\n")
        f.write(f"% Generated: {gen_at}\n")
        f.write(f"% Split: {split}, OUTPUT_DIR: {cfg.OUTPUT_DIR}\n")
        if missing:
            f.write("% MISSING inputs (macros skipped for these):\n")
            for m in missing:
                f.write(f"%   - {m}\n")
        f.write("%\n% In preamble or before \\begin{document}:\n")
        f.write("%   \\input{experiment_macros}  % path relative to your .tex file\n")
        f.write("%\n")
        f.writelines(lines_macro)
        f.write("\n")

    # Write experiment_tables.tex
    with open(tables_path, "w", encoding="utf-8") as f:
        f.write("% -*- coding: utf-8 -*-\n")
        f.write(f"% Auto-generated by experiments/export_paper_latex.py ({gen_at})\n")
        f.write("% Paste or \\input{} where appropriate. Adjust column spec as needed.\n\n")
        _write_header(f, "RQ1 main metrics (classification)")
        f.write("\\begin{table}[t]\n\\centering\n")
        f.write("\\begin{tabular}{lc}\n\\toprule\nMetric & Value \\\\\n\\midrule\n")
        d1 = _load(paths["rq1"])
        if d1 and "metrics" in d1:
            m = d1["metrics"]
            order = [
                ("Precision", "precision"),
                ("Recall", "recall"),
                ("F1", "f1"),
                ("AUC", "auc"),
                ("AUPRC", "auprc"),
                ("MCC", "mcc"),
            ]
            for label, key in order:
                if key in m:
                    f.write(f"{label} & {_fmt(m[key], nd)} \\\\\n")
        else:
            f.write("% (rq1 JSON missing)\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
        f.write(f"\\caption{{Main classification metrics (\\texttt{{{split}}} split).}}\n")
        f.write("\\end{table}\n\n")

        _write_header(f, "RQ2 explainability (overlap)")
        f.write("\\begin{table}[t]\n\\centering\n")
        f.write("\\begin{tabular}{lc}\n\\toprule\nMetric & Value \\\\\n\\midrule\n")
        d2 = _load(paths["rq2_overlap"])
        if d2:
            for label, key in [("MIoU", "MIoU"), ("MSP", "MSP"), ("MSR", "MSR")]:
                if key in d2:
                    f.write(f"{label} & {_fmt(d2[key], nd)} \\\\\n")
        else:
            f.write("% (rq2 overlap JSON missing)\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
        f.write("\\caption{Statement-level explanation overlap metrics.}\n\\end{table}\n\n")

        f.writelines(lines_table)

    # Small bundle file listing what to input
    with open(master_path, "w", encoding="utf-8") as f:
        f.write("% -*- coding: utf-8 -*-\n")
        f.write(f"% Paper experiment LaTeX bundle — generated {gen_at}\n")
        f.write("%\n")
        f.write("% 1) Copy this folder to your PonziSense paper tree, or symlink.\n")
        f.write("% 2) In preamble:\n")
        f.write("%      \\input{experiment_macros}\n")
        f.write("% 3) In body, use macros like \\ExpRQ1F1, \\ExpRQ2MIoU, \\ExpRQ4Taup5F1, ...\n")
        f.write("%    (run this script after each experiment batch to refresh numbers.)\n")
        f.write("% 4) Optional: \\input{experiment_tables} for ready-made tables.\n")
        if rq3_note:
            f.write(f"% RQ3 table source: {rq3_note}\n")
        f.write("%\n")
        f.write("\\input{experiment_macros}\n")
        f.write("% \\input{experiment_tables}\n")

    print("Wrote:")
    print(f"  {macros_path}")
    print(f"  {tables_path}")
    print(f"  {master_path}")
    if missing:
        print("Missing JSON (some macros skipped):", len(missing))


def main() -> None:
    ap = argparse.ArgumentParser(description="Export experiment JSON to LaTeX macros and tables.")
    ap.add_argument("--split", default="test", help="与 rq1/rq2/rq4 输出文件名中的 split 一致")
    ap.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="输出目录（默认: <OUTPUT_DIR>/paper_latex）",
    )
    ap.add_argument(
        "--rq3-merged",
        type=str,
        default="",
        help="RQ3 merged JSON 路径；默认取 outputs/logs/experiments/rq3_*_merged.json 中最新一个",
    )
    ap.add_argument("--decimals", type=int, default=4, help="浮点打印小数位")
    args = ap.parse_args()

    cfg = Config()
    out_dir = Path(args.out_dir) if args.out_dir else Path(cfg.OUTPUT_DIR) / "paper_latex"
    rq3 = Path(args.rq3_merged) if args.rq3_merged else None

    export_macros(cfg, args.split, out_dir, rq3, max(0, args.decimals))


if __name__ == "__main__":
    main()
