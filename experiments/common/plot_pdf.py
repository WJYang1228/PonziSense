"""Matplotlib PDF 导出（论文级矢量图）。"""
from __future__ import annotations

import os
from typing import Iterable, Sequence

from utils.io import ensure_dir


def _setup_agg():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def save_bar_pdf(
    path: str,
    labels: Sequence[str],
    values: Sequence[float],
    title: str,
    ylabel: str,
    figsize=(8, 4.5),
):
    plt = _setup_agg()
    ensure_dir(os.path.dirname(path) or ".")
    fig, ax = plt.subplots(figsize=figsize)
    x = range(len(labels))
    ax.bar(x, values, color="steelblue", edgecolor="black", linewidth=0.4)
    ax.set_xticks(list(x))
    ax.set_xticklabels(list(labels), rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def save_lines_pdf(
    path: str,
    xs: Sequence,
    series: dict[str, Iterable[float]],
    title: str,
    xlabel: str,
    ylabel: str,
    figsize=(7, 4.5),
):
    plt = _setup_agg()
    ensure_dir(os.path.dirname(path) or ".")
    fig, ax = plt.subplots(figsize=figsize)
    for name, ys in series.items():
        ax.plot(xs, list(ys), marker="o", linewidth=1.5, label=name)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def save_grouped_bar_pdf(
    path: str,
    variant_labels: Sequence[str],
    metric_names: Sequence[str],
    values: Sequence[Sequence[float]],
    title: str,
    ylabel: str = "Score",
    figsize=(10, 4.8),
):
    """
    values: len(variants) 行 × len(metric_names) 列；与论文消融柱状图（多指标分组）一致。
    """
    import numpy as np

    plt = _setup_agg()
    ensure_dir(os.path.dirname(path) or ".")
    arr = np.asarray(values, dtype=float)
    n_var, n_met = arr.shape
    if n_var != len(variant_labels) or n_met != len(metric_names):
        raise ValueError("values shape must match variant_labels × metric_names")

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(n_var)
    width = min(0.8 / max(1, n_met), 0.25)
    offsets = (np.arange(n_met) - (n_met - 1) / 2) * width
    cmap = plt.get_cmap("tab10")
    for j, name in enumerate(metric_names):
        ax.bar(
            x + offsets[j],
            arr[:, j],
            width,
            label=name,
            color=cmap(j % 10),
            edgecolor="black",
            linewidth=0.35,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(list(variant_labels), rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(ncol=min(4, n_met))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def save_table_pdf(
    path: str,
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    title: str,
    figsize: tuple[float, float] | None = None,
):
    plt = _setup_agg()
    ensure_dir(os.path.dirname(path) or ".")
    if figsize is None:
        figsize = (10, 0.55 * max(2, len(rows) + 2))
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    tbl = ax.table(
        cellText=rows,
        colLabels=list(headers),
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.3)
    ax.set_title(title, pad=12)
    fig.tight_layout()
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)
