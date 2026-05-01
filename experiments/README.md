# 论文实验复现脚本（按 RQ 组织）

> Paper source of truth: `ccs2026b-paper4715.pdf`. Main paper tables report PRE, REC, F1 for detection and MSP, MSR, MIoU for explanation; AUC/AUPRC/MCC remain supplementary script outputs.

对应 `PonziSense/section/5-new_experiment.tex` 中的 **RQ1–RQ5** 及效率/案例段落。所有命令均在**仓库根目录** `ponzi_exp/` 下执行。

## 前提

- 在**仓库根目录**执行下列命令（`python experiments/...` 已处理 `sys.path`，无需设置 `PYTHONPATH`）。
- 已训练：`outputs/checkpoints/best_model.pt`
- Python 依赖：根目录 `requirements.txt`；RQ5 另需 `umap-learn matplotlib`（见该节）

结果 JSON/图默认写入：`outputs/logs/experiments/`

---

## RQ1：合同级分类主结果

- **脚本**: `rq1/run_main_metrics.py`
- **指标**: Precision, Recall, F1 (AUC, AUPRC, MCC are supplementary outputs)（与论文一致）
- **说明**: 论文中 **Ridge-NC / SVM-NC / XGBoost-TF-IDF** 的 sklearn 复现见仓库 **`baseline/`**（`run_sklearn_baselines.py`）；其余深度基线（MulCas、SadPonzi 等）仍依赖外部代码，见 **`baseline/README.md`**。

```bash
python experiments/rq1/run_main_metrics.py --split test --threshold 0.5
```

---

## RQ2：可解释性（MSP/MSR/MIoU、FD、ARL）

| 脚本 | 内容 |
|------|------|
| `rq2/run_overlap_metrics.py` | MSP、MSR、MIoU（与 `utils/explain_metrics` 一致） |
| `rq2/run_fd_arl.py` | Faithfulness Drop (FD)、Average Rationale Length (ARL) |

```bash
python experiments/rq2/run_overlap_metrics.py --split test
python experiments/rq2/run_fd_arl.py --split test --max-samples 200
```

**长程依赖子集 / faithfulness 曲线**：论文为单独子集与作图；可在本目录扩展 CSV 过滤规则后复用上述脚本。

---

## RQ3：消融

见 **`rq3/README.md`**：说明 `USE_EXPLAIN_LOSS` 等与论文变体的对应；完整「w/o Cluster」等需论文级模型改动后对照训练。

---

## RQ4：鲁棒性（阈值、难负样本、时间划分）

| 脚本/文档 | 内容 |
|-----------|------|
| `rq4/run_threshold_sweep.py` | 多阈值下 P/R/F1/AUC/AUPRC/MCC |
| `rq4/diluted_negatives_README.md` | 难负样本与时间划分的数据层说明（无统一脚本） |

```bash
python experiments/rq4/run_threshold_sweep.py --split test --thresholds 0.3,0.5,0.7
```

---

## RQ5：嵌入空间（UMAP）

```bash
pip install umap-learn matplotlib
python experiments/rq5/run_umap_embeddings.py --split test --max-samples 800
```

论文中「聚类前/后」对比需**两个 checkpoint**；当前脚本对**当前单一权重**做 UMAP，用于定性查看类别分离。

---

## 案例与效率

```bash
python experiments/case_study/benchmark_latency.py --n 50
```

---

## 与开源策略

若论文仓库**不公开** `system/`，可仅保留本 `experiments/` 与根目录算法代码；`system/` 为内部演示，见仓库说明。