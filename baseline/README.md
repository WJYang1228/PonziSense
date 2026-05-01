# Baseline 复现实验（经典机器学习）

> Paper source of truth: `ccs2026b-paper4715.pdf`. Baseline tables should be read with Ponzi as the positive class and PRE/REC/F1 as the primary detection metrics.

本目录提供**可在本仓库数据上直接运行**的基线，与论文 `5-new_experiment.tex` 中部分方法**在设定上对应**，但**数值不一定与原文献实验一致**（特征维度、预处理、超参可能与 Zheng et al. 等原文不同）。

## 已实现（`run_sklearn_baselines.py`）

| 论文中的名称 | 本实现 |
|--------------|--------|
| **Ridge-NC** | 字符 n-gram **词袋计数** + `RidgeClassifier`，经 `CalibratedClassifierCV(sigmoid)` 得到概率，便于算 AUC/AUPRC |
| **SVM-NC** | 同上特征 + `LinearSVC` + 校准得到概率 |
| **XGBoost-TF-IDF** | 字符 n-gram **TF-IDF** + `XGBClassifier` |

数据与主工程一致：`configs/config.py` 中的 `TRAIN_PATH` / `VAL_PATH` / `TEST_PATH`、`POSITIVE_LABEL`。

评估指标与 RQ1 相同：Precision, Recall, F1 (AUC, AUPRC, MCC are supplementary outputs)（见 `experiments/common/classification_metrics.py`）。

## 未在此目录实现（需外部仓库或大规模移植）

以下在论文中作为对比列出，**无统一开源脚本**或实现成本过高，故不在本文件夹提供一键复现：

- **MulCas**、**SadPonzi**、**SourceP**、**CASPER**、**Coca**、**NEM-U** 等深度/专用结构。

若你获得官方实现，可将预测概率导出为 CSV，再用本仓库的 `classification_metrics` 或 `rq1/run_main_metrics.py` 的同一阈值逻辑对齐制表。

## 依赖

```bash
pip install -r baseline/requirements.txt
```

（根目录 `requirements.txt` 已含 scikit-learn；此处额外列出 **xgboost**。）

## 用法（仓库根目录）

```bash
python baseline/run_sklearn_baselines.py --split test --threshold 0.5
```

结果 JSON：`outputs/logs/baselines/sklearn_baselines_<split>.json`

可选：`--only ridge,svm` 跳过 XGBoost；`--max-features 30000` 控制词表大小以省内存。

## 与论文插图

RQ1 主结果柱状图当前默认仅画 PonziSense；若要将本脚本输出的 baseline 与神经网络结果并画，可后续把本 JSON 与 `rq1_main_metrics_test.json` 合并后扩展 `experiments/rq1/run_main_metrics.py` 或单独作图脚本。