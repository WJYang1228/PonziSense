# RQ4：更难负样本（Diluted negatives）

论文在「负样本被稀释 / 更难」设定下报告指标。实现要点：

1. 在 `datafiles/processed/` 中构造新 CSV：提高非 Ponzi 比例或混入近邻难例（需业务定义）。  
2. 在 `configs/config.py` 中把 `TEST_PATH`（或单独 `EVAL_*`）指向新划分。  
3. 重新运行 `python evaluate.py` 与 `experiments/rq1/run_main_metrics.py`。

本目录不提供统一脚本：难负样本构造与论文数据集 **Ponzi-E** 强相关，需按数据协议自定义预处理。

## Temporal split（时间划分）

若 CSV 含时间列，可在 `preprocess_dataset.py` 中按时间切分 train/val/test，再指向新路径。当前工程默认随机划分，**无内置时间列**时需先扩展数据格式。
