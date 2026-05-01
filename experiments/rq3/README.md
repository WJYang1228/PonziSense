# RQ3：组件消融（Component ablation）

论文表 `w/o Explainer`、`w/o Cluster`、`w/o Graph Weighting` 等。**完整 PonziSense** 含聚类正则、加权图扰动等；当前 `ponzi_exp` 工程实现为 **GraphCodeBERT + 语句解释头**，与论文模块一一对应关系如下。

## 与本仓库可操作的对应关系

| 论文变体 | 在本仓库中的近似做法 |
|----------|----------------------|
| **w/o Explainer** | 训练时设 `USE_EXPLAIN_LOSS = False`（仅分类损失），或冻结/移除 `StatementExplainer` 相关梯度（需改 `train.py` 跳过解释分支）。 |
| **w/o Cluster** | 当前代码**无**独立「聚类损失」模块；需在 `models/` 中实现论文所述聚类正则后，再通过开关对比。此处仅保留说明，**不自动生成对照权重**。 |
| **w/o Graph Weighting** | 当前使用 GraphCodeBERT 的 DFG 边；若需「无图权」对照，可尝试仅用代码 token、不拼接 DFG（需改 `data/feature_extractor.py`，工作量较大）。 |

## 建议复现流程

1. 复制 `configs/config.py` 为 `configs/config_ablation_no_explain.py`，将 `USE_EXPLAIN_LOSS` 设为 `False`，其余一致。  
2. 重新训练：`python train.py`（或通过 `PYTHONPATH` 指向自定义 config，需在 `train.py` 支持 `--config` 前，可临时直接改 `Config`）。  
3. 使用 `experiments/rq1/run_main_metrics.py` 与 `experiments/rq2/run_overlap_metrics.py` 分别得到分类与解释指标，填入消融表。

若需**自动化多配置训练**，可自行在 `experiments/rq3/` 下添加 shell 循环调用 `train.py`（并指定不同输出目录 `OUTPUT_DIR` 以免覆盖 checkpoint）。

## 消融表 / 图（PDF）

对各变体分别跑完 `experiments/rq1/run_main_metrics.py` 与 `experiments/rq2/` 下 overlap、fd_arl 后，编辑 `experiments/rq3/ablation_manifest.example.json`（复制为自有 manifest），列出每个变体的 JSON 路径，再执行：

`python experiments/rq3/run_ablation_plot.py --manifest <你的.json>`

将在 `outputs/figures/rq3/` 生成分组柱状图（F1、AUPRC、MIoU、FD），在 `outputs/tables/` 生成完整消融表 PDF。
