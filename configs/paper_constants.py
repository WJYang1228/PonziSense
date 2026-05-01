"""
论文 `section/3-new_methdology.tex` 与实验节中的符号与推荐超参（便于对齐与后续实现）。

主训练已实现 ``L_rep = L_cls + λ1 L_con + λ2 L_clu``（L_con 监督对比、L_clu DEC）；
边级解释目标仍以语句级 BCE 近似，见 `models/model.py`。
"""

# 实验节 (Implementation Details) 中给出的示例值 — 与代码默认值未必一致
RECOMMENDED_FROM_PAPER_EXPERIMENT = {
    "lambda1": 1.0,   # λ1 — L_con 系数
    "lambda2": 0.85,  # λ2 — L_clu 系数
    "mu": 0.55,       # μ — L_total = L_rep + μ L_exp
    "tau": 0.2,       # τ — InfoNCE 温度
    "K_clusters": 87, # K — 聚类中心数
}
