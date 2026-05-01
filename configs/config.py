from dataclasses import dataclass
import os


@dataclass
class Config:
    """
    与论文 ``3-new_methdology.tex`` 对齐的超参（实现细节可调）。

    - L_total = L_rep + μ L_exp；L_rep = L_cls + λ1 L_con + λ2 L_clu。
    - L_con: when USE_INFONCE_VIEWS is enabled, use the paper three-view InfoNCE objective; otherwise use USE_SUPCON_FALLBACK.
    - L_clu：DEC KL；``CLUSTER_USE_L2`` 时用欧氏距离软分配（式 39-41）。
    EXPLAIN_LOSS_MODE: str = "both"
    - 图分支：``USE_GRAPH_BRANCH`` 启用语句图消息传递并与 CLS 融合（式 79-83）。
    """

    TRAIN_PATH: str = r"./datafiles/processed/train.csv"
    VAL_PATH: str = r"./datafiles/processed/val.csv"
    TEST_PATH: str = r"./datafiles/processed/test.csv"

    OUTPUT_DIR: str = "./outputs"

    # Paper convention: Ponzi is the positive class. The dataset reader maps
    # raw labels equal to POSITIVE_LABEL into internal class 1.
    POSITIVE_LABEL: int = 1
    RANDOM_SEED: int = 42

    # Ponzi-E benchmark specification in the paper.
    PONZI_E_TOTAL_CONTRACTS: int = 8233
    PONZI_E_POSITIVE_CONTRACTS: int = 744
    PONZI_E_NEGATIVE_CONTRACTS: int = 7489
    TRAIN_RATIO: float = 0.60
    VAL_RATIO: float = 0.20
    TEST_RATIO: float = 0.20

    MODEL_NAME: str = "microsoft/graphcodebert-base"
    TOKENIZER_NAME: str = "microsoft/graphcodebert-base"
    CONFIG_NAME: str = "microsoft/graphcodebert-base"

    CODE_LENGTH: int = 384
    DATA_FLOW_LENGTH: int = 128
    MAX_LEN: int = CODE_LENGTH + DATA_FLOW_LENGTH

    TRAIN_BATCH_SIZE: int = 2
    EVAL_BATCH_SIZE: int = 4
    EPOCHS: int = 20
    LR: float = 1e-5
    WEIGHT_DECAY: float = 0.01
    ADAM_EPSILON: float = 1e-8
    MAX_GRAD_NORM: float = 1.0
    WARMUP_STEPS: int = 0
    WARMUP_RATIO: float = 0.06

    DEVICE: str = "cuda"

    NUM_WORKERS: int = 4
    # DataLoader 预取批次数（仅 num_workers>0 时生效）；提高 GPU 与数据流水线重叠。
    DATALOADER_PREFETCH_FACTOR: int = 4
    GRADIENT_ACCUMULATION_STEPS: int = 1
    USE_AMP: bool = True
    # 5090 / Ampere+ 上 bf16 通常比 fp16 更快且更稳；开启后不再使用 GradScaler（与 PyTorch 推荐一致）。
    USE_BF16_AMP: bool = False
    # AdamW CUDA fused 内核，数学与默认一致，通常更快。
    ADAMW_FUSED: bool = True
    # 固定形状训练时开启可略加速卷积/部分算子（Transformer 收益有限但无副作用）。
    CUDNN_BENCHMARK: bool = True

    LABEL_SMOOTHING: float = 0.05
    CLS_CLASS_WEIGHTS: tuple = (1.0, 2.0)

    EARLY_STOPPING_PATIENCE: int = 5
    EARLY_STOPPING_MIN_DELTA: float = 1e-4

    # --- 论文 Sec.3 表示学习 ---
    USE_CONTRASTIVE_LOSS: bool = True
    USE_CLUSTER_LOSS: bool = True
    USE_INFONCE_VIEWS: bool = True
    SEMANTIC_VIEW_COUNT: int = 3
    USE_SUPCON_FALLBACK: bool = False
    LAMBDA1: float = 1.0
    LAMBDA2: float = 0.85
    TAU_CONTRASTIVE: float = 0.2
    NUM_CLUSTERS_K: int = 87
    CLUSTER_LOSS_TEMP: float = 1.0
    CLUSTER_USE_L2: bool = True

    # --- 解释模块 L_exp（语句级；含 BCE 与论文稀疏/稳定项）---
    USE_EXPLAIN_LOSS: bool = True
    EXPLAIN_LOSS_WEIGHT: float = 0.55
    EXPLAIN_LOSS_MODE: str = "both"
    EXPLAIN_MAX_STATEMENTS: int = 64
    EXPLAIN_STMT_MAX_LEN: int = 96
    EXPLAIN_THRESHOLD: float = 0.5
    EXPLAIN_FORWARD_CHUNK: int = 24
    EXPLAIN_EMPTY_CUDA_CACHE: bool = False
    EXPLAIN_EVAL_TOP_K: int = 5
    EXPLAIN_EVAL_USE_PERTURBATION: bool = True

    PRED_THRESHOLD: float = 0.5

    LAMBDA_FID: float = 0.25
    LAMBDA_SPAR: float = 0.4
    LAMBDA_STAB: float = 0.15
    FID_MARGIN: float = 0.05
    FID_MAX_SAMPLES_PER_BATCH: int = 2
    EXPLAIN_BCE_IN_PAPER_MODE: float = 0.35

    AUGMENT_IDENTIFIER_MASK: bool = True
    AUGMENT_BLANK_LINES: bool = True
    AUGMENT_STMT_DROP: bool = False
    AUGMENT_ID_MASK_PROB: float = 0.12
    AUGMENT_BLANK_PROB: float = 0.35
    AUGMENT_STMT_DROP_PROB: float = 0.08

    USE_GRAPH_BRANCH: bool = True
    GRAPH_MAX_STATEMENTS: int = 24
    GRAPH_STATEMENT_ENCODE_CHUNK: int = 16

    GRAPH_WEIGHT_ALPHA: float = 1.0
    GRAPH_WEIGHT_BETA: float = 1.0
    GRAPH_WEIGHT_GAMMA: float = 1.0

    USE_TORCH_COMPILE: bool = False
    # torch.compile 模式：5090 上可试 "max-autograd" 或默认 "reduce-overhead"。
    TORCH_COMPILE_MODE: str = "reduce-overhead"
    # 显存充足时设为 False 可明显加速（关闭后梯度用存储换重算）；与论文目标一致，仅算力/显存权衡。
    USE_ENCODER_GRADIENT_CHECKPOINTING: bool = True

    # The paper treats edge scores as internal probes; returned explanations
    # remain source-level statements. Keep edge export opt-in for diagnostics.
    EXPORT_EDGE_EXPLANATIONS: bool = False
    EDGE_EXPLANATION_TOP_K: int = 25

    @property
    def CKPT_DIR(self):
        return os.path.join(self.OUTPUT_DIR, "checkpoints")

    @property
    def LOG_DIR(self):
        return os.path.join(self.OUTPUT_DIR, "logs")

    @property
    def PRED_DIR(self):
        return os.path.join(self.OUTPUT_DIR, "predictions")

    @property
    def FIGURES_DIR(self):
        return os.path.join(self.OUTPUT_DIR, "figures")

    @property
    def TABLES_DIR(self):
        return os.path.join(self.OUTPUT_DIR, "tables")
