import traceback
import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaConfig, RobertaForSequenceClassification

from configs.config import Config
from data.feature_extractor import convert_code_to_features, build_attention_mask
from data.dataset import load_datasets
from data.collate import collate_fn
from models.model import PonziModel


def print_section(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def check_environment(cfg):
    print_section("1. 环境检查")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Config DEVICE: {cfg.DEVICE}")

    try:
        tokenizer = RobertaTokenizer.from_pretrained(cfg.TOKENIZER_NAME)
        print("Tokenizer 加载成功")
    except Exception as e:
        raise RuntimeError(f"Tokenizer 加载失败: {e}")

    try:
        config = RobertaConfig.from_pretrained(cfg.CONFIG_NAME)
        print("RobertaConfig 加载成功")
    except Exception as e:
        raise RuntimeError(f"RobertaConfig 加载失败: {e}")

    try:
        encoder = RobertaForSequenceClassification.from_pretrained(cfg.MODEL_NAME, config=config)
        print("预训练模型加载成功")
    except Exception as e:
        raise RuntimeError(f"预训练模型加载失败: {e}")

    return tokenizer, config, encoder


def check_csv(cfg):
    print_section("2. CSV 读取检查")
    try:
        df = pd.read_csv(cfg.DATA_PATH, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(cfg.DATA_PATH, encoding="utf-8")

    df.columns = [c.strip().lower() for c in df.columns]

    required_cols = {"code", "label", "explain"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV 缺少必要字段: {missing}")

    print(f"CSV 读取成功: {cfg.DATA_PATH}")
    print(f"样本数量: {len(df)}")
    print(f"列名: {df.columns.tolist()}")
    print("label 分布:")
    print(df["label"].value_counts(dropna=False))

    if len(df) == 0:
        raise ValueError("CSV 为空，无法训练")

    return df


def check_single_feature(cfg, tokenizer, df):
    print_section("3. 单样本特征提取检查")

    row = df.iloc[0]
    code = str(row["code"])
    raw_label = int(row["label"])
    explain = "" if pd.isna(row["explain"]) else str(row["explain"])
    label = 1 if raw_label == cfg.POSITIVE_LABEL else 0

    feature = convert_code_to_features(
        code=code,
        label=label,
        explain=explain,
        tokenizer=tokenizer,
        cfg=cfg
    )
    attn_mask = build_attention_mask(feature, cfg, tokenizer)

    print(f"input_ids length: {len(feature.input_ids)}")
    print(f"position_idx length: {len(feature.position_idx)}")
    print(f"dfg_to_code length: {len(feature.dfg_to_code)}")
    print(f"dfg_to_dfg length: {len(feature.dfg_to_dfg)}")
    print(f"attn_mask shape: {attn_mask.shape}")

    expected_len = cfg.CODE_LENGTH + cfg.DATA_FLOW_LENGTH
    if len(feature.input_ids) != expected_len:
        raise ValueError(f"input_ids 长度异常，期望 {expected_len}，实际 {len(feature.input_ids)}")
    if len(feature.position_idx) != expected_len:
        raise ValueError(f"position_idx 长度异常，期望 {expected_len}，实际 {len(feature.position_idx)}")
    if attn_mask.shape != (expected_len, expected_len):
        raise ValueError(f"attn_mask 形状异常，期望 {(expected_len, expected_len)}，实际 {attn_mask.shape}")

    print("单样本特征提取成功")


def check_dataloader(cfg, tokenizer):
    print_section("4. DataLoader 检查")

    train_set, val_set, test_set = load_datasets(tokenizer, cfg)
    print(f"train size: {len(train_set)}")
    print(f"val size: {len(val_set)}")
    print(f"test size: {len(test_set)}")

    loader = DataLoader(
        train_set,
        batch_size=min(2, cfg.TRAIN_BATCH_SIZE),
        shuffle=True,
        collate_fn=collate_fn
    )
    batch = next(iter(loader))

    print(f"input_ids shape: {batch['input_ids'].shape}")
    print(f"position_idx shape: {batch['position_idx'].shape}")
    print(f"attn_mask shape: {batch['attn_mask'].shape}")
    print(f"labels shape: {batch['labels'].shape}")

    print("DataLoader 检查成功")
    return batch


def check_forward(cfg, encoder, config, batch):
    print_section("5. 模型前向传播检查")

    device = torch.device("cuda" if torch.cuda.is_available() and cfg.DEVICE == "cuda" else "cpu")
    model = PonziModel(encoder, config, num_clusters=cfg.NUM_CLUSTERS_K, cfg=cfg).to(device)
    model.eval()

    input_ids = batch["input_ids"].to(device)
    position_idx = batch["position_idx"].to(device)
    attn_mask = batch["attn_mask"].to(device)
    labels = batch["labels"].to(device)

    with torch.no_grad():
        loss, probs, logits, outputs = model(input_ids, position_idx, attn_mask, labels)

    print(f"loss: {loss.item():.6f}")
    print(f"probs shape: {probs.shape}")
    print(f"logits shape: {logits.shape}")
    print(f"outputs shape: {outputs.shape}")

    if probs.shape[-1] != 2:
        raise ValueError(f"分类输出维度异常，期望最后一维为 2，实际 {probs.shape}")

    print("模型前向传播成功")


def main():
    cfg = Config()

    try:
        tokenizer, config, encoder = check_environment(cfg)
        df = check_csv(cfg)
        check_single_feature(cfg, tokenizer, df)
        batch = check_dataloader(cfg, tokenizer)
        check_forward(cfg, encoder, config, batch)

        print_section("全部检查通过")
        print("现在可以开始小规模训练了。建议先把 EPOCHS 设为 1~2，BATCH_SIZE 设为 1~2 跑通。")

    except Exception as e:
        print_section("检查失败")
        print(f"错误信息: {e}")
        print("\n详细堆栈如下：")
        traceback.print_exc()


if __name__ == "__main__":
    main()