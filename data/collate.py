import torch


def collate_fn(batch):
    input_ids = torch.stack([x["input_ids"] for x in batch], dim=0)
    position_idx = torch.stack([x["position_idx"] for x in batch], dim=0)
    attn_mask = torch.stack([x["attn_mask"] for x in batch], dim=0)
    labels = torch.stack([x["label"] for x in batch], dim=0)

    codes = [x["code"] for x in batch]
    explains = [x["explain"] for x in batch]
    input_tokens = [x["input_tokens"] for x in batch]

    statements = [x["statements"] for x in batch]
    statement_labels = [x["statement_labels"] for x in batch]
    statement_meta = [x["statement_meta"] for x in batch]
    graph_adj = torch.stack([x["graph_adj"] for x in batch], dim=0)
    graph_mask = torch.stack([x["graph_mask"] for x in batch], dim=0)

    return {
        "input_ids": input_ids,
        "position_idx": position_idx,
        "attn_mask": attn_mask,
        "labels": labels,
        "codes": codes,
        "explains": explains,
        "input_tokens": input_tokens,

        "statements": statements,
        "statement_labels": statement_labels,
        "statement_meta": statement_meta,
        "graph_adj": graph_adj,
        "graph_mask": graph_mask,
    }