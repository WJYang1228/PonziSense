def edge_scores_to_statements(edge_index, edge_scores, num_nodes, topk_ratio=0.2):
    """
    根据高分边回溯到 statement 节点。
    """
    import torch

    E = edge_scores.shape[0]
    k = max(1, int(E * topk_ratio))
    topk = torch.topk(edge_scores, k=k).indices.tolist()

    stmt_ids = set()
    for eid in topk:
        src = int(edge_index[0, eid].item())
        dst = int(edge_index[1, eid].item())
        stmt_ids.add(src)
        stmt_ids.add(dst)
    return sorted(stmt_ids)


def stmt_ids_to_explain_text(stmt_ids, statements):
    parts = []
    for sid in stmt_ids:
        if 0 <= sid < len(statements):
            parts.append(statements[sid].strip())
    return "\n---\n".join(parts)