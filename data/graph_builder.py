import math
import torch
from data.parser import parse_program_units


def squash(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def defuse_score(u_i, u_j):
    defs_i = set(u_i.defs)
    uses_j = set(u_j.uses)
    if not defs_i or not uses_j:
        return 0.0
    inter = len(defs_i & uses_j)
    denom = max(1, len(defs_i | uses_j))
    return inter / denom


def valueflow_score(u_i, u_j):
    uses_i = set(u_i.uses)
    uses_j = set(u_j.uses)
    if not uses_i or not uses_j:
        return 0.0
    inter = len(uses_i & uses_j)
    denom = max(1, len(uses_i | uses_j))
    return inter / denom


def stateimpact_score(u_i, u_j):
    return (u_i.state_impact + u_j.state_impact) / 2.0


def build_cfg_edges(units):
    """
    CFG-dominant skeleton:
    1. 顺序边
    2. 分支到下一个/下下一个的近似边
    3. 循环回边近似
    """
    edges = []

    for i in range(len(units) - 1):
        edges.append((i, i + 1))

    for i, u in enumerate(units):
        if u.unit_type == "branch":
            if i + 2 < len(units):
                edges.append((i, i + 2))
        if u.unit_type == "loop":
            if i + 1 < len(units):
                edges.append((i, i + 1))
            if i - 1 >= 0:
                edges.append((i, i - 1))
    return list(set(edges))


def build_weighted_graph(code, alpha=1.0, beta=0.85, gamma=0.55):
    units = parse_program_units(code)
    edges = build_cfg_edges(units)

    edge_index = []
    edge_weight = []

    for i, j in edges:
        ui, uj = units[i], units[j]
        phi = (
            alpha * defuse_score(ui, uj) +
            beta * valueflow_score(ui, uj) +
            gamma * stateimpact_score(ui, uj)
        )
        w = squash(phi)
        edge_index.append([i, j])
        edge_weight.append(w)

    if len(edge_index) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_weight = torch.zeros((0,), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor(edge_weight, dtype=torch.float)

    statements = [u.text for u in units]
    unit_types = [u.unit_type for u in units]

    return {
        "statements": statements,
        "unit_types": unit_types,
        "edge_index": edge_index,
        "edge_weight": edge_weight,
        "num_nodes": len(statements),
    }