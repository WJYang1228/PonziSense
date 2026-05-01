import re
from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class ProgramUnit:
    idx: int
    text: str
    unit_type: str
    defs: list
    uses: list
    state_impact: float


STATE_KEYWORDS = {
    "balance", "balances", "investor", "investors", "participant", "participants",
    "queue", "payout", "pay", "paid", "owner", "fee", "fees", "reward", "rewards",
    "bonus", "deposit", "withdraw", "transfer", "send", "msg.value", "this.balance"
}


def clean_code(code: str) -> str:
    code = code.replace("\r\n", "\n").replace("\r", "\n")
    return code


def split_statements(code: str) -> List[str]:
    """
    一个工程化版本的 statement splitter。
    为了保证可运行性，这里不强依赖完整 AST。
    你后面可以替换成 tree-sitter 精确 statement span。
    """
    code = clean_code(code)
    lines = [x.strip() for x in code.split("\n")]
    lines = [x for x in lines if x]

    statements = []
    buf = []
    brace_depth = 0

    for line in lines:
        buf.append(line)
        brace_depth += line.count("{") - line.count("}")

        if line.endswith(";") or line.endswith("{") or line == "}":
            statements.append(" ".join(buf))
            buf = []

    if buf:
        statements.append(" ".join(buf))
    return statements


def detect_unit_type(stmt: str) -> str:
    s = stmt.lower()
    if "if " in s or s.startswith("if(") or s.startswith("if ("):
        return "branch"
    if "for " in s or s.startswith("for(") or s.startswith("for ("):
        return "loop"
    if "while " in s or s.startswith("while(") or s.startswith("while ("):
        return "loop"
    if "function " in s:
        return "function"
    if "return " in s:
        return "return"
    if ".transfer(" in s or ".send(" in s or ".call.value(" in s or "payable" in s:
        return "value_transfer"
    if "=" in s:
        return "assignment"
    return "statement"


def extract_defs_uses(stmt: str):
    # 粗粒度变量抽取
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_\.]*", stmt)
    defs, uses = set(), set()

    if "=" in stmt:
        left, right = stmt.split("=", 1)
        left_tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_\.]*", left)
        right_tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_\.]*", right)
        defs.update(left_tokens)
        uses.update(right_tokens)
    else:
        uses.update(tokens)

    keywords = {
        "if", "for", "while", "return", "function", "contract", "public",
        "private", "internal", "external", "view", "pure", "payable", "constant"
    }
    defs = [x for x in defs if x not in keywords]
    uses = [x for x in uses if x not in keywords]
    return defs, uses


def estimate_state_impact(stmt: str) -> float:
    s = stmt.lower()
    score = 0.0
    for kw in STATE_KEYWORDS:
        if kw.lower() in s:
            score += 1.0
    if ".send(" in s or ".transfer(" in s:
        score += 2.0
    if "msg.value" in s:
        score += 2.0
    if "owner" in s:
        score += 1.0
    return score


def parse_program_units(code: str) -> List[ProgramUnit]:
    statements = split_statements(code)
    units = []
    for i, stmt in enumerate(statements):
        defs, uses = extract_defs_uses(stmt)
        unit = ProgramUnit(
            idx=i,
            text=stmt,
            unit_type=detect_unit_type(stmt),
            defs=defs,
            uses=uses,
            state_impact=estimate_state_impact(stmt),
        )
        units.append(unit)
    return units