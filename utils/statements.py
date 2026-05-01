import re
from dataclasses import dataclass
from typing import List


@dataclass
class StatementBlock:
    stmt_id: int
    start_line: int
    end_line: int
    text: str


def normalize_newlines(code: str) -> str:
    return code.replace("\r\n", "\n").replace("\r", "\n")


def split_solidity_statements(code: str) -> List[StatementBlock]:
    code = normalize_newlines(code)
    lines = code.split("\n")

    blocks = []
    buffer = []
    start_line = None
    stmt_id = 0

    for i, line in enumerate(lines, start=1):
        raw = line.rstrip("\n")
        stripped = raw.strip()

        if start_line is None and stripped != "":
            start_line = i

        buffer.append(raw)

        end_stmt = False
        if stripped.endswith(";") or stripped.endswith("{") or stripped == "}":
            end_stmt = True

        if end_stmt:
            text = "\n".join(buffer).strip()
            if text:
                blocks.append(
                    StatementBlock(
                        stmt_id=stmt_id,
                        start_line=start_line if start_line is not None else i,
                        end_line=i,
                        text=text,
                    )
                )
                stmt_id += 1
            buffer = []
            start_line = None

    if buffer:
        text = "\n".join(buffer).strip()
        if text:
            blocks.append(
                StatementBlock(
                    stmt_id=stmt_id,
                    start_line=start_line if start_line is not None else len(lines),
                    end_line=len(lines),
                    text=text,
                )
            )

    return blocks


def normalize_statement_text(text: str) -> str:
    text = normalize_newlines(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def is_trivial_statement(text: str) -> bool:
    t = text.strip()
    if t == "":
        return True
    if re.fullmatch(r"[{};]+", t):
        return True
    if t in {"{", "}", ";", "else", "else {", "else{"}:
        return True
    if len(t) <= 2:
        return True
    return False


def build_statement_labels(code: str, explain: str):
    """
    根据 explain 弱监督构造 statement 标签，并过滤掉无语义语句。
    """
    blocks = split_solidity_statements(code)
    explain_norm = normalize_statement_text(explain)

    statements = []
    labels = []

    for b in blocks:
        stmt_norm = normalize_statement_text(b.text)
        statements.append(b.text)

        if not explain_norm:
            labels.append(0)
            continue

        hit = stmt_norm in explain_norm if stmt_norm else False

        if not hit and stmt_norm:
            stmt_tokens = set(stmt_norm.split())
            exp_tokens = set(explain_norm.split())
            if stmt_tokens:
                overlap = len(stmt_tokens & exp_tokens) / len(stmt_tokens)
                if overlap >= 0.8:
                    hit = True

        labels.append(1 if hit else 0)

    filtered_blocks = []
    filtered_statements = []
    filtered_labels = []

    for b, s, l in zip(blocks, statements, labels):
        if is_trivial_statement(s):
            continue
        filtered_blocks.append(b)
        filtered_statements.append(s)
        filtered_labels.append(l)

    return filtered_statements, filtered_labels, filtered_blocks