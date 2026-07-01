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
    """
    基于分号、花括号的工程化语句切分。
    返回带行号的 statement blocks。
    """
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


def remove_statement_by_id(code: str, stmt_id: int) -> str:
    """
    删除指定语句块，用于遮挡法解释。
    """
    blocks = split_solidity_statements(code)
    kept = [b.text for b in blocks if b.stmt_id != stmt_id]
    return "\n".join(kept)


def html_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
    )


def highlight_code_lines(code: str, highlighted_stmt_ids: List[int]) -> str:
    """
    把源码按行转成 HTML，并高亮 explain 对应 statement 涉及的行。
    """
    code = normalize_newlines(code)
    lines = code.split("\n")
    blocks = split_solidity_statements(code)

    highlight_lines = set()
    for b in blocks:
        if b.stmt_id in highlighted_stmt_ids:
            for ln in range(b.start_line, b.end_line + 1):
                highlight_lines.add(ln)

    html_lines = []
    for idx, line in enumerate(lines, start=1):
        escaped = html_escape(line)
        cls = "hl" if idx in highlight_lines else ""
        html_lines.append(
            f'<div class="code-line {cls}"><span class="ln">{idx:4d}</span><span class="src">{escaped}</span></div>'
        )
    return "\n".join(html_lines)