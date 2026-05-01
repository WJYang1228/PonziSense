import os
import re
import threading
from tree_sitter import Language, Parser

from parser.DFG import DFG_solidity
from utils.paths import solidity_grammar_so


def remove_comments_and_docstrings(source: str, lang="solidity"):
    def replacer(match):
        s = match.group(0)
        if s.startswith('/'):
            return " "
        return s

    pattern = re.compile(
        r'//.*?$|/\*.*?\*/|"(?:\\.|[^\\"])*"|\'(?:\\.|[^\\\'])*\'',
        re.DOTALL | re.MULTILINE
    )
    return re.sub(pattern, replacer, source)


def tree_to_token_index(root_node):
    if len(root_node.children) == 0:
        return [(root_node.start_point, root_node.end_point)]
    code_tokens = []
    for child in root_node.children:
        code_tokens += tree_to_token_index(child)
    return code_tokens


def index_to_code_token(index, code_lines):
    start_point = index[0]
    end_point = index[1]

    if start_point[0] == end_point[0]:
        return code_lines[start_point[0]][start_point[1]:end_point[1]]

    s = code_lines[start_point[0]][start_point[1]:]
    for i in range(start_point[0] + 1, end_point[0]):
        s += code_lines[i]
    s += code_lines[end_point[0]][:end_point[1]]
    return s


_parser_lock = threading.Lock()
_parser = None


def _get_parser():
    global _parser
    with _parser_lock:
        if _parser is None:
            so_path = solidity_grammar_so()
            if not os.path.isfile(so_path):
                raise FileNotFoundError(
                    f"未找到 Solidity 语法库: {so_path}\n"
                    "若使用打包版，请确认 my-languages.so 已打入程序目录。"
                )
            lang = Language(so_path, "solidity")
            _parser = Parser()
            _parser.set_language(lang)
        return _parser


def extract_dataflow(code: str):
    try:
        code = remove_comments_and_docstrings(code, "solidity")
    except Exception:
        pass

    try:
        parser = _get_parser()
        tree = parser.parse(bytes(code, "utf8"))
        root_node = tree.root_node
        tokens_index = tree_to_token_index(root_node)
        code_lines = code.split("\n")
        code_tokens = [index_to_code_token(x, code_lines) for x in tokens_index]

        index_to_code = {}
        for idx, (index, token) in enumerate(zip(tokens_index, code_tokens)):
            index_to_code[index] = (idx, token)

        try:
            dfg, _ = DFG_solidity(root_node, index_to_code, {})
        except Exception:
            dfg = []

        dfg = sorted(dfg, key=lambda x: x[1])

        valid_indices = set()
        for d in dfg:
            if len(d[-1]) != 0:
                valid_indices.add(d[1])
            for x in d[-1]:
                valid_indices.add(x)

        new_dfg = []
        for d in dfg:
            if d[1] in valid_indices:
                new_dfg.append(d)
        dfg = new_dfg

    except Exception:
        code_tokens = []
        dfg = []

    return code_tokens, dfg
