"""
路径解析：开发目录 vs PyInstaller 打包（sys._MEIPASS）。
"""
import os
import sys


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False)) and hasattr(sys, "_MEIPASS")


def get_bundle_dir() -> str:
    """资源根目录：打包后为 _MEIPASS；开发时为项目根目录。"""
    if is_frozen():
        return sys._MEIPASS
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def get_app_dir() -> str:
    """可写/用户数据目录：打包后为 exe 所在目录；开发时为当前工作目录。"""
    if is_frozen():
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.getcwd()


def resource_path(*parts: str) -> str:
    return os.path.join(get_bundle_dir(), *parts)


def solidity_grammar_so() -> str:
    return resource_path("parser", "my-languages.so")


def resolve_checkpoint_path() -> str:
    """
    权重查找顺序:
    1) 环境变量 PONZI_CKPT（完整路径）
    2) <app_dir>/outputs/checkpoints/best_model.pt
    3) <app_dir>/best_model.pt
    4) 项目内 ./outputs/checkpoints/best_model.pt（开发）
    """
    env = os.environ.get("PONZI_CKPT", "").strip()
    if env and os.path.isfile(env):
        return env

    app = get_app_dir()
    candidates = [
        os.path.join(app, "outputs", "checkpoints", "best_model.pt"),
        os.path.join(app, "best_model.pt"),
        os.path.join(get_bundle_dir(), "outputs", "checkpoints", "best_model.pt"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p

    return os.path.join(app, "outputs", "checkpoints", "best_model.pt")


def setup_demo_runtime() -> None:
    """在启动 Demo 前调用：HF 缓存、可选离线模型目录。"""
    app = get_app_dir()
    hf = os.path.join(app, ".cache", "huggingface")
    os.makedirs(hf, exist_ok=True)
    os.environ.setdefault("HF_HOME", hf)
    os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(hf, "hub"))
    torch_home = os.path.join(app, ".cache", "torch")
    os.makedirs(torch_home, exist_ok=True)
    os.environ.setdefault("TORCH_HOME", torch_home)
