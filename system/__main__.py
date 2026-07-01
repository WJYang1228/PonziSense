"""
开发启动：请在仓库根目录执行  
  python -m system
"""
from __future__ import annotations

import os


def main() -> None:
    import multiprocessing

    multiprocessing.freeze_support()

    from system.bootstrap import ensure_algorithm_on_path

    ensure_algorithm_on_path()

    from system.webapp import create_app

    application = create_app()
    host = os.environ.get("PONZI_HOST", "0.0.0.0")
    port = int(os.environ.get("PONZI_PORT", "7860"))
    application.run(host=host, port=port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
