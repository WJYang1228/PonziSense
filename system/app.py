"""
WSGI / 生产入口：gunicorn 示例  
  gunicorn -w 2 -b 0.0.0.0:7860 'system.app:app'
"""
from system.bootstrap import ensure_algorithm_on_path

ensure_algorithm_on_path()

from system.webapp import create_app

app = create_app()
