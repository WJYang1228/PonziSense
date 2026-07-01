import time

from flask import Blueprint, jsonify, render_template, current_app

bp = Blueprint("system", __name__, url_prefix="/system")

_start = time.monotonic()


@bp.route("/health")
def health():
    """企业环境常用的存活与版本探测（可对接负载均衡 / 监控）。"""
    cfg = current_app.config["WEB_CFG"]
    return jsonify(
        {
            "status": "ok",
            "product": cfg.PRODUCT_CODE,
            "version": cfg.VERSION,
            "environment": cfg.ENV,
            "uptime_seconds": round(time.monotonic() - _start, 2),
        }
    )


@bp.route("/about")
def about():
    return render_template("system/about.html")
