from flask import Blueprint, current_app, render_template, url_for

from system.webapp.i18n import translate as t

bp = Blueprint("portal", __name__)


def _app_url(endpoint, **values):
    path = url_for(endpoint, **values)
    prefix = getattr(current_app.config["WEB_CFG"], "URL_PREFIX", "")
    return f"{prefix}{path}" if prefix and path.startswith("/") else path


@bp.route("/")
def dashboard():
    return render_template(
        "dashboard.html",
        modules=[
            {
                "id": "analysis",
                "title": t("module.analysis.title"),
                "desc": t("module.analysis.desc"),
                "href": _app_url("analysis.workspace"),
                "badge": t("module.badge.core"),
            },
            {
                "id": "explain",
                "title": t("module.explain.title"),
                "desc": t("module.explain.desc"),
                "href": _app_url("analysis.workspace") + "#module-explainability",
                "badge": t("module.badge.feature"),
            },
            {
                "id": "system",
                "title": t("module.system.title"),
                "desc": t("module.system.desc"),
                "href": _app_url("system.health"),
                "badge": t("module.badge.ops"),
            },
        ],
    )
