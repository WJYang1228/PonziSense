from flask import Blueprint, current_app, render_template, request

from system.webapp.i18n import translate as t
from system.webapp.services.analysis_service import AnalysisService

bp = Blueprint("analysis", __name__, url_prefix="/workspace")


@bp.route("/analysis", methods=["GET", "POST"])
def workspace():
    service = AnalysisService()
    code = ""
    result = None
    error = None
    top_k = current_app.config["WEB_CFG"].ANALYSIS_DEFAULT_TOP_K

    if request.method == "POST":
        code = request.form.get("code", "") or ""
        top_k = int(request.form.get("top_k") or top_k)
        top_k = max(1, min(top_k, 20))
        if code.strip():
            try:
                result = service.analyze(code, top_k=top_k)
            except FileNotFoundError as e:
                error = str(e)
            except ValueError as e:
                error = str(e)
            except Exception as e:
                error = t("analysis.error.service", error=e)

    return render_template(
        "analysis/workspace.html",
        code=code,
        result=result,
        error=error,
        top_k=top_k,
    )
