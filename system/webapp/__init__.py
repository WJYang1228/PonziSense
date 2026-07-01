from pathlib import Path

from flask import Flask, request, session, url_for
from werkzeug.middleware.proxy_fix import ProxyFix

from system.bootstrap import ensure_algorithm_on_path
from system.webapp.config import WebConfig
from system.webapp.i18n import LANGUAGE_META, SUPPORTED_LANGUAGES, get_language, persist_requested_language, translate


def create_app() -> Flask:
    ensure_algorithm_on_path()
    system_root = Path(__file__).resolve().parent.parent
    app = Flask(
        __name__,
        template_folder=str(system_root / "templates"),
        static_folder=str(system_root / "static"),
    )
    app.config["WEB_CFG"] = WebConfig
    app.config["SECRET_KEY"] = WebConfig.SECRET_KEY
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

    from system.webapp.blueprints.portal import bp as portal_bp
    from system.webapp.blueprints.analysis import bp as analysis_bp
    from system.webapp.blueprints.system import bp as system_bp

    app.register_blueprint(portal_bp)
    app.register_blueprint(analysis_bp)
    app.register_blueprint(system_bp)

    @app.before_request
    def remember_language():
        persist_requested_language(app.config["WEB_CFG"].DEFAULT_LANGUAGE)

    @app.context_processor
    def inject_product():
        cfg = app.config["WEB_CFG"]
        lang = get_language(cfg.DEFAULT_LANGUAGE)

        def app_url(endpoint, **values):
            path = url_for(endpoint, **values)
            prefix = getattr(cfg, "URL_PREFIX", "")
            return f"{prefix}{path}" if prefix and path.startswith("/") else path

        def language_url(target_lang):
            target_lang = target_lang if target_lang in SUPPORTED_LANGUAGES else cfg.DEFAULT_LANGUAGE
            endpoint = request.endpoint or "portal.dashboard"
            values = dict(request.view_args or {})
            values.update(request.args.to_dict(flat=True))
            values["lang"] = target_lang
            try:
                return app_url(endpoint, **values)
            except Exception:
                return app_url("portal.dashboard", lang=target_lang)

        def t(key, **values):
            return translate(key, lang=lang, **values)

        return {
            "product_name": t("product.name"),
            "product_code": cfg.PRODUCT_CODE,
            "app_version": cfg.VERSION,
            "app_env": cfg.ENV,
            "legal_disclaimer": t("legal.disclaimer"),
            "app_url": app_url,
            "t": t,
            "ui_lang": lang,
            "html_lang": LANGUAGE_META[lang]["html_lang"],
            "language_options": [
                {"code": code, **LANGUAGE_META[code]} for code in SUPPORTED_LANGUAGES
            ],
            "language_url": language_url,
        }

    return app
