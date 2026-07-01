import os


class WebConfig:
    """Web 层配置（与训练 configs.config 分离）。"""

    PRODUCT_NAME = os.environ.get("APP_PRODUCT_NAME", "合约风险智能研判平台")
    PRODUCT_CODE = os.environ.get("APP_PRODUCT_CODE", "SCRA")
    VERSION = os.environ.get("APP_VERSION", "1.0.0")
    ENV = os.environ.get("APP_ENV", "development")
    URL_PREFIX = os.environ.get("PONZI_URL_PREFIX", "").rstrip("/")
    DEFAULT_LANGUAGE = os.environ.get("PONZI_DEFAULT_LANGUAGE", "zh")

    SECRET_KEY = os.environ.get("FLASK_SECRET_KEY", "dev-change-in-production")

    ANALYSIS_DEFAULT_TOP_K = int(os.environ.get("PONZI_EXPLAIN_TOP_K", "5"))

    LEGAL_DISCLAIMER = (
        "本系统输出仅供安全研究与辅助研判，不构成法律、投资或合规结论；"
        "最终结论须由具备资质的人员结合业务与监管要求作出。"
    )
