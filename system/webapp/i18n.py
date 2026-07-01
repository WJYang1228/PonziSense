from __future__ import annotations

from flask import request, session

SUPPORTED_LANGUAGES = ("zh", "en")
LANGUAGE_META = {
    "zh": {"label": "中文", "short_label": "中", "html_lang": "zh-CN"},
    "en": {"label": "English", "short_label": "EN", "html_lang": "en"},
}

TRANSLATIONS = {
    "zh": {
        "product.name": "合约风险智能研判平台",
        "product.short_name": "合约风险平台",
        "legal.disclaimer": "本系统输出仅供安全研究与辅助研判，不构成法律、投资或合规结论；最终结论须由具备资质的人员结合业务与监管要求作出。",
        "language.label": "语言",
        "language.selector": "语言选择",
        "nav.dashboard": "工作台概览",
        "nav.analysis": "合约分析工作台",
        "nav.system": "系统",
        "nav.about": "关于与合规说明",
        "nav.health": "健康检查 (JSON)",
        "topbar.default": "工作台",
        "dashboard.title": "工作台概览",
        "dashboard.meta": "按业务模块组织功能，可解释性为分析链路中的内置能力",
        "dashboard.welcome": "欢迎使用 {product_name}",
        "dashboard.lead": "面向智能合约的风险研判与复核流程：先完成合约级分类，再通过可解释性子模块定位可疑语句，并在源码视图中高亮对照。",
        "module.analysis.title": "合约分析工作台",
        "module.analysis.desc": "上传或粘贴 Solidity 源码，完成风险分类，并联动可解释性与源码视图。",
        "module.explain.title": "可解释性（子模块）",
        "module.explain.desc": "在分析结果中自动输出语句级可疑度排序与源码高亮，支撑人工复核。",
        "module.system.title": "系统与运维",
        "module.system.desc": "健康检查与版本信息，便于对接监控与发布流程。",
        "module.badge.core": "核心",
        "module.badge.feature": "功能模块",
        "module.badge.ops": "运维",
        "common.enter_module": "进入模块 →",
        "analysis.title": "合约分析工作台",
        "analysis.meta": "风险研判 · 可解释性 · 源码审阅 同一流程闭环",
        "analysis.submit.title": "提交分析任务",
        "analysis.submit.hint": "粘贴完整或片段 Solidity 源码。提交后依次生成：风险结论 → 可解释性语句列表 → 高亮源码。",
        "analysis.code.placeholder": "在此粘贴 Solidity 合约代码…",
        "analysis.topk.label": "可解释性 Top-K 语句",
        "analysis.run": "运行分析",
        "analysis.failed": "任务失败",
        "analysis.risk.tag": "模块 · 风险研判",
        "analysis.risk.title": "合约级风险结论",
        "analysis.risk.label": "研判标签",
        "analysis.risk.probability": "疑似 Ponzi 概率",
        "analysis.risk.threshold": "判定阈值",
        "analysis.softmax": "Softmax 分布",
        "analysis.explain.tag": "子模块 · 可解释性",
        "analysis.explain.hint": "对语句进行可疑度打分并排序（Top {top_k}），供安全与业务人员快速定位复核点。",
        "analysis.statement": "语句 #{stmt_id}",
        "analysis.line_range": "行 {start}–{end}",
        "analysis.score": "可疑度 {score}",
        "analysis.empty": "当前未检出需优先展示的语句级归因（或均为过滤后的琐碎语句）。",
        "analysis.source.tag": "模块 · 源码审阅",
        "analysis.source.title": "高亮对照视图",
        "analysis.source.hint": "共 {line_count} 行；高亮与上方可解释性子模块联动。",
        "analysis.error.service": "分析服务异常: {error}",
        "about.title": "关于",
        "about.header": "关于与合规说明",
        "about.product.title": "产品信息",
        "about.name": "名称",
        "about.code": "代号",
        "about.version": "版本",
        "about.environment": "运行环境",
        "about.compliance.title": "合规与使用边界",
        "about.compliance.note": "可解释性模块输出为模型内部归因分数，不单独作为定案依据；应与人工审计、静态分析与链上行为等综合使用。",
        "about.ops.title": "运维接口",
        "about.ops.text": "监控探活可请求",
        "about.ops.suffix": "，返回 JSON 状态与版本字段。",
    },
    "en": {
        "product.name": "PonziSense Contract Risk Assessment Platform",
        "product.short_name": "PonziSense Platform",
        "legal.disclaimer": "System outputs are intended only for security research and assisted review. They do not constitute legal, investment, or compliance conclusions; final decisions must be made by qualified personnel together with business and regulatory requirements.",
        "language.label": "Language",
        "language.selector": "Language selector",
        "nav.dashboard": "Dashboard",
        "nav.analysis": "Contract Analysis",
        "nav.system": "System",
        "nav.about": "About & Compliance",
        "nav.health": "Health Check (JSON)",
        "topbar.default": "Workspace",
        "dashboard.title": "Dashboard",
        "dashboard.meta": "Capabilities are organized by business module; explainability is embedded in the analysis workflow.",
        "dashboard.welcome": "Welcome to {product_name}",
        "dashboard.lead": "A smart-contract risk assessment and review workflow: contract-level classification, rationale extraction, and source-code highlighting for human verification.",
        "module.analysis.title": "Contract Analysis Workspace",
        "module.analysis.desc": "Upload or paste Solidity source code to run risk classification with linked explainability and source-code review.",
        "module.explain.title": "Explainability Submodule",
        "module.explain.desc": "Automatically ranks suspicious statements and highlights source lines to support manual review.",
        "module.system.title": "System & Operations",
        "module.system.desc": "Health checks and version metadata for monitoring and release workflows.",
        "module.badge.core": "Core",
        "module.badge.feature": "Feature",
        "module.badge.ops": "Ops",
        "common.enter_module": "Open Module →",
        "analysis.title": "Contract Analysis Workspace",
        "analysis.meta": "Risk assessment · Explainability · Source review in one closed loop",
        "analysis.submit.title": "Submit Analysis Task",
        "analysis.submit.hint": "Paste complete or partial Solidity source code. The system returns a risk decision, statement-level explanations, and highlighted source code.",
        "analysis.code.placeholder": "Paste Solidity contract code here...",
        "analysis.topk.label": "Explainability Top-K statements",
        "analysis.run": "Run Analysis",
        "analysis.failed": "Task Failed",
        "analysis.risk.tag": "Module · Risk Assessment",
        "analysis.risk.title": "Contract-Level Risk Decision",
        "analysis.risk.label": "Decision Label",
        "analysis.risk.probability": "Suspected Ponzi Probability",
        "analysis.risk.threshold": "Decision Threshold",
        "analysis.softmax": "Softmax distribution",
        "analysis.explain.tag": "Submodule · Explainability",
        "analysis.explain.hint": "Statements are scored and ranked by suspiciousness (Top {top_k}) so security and business reviewers can quickly locate evidence.",
        "analysis.statement": "Statement #{stmt_id}",
        "analysis.line_range": "Lines {start}–{end}",
        "analysis.score": "Suspiciousness {score}",
        "analysis.empty": "No statement-level attributions are currently available for prioritized display, or all candidates were filtered as trivial statements.",
        "analysis.source.tag": "Module · Source Review",
        "analysis.source.title": "Highlighted Source View",
        "analysis.source.hint": "{line_count} lines in total; highlights are linked with the explainability submodule above.",
        "analysis.error.service": "Analysis service error: {error}",
        "about.title": "About",
        "about.header": "About & Compliance",
        "about.product.title": "Product Information",
        "about.name": "Name",
        "about.code": "Code",
        "about.version": "Version",
        "about.environment": "Environment",
        "about.compliance.title": "Compliance and Usage Boundaries",
        "about.compliance.note": "Explainability scores are model attributions and should not be used as stand-alone determinations. They should be combined with manual auditing, static analysis, and on-chain behavior.",
        "about.ops.title": "Operations Endpoint",
        "about.ops.text": "Monitoring probes can request",
        "about.ops.suffix": "to receive JSON status and version fields.",
    },
}


def normalize_language(language: str | None, default: str = "zh") -> str:
    if language in SUPPORTED_LANGUAGES:
        return language
    return default if default in SUPPORTED_LANGUAGES else "zh"


def persist_requested_language(default: str = "zh") -> str:
    requested = request.args.get("lang")
    if requested in SUPPORTED_LANGUAGES:
        session["lang"] = requested
        return requested
    saved = session.get("lang")
    if saved in SUPPORTED_LANGUAGES:
        return saved
    return normalize_language(default)


def get_language(default: str = "zh") -> str:
    requested = request.args.get("lang")
    if requested in SUPPORTED_LANGUAGES:
        return requested
    saved = session.get("lang")
    if saved in SUPPORTED_LANGUAGES:
        return saved
    return normalize_language(default)


def translate(key: str, lang: str | None = None, **values) -> str:
    active = normalize_language(lang) if lang else get_language()
    text = TRANSLATIONS.get(active, {}).get(key)
    if text is None:
        text = TRANSLATIONS["zh"].get(key, key)
    return text.format(**values) if values else text
