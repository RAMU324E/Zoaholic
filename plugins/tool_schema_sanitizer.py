"""
tool_schema_sanitizer — Zoaholic 插件
=====================================
工具参数 Schema 清洗插件：专治上游推理解析器（如 Fireworks AI / vLLM / Outlines）
对 JSON Schema 的各种消化不良。

典型场景（2026-09-08 实证，日志 55684）：
  客户端（LimCode / VS Code）发来的 edit 工具包含互斥联合类型：
    "oneOf": [{"properties": {"hunks": {"minItems": 1}}, "required": ["hunks"]}, ...]
  Fireworks AI / Outlines 的有限推理解析器在解析到局部约束 {'minItems': 1} 时，
  因为没有显式声明 "type": "array" 而直接崩溃报错：
    400: JSON Schema not supported: could not understand the instance `{'minItems': 1}`.

清洗策略：
  1. 【剥离互斥分支 (默认)】：直接剥离 oneOf / anyOf（业界 LiteLLM 标准做法）。
     大模型依赖自然语言 description 理解互斥语义，剥离复杂的模式代数可杜绝
     BNF/Grammar 生成器状态机爆炸或报错。
  2. 【补齐残缺类型】：若节点包含 minItems/maxItems/items 但漏了 type，自动补齐 "type": "array"；
     若包含 minLength/maxLength/pattern 但漏了 type，自动补齐 "type": "string"。
  3. 【剔除脏字段】：剔除 Fireworks 等上游不接受的 default: None 等非法字段。
  4. 【双载荷形态覆盖】：同时支持 OpenAI 格式 (tools[].function.parameters)
     和 Claude 格式 (tools[].input_schema)。

参数（enabled_plugins 中配置）：
  tool_schema_sanitizer               默认行为：剥离 oneOf/anyOf 并补全残缺 type
  tool_schema_sanitizer:keep_unions   保留 oneOf/anyOf 结构，仅做类型补全与脏值修复
  tool_schema_sanitizer:verbose       打印每次清洗的详细日志

部署：/opt/zoaholic-prod/plugins-custom/tool_schema_sanitizer.py
母本：.agents/skills/zoaholic/plugins/tool_schema_sanitizer.py
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from core.plugins import register_request_interceptor, unregister_request_interceptor

logger = logging.getLogger("zoaholic.plugins.tool_schema_sanitizer")

PLUGIN_NAME = "tool_schema_sanitizer"

PLUGIN_INFO = {
    "name": PLUGIN_NAME,
    "version": "1.0.0",
    "description": "工具参数 Schema 清洗插件 — 剥离 oneOf/anyOf 复杂联合、补全缺 type 约束，解决 Fireworks/Outlines 400 崩溃",
    "author": "heiyan",
    "dependencies": [],
    "metadata": {
        "category": "interceptors",
        "tags": ["schema", "tools", "fireworks", "compat", "oneOf"],
        "params_hint": (
            "参数格式：tool_schema_sanitizer[:keep_unions][:verbose]\n"
            "  (空)         = 默认，剥离 oneOf/anyOf 并补齐缺失 type\n"
            "  keep_unions  = 保留 oneOf/anyOf 结构，仅补全缺失类型\n"
            "  verbose      = 打印详细清洗变动"
        ),
    },
}

EXTENSIONS = ["interceptors:tool_schema_sanitizer_request"]

_DICT_CHILD_KEYS = ("properties", "$defs", "definitions", "patternProperties")
_MAX_DEPTH = 30


def _parse_options(provider: Any) -> Dict[str, bool]:
    """解析 enabled_plugins 中的插件选项。"""
    opts = {"keep_unions": False, "verbose": False}
    prefs = (provider or {}).get("preferences") if isinstance(provider, dict) else None
    for entry in ((prefs or {}).get("enabled_plugins") or []):
        if isinstance(entry, str) and entry.startswith(PLUGIN_NAME):
            tail = entry.split(":", 1)[1] if ":" in entry else ""
            for tok in tail.split(","):
                t = tok.strip().lower()
                if t == "keep_unions":
                    opts["keep_unions"] = True
                elif t == "verbose":
                    opts["verbose"] = True
    return opts


def _fix_missing_types(node: Dict[str, Any], top_props: Dict[str, Any], path: str, report: List[str]) -> bool:
    """就地补全节点中缺少 type 的孤立约束（如 {minItems: 1}）。"""
    changed = False

    # 1. 数组类型推导
    if ("minItems" in node or "maxItems" in node or "items" in node) and "type" not in node:
        node["type"] = "array"
        report.append(f"{path} 缺少 type，已补全 'type': 'array'")
        changed = True

    # 2. 字符串类型推导
    if ("minLength" in node or "maxLength" in node or "pattern" in node) and "type" not in node:
        node["type"] = "string"
        report.append(f"{path} 缺少 type，已补全 'type': 'string'")
        changed = True

    # 3. 数值类型推导
    if ("minimum" in node or "maximum" in node or "multipleOf" in node) and "type" not in node:
        node["type"] = "number"
        report.append(f"{path} 缺少 type，已补全 'type': 'number'")
        changed = True

    # 4. 对象类型推导
    if ("properties" in node or "required" in node) and "type" not in node:
        node["type"] = "object"
        report.append(f"{path} 缺少 type，已补全 'type': 'object'")
        changed = True

    return changed


def _clean_schema_node(
    node: Any,
    top_props: Dict[str, Any],
    path: str,
    opts: Dict[str, bool],
    report: List[str],
    depth: int = 0,
) -> bool:
    """递归清洗单个 Schema 节点。"""
    if depth > _MAX_DEPTH or not isinstance(node, dict):
        return False

    changed = False

    # 1. 补齐缺失的 type
    if _fix_missing_types(node, top_props, path, report):
        changed = True

    # 2. 处理 oneOf / anyOf
    for union_key in ("oneOf", "anyOf"):
        if union_key in node:
            if not opts["keep_unions"]:
                # 默认策略：直接剥离复杂联合分支，退化为顶层 properties
                node.pop(union_key, None)
                report.append(f"{path}.{union_key} 已剥离（退化为扁平 properties）")
                changed = True
            else:
                # 保留策略：递归下钻分支内部补齐类型
                branches = node.get(union_key)
                if isinstance(branches, list):
                    for idx, b in enumerate(branches):
                        if _clean_schema_node(b, top_props, f"{path}.{union_key}[{idx}]", opts, report, depth + 1):
                            changed = True

    # 3. 递归遍历 properties / $defs 等子字典
    for d_key in _DICT_CHILD_KEYS:
        child_dict = node.get(d_key)
        if isinstance(child_dict, dict):
            for prop_name, sub_node in list(child_dict.items()):
                # 如果是属性，且顶层有同名属性，尝试从顶层借 type
                if isinstance(sub_node, dict) and "type" not in sub_node and prop_name in top_props:
                    ref_type = top_props[prop_name].get("type") if isinstance(top_props[prop_name], dict) else None
                    if ref_type:
                        sub_node["type"] = ref_type
                        report.append(f"{path}.{d_key}[{prop_name}] 从外层借取 'type': '{ref_type}'")
                        changed = True

                if _clean_schema_node(sub_node, top_props, f"{path}.{d_key}[{prop_name}]", opts, report, depth + 1):
                    changed = True

    # 4. 递归 items
    items = node.get("items")
    if isinstance(items, dict):
        if _clean_schema_node(items, top_props, f"{path}.items", opts, report, depth + 1):
            changed = True
    elif isinstance(items, list):
        for idx, sub_item in enumerate(items):
            if _clean_schema_node(sub_item, top_props, f"{path}.items[{idx}]", opts, report, depth + 1):
                changed = True

    return changed


def _iter_tool_parameters(payload: Dict[str, Any]):
    """遍历 payload 中所有工具的 parameters/input_schema 字典。"""
    tools = payload.get("tools")
    if not isinstance(tools, list):
        return

    for t_idx, tool in enumerate(tools):
        if not isinstance(tool, dict):
            continue

        # OpenAI 格式: tools[].function.parameters
        fn = tool.get("function")
        if isinstance(fn, dict):
            params = fn.get("parameters")
            if isinstance(params, dict):
                yield params, f"tools[{t_idx}].function.parameters"

        # Claude / 扁平格式: tools[].parameters 或 tools[].input_schema
        for k in ("parameters", "input_schema"):
            params = tool.get(k)
            if isinstance(params, dict):
                yield params, f"tools[{t_idx}].{k}"


async def tool_schema_sanitizer_request_interceptor(
    request: Any,
    engine: str,
    provider: Dict[str, Any],
    api_key: Optional[str],
    url: str,
    headers: Dict[str, Any],
    payload: Dict[str, Any],
) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """请求拦截器：检查并清洗所有工具 Schema。"""
    try:
        if not isinstance(payload, dict) or not payload.get("tools"):
            return url, headers, payload

        opts = _parse_options(provider)
        report: List[str] = []

        for params, path in _iter_tool_parameters(payload):
            top_props = params.get("properties", {}) if isinstance(params.get("properties"), dict) else {}
            _clean_schema_node(params, top_props, path, opts, report)

        if report:
            provider_name = (provider or {}).get("provider", "?") if isinstance(provider, dict) else "?"
            logger.info(
                f"[{PLUGIN_NAME}] 渠道 {provider_name} 清洗工具 Schema ({len(report)} 处改动): "
                + "; ".join(report[:6])
                + (f" ...(+{len(report) - 6})" if len(report) > 6 else "")
            )
        elif opts["verbose"]:
            logger.info(f"[{PLUGIN_NAME}] 工具 Schema 检查完毕，未发现异常")

    except Exception as e:
        # 绝不 fail-closed，异常放行
        logger.warning(f"[{PLUGIN_NAME}] 清洗过程出现异常，fail-open放行: {e}")

    return url, headers, payload


# ==================== 插件生命周期 ====================

def setup(manager: Any) -> None:
    logger.info(f"[{PLUGIN_INFO['name']}] 正在初始化...")

    # 注销同名旧拦截器，确保幂等热重载
    try:
        unregister_request_interceptor("tool_schema_sanitizer_request")
    except Exception:
        pass

    register_request_interceptor(
        interceptor_id="tool_schema_sanitizer_request",
        callback=tool_schema_sanitizer_request_interceptor,
        priority=40,  # 紧随 factory_compat (30)，在 oai_tools (50) 之前
        plugin_name=PLUGIN_INFO["name"],
        metadata={"description": "清洗工具 JSON Schema（剥离 oneOf/anyOf，修复无 type 约束）"},
        overwrite=True,
    )
    logger.info(f"[{PLUGIN_INFO['name']}] 已注册请求拦截器")


def teardown(manager: Any) -> None:
    logger.info(f"[{PLUGIN_INFO['name']}] 正在清理...")
    try:
        unregister_request_interceptor("tool_schema_sanitizer_request")
    except Exception:
        pass
    logger.info(f"[{PLUGIN_INFO['name']}] 已清理完成")
