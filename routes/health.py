from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from db import DISABLE_DATABASE

router = APIRouter()


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_admin_request(request: Request) -> bool:
    auth = request.headers.get("Authorization") or ""
    token = auth.split(" ", 1)[1].strip() if auth.startswith("Bearer ") else (request.headers.get("x-api-key") or "").strip()
    if not token:
        return False
    try:
        from core.jwt_utils import is_admin_jwt
        if is_admin_jwt(token):
            return True
    except Exception:
        pass
    admin_keys = getattr(request.app.state, "admin_api_key", None) or []
    return token in admin_keys


def _build_health_payload(app, *, readiness: bool, include_details: bool = False) -> tuple[dict[str, Any], int]:
    """构建健康检查载荷。

    readiness=False  → /healthz（存活探针）：只要进程还在、事件循环没有完全失控就返回 200。
    readiness=True   → /readyz（就绪探针）：额外要求 startup 完成、config 已加载、
                        client_manager 和 channel_manager 已初始化。
    """
    state = getattr(app, "state", None)
    startup_completed = bool(getattr(state, "startup_completed", False))
    started_at = getattr(state, "started_at", None)
    if isinstance(started_at, datetime):
        uptime_seconds = max(0.0, (datetime.now(timezone.utc) - started_at).total_seconds())
        started_at_iso = started_at.astimezone(timezone.utc).isoformat()
    else:
        uptime_seconds = None
        started_at_iso = None

    version = getattr(state, "version", "unknown")
    needs_setup = bool(getattr(state, "needs_setup", False))
    config = getattr(state, "config", None)
    providers = len((config or {}).get("providers") or []) if isinstance(config, dict) else 0
    api_keys = len((config or {}).get("api_keys") or []) if isinstance(config, dict) else 0

    watchdog = getattr(state, "event_loop_watchdog", None)
    event_loop = watchdog.snapshot() if watchdog else {
        "enabled": False,
        "status": "missing",
        "healthy": True,
    }

    checks = {
        "startup": {
            "status": "ok" if startup_completed else "error",
            "startup_completed": startup_completed,
            "started_at": started_at_iso,
            "uptime_seconds": uptime_seconds,
        },
        "config": {
            "status": "ok" if config is not None else "error",
            "loaded": config is not None,
            "needs_setup": needs_setup,
            "provider_count": providers,
            "api_key_count": api_keys,
        },
        "client_manager": {
            "status": "ok" if hasattr(state, "client_manager") else "error",
            "initialized": hasattr(state, "client_manager"),
        },
        "channel_manager": {
            "status": "ok" if hasattr(state, "channel_manager") else "error",
            "initialized": hasattr(state, "channel_manager"),
        },
        "database": {
            "status": "disabled" if DISABLE_DATABASE else "ok",
            "enabled": not DISABLE_DATABASE,
        },
        "event_loop": event_loop,
    }

    blocking_error = event_loop.get("status") == "critical"
    blocking_warning = event_loop.get("status") == "warning"
    missing_runtime = (
        not startup_completed
        or config is None
        or not hasattr(state, "client_manager")
        or not hasattr(state, "channel_manager")
    )

    if readiness:
        # /readyz：就绪探针 —— 必须完成启动且所有运行时组件就绪
        if missing_runtime or blocking_error:
            overall_status = "error"
            status_code = 503
        elif blocking_warning or needs_setup:
            overall_status = "degraded"
            status_code = 200
        else:
            overall_status = "ok"
            status_code = 200
    else:
        # /healthz：存活探针 —— 只要事件循环没有完全失控就算存活
        if blocking_error:
            overall_status = "error"
            status_code = 503
        elif blocking_warning:
            overall_status = "degraded"
            status_code = 200
        else:
            overall_status = "ok"
            status_code = 200

    if not include_details:
        return {
            "status": overall_status,
            "service": "zoaholic",
            "timestamp": _utcnow_iso(),
            "probe": "readyz" if readiness else "healthz",
        }, status_code

    payload = {
        "status": overall_status,
        "service": "zoaholic",
        "version": version,
        "timestamp": _utcnow_iso(),
        "probe": "readyz" if readiness else "healthz",
        "checks": checks,
    }

    # ── 运行时指标（metrics）──
    try:
        from core.metrics import get_full_metrics_snapshot
        payload["metrics"] = get_full_metrics_snapshot(state)
    except Exception:
        payload["metrics"] = {"available": False}

    return payload, status_code


@router.get("/healthz")
async def healthz(request: Request):
    """存活探针：进程存活且事件循环正常即返回 200（未鉴权仅返回最小状态，防泄露上游域名）"""
    payload, status_code = _build_health_payload(request.app, readiness=False, include_details=_is_admin_request(request))
    return JSONResponse(status_code=status_code, content=payload)


@router.get("/readyz")
async def readyz(request: Request):
    """就绪探针：要求启动完成、配置已加载、运行时组件已初始化（未鉴权仅返回最小状态）"""
    payload, status_code = _build_health_payload(request.app, readiness=True, include_details=_is_admin_request(request))
    return JSONResponse(status_code=status_code, content=payload)
