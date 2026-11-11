"""Kaggle Verifier Pool Plugin (v2.0 - 490+ Account Pool Edition)

专为 Kaggle ModelProxy (`https://mp-staging.kaggle.net/models`) 打造的成熟号池插件（可直接单文件分享给朋友）：

核心特性：
1. **多格式号源通吃（支持 490+ 混合号池）**：
   - 支持 `KGAT_<32hex>` Bearer 令牌（如 `kgat_bearer_tokens` / `kgat_keys`）；
   - 支持传统 `kaggle.json` 账号（`{"username": "...", "key": "..."}` 或 `username:key` 行格式，自动走 HTTP Basic Auth 并携带 `ownerSlug` 跑通 `BenchmarkTasks` 签发 12h Token）；
   - 号源可放在 `/home/data/kaggle_keys.json`（可通过环境变量 `KAGGLE_KEYS_FILE` 覆盖），也可直接粘贴在渠道 API Keys 框中。
2. **用 1 备 1（Active + Standby）+ 12 小时生命周期管理**：
   - 始终仅维护 1 张主力 `active` + 1 张热备 `standby` 的 12 小时 `kaggleweb:...` 临时 Token，绝不一上来把几百张卡全激活浪费 12h 寿命；
   - 距 12h 到期不足 30 分钟时，若当前母卡余额充足则同卡无缝续签 12h；若余额不足则自动提升 `standby` 为 `active` 并后台补签新 `standby`。
3. **精确额度解析 + 官方 `refillTime` 智能冷却 + 失效隔离**：
   - 完整解析 Kaggle `/api/v1/models/proxy/quota` 返回的 `daily`（$10/日）与 `monthly`（$100/月）数组及 `refillTime`（ISO 刷新时间）；
   - 当某张母卡余额低于阈值或请求返回 403/429 时，0 秒切换 `standby`，并将旧母卡按官方 `refillTime`（无则默认 6 小时）放入冷却池，到期自动解封回池；
   - 当母卡认证返回 401/403 永久无权限时，自动记入 `invalid_keys` 隔离，避免反复重试死卡。
4. **号池容量与 Token 直观可见**：
   - 自动将当前 `active` / `standby` 的 `kaggleweb:...` Token 与号池统计同步到渠道 API Keys 列表；
   - 点击渠道“余额查询”即可在行内看到 **`池 490 (可用488/冷却2) · 主#490 $10.00 715m`** 及一键复制 `active` / `standby` Token 按钮。
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import random
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import httpx

from core.log_config import logger
from core.plugins import (
    get_current_plugin_options,
    get_plugin_options,
    register_balance_enricher,
    register_request_interceptor,
    register_response_interceptor,
    unregister_balance_enricher,
    unregister_request_interceptor,
    unregister_response_interceptor,
)

PLUGIN_NAME = "kaggle_verifier_pool"
_REQ_ID = "kaggle_verifier_pool_request"
_RESP_ID = "kaggle_verifier_pool_response"
_BAL_ID = "kaggle_verifier_pool_balance"

DEFAULT_KEYS_FILE = os.environ.get("KAGGLE_KEYS_FILE", "/home/data/kaggle_keys.json")
DEFAULT_STATE_FILE = os.environ.get("KAGGLE_STATE_FILE", "/home/data/kaggle_verifier_state.json")

TOKEN_TTL_SECONDS = int(11.5 * 3600)   # 11.5h 保守有效期（官方 12h）
RENEW_AHEAD_SECONDS = 30 * 60          # 距到期 <30 分钟触发续签
DEFAULT_COOLDOWN_SECONDS = 6 * 3600    # 无官方 refillTime 时默认冷却 6 小时

PARAMS_SCHEMA = [
    {
        "key": "pin_key",
        "label": "手动指定母卡 (留空=自动)",
        "type": "text",
        "default": "",
        "placeholder": "填序号(1~490)、用户名 或 KGAT_ 前后缀",
        "hint": "留空则按用1备1自动轮换；填入序号(如 490)、用户名或 KGAT 前缀可强制锁定该母卡",
        "serialize": "key_value",
    },
    {
        "key": "pick_order",
        "label": "自动挑卡顺序 (多人共用默认随机防撞)",
        "type": "select",
        "default": "random",
        "options": [
            {"value": "random", "label": "随机洗牌 (多人零配置防撞推荐)"},
            {"value": "reverse", "label": "倒序 (N → 1)"},
            {"value": "forward", "label": "正序 (1 → N)"},
        ],
        "serialize": "key_value",
    },
    {
        "key": "max_tokens_cap",
        "label": "max_tokens 安全封顶",
        "type": "number",
        "min": 512,
        "max": 65536,
        "default": 65536,
        "hint": "Kaggle 网关实测支持 65536；若单卡余额较低时遇预冻结 403 可调低为 16384 或 8192",
        "serialize": "key_value",
    },
    {
        "key": "min_quota",
        "label": "母卡最低可用余额 ($)",
        "type": "number",
        "min": 0.2,
        "max": 9.5,
        "default": 1.5,
        "hint": "低于该美金额度的母卡自动进入冷却并跳过，签发下一把",
        "serialize": "key_value",
    },
    {
        "key": "rotate_nonce",
        "label": "强制换卡触发号",
        "type": "text",
        "default": "0",
        "placeholder": "随便改个数字并保存即可立即切到热备卡",
        "hint": "修改此值后下一次请求或查余额会立即将热备卡扶正并后台补签",
        "serialize": "key_value",
    },
]

PLUGIN_INFO = {
    "name": PLUGIN_NAME,
    "version": "2.0.0",
    "description": "Kaggle 裁判号池 v2.0 (支持 490+ KGAT/kaggle.json 混合号池)：用1备1自动签发 12h kaggleweb Token、官方 refillTime 智能冷却、0 秒热备切换、池容量/Token 面板直看",
    "author": "Zoaholic Team",
    "dependencies": [],
    "metadata": {
        "display_name": "Kaggle 裁判号池 v2.0 (用1备1 · 智能冷却)",
        "category": "interceptors",
        "tags": ["kaggle", "auth", "pool", "token", "gemini", "claude", "openai"],
        "params_hint": "支持 KGAT_ 与 legacy kaggle.json 混合号池；点余额查询可看池子总数/可用/冷却及主备 Token",
        "params_schema": PARAMS_SCHEMA,
    },
}

EXTENSIONS = [
    f"interceptors:{_REQ_ID}",
    f"interceptors:{_RESP_ID}",
    f"balance_enrichers:{_BAL_ID}",
]

_state_lock = asyncio.Lock()
_bg_mint_task: Optional[asyncio.Task] = None
_cached_state: Optional[Dict[str, Any]] = None
_last_proxy: Dict[str, Any] = {"url": None, "require": False}


def _refresh_proxy_state(provider: Optional[Dict[str, Any]]) -> None:
    if not isinstance(provider, dict):
        return
    try:
        from core.utils import is_warp_only_provider, resolve_provider_proxy
        if is_warp_only_provider(provider):
            _last_proxy.update(url=resolve_provider_proxy(provider), require=True)
        else:
            _last_proxy.update(url=None, require=False)
    except ImportError:
        _last_proxy.update(url=None, require=False)
    except Exception as exc:
        logger.warning(f"[{PLUGIN_NAME}] proxy resolve failed: {exc}")


def _proxy_ctx(provider: Optional[Dict[str, Any]]):
    url: Any = None
    require = False
    try:
        if isinstance(provider, dict):
            from core.utils import is_warp_only_provider, resolve_provider_proxy
            if is_warp_only_provider(provider):
                url, require = resolve_provider_proxy(provider), True
    except ImportError:
        pass
    if not require and _last_proxy.get("require"):
        url, require = _last_proxy.get("url"), True
    if require:
        from core.http import proxy_context
        return proxy_context(url, require_proxy=True)
    return nullcontext()


def parse_options(raw_options: Any) -> Dict[str, Any]:
    cfg = {
        "pin_key": "",
        "pick_order": "random",
        "max_tokens_cap": 65536,
        "min_quota": 1.5,
        "rotate_nonce": "0",
    }
    if isinstance(raw_options, dict):
        for k in cfg:
            if k in raw_options and raw_options[k] is not None:
                cfg[k] = raw_options[k]
    elif isinstance(raw_options, str) and raw_options.strip():
        for part in raw_options.split(","):
            part = part.strip()
            if "=" in part:
                k, v = part.split("=", 1)
                k, v = k.strip(), v.strip()
                if k in cfg:
                    cfg[k] = v
    try:
        cfg["max_tokens_cap"] = max(256, int(float(cfg["max_tokens_cap"])))
    except Exception:
        cfg["max_tokens_cap"] = 65536
    try:
        cfg["min_quota"] = max(0.1, float(cfg["min_quota"]))
    except Exception:
        cfg["min_quota"] = 1.5
    cfg["pin_key"] = str(cfg.get("pin_key") or "").strip()
    po = str(cfg.get("pick_order") or "random").strip().lower()
    cfg["pick_order"] = po if po in ("random", "reverse", "forward") else "random"
    cfg["rotate_nonce"] = str(cfg.get("rotate_nonce") or "0").strip()
    return cfg


# ==================== 多格式母卡归一化 ====================
# 统一内部凭证格式：
# - KGAT Bearer:  "KGAT_<hex>"
# - Basic 账号:   "KAGGLE_BASIC:<username>:<key>"

def _normalize_cred_entry(entry: Any) -> Optional[str]:
    if isinstance(entry, str):
        s = entry.strip().lstrip("!")
        if not s:
            return None
        if s.startswith("KGAT_"):
            return s
        if s.startswith("KAGGLE_BASIC:"):
            parts = s.split(":", 2)
            if len(parts) == 3 and parts[1] and parts[2]:
                return f"KAGGLE_BASIC:{parts[1].strip()}:{parts[2].strip()}"
        if s.startswith("{") and s.endswith("}"):
            try:
                obj = json.loads(s)
                return _normalize_cred_entry(obj)
            except Exception:
                return None
        if ":" in s and not s.startswith("kaggleweb:") and not s.startswith("http"):
            u, k = s.split(":", 1)
            u, k = u.strip(), k.strip()
            if u and len(k) >= 16 and " " not in u and " " not in k:
                return f"KAGGLE_BASIC:{u}:{k}"
        return None
    if isinstance(entry, dict):
        tok = entry.get("token") or entry.get("kgat")
        if isinstance(tok, str) and tok.strip().startswith("KGAT_"):
            return tok.strip()
        kj = entry.get("kaggle_json") if isinstance(entry.get("kaggle_json"), dict) else entry
        u = str(kj.get("username") or entry.get("username") or "").strip()
        k = str(kj.get("key") or entry.get("key") or "").strip()
        if k.startswith("KGAT_"):
            return k
        if u and k:
            return f"KAGGLE_BASIC:{u}:{k}"
    return None


def _cred_auth_header(cred: str) -> str:
    if cred.startswith("KAGGLE_BASIC:"):
        _, u, k = cred.split(":", 2)
        b64 = base64.b64encode(f"{u}:{k}".encode("utf-8")).decode("ascii")
        return f"Basic {b64}"
    return f"Bearer {cred}"


def _cred_owner_slug(cred: str) -> Optional[str]:
    if cred.startswith("KAGGLE_BASIC:"):
        parts = cred.split(":", 2)
        return parts[1] if len(parts) >= 2 else None
    return None


def _cred_masked(cred: str) -> str:
    if cred.startswith("KAGGLE_BASIC:"):
        _, u, k = cred.split(":", 2)
        tail = k[-4:] if len(k) >= 4 else k
        return f"json:{u}(..{tail})"
    if len(cred) > 16:
        return f"{cred[:11]}...{cred[-4:]}"
    return cred


def _cred_short_tail(cred: str) -> str:
    if cred.startswith("KAGGLE_BASIC:"):
        _, u, _ = cred.split(":", 2)
        return u[:10]
    return f"..{cred[-4:]}" if len(cred) >= 4 else cred


def _load_keys_source(provider: Optional[Dict[str, Any]] = None) -> Tuple[List[str], List[str]]:
    """Return (mother_creds, seed_verifier_tokens) from provider config + DEFAULT_KEYS_FILE."""
    creds: List[str] = []
    seen: set[str] = set()
    seed_tokens: List[str] = []

    def add_cred(item: Any) -> None:
        norm = _normalize_cred_entry(item)
        if norm and norm not in seen:
            seen.add(norm)
            creds.append(norm)

    if isinstance(provider, dict):
        api_val = provider.get("api") or provider.get("api_keys")
        raw_list = api_val if isinstance(api_val, list) else ([api_val] if isinstance(api_val, str) else [])
        for item in raw_list:
            if isinstance(item, dict) and len(item) == 1:
                k_only = str(next(iter(item.keys()))).strip().lstrip("!")
                if k_only.startswith("kaggleweb:"):
                    continue
                add_cred(k_only)
                continue
            for line in str(item or "").replace(",", "\n").splitlines():
                s = line.strip()
                if not s:
                    continue
                clean = s.lstrip("!")
                if clean.startswith("kaggleweb:"):
                    continue
                add_cred(clean)

    p = Path(DEFAULT_KEYS_FILE)
    if p.is_file():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                for s in data.get("kgat_keys") or []:
                    add_cred(s)
                for item in data.get("kgat_bearer_tokens") or []:
                    add_cred(item)
                for item in data.get("legacy_kaggle_json_accounts") or []:
                    add_cred(item)
                for item in data.get("accounts") or []:
                    add_cred(item)
                for s in data.get("active_12h_tokens") or []:
                    if isinstance(s, str) and s.strip().startswith("kaggleweb:") and s.strip() not in seed_tokens:
                        seed_tokens.append(s.strip())
            elif isinstance(data, list):
                for item in data:
                    add_cred(item)
        except Exception as exc:
            logger.warning(f"[{PLUGIN_NAME}] failed reading {p}: {exc}")

    return creds, seed_tokens


def _normalize_cooldowns(raw_ex: Any) -> Dict[str, Dict[str, Any]]:
    """Upgrade legacy {cred: timestamp} or {cred: {until, reason}} into a clean dict."""
    out: Dict[str, Dict[str, Any]] = {}
    if not isinstance(raw_ex, dict):
        return out
    for k, v in raw_ex.items():
        if isinstance(v, (int, float)):
            # 旧版存的是触发时间戳，统一换算为触发后 6 小时解封
            out[str(k)] = {"until": float(v) + DEFAULT_COOLDOWN_SECONDS, "reason": "quota_exhausted"}
        elif isinstance(v, dict):
            until = float(v.get("until") or 0)
            out[str(k)] = {"until": until, "reason": str(v.get("reason") or "cooldown")}
    return out


def _is_in_cooldown(cred: str, state: Dict[str, Any], now: Optional[float] = None) -> bool:
    now_ts = now or time.time()
    invalid = state.get("invalid_keys")
    if isinstance(invalid, dict) and cred in invalid:
        return True
    cd = _normalize_cooldowns(state.get("exhausted_kgats"))
    entry = cd.get(cred)
    if not entry:
        return False
    return float(entry.get("until") or 0) > now_ts


def _mark_cooldown(
    state: Dict[str, Any],
    cred: str,
    reason: str = "quota_exhausted",
    until: Optional[float] = None,
) -> None:
    if not cred:
        return
    cd = _normalize_cooldowns(state.get("exhausted_kgats"))
    cd[cred] = {
        "until": float(until) if until and until > time.time() else (time.time() + DEFAULT_COOLDOWN_SECONDS),
        "reason": reason,
        "marked_at": time.time(),
    }
    state["exhausted_kgats"] = cd


def _mark_invalid(state: Dict[str, Any], cred: str, reason: str = "auth_failed") -> None:
    if not cred:
        return
    inv = state.get("invalid_keys")
    if not isinstance(inv, dict):
        inv = {}
    inv[cred] = {"reason": reason, "marked_at": time.time()}
    state["invalid_keys"] = inv


def _pool_counts(creds: List[str], state: Dict[str, Any], now: Optional[float] = None) -> Dict[str, int]:
    now_ts = now or time.time()
    total = len(creds)
    kgat_count = sum(1 for c in creds if c.startswith("KGAT_"))
    basic_count = total - kgat_count
    inv = state.get("invalid_keys") if isinstance(state.get("invalid_keys"), dict) else {}
    cd = _normalize_cooldowns(state.get("exhausted_kgats"))
    invalid_n = 0
    cooling_n = 0
    for c in creds:
        if c in inv:
            invalid_n += 1
        elif c in cd and float(cd[c].get("until") or 0) > now_ts:
            cooling_n += 1
    available_n = max(0, total - invalid_n - cooling_n)
    return {
        "total": total,
        "available": available_n,
        "cooling": cooling_n,
        "invalid": invalid_n,
        "kgat_count": kgat_count,
        "legacy_json_count": basic_count,
    }


def _load_state(kgat_keys: List[str], seed_tokens: List[str]) -> Dict[str, Any]:
    global _cached_state
    if _cached_state is not None:
        return _cached_state

    p = Path(DEFAULT_STATE_FILE)
    if p.is_file():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                data["exhausted_kgats"] = _normalize_cooldowns(data.get("exhausted_kgats"))
                if not isinstance(data.get("invalid_keys"), dict):
                    data["invalid_keys"] = {}
                _cached_state = data
                return _cached_state
        except Exception as exc:
            logger.warning(f"[{PLUGIN_NAME}] failed loading state {p}: {exc}")

    now = time.time()
    active_slot = None
    standby_slot = None
    if seed_tokens:
        active_slot = {
            "verifier_token": seed_tokens[0],
            "kgat": kgat_keys[-1] if kgat_keys else "seed_1",
            "index": len(kgat_keys) if kgat_keys else 1,
            "minted_at": now,
            "expires_at": now + 6 * 3600,
            "quota_remaining": 10.0,
            "source": "seed",
        }
    if len(seed_tokens) >= 2:
        standby_slot = {
            "verifier_token": seed_tokens[1],
            "kgat": kgat_keys[-2] if len(kgat_keys) >= 2 else "seed_2",
            "index": max(1, len(kgat_keys) - 1) if kgat_keys else 2,
            "minted_at": now,
            "expires_at": now + 6 * 3600,
            "quota_remaining": 10.0,
            "source": "seed",
        }

    _cached_state = {
        "active": active_slot,
        "standby": standby_slot,
        "cursor_index": None,
        "rotate_nonce": "0",
        "exhausted_kgats": {},
        "invalid_keys": {},
        "updated_at": now,
    }
    _save_state(_cached_state)
    return _cached_state


def _save_state(state: Dict[str, Any]) -> None:
    global _cached_state
    state["updated_at"] = time.time()
    _cached_state = state
    try:
        p = Path(DEFAULT_STATE_FILE)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(p)
    except Exception as exc:
        logger.warning(f"[{PLUGIN_NAME}] failed saving state: {exc}")
    _sync_state_to_runtime_provider(state)


def _format_key_label(prefix: str, slot: Dict[str, Any], pool_summary: str = "") -> str:
    idx = slot.get("index", "?")
    usd = float(slot.get("quota_remaining") or 0.0)
    rem_min = max(0, int((float(slot.get("expires_at") or 0) - time.time()) / 60))
    cred = str(slot.get("kgat") or "")
    tail = _cred_short_tail(cred)
    extra = f" · {pool_summary}" if pool_summary else ""
    return f"{prefix}#{idx}({tail}) · ${usd:.2f} · {rem_min}m{extra}"


def _build_provider_api_entries(state: Dict[str, Any], creds: Optional[List[str]] = None) -> Tuple[List[str], Dict[str, str]]:
    """Build runtime api list + _api_labels so the admin UI directly shows pool size + active & standby tokens."""
    if creds is None:
        creds, _ = _load_keys_source()
    counts = _pool_counts(creds, state)
    pool_tag = f"池{counts['total']}把(可用{counts['available']}/冷{counts['cooling']})" if counts["total"] else ""

    api_list: List[str] = []
    labels: Dict[str, str] = {}
    act = state.get("active")
    sb = state.get("standby")
    if isinstance(act, dict) and act.get("verifier_token"):
        tok = str(act["verifier_token"]).strip()
        api_list.append(tok)
        labels[tok] = _format_key_label("主卡", act, pool_tag)
    if isinstance(sb, dict) and sb.get("verifier_token"):
        tok = str(sb["verifier_token"]).strip()
        if tok not in labels:
            api_list.append(f"!{tok}")
            labels[tok] = _format_key_label("热备", sb)
    return api_list, labels


def _sync_state_to_runtime_provider(state: Dict[str, Any]) -> None:
    """Keep the channel's API Keys list in sync with active/standby verifier tokens for easy viewing in UI."""
    try:
        api_list, labels = _build_provider_api_entries(state)
        if not api_list:
            return
        from core.plugins.interceptors import parse_enabled_plugins
        from main import get_app
        app = get_app()
        cfg = getattr(app.state, "config", None) or {}
        changed = False
        for p in cfg.get("providers") or []:
            if not isinstance(p, dict):
                continue
            ep = ((p.get("preferences") or {}).get("enabled_plugins")) or []
            if PLUGIN_NAME in parse_enabled_plugins(ep):
                if p.get("api") != api_list or p.get("_api_labels") != labels:
                    p["api"] = list(api_list)
                    p["_api_labels"] = dict(labels)
                    changed = True
        if changed:
            try:
                loop = asyncio.get_running_loop()
                from core.config.db_store import save_config_to_db
                loop.create_task(save_config_to_db(cfg))
            except Exception:
                pass
    except Exception:
        pass


# ==================== Kaggle Quota & RefillTime 解析 ====================

def _parse_iso_timestamp(iso_str: Any) -> Optional[float]:
    if not isinstance(iso_str, str) or not iso_str.strip():
        return None
    s = iso_str.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    # Python 3.10 compatibility for fractional seconds != 3 or 6 digits
    if "." in s and "+" in s:
        head, tz = s.split("+", 1)
        base, frac = head.split(".", 1)
        frac = (frac + "000000")[:6]
        s = f"{base}.{frac}+{tz}"
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        return None


def _parse_quota_details(quota_obj: Any) -> Dict[str, Any]:
    """Parse Kaggle /api/v1/models/proxy/quota response (both list and dict formats).

    Returns:
      {
        "remaining": float (min of daily & monthly remaining USD),
        "daily_remaining": float,
        "monthly_remaining": float,
        "refill_until": Optional[float] (epoch timestamp when exhausted bucket refills),
      }
    """
    if isinstance(quota_obj, list):
        daily_rem: Optional[float] = None
        monthly_rem: Optional[float] = None
        daily_refill: Optional[float] = None
        monthly_refill: Optional[float] = None
        for item in quota_obj:
            if not isinstance(item, dict):
                continue
            allowed = float(item.get("totalQuotaAllowed") or 0.0)
            used = float(item.get("quotaUsed") or 0.0)
            rem = max(0.0, allowed - used)
            period = str(item.get("refillPeriod") or "").lower()
            refill_ts = _parse_iso_timestamp(item.get("refillTime"))
            if "day" in period or "daily" in period or allowed <= 15.0:
                daily_rem = rem
                daily_refill = refill_ts
            else:
                monthly_rem = rem
                monthly_refill = refill_ts
        d_val = daily_rem if daily_rem is not None else 10.0
        m_val = monthly_rem if monthly_rem is not None else 100.0
        eff = min(d_val, m_val)
        refill = monthly_refill if m_val < d_val else daily_refill
        return {
            "remaining": eff,
            "daily_remaining": d_val,
            "monthly_remaining": m_val,
            "refill_until": refill,
        }

    if isinstance(quota_obj, dict):
        for key in ("remainingQuota", "remaining_quota", "availableQuota", "available", "remaining"):
            if key in quota_obj:
                try:
                    val = float(quota_obj[key])
                    return {"remaining": val, "daily_remaining": val, "monthly_remaining": val, "refill_until": None}
                except Exception:
                    pass
        for limit_k, used_k in (
            ("dailyLimit", "dailyUsed"),
            ("dailyQuota", "dailyUsed"),
            ("totalQuotaAllowed", "quotaUsed"),
            ("limit", "used"),
            ("maxQuota", "usedQuota"),
        ):
            if limit_k in quota_obj and used_k in quota_obj:
                try:
                    val = max(0.0, float(quota_obj[limit_k]) - float(quota_obj[used_k]))
                    return {
                        "remaining": val,
                        "daily_remaining": val,
                        "monthly_remaining": val,
                        "refill_until": _parse_iso_timestamp(quota_obj.get("refillTime")),
                    }
                except Exception:
                    pass
    return {"remaining": 10.0, "daily_remaining": 10.0, "monthly_remaining": 100.0, "refill_until": None}


def _extract_quota_remaining(quota_obj: Any) -> float:
    return float(_parse_quota_details(quota_obj)["remaining"])


async def check_kaggle_quota(client: httpx.AsyncClient, cred: str) -> Tuple[bool, float, Any]:
    url = "https://www.kaggle.com/api/v1/models/proxy/quota"
    try:
        resp = await client.get(
            url,
            headers={"Authorization": _cred_auth_header(cred), "Accept": "application/json"},
            timeout=15.0,
        )
        if resp.status_code != 200:
            return False, 0.0, f"HTTP {resp.status_code}"
        data = resp.json()
        return True, _extract_quota_remaining(data), data
    except Exception as exc:
        return False, 0.0, str(exc)


def _make_notebook(code_str: str) -> str:
    return json.dumps({
        "cells": [{"cell_type": "code", "execution_count": None, "id": "c1", "metadata": {}, "outputs": [], "source": code_str}],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    })


async def _kaggle_rpc(client: httpx.AsyncClient, cred: str, service: str, method: str, payload: Any = None) -> Tuple[int, Any]:
    url = f"https://www.kaggle.com/api/v1/{service}/{method}"
    try:
        resp = await client.post(
            url,
            json=payload or {},
            headers={
                "Authorization": _cred_auth_header(cred),
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            timeout=30.0,
        )
        try:
            return resp.status_code, resp.json()
        except Exception:
            return resp.status_code, resp.text[:1000]
    except Exception as exc:
        return 0, str(exc)


async def mint_verifier_token_async(
    cred: str,
    slug: str = "sanity-bench-check",
    provider: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[str], float, Any]:
    """Mint a 12-hour kaggleweb:... verifier token from either a KGAT_ key or a KAGGLE_BASIC:user:key account."""
    extract_code = """import os, json
import kaggle_benchmarks as kbench

raw_keys = os.environ.get("MODEL_PROXY_API_KEYS", "[]")
print("VERIFIER_JSON_START:" + raw_keys + ":VERIFIER_JSON_END")

@kbench.task(name="sanity-bench-check", description="Sanity check")
def sanity_bench_check(llm) -> None:
    kbench.assertions.assert_true(True)

sanity_bench_check.run(kbench.llm)
"""
    clean_code = """import kaggle_benchmarks as kbench
@kbench.task(name="sanity-bench-check", description="Sanity check")
def sanity_bench_check(llm) -> None:
    kbench.assertions.assert_true(True)
sanity_bench_check.run(kbench.llm)
"""
    with _proxy_ctx(provider):
        async with httpx.AsyncClient(timeout=30.0) as client:
            ok, rem_quota, raw_quota = await check_kaggle_quota(client, cred)
            if not ok:
                return None, 0.0, raw_quota

            st, resp = await _kaggle_rpc(
                client,
                cred,
                "benchmarks.BenchmarkTasksApiService",
                "CreateBenchmarkTask",
                {"slug": slug, "text": _make_notebook(extract_code)},
            )
            if st != 200:
                logger.warning(f"[{PLUGIN_NAME}] CreateBenchmarkTask failed (HTTP {st}): {resp}")
                return None, rem_quota, resp

            # Basic Auth 账号调 GetBenchmarkTask / ListBenchmarkTaskRuns 必须带 ownerSlug
            owner_slug = _cred_owner_slug(cred)
            if isinstance(resp, dict) and isinstance(resp.get("slug"), dict):
                owner_slug = resp["slug"].get("ownerSlug") or owner_slug
            slug_Locator: Dict[str, Any] = {"taskSlug": slug}
            if owner_slug:
                slug_Locator["ownerSlug"] = owner_slug

            verifier_token: Optional[str] = None
            try:
                for _ in range(45):
                    await asyncio.sleep(4.0)
                    _, gresp = await _kaggle_rpc(
                        client,
                        cred,
                        "benchmarks.BenchmarkTasksApiService",
                        "GetBenchmarkTask",
                        {"slug": slug_Locator},
                    )
                    cstate = gresp.get("creationState", "") if isinstance(gresp, dict) else ""
                    if "COMPLETED" in cstate or "ERRORED" in cstate:
                        break

                _, rresp = await _kaggle_rpc(
                    client,
                    cred,
                    "benchmarks.BenchmarkTasksApiService",
                    "ListBenchmarkTaskRuns",
                    {"taskSlug": slug_Locator},
                )
                runs = rresp.get("runs", []) if isinstance(rresp, dict) else []
                for run in runs:
                    rid = run.get("id")
                    _, logs = await _kaggle_rpc(
                        client,
                        cred,
                        "benchmarks.BenchmarkTasksApiService",
                        "GetBenchmarkTaskRunLogs",
                        {"runId": rid},
                    )
                    if isinstance(logs, list):
                        stdout_text = "".join(
                            x.get("data", "") for x in logs if isinstance(x, dict) and x.get("stream_name") == "stdout"
                        )
                    else:
                        stdout_text = str(logs)
                    if "VERIFIER_JSON_START:" in stdout_text:
                        seg = stdout_text.split("VERIFIER_JSON_START:", 1)[1].split(":VERIFIER_JSON_END", 1)[0]
                        arr = json.loads(seg)
                        if isinstance(arr, list) and len(arr) >= 2:
                            item = arr[1]
                            tok = (
                                item["apiKey"]
                                if isinstance(item, dict) and "apiKey" in item
                                else (item["key"] if isinstance(item, dict) and "key" in item else item)
                            )
                            if isinstance(tok, dict):
                                tok = next(
                                    (v for v in tok.values() if isinstance(v, str) and v.startswith("kaggleweb:")),
                                    str(tok),
                                )
                            if isinstance(tok, str) and tok.startswith("kaggleweb:"):
                                verifier_token = tok
                                break
            finally:
                await _kaggle_rpc(
                    client,
                    cred,
                    "benchmarks.BenchmarkTasksApiService",
                    "CreateBenchmarkTask",
                    {"slug": slug, "text": _make_notebook(clean_code)},
                )

            return verifier_token, rem_quota, raw_quota


def _slot_valid(slot: Optional[Dict[str, Any]], now: Optional[float] = None) -> bool:
    if not isinstance(slot, dict):
        return False
    tok = str(slot.get("verifier_token") or "")
    exp = float(slot.get("expires_at") or 0)
    return tok.startswith("kaggleweb:") and exp > (now or time.time()) + 60


def _slot_needs_renewal(slot: Optional[Dict[str, Any]], now: Optional[float] = None) -> bool:
    if not _slot_valid(slot, now):
        return True
    exp = float(slot.get("expires_at") or 0)
    return exp <= (now or time.time()) + RENEW_AHEAD_SECONDS


def _resolve_pinned_kgat(kgat_keys: List[str], pin_key: str) -> Optional[Tuple[int, str]]:
    if not pin_key or not kgat_keys:
        return None
    if pin_key.isdigit():
        idx = int(pin_key)
        if 1 <= idx <= len(kgat_keys):
            return idx, kgat_keys[idx - 1]
    for i, k in enumerate(kgat_keys, start=1):
        if k.startswith(pin_key) or k.endswith(pin_key) or (f":{pin_key}:" in k):
            return i, k
    return None


def _candidate_order(
    kgat_keys: List[str],
    state: Dict[str, Any],
    opts: Dict[str, Any],
    exclude_kgat: str = "",
) -> List[Tuple[int, str]]:
    total = len(kgat_keys)
    if total == 0:
        return []
    pinned = _resolve_pinned_kgat(kgat_keys, opts.get("pin_key", ""))
    ordered: List[Tuple[int, str]] = []
    if pinned and pinned[1] != exclude_kgat:
        ordered.append(pinned)

    order_mode = opts.get("pick_order") or "random"
    if order_mode == "random":
        indices = list(range(1, total + 1))
        random.SystemRandom().shuffle(indices)
    elif order_mode == "reverse":
        indices = list(range(total, 0, -1))
        cursor = state.get("cursor_index")
        if isinstance(cursor, int) and cursor in indices:
            pos = indices.index(cursor)
            indices = indices[pos:] + indices[:pos]
    else:
        indices = list(range(1, total + 1))
        cursor = state.get("cursor_index")
        if isinstance(cursor, int) and cursor in indices:
            pos = indices.index(cursor)
            indices = indices[pos:] + indices[:pos]

    now = time.time()
    for idx in indices:
        k = kgat_keys[idx - 1]
        if k == exclude_kgat:
            continue
        if any(k == existing_k for _, existing_k in ordered):
            continue
        if _is_in_cooldown(k, state, now):
            continue
        ordered.append((idx, k))
    return ordered


async def _mint_from_candidates(
    kgat_keys: List[str],
    state: Dict[str, Any],
    opts: Dict[str, Any],
    exclude_kgat: str = "",
    provider: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    candidates = _candidate_order(kgat_keys, state, opts, exclude_kgat=exclude_kgat)
    min_quota = float(opts.get("min_quota") or 1.5)
    with _proxy_ctx(provider):
        async with httpx.AsyncClient(timeout=15.0) as client:
            for idx, cred in candidates:
                ok, rem, raw_q = await check_kaggle_quota(client, cred)
                if not ok:
                    if isinstance(raw_q, str) and ("HTTP 401" in raw_q or "HTTP 403" in raw_q):
                        _mark_invalid(state, cred, reason=raw_q)
                        _save_state(state)
                    continue
                q_details = _parse_quota_details(raw_q)
                if rem < min_quota:
                    _mark_cooldown(
                        state,
                        cred,
                        reason=f"low_quota(${rem:.2f})",
                        until=q_details.get("refill_until"),
                    )
                    _save_state(state)
                    continue
                logger.info(
                    f"[{PLUGIN_NAME}] minting 12h verifier token from mother key #{idx} "
                    f"({_cred_masked(cred)}, quota=${rem:.2f})"
                )
                tok, rem2, _ = await mint_verifier_token_async(cred, provider=provider)
                if tok:
                    now = time.time()
                    state["cursor_index"] = idx
                    return {
                        "verifier_token": tok,
                        "kgat": cred,
                        "index": idx,
                        "minted_at": now,
                        "expires_at": now + TOKEN_TTL_SECONDS,
                        "quota_remaining": rem2,
                        "quota_daily": q_details.get("daily_remaining", rem2),
                        "quota_monthly": q_details.get("monthly_remaining", 100.0),
                        "refill_until": q_details.get("refill_until"),
                        "quota_raw": raw_q,
                        "source": "minted",
                    }
    return None


async def _ensure_pool_Task(
    kgat_keys: List[str],
    seed_tokens: List[str],
    opts: Dict[str, Any],
    provider: Optional[Dict[str, Any]] = None,
) -> None:
    """Background task to keep active + standby populated and renewed."""
    async with _state_lock:
        state = _load_state(kgat_keys, seed_tokens)
        now = time.time()
        active = state.get("active")
        standby = state.get("standby")

        # 1. If active is missing or expired, promote standby if valid
        if not _slot_valid(active, now) and _slot_valid(standby, now):
            logger.info(f"[{PLUGIN_NAME}] promoting standby key #{standby.get('index')} to active")
            state["active"] = standby
            state["standby"] = None
            active = state["active"]
            standby = None
            _save_state(state)

        # 2. If active is still invalid, mint one now
        if not _slot_valid(active, now):
            minted = await _mint_from_candidates(kgat_keys, state, opts, provider=provider)
            if minted:
                state["active"] = minted
                active = minted
                _save_state(state)

        # 3. If active is valid but nearing 12h expiry (<30m), try same-key renewal first
        elif _slot_needs_renewal(active, now):
            act_cred = str(active.get("kgat") or "")
            renewed = None
            if act_cred and not act_cred.startswith("seed_"):
                with _proxy_ctx(provider):
                    async with httpx.AsyncClient(timeout=15.0) as client:
                        ok, rem, raw_q = await check_kaggle_quota(client, act_cred)
                if ok and rem >= float(opts.get("min_quota") or 1.5):
                    logger.info(f"[{PLUGIN_NAME}] same-key renewing active #{active.get('index')} ({_cred_masked(act_cred)})")
                    tok, rem2, _ = await mint_verifier_token_async(act_cred, provider=provider)
                    if tok:
                        q_details = _parse_quota_details(raw_q)
                        renewed = {
                            **active,
                            "verifier_token": tok,
                            "minted_at": time.time(),
                            "expires_at": time.time() + TOKEN_TTL_SECONDS,
                            "quota_remaining": rem2,
                            "quota_daily": q_details.get("daily_remaining", rem2),
                            "quota_monthly": q_details.get("monthly_remaining", 100.0),
                            "refill_until": q_details.get("refill_until"),
                            "quota_raw": raw_q,
                            "source": "renewed",
                        }
                elif ok:
                    q_details = _parse_quota_details(raw_q)
                    _mark_cooldown(state, act_cred, reason="renewal_low_quota", until=q_details.get("refill_until"))
            if renewed:
                state["active"] = renewed
                active = renewed
                _save_state(state)
            elif _slot_valid(standby, now):
                logger.info(f"[{PLUGIN_NAME}] active key near expiry & low quota; promoting standby #{standby.get('index')}")
                state["active"] = standby
                state["standby"] = None
                active = state["active"]
                standby = None
                _save_state(state)

        # 4. Ensure standby is valid and from a different mother key
        if not _slot_valid(standby, now):
            exclude = str((state.get("active") or {}).get("kgat") or "")
            minted_sb = await _mint_from_candidates(kgat_keys, state, opts, exclude_kgat=exclude, provider=provider)
            if minted_sb:
                state["standby"] = minted_sb
                _save_state(state)


def _trigger_bg_maintenance(
    kgat_keys: List[str],
    seed_tokens: List[str],
    opts: Dict[str, Any],
    provider: Optional[Dict[str, Any]] = None,
) -> None:
    global _bg_mint_task
    if _bg_mint_task is not None and not _bg_mint_task.done():
        return
    try:
        loop = asyncio.get_running_loop()
        _bg_mint_task = loop.create_task(_ensure_pool_Task(kgat_keys, seed_tokens, opts, provider=provider))
    except RuntimeError:
        pass


def _rotate_on_quota_exhausted(reason: str) -> bool:
    """Immediately promote standby to active when active hits 403/401/429."""
    kgat_keys, seed_tokens = _load_keys_source()
    state = _load_state(kgat_keys, seed_tokens)
    active = state.get("active")
    standby = state.get("standby")
    if isinstance(active, dict):
        act_cred = str(active.get("kgat") or "")
        if act_cred and not act_cred.startswith("seed_"):
            _mark_cooldown(state, act_cred, reason=reason, until=active.get("refill_until"))
    if _slot_valid(standby):
        logger.warning(
            f"[{PLUGIN_NAME}] active key #{active.get('index') if isinstance(active, dict) else '?'} "
            f"exhausted ({reason}); 0s promoting standby #{standby.get('index')}"
        )
        state["active"] = standby
        state["standby"] = None
        _save_state(state)
        _trigger_bg_maintenance(kgat_keys, seed_tokens, parse_options(None))
        return True
    logger.warning(f"[{PLUGIN_NAME}] active key exhausted ({reason}) and no valid standby yet; triggering background mint")
    state["active"] = None
    _save_state(state)
    _trigger_bg_maintenance(kgat_keys, seed_tokens, parse_options(None))
    return False


def _strip_url_key_param(url: str) -> str:
    if "key=" not in url:
        return url
    parsed = urlparse(url)
    q = [(k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=True) if k.lower() != "key"]
    return urlunparse(parsed._replace(query=urlencode(q)))


def _cap_max_tokens(payload: Dict[str, Any], cap: int) -> None:
    if not isinstance(payload, dict) or cap <= 0:
        return
    for field in ("max_tokens", "max_completion_tokens", "max_output_tokens"):
        if field in payload:
            try:
                if int(payload[field]) > cap:
                    payload[field] = cap
            except Exception:
                payload[field] = cap
    if "contents" in payload:
        gen_cfg = payload.get("generationConfig")
        if not isinstance(gen_cfg, dict):
            gen_cfg = {}
            payload["generationConfig"] = gen_cfg
        cur = gen_cfg.get("maxOutputTokens")
        try:
            if cur is None or int(cur) > cap:
                gen_cfg["maxOutputTokens"] = cap
        except Exception:
            gen_cfg["maxOutputTokens"] = cap
    elif "messages" in payload and "max_tokens" not in payload and "max_completion_tokens" not in payload:
        payload["max_tokens"] = cap


async def kaggle_verifier_pool_request_interceptor(
    request: Any,
    engine: str,
    provider: Dict[str, Any],
    api_key: Optional[str],
    url: str,
    headers: Dict[str, Any],
    payload: Dict[str, Any],
) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    opts = parse_options(get_plugin_options(PLUGIN_NAME, provider))
    kgat_keys, seed_tokens = _load_keys_source(provider)
    _refresh_proxy_state(provider)
    state = _load_state(kgat_keys, seed_tokens)

    if opts["rotate_nonce"] != str(state.get("rotate_nonce") or "0"):
        state["rotate_nonce"] = opts["rotate_nonce"]
        if _slot_valid(state.get("standby")):
            state["active"] = state.get("standby")
            state["standby"] = None
        else:
            state["active"] = None
        _save_state(state)

    pinned = _resolve_pinned_kgat(kgat_keys, opts["pin_key"])
    active = state.get("active")
    if pinned and isinstance(active, dict) and active.get("kgat") != pinned[1]:
        sb = state.get("standby")
        if isinstance(sb, dict) and sb.get("kgat") == pinned[1] and _slot_valid(sb):
            state["active"], state["standby"] = sb, active
            active = state["active"]
            _save_state(state)
        else:
            await _ensure_pool_Task(kgat_keys, seed_tokens, opts, provider=provider)
            state = _load_state(kgat_keys, seed_tokens)
            active = state.get("active")

    if not _slot_valid(active):
        await _ensure_pool_Task(kgat_keys, seed_tokens, opts, provider=provider)
        state = _load_state(kgat_keys, seed_tokens)
        active = state.get("active")
    elif _slot_needs_renewal(active) or not _slot_valid(state.get("standby")):
        _trigger_bg_maintenance(kgat_keys, seed_tokens, opts, provider=provider)

    verifier_token = str((active or {}).get("verifier_token") or "").strip()
    if not verifier_token and isinstance(api_key, str) and api_key.startswith("kaggleweb:"):
        verifier_token = api_key

    if verifier_token:
        url = _strip_url_key_param(url)
        low_url = url.lower()
        eng = str(engine or "").lower()
        if "genai" in low_url or ":generatecontent" in low_url or ":streamgeneratecontent" in low_url or eng == "gemini":
            headers.pop("Authorization", None)
            headers.pop("x-api-key", None)
            headers["x-goog-api-key"] = verifier_token
        elif "/anthropic" in low_url or eng in {"claude", "anthropic"}:
            headers.pop("Authorization", None)
            headers.pop("x-goog-api-key", None)
            headers["x-api-key"] = verifier_token
        else:
            headers.pop("x-goog-api-key", None)
            headers.pop("x-api-key", None)
            headers["Authorization"] = f"Bearer {verifier_token}"

    _cap_max_tokens(payload, opts["max_tokens_cap"])
    return url, headers, payload


def _is_kaggle_quota_or_auth_error(chunk: Any) -> Optional[str]:
    if isinstance(chunk, dict):
        status = chunk.get("status_code")
        blob = json.dumps(chunk, ensure_ascii=False, default=str).lower()
        if status in (401, 403, 429) or "exceeds your available quota" in blob or ("quota" in blob and "403" in blob):
            if any(k in blob for k in ("quota", "unauthorized", "unauthenticated", "expired", "invalid", "ip address", "401", "403", "429")):
                return f"status={status}"
    elif isinstance(chunk, str):
        low = chunk.lower()
        if "exceeds your available quota" in low or ("kaggle" in low and ("401" in low or "403" in low)):
            return "stream_quota_exceeded"
    return None


async def kaggle_verifier_pool_response_interceptor(
    response_chunk: Any,
    engine: str,
    model: str,
    is_stream: bool,
) -> Any:
    reason = _is_kaggle_quota_or_auth_error(response_chunk)
    if reason:
        _rotate_on_quota_exhausted(reason)
        if isinstance(response_chunk, dict):
            rewritten = dict(response_chunk)
            if rewritten.get("status_code") in (401, 403):
                rewritten["status_code"] = 502
            return rewritten
    return response_chunk


def _format_slot_summary(slot: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(slot, dict):
        return None
    now = time.time()
    exp = float(slot.get("expires_at") or 0)
    rem_min = max(0, int((exp - now) / 60))
    cred = str(slot.get("kgat") or "")
    return {
        "index": slot.get("index"),
        "kgat_masked": _cred_masked(cred),
        "auth_type": "basic_json" if cred.startswith("KAGGLE_BASIC:") else "kgat_bearer",
        "quota_remaining_usd": round(float(slot.get("quota_remaining") or 0.0), 2),
        "quota_daily_usd": round(float(slot.get("quota_daily") or slot.get("quota_remaining") or 0.0), 2),
        "quota_monthly_usd": round(float(slot.get("quota_monthly") or 100.0), 2),
        "expires_in_minutes": rem_min,
        "valid": _slot_valid(slot, now),
        "verifier_token": slot.get("verifier_token"),
    }


async def kaggle_verifier_pool_balance_enricher(
    result: Dict[str, Any],
    engine: str,
    provider: Dict[str, Any],
) -> Dict[str, Any]:
    opts = parse_options(get_plugin_options(PLUGIN_NAME, provider) or get_current_plugin_options(PLUGIN_NAME))
    kgat_keys, seed_tokens = _load_keys_source(provider)
    state = _load_state(kgat_keys, seed_tokens)
    active = state.get("active")

    if isinstance(active, dict):
        act_cred = str(active.get("kgat") or "")
        if act_cred and not act_cred.startswith("seed_"):
            with _proxy_ctx(provider):
                async with httpx.AsyncClient(timeout=10.0) as client:
                    ok, rem, raw_q = await check_kaggle_quota(client, act_cred)
            if ok:
                q_details = _parse_quota_details(raw_q)
                active["quota_remaining"] = rem
                active["quota_daily"] = q_details.get("daily_remaining", rem)
                active["quota_monthly"] = q_details.get("monthly_remaining", 100.0)
                active["refill_until"] = q_details.get("refill_until")
                active["quota_raw"] = raw_q
                _save_state(state)

    if not _slot_valid(active) or not _slot_valid(state.get("standby")):
        _trigger_bg_maintenance(kgat_keys, seed_tokens, opts, provider=provider)

    counts = _pool_counts(kgat_keys, state)
    act_info = _format_slot_summary(state.get("active"))
    sb_info = _format_slot_summary(state.get("standby"))
    rem_usd = act_info["quota_remaining_usd"] if act_info else 0.0
    act_idx = act_info["index"] if act_info else "?"
    exp_min = act_info["expires_in_minutes"] if act_info else 0

    enriched = dict(result or {})
    enriched["supported"] = True
    enriched["error"] = None
    enriched["value_type"] = "amount"
    enriched["available"] = rem_usd
    enriched["total"] = 10.0
    enriched["used"] = round(max(0.0, 10.0 - rem_usd), 2)
    enriched["unit"] = "USD"
    enriched["tier"] = f"池{counts['total']}把(可用{counts['available']}/冷{counts['cooling']}) · 主#{act_idx} ({exp_min}m)"
    enriched["kaggle_pool"] = {
        "total_mother_keys": counts["total"],
        "available_keys": counts["available"],
        "cooling_keys": counts["cooling"],
        "invalid_keys": counts["invalid"],
        "kgat_bearer_count": counts["kgat_count"],
        "legacy_json_count": counts["legacy_json_count"],
        "pick_order": opts["pick_order"],
        "pin_key": opts["pin_key"] or "auto",
        "max_tokens_cap": opts["max_tokens_cap"],
        "active": act_info,
        "standby": sb_info,
    }
    enriched["raw"] = enriched["kaggle_pool"]
    return enriched


KAGGLE_POOL_QUOTA_DISPLAY = """
export default function render(ctx) {
    const { el, data, context } = ctx || {};
    if (!el) return;
    const mode = context?.mode || 'row';
    const pool = data?.kaggle_pool || context?.balance?.kaggle_pool || data?.raw || {};
    const act = pool?.active;
    const sb = pool?.standby;
    const tier = data?.tier || context?.balance?.tier || '';
    const total = pool?.total_mother_keys ?? 0;
    const avail = pool?.available_keys ?? total;
    const cool = pool?.cooling_keys ?? 0;
    if (mode === 'rack') {
        el.textContent = act ? `#${act.index ?? '?'}/${total} $${(act.quota_remaining_usd ?? 0).toFixed(0)}` : (tier || '');
        el.className = 'text-[10px] font-semibold font-mono text-emerald-500';
        return;
    }
    el.innerHTML = '';
    el.className = 'flex items-center gap-1 flex-wrap';
    if (total > 0) {
        const pBadge = document.createElement('span');
        pBadge.textContent = `池 ${total} (可用${avail}/冷${cool})`;
        pBadge.title = `KGAT Bearer: ${pool?.kgat_bearer_count ?? 0} | Legacy JSON: ${pool?.legacy_json_count ?? 0} | 失效: ${pool?.invalid_keys ?? 0}`;
        pBadge.className = 'text-[10px] font-semibold font-mono px-1.5 py-0.5 rounded text-cyan-400 bg-cyan-500/15';
        el.appendChild(pBadge);
    }
    const badge = document.createElement('span');
    const usd = act?.quota_remaining_usd ?? data?.available ?? 0;
    const mins = act?.expires_in_minutes ?? 0;
    badge.textContent = act ? `主#${act.index} · $${Number(usd).toFixed(2)} · ${mins}m` : (tier || '未就绪');
    badge.className = 'text-[10px] font-semibold font-mono px-1.5 py-0.5 rounded text-emerald-500 bg-emerald-500/15';
    if (act?.verifier_token) {
        badge.title = `Active(#${act.index} ${act.kgat_masked || ''}): ${act.verifier_token}`;
    }
    el.appendChild(badge);
    const mkBtn = (label, slot, cls) => {
        if (!slot?.verifier_token) return;
        const btn = document.createElement('button');
        btn.type = 'button';
        const shortTok = String(slot.verifier_token).slice(0, 16) + '...';
        btn.textContent = `${label}#${slot.index}:${shortTok}`;
        btn.title = `点击复制完整 Token (${slot.kgat_masked || ''}):\\n${slot.verifier_token}`;
        btn.className = `text-[10px] font-mono px-1.5 py-0.5 rounded border cursor-pointer ${cls}`;
        btn.onclick = (e) => {
            e.stopPropagation();
            if (navigator?.clipboard?.writeText) {
                navigator.clipboard.writeText(slot.verifier_token);
                const old = btn.textContent;
                btn.textContent = `✓ 已复制 #${slot.index}`;
                setTimeout(() => { btn.textContent = old; }, 1500);
            }
        };
        el.appendChild(btn);
    };
    mkBtn('主', act, 'border-blue-500/40 bg-blue-500/10 text-blue-400 hover:bg-blue-500/20');
    mkBtn('备', sb, 'border-purple-500/40 bg-purple-500/10 text-purple-400 hover:bg-purple-500/20');
}
"""


def setup(manager):
    register_request_interceptor(
        interceptor_id=_REQ_ID,
        callback=kaggle_verifier_pool_request_interceptor,
        priority=20,
        plugin_name=PLUGIN_NAME,
        overwrite=True,
        metadata={"description": "Kaggle 12h 裁判号池用1备1注入与 max_tokens 封顶"},
    )
    register_response_interceptor(
        interceptor_id=_RESP_ID,
        callback=kaggle_verifier_pool_response_interceptor,
        priority=20,
        plugin_name=PLUGIN_NAME,
        overwrite=True,
        metadata={"description": "Kaggle 额度耗尽 0 秒切换热备卡并按 refillTime 冷却"},
    )
    register_balance_enricher(
        enricher_id=_BAL_ID,
        callback=kaggle_verifier_pool_balance_enricher,
        priority=20,
        plugin_name=PLUGIN_NAME,
        overwrite=True,
        metadata={"description": "Kaggle 裁判号池容量/冷却统计与当前临时 Token 查看"},
    )
    try:
        from core.ui_slots.registry import register_ui_slot
        register_ui_slot(
            slot_id=f"{PLUGIN_NAME}.quota_display",
            slot="quota_display",
            script=KAGGLE_POOL_QUOTA_DISPLAY,
            source=PLUGIN_NAME,
            priority=80,
            engines=["gemini", "openai", "claude", "anthropic", "openai-responses"],
            auth_types=["api_key"],
            enabled_plugin=PLUGIN_NAME,
        )
    except Exception as exc:
        logger.warning(f"[{PLUGIN_NAME}] register_ui_slot failed: {exc}")


def teardown(manager):
    unregister_request_interceptor(_REQ_ID)
    unregister_response_interceptor(_RESP_ID)
    unregister_balance_enricher(_BAL_ID)
    try:
        from core.ui_slots.registry import unregister_ui_slot
        unregister_ui_slot(f"{PLUGIN_NAME}.quota_display")
    except Exception:
        pass
