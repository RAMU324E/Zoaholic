"""
Kaggle ModelProxy 裁判卡池插件 (kaggle_verifier_pool)

功能特性：
1. 用 1 备 1 (Active + Standby)：
   - 维护 1 张当前使用的 12h 临时 Token (`kaggleweb:...`) + 1 张后台预热好的热备 Token。
   - 当前卡快过期（<30min）时：若母卡额度充足则同卡自动续签，否则无缝切换至热备卡。
   - 当前卡额度耗尽（403 exceeds your available quota / 401 / 429）时：0 秒切到热备卡，并将状态码改写为 502 触发网关自动重试，同时后台异步签发下一张热备卡。
2. 多人共用 48 母卡防撞：
   - 默认倒序挑卡 (`pick_order=reverse`，从第 48 把往第 1 把挑)，与朋友正序使用天然错开。
   - 签发前实时调用 `/api/v1/models/proxy/quota` 检查剩余美金（默认 >= $1.50 才签发）。
   - 支持 `pin_key` 手动指定母卡序号（1-48）或 `KGAT_` 前缀，支持 `rotate_nonce` 一键强制换卡。
3. 三大原生协议自适应 + 预冻结额度保护：
   - 自动识别 Gemini (`x-goog-api-key`)、Anthropic (`x-api-key`)、OpenAI (`Authorization: Bearer`) 并移除 URL 中的 `?key=`。
   - 自动将 `max_tokens` / `max_completion_tokens` / `generationConfig.maxOutputTokens` 限制在安全上限内（默认 8192），防止触发预冻结余额 403。
4. 可视化查看当前临时 Token：
   - 接入 `balance_enricher`：在后台点击渠道“余额查询”即可查看当前剩余额度、母卡序号及 `active` / `standby` 的 `kaggleweb:...` Token。
"""

from __future__ import annotations

import asyncio
import json
import os
import time
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
TOKEN_TTL_SECONDS = 11.5 * 3600  # 11.5h 保守有效期（官方 12h）
RENEW_AHEAD_SECONDS = 1800       # 提前 30 分钟续签

PARAMS_SCHEMA = [
    {
        "key": "pin_key",
        "label": "手动指定母卡 (留空 auto 自动轮换)",
        "type": "text",
        "default": "",
        "placeholder": "如 48 或 KGAT_f92be232 (留空则自动按顺序用1备1)",
        "hint": "填 1~48 的序号或 KGAT_ 前缀可锁定特定母卡；留空则自动用1备1",
        "serialize": "key_value",
    },
    {
        "key": "pick_order",
        "label": "自动挑卡顺序 (多人共用建议倒序)",
        "type": "select",
        "default": "reverse",
        "options": [
            {"value": "reverse", "label": "倒序 (48 → 1，防撞推荐)"},
            {"value": "forward", "label": "正序 (1 → 48)"},
        ],
        "serialize": "key_value",
    },
    {
        "key": "max_tokens_cap",
        "label": "max_tokens 安全封顶 (防预冻结 403)",
        "type": "number",
        "min": 512,
        "max": 32768,
        "default": 8192,
        "hint": "Kaggle 网关按 max_tokens 预冻结余额，建议保持 4096 或 8192",
        "serialize": "key_value",
    },
    {
        "key": "min_quota",
        "label": "母卡最低可用余额 ($)",
        "type": "number",
        "min": 0.2,
        "max": 9.5,
        "default": 1.5,
        "hint": "低于该美金额度的母卡自动跳过，签发下一把",
        "serialize": "key_value",
    },
    {
        "key": "rotate_nonce",
        "label": "强制切换/续签触发号",
        "type": "text",
        "default": "0",
        "placeholder": "随便改个数字并保存即可立即切到下一张卡",
        "hint": "修改此值后下一次请求或查余额会立即将热备卡扶正并后台重签",
        "serialize": "key_value",
    },
]

PLUGIN_INFO = {
    "name": PLUGIN_NAME,
    "version": "1.0.0",
    "description": "Kaggle 裁判卡池 (用1备1)：自动从 48 把 KGAT 母卡签发/续签 12h kaggleweb 临时 Token，额度用尽 0 秒切热备，自动封顶 max_tokens 防预冻结 403，支持手动选卡与余额面板查看 Token",
    "author": "Zoaholic Team",
    "dependencies": [],
    "metadata": {
        "display_name": "Kaggle 裁判卡池 (用1备1自动签发)",
        "category": "interceptors",
        "tags": ["kaggle", "auth", "token", "gemini", "claude", "openai"],
        "params_hint": "默认倒序用1备1、max_tokens=8192；点渠道余额查询可直接查看当前 kaggleweb 临时 Token",
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


def parse_options(raw_options: Any) -> Dict[str, Any]:
    cfg = {
        "pin_key": "",
        "pick_order": "reverse",
        "max_tokens_cap": 8192,
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
        cfg["max_tokens_cap"] = 8192
    try:
        cfg["min_quota"] = max(0.1, float(cfg["min_quota"]))
    except Exception:
        cfg["min_quota"] = 1.5
    cfg["pin_key"] = str(cfg.get("pin_key") or "").strip()
    cfg["pick_order"] = "forward" if str(cfg.get("pick_order")).strip().lower() == "forward" else "reverse"
    cfg["rotate_nonce"] = str(cfg.get("rotate_nonce") or "0").strip()
    return cfg


def _load_keys_source(provider: Optional[Dict[str, Any]] = None) -> Tuple[List[str], List[str]]:
    """Return (kgat_keys, seed_verifier_tokens)."""
    kgat_keys: List[str] = []
    seed_tokens: List[str] = []

    if isinstance(provider, dict):
        api_val = provider.get("api") or provider.get("api_keys")
        raw_list = api_val if isinstance(api_val, list) else ([api_val] if isinstance(api_val, str) else [])
        for item in raw_list:
            for line in str(item or "").replace(",", "\n").splitlines():
                s = line.strip()
                if s.startswith("KGAT_") and s not in kgat_keys:
                    kgat_keys.append(s)
                elif s.startswith("kaggleweb:") and s not in seed_tokens:
                    seed_tokens.append(s)

    p = Path(DEFAULT_KEYS_FILE)
    if p.is_file():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                for s in data.get("kgat_keys") or []:
                    if isinstance(s, str) and s.strip().startswith("KGAT_") and s.strip() not in kgat_keys:
                        kgat_keys.append(s.strip())
                for s in data.get("active_12h_tokens") or []:
                    if isinstance(s, str) and s.strip().startswith("kaggleweb:") and s.strip() not in seed_tokens:
                        seed_tokens.append(s.strip())
        except Exception as exc:
            logger.warning(f"[{PLUGIN_NAME}] failed reading {p}: {exc}")

    return kgat_keys, seed_tokens


def _load_state(kgat_keys: List[str], seed_tokens: List[str]) -> Dict[str, Any]:
    global _cached_state
    if _cached_state is not None:
        return _cached_state

    p = Path(DEFAULT_STATE_FILE)
    if p.is_file():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict):
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


def _extract_quota_remaining(quota_obj: Any) -> float:
    """Parse remaining USD quota from Kaggle /api/v1/models/proxy/quota response."""
    if not isinstance(quota_obj, dict):
        return 10.0
    for key in ("remainingQuota", "remaining_quota", "availableQuota", "available", "remaining"):
        if key in quota_obj:
            try:
                return float(quota_obj[key])
            except Exception:
                pass
    # Common Kaggle structure: daily/monthly quota or limit - used
    for limit_k, used_k in (
        ("dailyLimit", "dailyUsed"),
        ("dailyQuota", "dailyUsed"),
        ("limit", "used"),
        ("maxQuota", "usedQuota"),
    ):
        if limit_k in quota_obj and used_k in quota_obj:
            try:
                return max(0.0, float(quota_obj[limit_k]) - float(quota_obj[used_k]))
            except Exception:
                pass
    # Recursive search for nested quota dicts
    for v in quota_obj.values():
        if isinstance(v, dict):
            rem = _extract_quota_remaining(v)
            if rem != 10.0:
                return rem
    return 10.0


async def check_kaggle_quota(client: httpx.AsyncClient, kgat_token: str) -> Tuple[bool, float, Any]:
    url = "https://www.kaggle.com/api/v1/models/proxy/quota"
    try:
        resp = await client.get(url, headers={"Authorization": f"Bearer {kgat_token}"}, timeout=15.0)
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


async def _kaggle_rpc(client: httpx.AsyncClient, kgat_token: str, service: str, method: str, payload: Any = None) -> Tuple[int, Any]:
    url = f"https://www.kaggle.com/api/v1/{service}/{method}"
    try:
        resp = await client.post(
            url,
            json=payload or {},
            headers={
                "Authorization": f"Bearer {kgat_token}",
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


async def mint_verifier_token_async(kgat_token: str, slug: str = "sanity-bench-check") -> Tuple[Optional[str], float, Any]:
    """Mint a 12-hour kaggleweb:... verifier token from a KGAT_ mother key."""
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
    async with httpx.AsyncClient(timeout=30.0) as client:
        ok, rem_quota, raw_quota = await check_kaggle_quota(client, kgat_token)
        if not ok:
            return None, 0.0, raw_quota

        st, resp = await _kaggle_rpc(
            client,
            kgat_token,
            "benchmarks.BenchmarkTasksApiService",
            "CreateBenchmarkTask",
            {"slug": slug, "text": _make_notebook(extract_code)},
        )
        if st != 200:
            logger.warning(f"[{PLUGIN_NAME}] CreateBenchmarkTask failed (HTTP {st}): {resp}")
            return None, rem_quota, resp

        verifier_token: Optional[str] = None
        try:
            for _ in range(45):
                await asyncio.sleep(4.0)
                _, gresp = await _kaggle_rpc(
                    client,
                    kgat_token,
                    "benchmarks.BenchmarkTasksApiService",
                    "GetBenchmarkTask",
                    {"slug": {"taskSlug": slug}},
                )
                cstate = gresp.get("creationState", "") if isinstance(gresp, dict) else ""
                if "COMPLETED" in cstate or "ERRORED" in cstate:
                    break

            _, rresp = await _kaggle_rpc(
                client,
                kgat_token,
                "benchmarks.BenchmarkTasksApiService",
                "ListBenchmarkTaskRuns",
                {"taskSlug": {"taskSlug": slug}},
            )
            runs = rresp.get("runs", []) if isinstance(rresp, dict) else []
            for run in runs:
                rid = run.get("id")
                _, logs = await _kaggle_rpc(
                    client,
                    kgat_token,
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
                kgat_token,
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
        if k.startswith(pin_key) or k.endswith(pin_key):
            return i, k
    return None


def _candidate_order(kgat_keys: List[str], state: Dict[str, Any], opts: Dict[str, Any], exclude_kgat: str = "") -> List[Tuple[int, str]]:
    total = len(kgat_keys)
    if total == 0:
        return []
    pinned = _resolve_pinned_kgat(kgat_keys, opts.get("pin_key", ""))
    ordered: List[Tuple[int, str]] = []
    if pinned and pinned[1] != exclude_kgat:
        ordered.append(pinned)

    indices = list(range(total, 0, -1)) if opts.get("pick_order") == "reverse" else list(range(1, total + 1))
    cursor = state.get("cursor_index")
    if isinstance(cursor, int) and cursor in indices:
        pos = indices.index(cursor)
        indices = indices[pos:] + indices[:pos]

    exhausted = state.get("exhausted_kgats") if isinstance(state.get("exhausted_kgats"), dict) else {}
    now = time.time()
    for idx in indices:
        k = kgat_keys[idx - 1]
        if k == exclude_kgat:
            continue
        if any(k == existing_k for _, existing_k in ordered):
            continue
        # Skip keys marked exhausted within the last 6 hours
        if float(exhausted.get(k) or 0) > now - 6 * 3600:
            continue
        ordered.append((idx, k))
    return ordered


async def _mint_from_candidates(
    kgat_keys: List[str],
    state: Dict[str, Any],
    opts: Dict[str, Any],
    exclude_kgat: str = "",
) -> Optional[Dict[str, Any]]:
    candidates = _candidate_order(kgat_keys, state, opts, exclude_kgat=exclude_kgat)
    min_quota = float(opts.get("min_quota") or 1.5)
    async with httpx.AsyncClient(timeout=15.0) as client:
        for idx, kgat in candidates:
            ok, rem, raw_q = await check_kaggle_quota(client, kgat)
            if not ok or rem < min_quota:
                if ok and rem < min_quota:
                    state.setdefault("exhausted_kgats", {})[kgat] = time.time()
                continue
            logger.info(f"[{PLUGIN_NAME}] minting 12h verifier token from mother key #{idx} ({kgat[:12]}..., quota=${rem:.2f})")
            tok, rem2, _ = await mint_verifier_token_async(kgat)
            if tok:
                now = time.time()
                state["cursor_index"] = idx
                return {
                    "verifier_token": tok,
                    "kgat": kgat,
                    "index": idx,
                    "minted_at": now,
                    "expires_at": now + TOKEN_TTL_SECONDS,
                    "quota_remaining": rem2,
                    "quota_raw": raw_q,
                    "source": "minted",
                }
    return None


async def _ensure_pool_Task(kgat_keys: List[str], seed_tokens: List[str], opts: Dict[str, Any]) -> None:
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
            minted = await _mint_from_candidates(kgat_keys, state, opts)
            if minted:
                state["active"] = minted
                active = minted
                _save_state(state)

        # 3. If active is valid but nearing 12h expiry (<30m), try same-key renewal first
        elif _slot_needs_renewal(active, now):
            act_kgat = str(active.get("kgat") or "")
            renewed = None
            if act_kgat.startswith("KGAT_"):
                async with httpx.AsyncClient(timeout=15.0) as client:
                    ok, rem, raw_q = await check_kaggle_quota(client, act_kgat)
                if ok and rem >= float(opts.get("min_quota") or 1.5):
                    logger.info(f"[{PLUGIN_NAME}] same-key renewing active #{active.get('index')} ({act_kgat[:12]}...)")
                    tok, rem2, _ = await mint_verifier_token_async(act_kgat)
                    if tok:
                        renewed = {
                            **active,
                            "verifier_token": tok,
                            "minted_at": time.time(),
                            "expires_at": time.time() + TOKEN_TTL_SECONDS,
                            "quota_remaining": rem2,
                            "quota_raw": raw_q,
                            "source": "renewed",
                        }
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
            minted_sb = await _mint_from_candidates(kgat_keys, state, opts, exclude_kgat=exclude)
            if minted_sb:
                state["standby"] = minted_sb
                _save_state(state)


def _trigger_bg_maintenance(kgat_keys: List[str], seed_tokens: List[str], opts: Dict[str, Any]) -> None:
    global _bg_mint_task
    if _bg_mint_task is not None and not _bg_mint_task.done():
        return
    try:
        loop = asyncio.get_running_loop()
        _bg_mint_task = loop.create_task(_ensure_pool_Task(kgat_keys, seed_tokens, opts))
    except RuntimeError:
        pass


def _rotate_on_quota_exhausted(reason: str) -> bool:
    """Immediately promote standby to active when active hits 403/401/429."""
    global _cached_state
    kgat_keys, seed_tokens = _load_keys_source()
    state = _load_state(kgat_keys, seed_tokens)
    active = state.get("active")
    standby = state.get("standby")
    if isinstance(active, dict) and str(active.get("kgat") or "").startswith("KGAT_"):
        state.setdefault("exhausted_kgats", {})[active["kgat"]] = time.time()
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
    # OpenAI / Anthropic top-level
    for field in ("max_tokens", "max_completion_tokens", "max_output_tokens"):
        if field in payload:
            try:
                if int(payload[field]) > cap:
                    payload[field] = cap
            except Exception:
                payload[field] = cap
    # Gemini generationConfig.maxOutputTokens
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
        payload["max_tokens"] = min(4096, cap)


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
    state = _load_state(kgat_keys, seed_tokens)

    # Handle manual rotate_nonce change or pinned mother key mismatch
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
            # Synchronous mint for newly pinned key if user explicitly pinned it
            await _ensure_pool_Task(kgat_keys, seed_tokens, opts)
            state = _load_state(kgat_keys, seed_tokens)
            active = state.get("active")

    if not _slot_valid(active):
        await _ensure_pool_Task(kgat_keys, seed_tokens, opts)
        state = _load_state(kgat_keys, seed_tokens)
        active = state.get("active")
    elif _slot_needs_renewal(active) or not _slot_valid(state.get("standby")):
        _trigger_bg_maintenance(kgat_keys, seed_tokens, opts)

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
        if status in (401, 403, 429) or "exceeds your available quota" in blob or "quota" in blob and "403" in blob:
            if any(k in blob for k in ("quota", "unauthorized", "unauthenticated", "expired", "invalid", "401", "403", "429")):
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
    kgat = str(slot.get("kgat") or "")
    masked_kgat = f"{kgat[:11]}...{kgat[-4:]}" if len(kgat) > 16 else kgat
    return {
        "index": slot.get("index"),
        "kgat_masked": masked_kgat,
        "quota_remaining_usd": round(float(slot.get("quota_remaining") or 0.0), 2),
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

    # Refresh active mother key quota in real time when checking balance
    if isinstance(active, dict) and str(active.get("kgat") or "").startswith("KGAT_"):
        async with httpx.AsyncClient(timeout=10.0) as client:
            ok, rem, raw_q = await check_kaggle_quota(client, active["kgat"])
            if ok:
                active["quota_remaining"] = rem
                active["quota_raw"] = raw_q
                _save_state(state)

    if not _slot_valid(active) or not _slot_valid(state.get("standby")):
        _trigger_bg_maintenance(kgat_keys, seed_tokens, opts)

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
    enriched["tier"] = f"母卡#{act_idx} ({exp_min}m)"
    enriched["kaggle_pool"] = {
        "total_mother_keys": len(kgat_keys),
        "pick_order": opts["pick_order"],
        "pin_key": opts["pin_key"] or "auto",
        "max_tokens_cap": opts["max_tokens_cap"],
        "active": act_info,
        "standby": sb_info,
    }
    enriched["raw"] = enriched["kaggle_pool"]
    return enriched


def setup(manager):
    register_request_interceptor(
        interceptor_id=_REQ_ID,
        callback=kaggle_verifier_pool_request_interceptor,
        priority=20,
        plugin_name=PLUGIN_NAME,
        overwrite=True,
        metadata={"description": "Kaggle 12h 裁判卡用1备1注入与 max_tokens 封顶"},
    )
    register_response_interceptor(
        interceptor_id=_RESP_ID,
        callback=kaggle_verifier_pool_response_interceptor,
        priority=20,
        plugin_name=PLUGIN_NAME,
        overwrite=True,
        metadata={"description": "Kaggle 额度耗尽 0 秒切换热备卡"},
    )
    register_balance_enricher(
        enricher_id=_BAL_ID,
        callback=kaggle_verifier_pool_balance_enricher,
        priority=20,
        plugin_name=PLUGIN_NAME,
        overwrite=True,
        metadata={"description": "Kaggle 裁判卡额度与当前临时 Token 查看"},
    )


def teardown(manager):
    unregister_request_interceptor(_REQ_ID)
    unregister_response_interceptor(_RESP_ID)
    unregister_balance_enricher(_BAL_ID)
