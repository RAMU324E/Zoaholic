import asyncio
import base64
import ipaddress
import mimetypes
import socket
import urllib.parse
from typing import Optional

import httpx
from fastapi import HTTPException


_shared_fetch_client: Optional[httpx.AsyncClient] = None
_shared_fetch_client_lock = asyncio.Lock()
# 响应体上限：防止恶意 URL 返回超大内容耗尽网关内存（图片/文件类输入远低于此值）。
_MAX_FETCH_BYTES = 32 * 1024 * 1024


def parse_data_uri(data_uri: str) -> tuple[str, str]:
    if not data_uri.startswith("data:"):
        raise ValueError("Invalid data URI")
    header, data = data_uri.split(",", 1)
    mime_type = header[5:].split(";")[0]
    return mime_type, data


def build_data_uri(mime_type: str, base64_data: str) -> str:
    return f"data:{mime_type};base64,{base64_data}"


def split_data_uri_prefix_and_data(data_uri_or_b64: str, default_mime_type: str = "image/png") -> tuple[str, str]:
    if isinstance(data_uri_or_b64, str) and data_uri_or_b64.startswith("data:") and "," in data_uri_or_b64:
        header, data = data_uri_or_b64.split(",", 1)
        return header + ",", data
    return f"data:{default_mime_type};base64,", data_uri_or_b64


def extract_base64_data(data_uri_or_b64: str) -> str:
    if isinstance(data_uri_or_b64, str) and data_uri_or_b64.startswith("data:"):
        _, data = parse_data_uri(data_uri_or_b64)
        return data
    return data_uri_or_b64


def guess_mime_type(filename: str = None, default="application/octet-stream") -> str:
    if filename:
        mime_type, _ = mimetypes.guess_type(filename)
        if mime_type:
            return mime_type
    return default


def _encode_bytes_to_base64_text(content: bytes) -> str:
    return base64.b64encode(content).decode("utf-8")


async def _get_shared_fetch_client() -> httpx.AsyncClient:
    global _shared_fetch_client
    if _shared_fetch_client is not None:
        return _shared_fetch_client

    async with _shared_fetch_client_lock:
        if _shared_fetch_client is not None:
            return _shared_fetch_client

        transport = httpx.AsyncHTTPTransport(http2=True, verify=False, retries=1)
        limits = httpx.Limits(max_connections=50, max_keepalive_connections=20)
        timeout = httpx.Timeout(connect=15.0, read=30.0, write=30.0, pool=10.0)
        _shared_fetch_client = httpx.AsyncClient(
            transport=transport,
            limits=limits,
            timeout=timeout,
            follow_redirects=False,
        )
        return _shared_fetch_client


def _validate_safe_public_url(url: str) -> None:
    """校验 URL 仅指向公网 HTTP(S) 地址，防止 SSRF 探测本机/局域网/容器内网/云元数据。"""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("Only public http/https URLs are allowed")
    host = parsed.hostname.strip().lower()
    if host in ("localhost", "localhost.localdomain") or host.endswith(".local") or host.endswith(".internal"):
        raise ValueError("Internal hostnames are not allowed")
    infos = socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80), proto=socket.IPPROTO_TCP)
    if not infos:
        raise ValueError("Unable to resolve host")
    for info in infos:
        # 修改原因：IPv4-mapped IPv6（如 ::ffff:127.0.0.1）的 is_private 等属性按 IPv6 判断，可能绕过内网拦截。
        # 修改方式：先剥离 scope id 并还原 ipv4_mapped，再统一用 is_global 做公网判定。
        # 目的：阻止通过映射地址或其它非公网段访问本机/内网/元数据。
        ip_str = str(info[4][0]).split("%", 1)[0]
        ip_obj = ipaddress.ip_address(ip_str)
        if getattr(ip_obj, "ipv4_mapped", None) is not None:
            ip_obj = ip_obj.ipv4_mapped
        if (
            not ip_obj.is_global
            or ip_obj.is_loopback
            or ip_obj.is_private
            or ip_obj.is_link_local
            or ip_obj.is_multicast
            or ip_obj.is_reserved
            or ip_obj.is_unspecified
        ):
            raise ValueError("Access to private or internal IP addresses is forbidden")


async def close_shared_fetch_client() -> None:
    global _shared_fetch_client
    async with _shared_fetch_client_lock:
        if _shared_fetch_client is not None:
            await _shared_fetch_client.aclose()
            _shared_fetch_client = None


async def fetch_url_content(url: str) -> tuple[bytes, str]:
    client = await _get_shared_fetch_client()
    try:
        current_url = url
        for _ in range(4):
            await asyncio.to_thread(_validate_safe_public_url, current_url)
            # 修改原因：一次性读取 response.content 会被超大响应体耗尽内存。
            # 修改方式：流式读取并累计字节数，超过 _MAX_FETCH_BYTES 立即中止。
            # 目的：在保持原有重定向校验的同时限制单次 URL 拉取的响应体大小。
            async with client.stream("GET", current_url) as response:
                if response.is_redirect and response.headers.get("location"):
                    current_url = urllib.parse.urljoin(current_url, response.headers["location"])
                    continue
                response.raise_for_status()
                content_type = response.headers.get("Content-Type", "").split(";")[0].strip()
                content_length = response.headers.get("Content-Length")
                if content_length:
                    try:
                        declared_size = int(content_length)
                    except ValueError:
                        declared_size = 0
                    if declared_size > _MAX_FETCH_BYTES:
                        raise ValueError("Response body is too large")
                body = bytearray()
                async for chunk in response.aiter_bytes():
                    body.extend(chunk)
                    if len(body) > _MAX_FETCH_BYTES:
                        raise ValueError("Response body is too large")
                return bytes(body), content_type
        raise ValueError("Too many redirects")
    except Exception:
        raise HTTPException(status_code=400, detail="Failed to fetch file from the provided URL.")


async def get_base64_file(file_url_or_data: str) -> tuple[str, str]:
    """返回 (base64_data_with_prefix, mime_type)"""
    if file_url_or_data.startswith("http://") or file_url_or_data.startswith("https://"):
        content, content_type = await fetch_url_content(file_url_or_data)
        b64_data = await asyncio.to_thread(_encode_bytes_to_base64_text, content)
        if not content_type:
            content_type = guess_mime_type(file_url_or_data)
        return build_data_uri(content_type, b64_data), content_type
    elif file_url_or_data.startswith("data:"):
        mime_type, _ = parse_data_uri(file_url_or_data)
        return file_url_or_data, mime_type
    else:
        return file_url_or_data, "application/octet-stream"
