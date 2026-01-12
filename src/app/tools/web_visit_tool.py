# web_visit_tool.py
from __future__ import annotations
import os, re, time
from typing import Any, Dict, Optional
import httpx
from .utils import normalize_url
from ..config import get_settings

def _html_to_text_basic(html: str) -> Dict[str, Any]:
    m = re.search(r"(?is)<title[^>]*>(.*?)</title>", html)
    title = re.sub(r"\s+", " ", m.group(1)).strip() if m else None
    html = re.sub(r"(?is)<(script|style|noscript).*?>.*?</\1>", " ", html)
    text = re.sub(r"(?s)<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text).strip()
    return {"title": title, "text": text}

class WebVisitClient:
    def __init__(self, timeout_s: Optional[float] = None, max_chars: Optional[int] = None):
        settings = get_settings()
        self.timeout_s = timeout_s if timeout_s is not None else settings.web_visit_timeout
        self.max_chars = max_chars if max_chars is not None else settings.web_visit_max_chars
        self._client = httpx.AsyncClient(timeout=self.timeout_s, follow_redirects=True)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def visit(self, url: str, max_chars: Optional[int] = None, use_jina: bool = True) -> Dict[str, Any]:
        if max_chars is None:
            max_chars = self.max_chars
        url = normalize_url(url)
        if not url:
            return {"error": "empty url"}

        fetched_at = int(time.time())
        headers = {}
        jina_key = os.getenv("JINA_API_KEY")
        if jina_key:
            headers["Authorization"] = f"Bearer {jina_key}"

        # 1) Jina
        if use_jina:
            try:
                jina_url = f"https://r.jina.ai/{url}"
                r = await self._client.get(jina_url, headers=headers)
                r.raise_for_status()
                text = r.text.strip()
                if len(text) > max_chars:
                    text = text[:max_chars]
                return {"url": url, "title": None, "text": text, "fetched_at": fetched_at, "source": "jina"}
            except Exception:
                pass

        # 2) direct html
        try:
            r = await self._client.get(url, headers=headers)
            r.raise_for_status()
            parsed = _html_to_text_basic(r.text)
            text = parsed["text"]
            if len(text) > max_chars:
                text = text[:max_chars]
            return {"url": url, "title": parsed.get("title"), "text": text, "fetched_at": fetched_at, "source": "direct_html"}
        except Exception as e:
            return {"url": url, "error": repr(e)}

_web_visit_singleton: Optional[WebVisitClient] = None
def get_web_visit_tool() -> WebVisitClient:
    global _web_visit_singleton
    if _web_visit_singleton is None:
        _web_visit_singleton = WebVisitClient()
    return _web_visit_singleton
