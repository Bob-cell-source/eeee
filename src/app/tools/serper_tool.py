from __future__ import annotations

import os
import re
import asyncio
import random
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
load_dotenv()
import httpx

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")

def _is_chinese(text: str) -> bool:
    return bool(_CJK_RE.search(text))


@dataclass
class SerperOrganicResult:
    title: str
    url: str
    snippet: str = ""
    date: Optional[str] = None
    position: Optional[int] = None


class SerperClient:
    """
    serper搜索web
    Env:
      - SERPER_API_KEY: 必需
      - SERPER_SEARCH_URL: 默认 https://google.serper.dev/search
      - SERPER_TIMEOUT: 可选, seconds
      - SERPER_CONCURRENCY: 可选, 并发数
      - SERPER_MAX_RETRIES: 可选, 最大重试次数
      - SERPER_BACKOFF_BASE: 可选, 退避基数秒
    """
    def __init__(
        self,
        api_key: Optional[str] = None,
        search_url: Optional[str] = None,
        timeout_s: Optional[float] = None,
        concurrency: Optional[int] = None,
        max_retries: Optional[int] = None,
        backoff_base: Optional[float] = None,
    ):
        self.api_key = api_key or os.getenv("SERPER_API_KEY") or os.getenv("SERPER_KEY_ID")
        self.search_url = search_url or os.getenv("SERPER_SEARCH_URL") or "https://google.serper.dev/search"
        self.timeout_s = timeout_s or float(os.getenv("SERPER_TIMEOUT", "20"))

        if not self.api_key:
            raise ValueError("没有 SERPER_API_KEY (or SERPER_KEY_ID). 在使用搜索前请设置环境变量")

        # ✅ 1) 复用 AsyncClient（连接池复用）
        self._client = httpx.AsyncClient(
            timeout=self.timeout_s,
            headers={
                "X-API-KEY": self.api_key,
                "Content-Type": "application/json",
            },
        )

        # （可选）实例级并发限流：同一个 SerperClient 全局共享
        self._concurrency = concurrency or int(os.getenv("SERPER_CONCURRENCY", "5"))
        self._sem = asyncio.Semaphore(max(1, self._concurrency))

        # ✅ 2) 重试退避参数
        self._max_retries = max_retries if max_retries is not None else int(os.getenv("SERPER_MAX_RETRIES", "3"))
        self._backoff_base = backoff_base if backoff_base is not None else float(os.getenv("SERPER_BACKOFF_BASE", "0.8"))

    async def aclose(self) -> None:
        """如果你是长期服务（FastAPI等），建议在 shutdown 时调用。"""
        await self._client.aclose()

    def _compute_backoff(self, attempt: int, retry_after: Optional[str] = None) -> float:
        # 优先尊重 Retry-After（常见于 429）
        if retry_after:
            try:
                return max(0.0, float(retry_after))
            except ValueError:
                pass

        # 指数退避 + 抖动 jitter
        base = self._backoff_base * (2 ** attempt)
        jitter = random.uniform(0, 0.25 * base)
        return base + jitter

    async def _post_with_retry(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        last_exc: Optional[BaseException] = None

        for attempt in range(self._max_retries + 1):
            try:
                resp = await self._client.post(self.search_url, json=payload)

                # 需要重试的状态码：429 + 常见 5xx
                if resp.status_code in (429, 500, 502, 503, 504):
                    retry_after = resp.headers.get("Retry-After")
                    wait_s = self._compute_backoff(attempt, retry_after=retry_after)

                    # 最后一次就不睡了，直接 raise
                    if attempt >= self._max_retries:
                        resp.raise_for_status()

                    await asyncio.sleep(wait_s)
                    continue

                resp.raise_for_status()
                return resp.json()

            except (httpx.TimeoutException, httpx.TransportError) as e:
                # 网络/超时类错误：退避重试
                last_exc = e
                if attempt >= self._max_retries:
                    raise
                await asyncio.sleep(self._compute_backoff(attempt))
            except httpx.HTTPStatusError as e:
                # 其他 4xx 一般不重试（比如 401/403/400）
                raise

        # 理论上走不到这里
        if last_exc:
            raise last_exc
        raise RuntimeError("Serper request failed unexpectedly")

    async def search_one(
        self,
        query: str,
        max_results: int = 10,
        country: Optional[str] = None,
        locale: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not query.strip():
            return {"query": query, "organic": [], "error": "empty query"}

        # 分别补齐，更直觉
        if country is None:
            country = "cn" if _is_chinese(query) else "us"
        if locale is None:
            locale = "zh-cn" if _is_chinese(query) else "en"

        max_results = max(1, min(int(max_results), 100))

        payload = {"q": query, "num": max_results, "gl": country, "hl": locale}

        data = await self._post_with_retry(payload)

        organic: List[SerperOrganicResult] = []
        for i, item in enumerate(data.get("organic", []) or []):
            organic.append(
                SerperOrganicResult(
                    title=item.get("title", "") or "",
                    url=item.get("link", "") or item.get("url", "") or "",
                    snippet=item.get("snippet", "") or "",
                    date=item.get("date"),
                    position=item.get("position", i + 1),
                )
            )

        return {
            "query": query,
            "gl": country,
            "hl": locale,
            "organic": [asdict(x) for x in organic[:max_results]],
            "answer_box": data.get("answerBox"),
            "knowledge_graph": data.get("knowledgeGraph"),
        }

    async def search_many(
        self,
        queries: List[str],
        max_results: int = 10,
        country: Optional[str] = None,
        locale: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        # ✅ 这里改成：内部吞异常，返回 dict（避免你之前 Pylance 的 BaseException 联合类型报错）
        async def _run(q: str) -> Dict[str, Any]:
            async with self._sem:
                try:
                    return await self.search_one(
                        query=q,
                        max_results=max_results,
                        country=country,
                        locale=locale,
                    )
                except Exception as e:
                    return {"query": q, "error": repr(e)}

        return await asyncio.gather(*[_run(q) for q in queries])
_serper_singleton:Optional[SerperClient]=None
def get_serper_tool()->SerperClient:
    global _serper_singleton
    if _serper_singleton is None:
        _serper_singleton = SerperClient()
    return _serper_singleton