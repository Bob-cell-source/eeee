# image_parse_tool.py
from __future__ import annotations
import base64
from typing import Any, Dict, Callable, Awaitable, Optional
import httpx

async def _fetch_bytes(url: str, timeout_s: int = 20) -> bytes:
    async with httpx.AsyncClient(timeout=timeout_s) as client:
        r = await client.get(url); r.raise_for_status()
        return r.content

def _to_data_url(image_bytes: bytes, mime: str = "image/png") -> str:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64}"

class ImageParseClient:
    def __init__(self, generate_vision: Callable[[str, str], Awaitable[str]]):
        self._gen = generate_vision

    async def parse(
        self,
        question: str,
        image_url: Optional[str] = None,
        image_path: Optional[str] = None,
        image_base64: Optional[str] = None,
    ) -> Dict[str, Any]:
        if sum(x is not None for x in [image_url, image_path, image_base64]) != 1:
            return {"error": "image_url/image_path/image_base64 必须且只能提供一个"}

        if image_url:
            b = await _fetch_bytes(image_url)
            data_url = _to_data_url(b)
        elif image_path:
            with open(image_path, "rb") as f:
                data_url = _to_data_url(f.read())
        else:
            assert image_base64 is not None, "image_base64 should be set here"
            try:
                b = base64.b64decode(image_base64)
            except Exception:
                return {"error": "image_base64 解码失败"}
            data_url = _to_data_url(b)

        text = await self._gen(question, data_url)
        return {"text": text}


# 单例模式
_image_parser_singleton: Optional[ImageParseClient] = None

async def _default_generate_vision(question: str, data_url: str) -> str:
    """默认的VLM实现（使用qwen-vl-max或其他多模态模型）"""
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage
        import os
        from ..config import get_settings
        
        settings = get_settings()
        llm = ChatOpenAI(
            model="qwen3-vl-flash",  # 或其他支持视觉的模型
            temperature=0,
            base_url=os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            api_key=os.getenv("LLM_API_KEY"), # type: ignore
        )
        
        message = HumanMessage(
            content=[
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]
        )
        
        response = await llm.ainvoke([message])
        return response.content if isinstance(response.content, str) else str(response.content)
    except Exception as e:
        return f"图片解析失败: {e}"

def get_image_parser() -> ImageParseClient:
    """获取图片解析器单例"""
    global _image_parser_singleton
    if _image_parser_singleton is None:
        _image_parser_singleton = ImageParseClient(_default_generate_vision)
    return _image_parser_singleton
