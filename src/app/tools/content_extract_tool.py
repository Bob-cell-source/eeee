# content_extract_tool.py
from __future__ import annotations
import json, re
from typing import Any, Dict, List, Callable, Awaitable, Optional
from ..config import get_settings

def _split_paragraphs(text: str) -> List[str]:
    """Split text into paragraphs by double newlines"""
    parts = re.split(r"\n\s*\n+", text)
    return [p.strip() for p in parts if p.strip()]

def _select_top_paragraphs(question: str, paragraphs: List[str], top_k: Optional[int] = None) -> List[str]:
    """Select most relevant paragraphs based on keyword matching"""
    if top_k is None:
        top_k = get_settings().content_top_paragraphs
    q_terms = [t for t in re.split(r"\W+", question.lower()) if t and len(t) >= 2]
    if not q_terms:
        return paragraphs[:top_k]
    scored = []
    for p in paragraphs:
        low = p.lower()
        score = sum(low.count(t) for t in q_terms)
        if score > 0:
            scored.append((score, p))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored[:top_k]] if scored else paragraphs[:top_k]

class ContentExtractClient:
    def __init__(self, generate: Callable[[str], Awaitable[str]]):
        self._generate = generate

    async def extract(self, question: str, document_text: str, max_quotes: Optional[int] = None) -> Dict[str, Any]:
        """Extract key points and evidence from document"""
        if not question or not question.strip():
            return {"error": "Question cannot be empty"}
        if not document_text or not document_text.strip():
            return {"error": "Document text cannot be empty"}
        
        if max_quotes is None:
            max_quotes = get_settings().content_max_quotes
        paras = _split_paragraphs(document_text)
        picked = _select_top_paragraphs(question, paras, top_k=12)
        context = "\n\n".join([f"[P{i+1}] {p}" for i, p in enumerate(picked)])

        prompt = f"""
你是一个研究助理。请从给定材料中抽取与问题最相关的要点与“可引用证据”。
要求：
- key_points：简洁中文（<=10条）
- evidence_quotes：最多 {max_quotes} 条，必须是原文片段（quote），并说明 why_relevant
- location_hint：来自哪个段落标记 P1/P2...

输出必须是严格 JSON：
{{
  "key_points": ["..."],
  "evidence_quotes": [
    {{"quote": "...", "why_relevant": "...", "location_hint": "P3"}}
  ]
}}

问题：
{question}

材料：
{context}
""".strip()

        raw = (await self._generate(prompt)).strip()
        try:
            return json.loads(raw)
        except Exception:
            m = re.search(r"\{.*\}", raw, flags=re.S)
            if m:
                try:
                    return json.loads(m.group(0))
                except Exception:
                    pass
            return {"error": "LLM output is not valid JSON", "raw": raw[:2000]}
