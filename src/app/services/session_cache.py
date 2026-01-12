"""
Session Cache 管理模块

负责管理单次研究会话中的原文存储（RawDoc）和结构化证据笔记（ResearchNote）
"""
import logging
from typing import Dict, List, Optional, Any, TypedDict
from datetime import datetime
from dataclasses import dataclass, field, asdict
from ..utils.chunking import DocChunk

logger = logging.getLogger(__name__)


class EvidenceQuote(TypedDict):
    """证据引用（来自 content_extract）"""
    quote: str  # 原文引用
    why_relevant: str  # 相关性说明
    location_hint: Optional[str]  # 位置提示（如 "P3" / "Introduction" / "chunk_5"）


class ResearchNote(TypedDict, total=False):
    """
    结构化证据笔记（取代原 EvidencePack）
    
    这是从 RawDoc 提取后的高层摘要，用于 workspace 和 write_answer
    """
    pack_id: str
    source: str  # "arxiv"|"github"|"web"|"local_file"
    ref: str  # arxiv_id / repo full_name / url / file_path
    url: Optional[str]
    title: Optional[str]
    snippet: str  # <= 300 chars（总体摘要）
    key_points: List[str]  # 来自 content_extract
    evidence_quotes: List[EvidenceQuote]  # 来自 content_extract
    raw_pointer: str  # doc_id，用于按需取原文
    confidence: str  # "high"|"medium"|"low"
    relevance: str  # one sentence
    # 多模态
    media_type: str  # "text"|"image"
    image_url: Optional[str]
    # 元信息
    fetched_at: str  # ISO timestamp
    target_question: Optional[str]  # 对应的 open_question


# 辅助函数：安全访问 ResearchNote 可选字段（处理 total=False）
def get_note_field(note: ResearchNote, field: str, default: Any = None) -> Any:
    """安全获取 ResearchNote 字段（total=False 导致所有字段都是可选）"""
    return note.get(field, default)  # type: ignore


def safe_note_pack_id(note: ResearchNote) -> str:
    """安全获取 pack_id"""
    return get_note_field(note, 'pack_id', f"note_{datetime.now().timestamp()}")


def safe_note_source(note: ResearchNote) -> str:
    """安全获取 source"""
    return get_note_field(note, 'source', 'unknown')


def safe_note_evidence_quotes(note: ResearchNote) -> List[EvidenceQuote]:
    """安全获取 evidence_quotes"""
    return get_note_field(note, 'evidence_quotes', [])


@dataclass
class RawDoc:
    """
    原文文档（Session Cache 核心单元）
    
    存储完整原文和分块索引，不直接进入 LLM prompt
    """
    doc_id: str
    source: str  # "arxiv"|"github"|"web"|"local_file"|"local_kb"
    ref: str  # arxiv_id / repo / url / file_path
    url: Optional[str]
    title: Optional[str]
    fetched_at: str  # ISO timestamp
    text: str  # 完整原文（可能很长）
    chunks: List[DocChunk]  # 分块索引（用于 NEED_RAW 精准定位）
    metadata: Dict[str, Any] = field(default_factory=dict)
    images: List[Dict[str, Any]] = field(default_factory=list)  # 图片信息（如有）
    origin: str = "web"  # "local" 或 "web"（用于 Milvus 写入时区分权重）
    degraded: bool = False  # 标记是否降级（拉全文失败时）
    
    def to_dict(self) -> Dict[str, Any]:
        """转为字典（序列化）"""
        data = asdict(self)
        # chunks 转为简单字典
        data['chunks'] = [
            {
                'chunk_id': c.chunk_id,
                'start_char': c.start_char,
                'end_char': c.end_char,
                'text': c.text,
                'page_hint': c.page_hint,
                'section_hint': c.section_hint,
                'text_preview': c.text_preview,
                'element_type': c.element_type
            }
            for c in self.chunks
        ]
        return data
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'RawDoc':
        """从字典恢复（反序列化）"""
        from ..utils.chunking import DocChunk as DC
        
        # 恢复 chunks
        chunks = []
        for c in data.get('chunks', []):
            try:
                chunks.append(DC(
                    chunk_id=c.get('chunk_id', ''),
                    start_char=c.get('start_char', 0),
                    end_char=c.get('end_char', 0),
                    text=c.get('text', ''),
                    page_hint=c.get('page_hint'),
                    section_hint=c.get('section_hint'),
                    element_type=c.get('element_type'),
                    text_preview=c.get('text_preview')
                ))
            except Exception as e:
                logger.warning(f"Failed to restore chunk: {e}")
                continue
        
        return RawDoc(
            doc_id=data.get('doc_id', ''),
            source=data.get('source', 'web'),
            ref=data.get('ref', ''),
            url=data.get('url'),
            title=data.get('title'),
            fetched_at=data.get('fetched_at', datetime.now().isoformat()),
            text=data.get('text', ''),
            chunks=chunks,
            metadata=data.get('metadata', {}),
            images=data.get('images', []),
            origin=data.get('origin', 'web'),
            degraded=data.get('degraded', False)
        )
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[DocChunk]:
        """根据 chunk_id 获取分块"""
        for chunk in self.chunks:
            if chunk.chunk_id == chunk_id:
                return chunk
        return None
    
    def get_chunks_by_page(self, page: int) -> List[DocChunk]:
        """根据页码获取分块"""
        return [c for c in self.chunks if c.page_hint == page]
    
    def get_chunks_by_location_hint(self, hint: str) -> List[DocChunk]:
        """根据 location_hint 获取分块（支持 P3 / section name）"""
        # 尝试解析页码
        import re
        page_match = re.match(r'[Pp](\d+)', hint)
        if page_match:
            page = int(page_match.group(1))
            return self.get_chunks_by_page(page)
        
        # 尝试匹配 section_hint
        matching_chunks = []
        for chunk in self.chunks:
            if chunk.section_hint and hint.lower() in chunk.section_hint.lower():
                matching_chunks.append(chunk)
        
        return matching_chunks
    
    def search_chunks(self, query: str, top_k: int = 3) -> List[DocChunk]:
        """
        在 chunks 中搜索（简单关键词匹配）
        
        可选：实现轻量向量检索（懒加载 embedding）
        """
        query_lower = query.lower()
        scored_chunks = []
        
        for chunk in self.chunks:
            # 简单 TF 评分
            score = chunk.text.lower().count(query_lower)
            if score > 0:
                scored_chunks.append((score, chunk))
        
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in scored_chunks[:top_k]]


class SessionCache:
    """
    Session Cache 管理器
    
    管理单次研究会话的原文存储和结构化笔记
    """
    
    def __init__(self):
        self.docs: Dict[str, RawDoc] = {}  # doc_id -> RawDoc
        self.notes: Dict[str, ResearchNote] = {}  # pack_id -> ResearchNote
    
    def add_doc(self, doc: RawDoc):
        """添加原文文档"""
        self.docs[doc.doc_id] = doc
        logger.debug(f"Session Cache: 添加 doc {doc.doc_id}, {len(doc.text)} chars, {len(doc.chunks)} chunks")
    
    def get_doc(self, doc_id: str) -> Optional[RawDoc]:
        """获取原文文档"""
        return self.docs.get(doc_id)
    
    def add_note(self, note: ResearchNote):
        """添加结构化笔记"""
        pack_id = note.get("pack_id")
        if not isinstance(pack_id, str) or not pack_id:
            raise ValueError("note 缺少 pack_id")
        self.notes[pack_id] = note
        logger.debug(f"Session Cache: 添加 note {pack_id}")

    def get_note(self, pack_id: str) -> Optional[ResearchNote]:
        """获取结构化笔记"""
        return self.notes.get(pack_id)

    def get_all_notes(self) -> List[ResearchNote]:
        """获取所有笔记"""
        return list(self.notes.values())

    def get_notes_by_source(self, source: str) -> List[ResearchNote]:
        """按来源筛选笔记"""
        return [n for n in self.notes.values() if n.get("source") == source]

    
    def extract_chunks_for_need_raw(
        self,
        doc_id: str,
        location_hints: Optional[List[str]] = None,
        pack_ids: Optional[List[str]] = None,
        max_chunks: int = 3,
        max_chars: int = 3000
    ) -> List[Dict[str, Any]]:
        """
        为 NEED_RAW 请求提取原文片段
        
        Args:
            doc_id: 文档ID
            location_hints: 位置提示列表（如 ["P3", "Introduction"]）
            pack_ids: 笔记ID列表（用于定位对应 evidence_quotes）
            max_chunks: 最多返回分块数
            max_chars: 总字符数限制
        
        Returns:
            List[{chunk_id, text, location, source}]
        """
        doc = self.get_doc(doc_id)
        if not doc:
            logger.warning(f"NEED_RAW: doc_id {doc_id} not found in session cache")
            return []
        
        candidate_chunks: List[DocChunk] = []
        
        # 策略1: 根据 pack_id 找 evidence_quotes 的 location_hint
        if pack_ids:
            for pack_id in pack_ids:
                note = self.get_note(pack_id)
                if note and note.get('evidence_quotes'):
                    evidence_quotes = note.get("evidence_quotes") or []
                    for eq in evidence_quotes:
                        if not isinstance(eq, dict):
                            continue
                        hint = eq.get("location_hint")
                        if hint:
                            chunks = doc.get_chunks_by_location_hint(hint)
                            candidate_chunks.extend(chunks)
        
        # 策略2: 直接使用 location_hints
        if location_hints:
            for hint in location_hints:
                chunks = doc.get_chunks_by_location_hint(hint)
                candidate_chunks.extend(chunks)
        
        # 策略3: 如果都没有，返回前 N 个 chunks
        if not candidate_chunks:
            candidate_chunks = doc.chunks[:max_chunks]
        
        # 去重并限制数量
        seen_ids = set()
        unique_chunks = []
        for chunk in candidate_chunks:
            if chunk.chunk_id not in seen_ids:
                seen_ids.add(chunk.chunk_id)
                unique_chunks.append(chunk)
        
        unique_chunks = unique_chunks[:max_chunks]
        
        # 限制总字符数
        result = []
        total_chars = 0
        for chunk in unique_chunks:
            if total_chars + len(chunk.text) > max_chars:
                break
            result.append({
                'chunk_id': chunk.chunk_id,
                'text': chunk.text,
                'location': f"P{chunk.page_hint}" if chunk.page_hint else chunk.section_hint or "unknown",
                'source': f"{doc.source}:{doc.ref}"
            })
            total_chars += len(chunk.text)
        
        logger.info(f"NEED_RAW: 返回 {len(result)} chunks, {total_chars} chars for doc {doc_id}")
        return result
    
    def to_state_dict(self) -> Dict[str, Any]:
        """
        转为可序列化的状态字典（用于持久化）
        """
        return {
            'docs': {doc_id: doc.to_dict() for doc_id, doc in self.docs.items()},
            'notes': {pack_id: note for pack_id, note in self.notes.items()}
        }
    
    @staticmethod
    def from_state_dict(data: Dict[str, Any]) -> 'SessionCache':
        """从 State 反序列化（支持错误恢复）"""
        cache = SessionCache()
        
        # 恢复 docs
        for doc_id, doc_dict in data.get('docs', {}).items():
            try:
                doc = RawDoc.from_dict(doc_dict)
                cache.docs[doc_id] = doc
            except Exception as e:
                logger.warning(f"Failed to restore doc {doc_id}: {e}")
                continue
        
        # 恢复 notes
        for pack_id, note in data.get('notes', {}).items():
            try:
                cache.notes[pack_id] = note  # TypedDict 可直接使用
            except Exception as e:
                logger.warning(f"Failed to restore note {pack_id}: {e}")
                continue
        
        logger.info(f"SessionCache restored: {len(cache.docs)} docs, {len(cache.notes)} notes")
        return cache
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        total_text_len = sum(len(doc.text) for doc in self.docs.values())
        total_chunks = sum(len(doc.chunks) for doc in self.docs.values())
        
        return {
            'total_docs': len(self.docs),
            'total_notes': len(self.notes),
            'total_text_chars': total_text_len,
            'total_chunks': total_chunks,
            'sources': list(set(doc.source for doc in self.docs.values()))
        }
