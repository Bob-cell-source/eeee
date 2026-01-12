"""
文本分块模块 - 支持按结构/段落/固定长度分块

提供可插拔的 chunker backend：
- Backend A（默认）：规则分块（按段落/标题/长度）
- Backend B（可选）：Unstructured（如可用）提供结构化 elements
"""
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DocChunk:
    """文档分块单元"""
    chunk_id: str
    start_char: int
    end_char: int
    text: str
    page_hint: Optional[int] = None  # PDF 页码（如可得）
    section_hint: Optional[str] = None  # 章节标题（如可得）
    element_type: Optional[str] = None  # Title/Paragraph/Table/ListItem（如 Unstructured 可用）
    text_preview: Optional[str] = None  # 前120字预览


def create_chunk(
    chunk_id: str,
    start_char: int,
    end_char: int,
    text: str,
    page_hint: Optional[int] = None,
    section_hint: Optional[str] = None,
    element_type: Optional[str] = None
) -> DocChunk:
    """创建分块对象"""
    preview = text[:120] + "..." if len(text) > 120 else text
    return DocChunk(
        chunk_id=chunk_id,
        start_char=start_char,
        end_char=end_char,
        text=text,
        page_hint=page_hint,
        section_hint=section_hint,
        element_type=element_type,
        text_preview=preview
    )


class RuleBasedChunker:
    """规则分块器（Backend A - 默认）"""
    
    def __init__(
        self,
        chunk_size: int = 800,  # 中文约 800-1200 字
        chunk_overlap: int = 100,  # 10-20% overlap
        min_chunk_size: int = 100
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
    
    def split_text_to_chunks(
        self,
        text: str,
        doc_id: str,
        page_hints: Optional[Dict[int, Tuple[int, int]]] = None  # {page_num: (start_char, end_char)}
    ) -> List[DocChunk]:
        """
        分块策略：
        1. 优先按段落/标题边界切分
        2. 若段落过长则按长度切分（带 overlap）
        3. 记录 start_char/end_char 用于精准定位
        """
        if not text or len(text) < self.min_chunk_size:
            return []
        
        chunks: List[DocChunk] = []
        
        # 按段落分割（双换行或单换行+缩进）
        paragraphs = self._split_paragraphs(text)
        
        current_chunk = ""
        current_start = 0
        chunk_idx = 0
        
        for para in paragraphs:
            para_start = text.find(para, current_start)
            if para_start == -1:
                continue
            
            # 如果当前 chunk + 新段落不超过限制，合并
            if len(current_chunk) + len(para) <= self.chunk_size:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
                    current_start = para_start
            else:
                # 输出当前 chunk
                if current_chunk:
                    chunk_end = current_start + len(current_chunk)
                    page_hint = self._get_page_for_char(current_start, page_hints)
                    section_hint = self._extract_section_hint(current_chunk)
                    
                    chunk_id = f"{doc_id}_chunk_{chunk_idx}"
                    chunks.append(create_chunk(
                        chunk_id=chunk_id,
                        start_char=current_start,
                        end_char=chunk_end,
                        text=current_chunk,
                        page_hint=page_hint,
                        section_hint=section_hint
                    ))
                    chunk_idx += 1
                
                # 开始新 chunk（带 overlap）
                if len(para) > self.chunk_size:
                    # 段落本身过长，按固定长度切分
                    sub_chunks = self._split_long_paragraph(para, para_start, doc_id, chunk_idx)
                    chunks.extend(sub_chunks)
                    chunk_idx += len(sub_chunks)
                    current_chunk = ""
                    current_start = para_start + len(para)
                else:
                    current_chunk = para
                    current_start = para_start
        
        # 处理最后剩余的 chunk
        if current_chunk:
            chunk_end = current_start + len(current_chunk)
            page_hint = self._get_page_for_char(current_start, page_hints)
            section_hint = self._extract_section_hint(current_chunk)
            
            chunk_id = f"{doc_id}_chunk_{chunk_idx}"
            chunks.append(create_chunk(
                chunk_id=chunk_id,
                start_char=current_start,
                end_char=chunk_end,
                text=current_chunk,
                page_hint=page_hint,
                section_hint=section_hint
            ))
        
        logger.info(f"分块完成：{doc_id} -> {len(chunks)} chunks")
        return chunks
    
    def _split_paragraphs(self, text: str) -> List[str]:
        """按段落分割"""
        # 按双换行或单换行+明显缩进分割
        paragraphs = re.split(r'\n\s*\n|(?<=\n)(?=\s{4,})', text)
        return [p.strip() for p in paragraphs if p.strip()]
    
    def _split_long_paragraph(
        self,
        para: str,
        para_start: int,
        doc_id: str,
        start_idx: int
    ) -> List[DocChunk]:
        """切分过长段落"""
        chunks = []
        sentences = re.split(r'([。！？\.\!\?])', para)
        
        # 重新拼接句子（保留标点）
        reconstructed = []
        for i in range(0, len(sentences) - 1, 2):
            if i + 1 < len(sentences):
                reconstructed.append(sentences[i] + sentences[i + 1])
            else:
                reconstructed.append(sentences[i])
        
        current = ""
        char_offset = 0
        
        for sent in reconstructed:
            if len(current) + len(sent) <= self.chunk_size:
                current += sent
            else:
                if current:
                    chunk_id = f"{doc_id}_chunk_{start_idx + len(chunks)}"
                    chunks.append(create_chunk(
                        chunk_id=chunk_id,
                        start_char=para_start + char_offset,
                        end_char=para_start + char_offset + len(current),
                        text=current
                    ))
                    char_offset += len(current)
                current = sent
        
        if current:
            chunk_id = f"{doc_id}_chunk_{start_idx + len(chunks)}"
            chunks.append(create_chunk(
                chunk_id=chunk_id,
                start_char=para_start + char_offset,
                end_char=para_start + char_offset + len(current),
                text=current
            ))
        
        return chunks
    
    def _get_page_for_char(
        self,
        char_pos: int,
        page_hints: Optional[Dict[int, Tuple[int, int]]]
    ) -> Optional[int]:
        """根据字符位置查找页码"""
        if not page_hints:
            return None
        
        for page_num, (start, end) in page_hints.items():
            if start <= char_pos < end:
                return page_num
        return None
    
    def _extract_section_hint(self, text: str) -> Optional[str]:
        """提取章节标题（简单启发式）"""
        lines = text.split('\n')
        for line in lines[:3]:  # 只看前3行
            line = line.strip()
            # 匹配标题模式：短行、大写开头、或带编号
            if len(line) < 60 and (
                line[0].isupper() or
                re.match(r'^[0-9一二三四五六七八九十]+[\.、]', line) or
                re.match(r'^#+\s', line)  # Markdown 标题
            ):
                return line[:50]
        return None


class UnstructuredChunker:
    """基于 Unstructured 的结构化分块器（Backend B - 可选）"""
    
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._check_availability()
    
    def _check_availability(self):
        """检查 Unstructured 是否可用"""
        try:
            import unstructured
            self.available = True
            logger.info("Unstructured backend available")
        except ImportError:
            self.available = False
            logger.warning("Unstructured not installed, fallback to RuleBasedChunker")
    
    def split_text_to_chunks(
        self,
        text: str,
        doc_id: str,
        page_hints: Optional[Dict[int, Tuple[int, int]]] = None
    ) -> List[DocChunk]:
        """使用 Unstructured 分块"""
        if not self.available:
            # Fallback to rule-based
            return RuleBasedChunker(self.chunk_size, self.chunk_overlap).split_text_to_chunks(
                text, doc_id, page_hints
            )
        
        # TODO: 实现 Unstructured 集成
        # from unstructured.partition.text import partition_text
        # elements = partition_text(text=text)
        # 按 element 类型组织 chunks
        
        logger.warning("Unstructured integration not yet implemented, using fallback")
        return RuleBasedChunker(self.chunk_size, self.chunk_overlap).split_text_to_chunks(
            text, doc_id, page_hints
        )


def get_default_chunker() -> RuleBasedChunker:
    """获取默认分块器"""
    return RuleBasedChunker(chunk_size=800, chunk_overlap=100)


def get_chunker(backend: str = "rule") -> RuleBasedChunker:
    """
    获取指定 backend 的分块器
    
    Args:
        backend: "rule"（默认） 或 "unstructured"
    """
    if backend == "unstructured":
        chunker = UnstructuredChunker()
        if not chunker.available:
            logger.warning("Unstructured backend unavailable, using rule-based")
            return RuleBasedChunker()
        return chunker  # type: ignore
    else:
        return RuleBasedChunker()
