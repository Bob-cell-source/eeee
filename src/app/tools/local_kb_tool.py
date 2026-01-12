"""
本地知识库工具（占位接口）

未来用于集成本地向量数据库、文件索引等
"""
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class LocalKBHit:
    """本地 KB 命中结果"""
    def __init__(
        self,
        kb_id: str,
        doc_id: str,
        title: str,
        snippet: str,
        score: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.kb_id = kb_id
        self.doc_id = doc_id
        self.title = title
        self.snippet = snippet
        self.score = score
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "kb_id": self.kb_id,
            "doc_id": self.doc_id,
            "title": self.title,
            "snippet": self.snippet,
            "score": self.score,
            "metadata": self.metadata
        }


async def local_kb_search(
    query: str,
    kb_id: Optional[str] = None,
    top_k: int = 5,
    filters: Optional[Dict[str, Any]] = None
) -> List[LocalKBHit]:
    """
    搜索本地知识库（占位）
    
    Args:
        query: 查询文本
        kb_id: 知识库 ID（None 表示搜索所有）
        top_k: 返回结果数
        filters: 额外过滤条件
    
    Returns:
        LocalKBHit 列表
    
    Raises:
        NotImplementedError: 本功能尚未实现
    """
    logger.warning("local_kb_search called but not implemented yet")
    raise NotImplementedError(
        "Local KB search is not implemented. "
        "To enable, integrate with local vector DB (e.g., ChromaDB, FAISS) "
        "and implement the search logic here."
    )


async def local_kb_fetch(
    doc_id: str,
    kb_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    获取本地知识库文档全文（占位）
    
    Args:
        doc_id: 文档 ID
        kb_id: 知识库 ID
    
    Returns:
        文档数据字典 {doc_id, title, text, metadata}
    
    Raises:
        NotImplementedError: 本功能尚未实现
    """
    logger.warning("local_kb_fetch called but not implemented yet")
    raise NotImplementedError(
        "Local KB fetch is not implemented. "
        "To enable, implement document retrieval from local storage."
    )


async def local_kb_index_document(
    doc_id: str,
    title: str,
    text: str,
    kb_id: str = "default",
    metadata: Optional[Dict[str, Any]] = None
) -> bool:
    """
    索引文档到本地知识库（占位）
    
    Args:
        doc_id: 文档 ID
        title: 标题
        text: 正文
        kb_id: 知识库 ID
        metadata: 元数据
    
    Returns:
        是否成功
    
    Raises:
        NotImplementedError: 本功能尚未实现
    """
    logger.warning("local_kb_index_document called but not implemented yet")
    raise NotImplementedError(
        "Local KB indexing is not implemented. "
        "To enable, implement document embedding and storage."
    )


# 集成建议：
# 1. 使用 ChromaDB / FAISS / Milvus 作为向量存储
# 2. 支持 PDF/Word/Markdown 文档解析
# 3. 与 materialize_round_cache_node 集成：
#    - 在 Phase 1 添加 "kb_hit" 类型识别
#    - 在 Phase 2 调用 local_kb_fetch 获取全文
#    - 标记 origin="local" 提高权重
# 4. 权限控制：按 task_id / user_id 隔离知识库
