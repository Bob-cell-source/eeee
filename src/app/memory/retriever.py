"""
Milvus Retriever - Search and retrieve from Milvus collections (支持混合检索和重排序)
"""
import json
import logging
from typing import List, Dict, Any, Optional
from .milvus_client import get_milvus_client

logger = logging.getLogger(__name__)


def rerank_results(query: str, results: List[Dict[str, Any]], top_k: int = 10) -> List[Dict[str, Any]]:
    """
    重排序检索结果（使用Cross-Encoder提升准确性）
    
    Args:
        query: 查询文本
        results: 候选结果列表
        top_k: 返回top-k结果
    
    Returns:
        重排序后的top-k结果
    
    注意：这是占位接口，可以集成：
    - sentence-transformers Cross-Encoder
    - BGE-Reranker
    - Cohere Rerank API
    """
    try:
        # TODO: 集成实际的重排序模型
        # 示例：使用 sentence-transformers/ms-marco-MiniLM-L-12-v2
        # from sentence_transformers import CrossEncoder
        # model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')
        # pairs = [[query, r['text']] for r in results]
        # scores = model.predict(pairs)
        # for i, r in enumerate(results):
        #     r['rerank_score'] = float(scores[i])
        # results.sort(key=lambda x: x['rerank_score'], reverse=True)
        
        # 当前实现：基于向量相似度分数的简单排序（占位）
        logger.debug(f"Reranking {len(results)} results (using placeholder logic)")
        results_sorted = sorted(results, key=lambda x: x.get('score', 0), reverse=True)
        return results_sorted[:top_k]
    except Exception as e:
        logger.error(f"Reranking failed: {e}, returning original results")
        return results[:top_k]


class MilvusRetriever:
    """Retrieve data from Milvus collections (支持多模态和高级检索)"""
    
    def __init__(self):
        self.client = get_milvus_client()
    
    def search_evidence(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        source_filter: Optional[str] = None,  # upload|arxiv|github|local_file|web
        doc_ids: Optional[List[str]] = None,
        task_id: Optional[str] = None,
        media_type: Optional[str] = None,  # text|image|None(all)
        year_filter: Optional[int] = None,  # 按年份过滤
        enable_rerank: bool = False,  # 是否启用重排序
        query_text: Optional[str] = None,  # 重排序需要原始查询文本
    ) -> List[Dict[str, Any]]:
        """
        搜索相似证据块（支持元数据过滤和重排序）
        
        Args:
            query_embedding: 查询向量
            top_k: 最终返回数量
            source_filter: 来源过滤
            doc_ids: 文档ID过滤
            task_id: 任务ID过滤
            media_type: 媒体类型过滤 (text/image)
            year_filter: 年份过滤
            enable_rerank: 是否启用两阶段重排序
            query_text: 原始查询文本（重排序时需要）
        
        Returns:
            包含 id, text, source, ref, snippet, metadata, score, media_type, image_url
        """
        try:
            collection = self.client.get_evidence_collection()
            
            # 构建过滤表达式
            filters = []
            if source_filter:
                filters.append(f'source == "{source_filter}"')
            if doc_ids:
                doc_ids_str = ", ".join([f'"{d}"' for d in doc_ids])
                filters.append(f'ref in [{doc_ids_str}]')
            if task_id:
                filters.append(f'task_id == "{task_id}"')
            if media_type:
                filters.append(f'media_type == "{media_type}"')
            # 年份过滤需要metadata中包含year字段
            if year_filter:
                # Milvus不支持JSON字段查询，需要在后处理中过滤
                pass
            
            expr = " and ".join(filters) if filters else ""
            
            # 第一阶段：向量检索（取top_k*5个候选）
            candidate_limit = top_k * 5 if enable_rerank else top_k
            search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
            
            results = collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=candidate_limit,
                expr=expr if expr else None,
                output_fields=["id", "text", "source", "ref", "snippet", "metadata", 
                              "task_id", "round_id", "media_type", "image_url"],
            )
            
            chunks = []
            for hits in results:
                for hit in hits:
                    entity = hit.entity
                    metadata = json.loads(entity.get("metadata", "{}"))
                    
                    # 年份后过滤
                    if year_filter and metadata.get("year"):
                        if int(metadata.get("year")) != year_filter:
                            continue
                    
                    chunks.append({
                        "id": entity.get("id"),
                        "text": entity.get("text"),
                        "source": entity.get("source"),
                        "ref": entity.get("ref"),
                        "snippet": entity.get("snippet"),
                        "metadata": metadata,
                        "task_id": entity.get("task_id"),
                        "round_id": entity.get("round_id"),
                        "score": hit.score,
                        "media_type": entity.get("media_type", "text"),
                        "image_url": entity.get("image_url", ""),
                    })
            
            # 第二阶段：重排序（如果启用）
            if enable_rerank and query_text and len(chunks) > top_k:
                logger.debug(f"Reranking {len(chunks)} candidates to top {top_k}")
                chunks = rerank_results(query_text, chunks, top_k)
            
            logger.debug(f"Found {len(chunks)} evidence chunks (media types: {set(c['media_type'] for c in chunks)})")
            return chunks
        except Exception as e:
            logger.error(f"Failed to search evidence: {e}")
            return []
    
    def search_reports(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        task_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar research reports
        
        Returns list of dicts with:
        - id, task_id, round_id, summary, findings, open_questions, evidence_ids, score
        """
        try:
            collection = self.client.get_reports_collection()
            
            expr = f'task_id == "{task_id}"' if task_id else None
            
            search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
            
            results = collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                expr=expr,
                output_fields=["id", "task_id", "round_id", "summary", "findings", "open_questions", "evidence_ids"],
            )
            
            reports = []
            for hits in results:
                for hit in hits:
                    entity = hit.entity
                    reports.append({
                        "id": entity.get("id"),
                        "task_id": entity.get("task_id"),
                        "round_id": entity.get("round_id"),
                        "summary": entity.get("summary"),
                        "findings": json.loads(entity.get("findings", "[]")),
                        "open_questions": json.loads(entity.get("open_questions", "[]")),
                        "evidence_ids": json.loads(entity.get("evidence_ids", "[]")),
                        "score": hit.score,
                    })
            
            logger.debug(f"Found {len(reports)} reports")
            return reports
        except Exception as e:
            logger.error(f"Failed to search reports: {e}")
            return []
    
    def get_evidence_by_ids(self, chunk_ids: List[str]) -> List[Dict[str, Any]]:
        """Get evidence chunks by IDs"""
        if not chunk_ids:
            return []
        
        try:
            collection = self.client.get_evidence_collection()
            
            ids_str = ", ".join([f'"{cid}"' for cid in chunk_ids])
            expr = f'id in [{ids_str}]'
            
            results = collection.query(
                expr=expr,
                output_fields=["id", "text", "source", "ref", "snippet", "metadata"],
            )
            
            chunks = []
            for entity in results:
                chunks.append({
                    "id": entity.get("id"),
                    "text": entity.get("text"),
                    "source": entity.get("source"),
                    "ref": entity.get("ref"),
                    "snippet": entity.get("snippet"),
                    "metadata": json.loads(entity.get("metadata", "{}")),
                })
            
            return chunks
        except Exception as e:
            logger.error(f"Failed to get evidence by IDs: {e}")
            return []


def get_retriever() -> MilvusRetriever:
    """Get retriever instance"""
    return MilvusRetriever()
