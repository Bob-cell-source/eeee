"""
Milvus Writer - Write data to Milvus collections
"""
import json
import time
import logging
from typing import List, Dict, Any, Optional
from .milvus_client import get_milvus_client

logger = logging.getLogger(__name__)


class MilvusWriter:
    """Write data to Milvus collections"""
    
    def __init__(self):
        self.client = get_milvus_client()
    
    def write_evidence(
        self,
        chunk_id: str,
        embedding: List[float],
        text: str,
        source: str,  # upload|arxiv|github
        ref: str,  # doc_id/arxiv_id/repo
        snippet: str,
        task_id: str = "",
        round_id: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Write a single evidence chunk to Milvus"""
        try:
            collection = self.client.get_evidence_collection()
            
            data = [
                [chunk_id],
                [embedding],
                [text[:65535]],  # Truncate if needed
                [source],
                [ref],
                [task_id],
                [round_id],
                [snippet[:2048]],  # Truncate if needed
                [json.dumps(metadata or {})[:4096]],
                [int(time.time())],
            ]
            
            collection.insert(data)
            collection.flush()
            logger.debug(f"Wrote evidence chunk {chunk_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to write evidence chunk {chunk_id}: {e}")
            return False
    
    def write_evidence_batch(
        self,
        chunks: List[Dict[str, Any]],
    ) -> int:
        """
        Write multiple evidence chunks to Milvus
        
        Each chunk dict should have:
        - id: str
        - embedding: List[float]
        - text: str
        - source: str
        - ref: str
        - snippet: str
        - task_id: str (optional)
        - round_id: int (optional)
        - metadata: dict (optional)
        """
        if not chunks:
            return 0
        
        try:
            collection = self.client.get_evidence_collection()
            
            ids = []
            embeddings = []
            texts = []
            sources = []
            refs = []
            task_ids = []
            round_ids = []
            snippets = []
            metadatas = []
            created_ats = []
            
            current_time = int(time.time())
            
            for chunk in chunks:
                ids.append(chunk["id"])
                embeddings.append(chunk["embedding"])
                texts.append(chunk["text"][:65535])
                sources.append(chunk["source"])
                refs.append(chunk["ref"])
                task_ids.append(chunk.get("task_id", ""))
                round_ids.append(chunk.get("round_id", 0))
                snippets.append(chunk.get("snippet", chunk["text"][:200])[:2048])
                metadatas.append(json.dumps(chunk.get("metadata", {}))[:4096])
                created_ats.append(current_time)
            
            data = [ids, embeddings, texts, sources, refs, task_ids, round_ids, snippets, metadatas, created_ats]
            
            collection.insert(data)
            collection.flush()
            logger.info(f"Wrote {len(chunks)} evidence chunks to Milvus")
            return len(chunks)
        except Exception as e:
            logger.error(f"Failed to write evidence batch: {e}")
            return 0
    
    def write_report(
        self,
        report_id: str,
        embedding: List[float],
        task_id: str,
        round_id: int,
        summary: str,
        findings: List[str],
        open_questions: List[str],
        evidence_ids: List[str],
    ) -> bool:
        """Write a research report to Milvus"""
        try:
            collection = self.client.get_reports_collection()
            
            data = [
                [report_id],
                [embedding],
                [task_id],
                [round_id],
                [summary[:65535]],
                [json.dumps(findings)[:65535]],
                [json.dumps(open_questions)[:65535]],
                [json.dumps(evidence_ids)[:65535]],
                [int(time.time())],
            ]
            
            collection.insert(data)
            collection.flush()
            logger.info(f"Wrote report {report_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to write report {report_id}: {e}")
            return False


def get_writer() -> MilvusWriter:
    """Get writer instance"""
    return MilvusWriter()
