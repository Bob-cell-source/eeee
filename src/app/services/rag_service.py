"""
RAG Service - Retrieval Augmented Generation for uploaded documents
"""
import logging
from typing import Dict, Any, Optional, List

from ..tools.embeddings import get_embedding_provider
from ..tools.llm import get_llm_provider
from ..memory.retriever import get_retriever

logger = logging.getLogger(__name__)


class RAGService:
    """Service for RAG-based question answering on uploaded documents"""
    
    def __init__(self):
        self.embedding_provider = get_embedding_provider()
        self.llm_provider = get_llm_provider()
        self.retriever = get_retriever()
    
    async def answer(
        self,
        question: str,
        doc_ids: Optional[List[str]] = None,
        top_k: int = 5,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Answer a question using RAG on uploaded documents
        
        Args:
            question: User's question
            doc_ids: Optional list of document IDs to search within
            top_k: Number of chunks to retrieve
            session_id: Optional session ID for context
        
        Returns:
            {
                "answer": str,
                "citations": List[{source, ref, snippet, chunk_id}],
                "sources_used": int
            }
        """
        # Generate embedding for the question
        query_embedding = await self.embedding_provider.embed(question)
        
        # Retrieve relevant chunks from uploaded documents
        chunks = self.retriever.search_evidence(
            query_embedding=query_embedding,
            top_k=top_k,
            source_filter="upload",
            doc_ids=doc_ids,
        )
        
        if not chunks:
            return {
                "answer": "I couldn't find any relevant information in the uploaded documents to answer your question. Please make sure you've uploaded relevant documents first.",
                "citations": [],
                "sources_used": 0,
            }
        
        # Build context from retrieved chunks
        context_parts = []
        citations = []
        
        for i, chunk in enumerate(chunks):
            context_parts.append(f"[Source {i+1}]:\n{chunk['text']}")
            citations.append({
                "source": chunk["source"],
                "ref": chunk["ref"],
                "snippet": chunk["snippet"],
                "chunk_id": chunk["id"],
                "score": chunk.get("score", 0),
                "metadata": chunk.get("metadata", {}),
            })
        
        context = "\n\n".join(context_parts)
        
        # Generate answer using LLM
        system_prompt = """You are a helpful research assistant. Answer the user's question based on the provided context from their uploaded documents.

Rules:
1. Only use information from the provided context
2. If the context doesn't contain enough information, say so
3. Cite your sources using [Source N] notation
4. Be concise but thorough
5. If you're uncertain, indicate your level of confidence"""

        user_prompt = f"""Context from uploaded documents:

{context}

---

Question: {question}

Please provide a comprehensive answer based on the context above, citing your sources."""

        answer = await self.llm_provider.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
        )
        
        return {
            "answer": answer,
            "citations": citations,
            "sources_used": len(chunks),
        }
    
    async def summarize_document(
        self,
        doc_id: str,
        max_chunks: int = 10,
    ) -> Dict[str, Any]:
        """
        Summarize an uploaded document
        
        Args:
            doc_id: Document ID to summarize
            max_chunks: Maximum chunks to use for summary
        
        Returns:
            {
                "summary": str,
                "doc_id": str,
                "chunks_used": int
            }
        """
        # Get a dummy embedding (just to retrieve chunks by doc_id filter)
        # We'll retrieve by doc_id filter rather than similarity
        dummy_embedding = await self.embedding_provider.embed("document summary overview content")
        
        chunks = self.retriever.search_evidence(
            query_embedding=dummy_embedding,
            top_k=max_chunks,
            source_filter="upload",
            doc_ids=[doc_id],
        )
        
        if not chunks:
            return {
                "summary": "No content found for this document.",
                "doc_id": doc_id,
                "chunks_used": 0,
            }
        
        # Combine chunk texts
        full_text = "\n\n".join([chunk["text"] for chunk in chunks])
        
        # Generate summary
        system_prompt = "You are a helpful research assistant. Provide a comprehensive summary of the document content."
        
        user_prompt = f"""Please summarize the following document content:

{full_text}

Provide a structured summary including:
1. Main topics covered
2. Key findings or arguments
3. Important details
4. Conclusions (if any)"""

        summary = await self.llm_provider.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
        )
        
        return {
            "summary": summary,
            "doc_id": doc_id,
            "chunks_used": len(chunks),
        }


def get_rag_service() -> RAGService:
    """Get RAG service instance"""
    return RAGService()
