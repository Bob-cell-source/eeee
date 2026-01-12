"""
Hashing Utilities
"""
import hashlib
from typing import Union


def hash_text(text: str) -> str:
    """Generate SHA256 hash of text"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hash_bytes(data: bytes) -> str:
    """Generate SHA256 hash of bytes"""
    return hashlib.sha256(data).hexdigest()


def generate_chunk_id(doc_id: str, chunk_index: int) -> str:
    """Generate unique chunk ID"""
    return f"{doc_id}_chunk_{chunk_index}"


def generate_evidence_id(source: str, ref: str, text_hash: str) -> str:
    """Generate unique evidence ID"""
    combined = f"{source}:{ref}:{text_hash[:16]}"
    return hashlib.md5(combined.encode()).hexdigest()
