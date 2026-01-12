"""
DeepResearch Service Smoke Test

基础冒烟测试，验证系统主要组件可以正常运行
"""
import asyncio
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_session_cache_serialization():
    """测试 SessionCache 序列化/反序列化"""
    from backend.src.app.services.session_cache import SessionCache, RawDoc, ResearchNote
    from backend.src.app.utils.chunking import DocChunk
    
    logger.info("Testing SessionCache serialization...")
    
    # 创建测试数据
    cache = SessionCache()
    
    # 添加 RawDoc
    chunks = [
        DocChunk(
            chunk_id="chunk_0",
            start_char=0,
            end_char=100,
            text="This is a test chunk."
        )
    ]
    
    doc = RawDoc(
        doc_id="test_doc_1",
        source="web",
        ref="https://example.com",
        url="https://example.com",
        title="Test Document",
        fetched_at=datetime.now().isoformat(),
        text="This is a test document with some content.",
        chunks=chunks,
        metadata={"test": "value"},
        origin="web",
        degraded=False
    )
    cache.add_doc(doc)
    
    # 添加 ResearchNote
    note: ResearchNote = {
        "pack_id": "test_note_1",
        "source": "web",
        "ref": "https://example.com",
        "url": "https://example.com",
        "title": "Test Note",
        "snippet": "Test snippet",
        "key_points": ["Point 1", "Point 2"],
        "evidence_quotes": [
            {"quote": "Test quote", "why_relevant": "Test reason", "location_hint": "P1"}
        ],
        "raw_pointer": "test_doc_1",
        "confidence": "high",
        "relevance": "Test relevance",
        "media_type": "text",
        "image_url": None,
        "fetched_at": datetime.now().isoformat(),
        "target_question": "Test question"
    }
    cache.add_note(note)
    
    # 序列化
    state_dict = cache.to_state_dict()
    assert "docs" in state_dict
    assert "notes" in state_dict
    assert len(state_dict["docs"]) == 1
    assert len(state_dict["notes"]) == 1
    logger.info("✓ Serialization successful")
    
    # 反序列化
    restored_cache = SessionCache.from_state_dict(state_dict)
    assert len(restored_cache.docs) == 1
    assert len(restored_cache.notes) == 1
    
    restored_doc = restored_cache.get_doc("test_doc_1")
    assert restored_doc is not None
    assert restored_doc.doc_id == "test_doc_1"
    assert restored_doc.source == "web"
    assert len(restored_doc.chunks) == 1
    assert restored_doc.chunks[0].chunk_id == "chunk_0"
    
    restored_note = restored_cache.get_note("test_note_1")
    assert restored_note is not None
    assert restored_note["pack_id"] == "test_note_1"
    assert len(restored_note["key_points"]) == 2
    
    logger.info("✓ Deserialization successful")
    logger.info("✓ SessionCache round-trip test passed")


async def test_chunking():
    """测试文本分块功能"""
    from backend.src.app.utils.chunking import RuleBasedChunker
    
    logger.info("Testing text chunking...")
    
    chunker = RuleBasedChunker(chunk_size=200, overlap=50)
    
    text = """This is a test document.

It has multiple paragraphs.

Each paragraph should be preserved where possible.

The chunker should split long paragraphs intelligently.

This is the final paragraph of our test document."""
    
    chunks = chunker.split_text_to_chunks(text, "test_doc")
    
    assert len(chunks) > 0
    assert all(c.chunk_id.startswith("test_doc_chunk_") for c in chunks)
    assert all(len(c.text) <= 250 for c in chunks)  # Allow some margin
    
    logger.info(f"✓ Created {len(chunks)} chunks")
    logger.info("✓ Chunking test passed")


async def test_safe_note_access():
    """测试 ResearchNote 安全访问函数"""
    from backend.src.app.services.session_cache import (
        ResearchNote, 
        safe_note_pack_id, 
        safe_note_source, 
        safe_note_evidence_quotes
    )
    
    logger.info("Testing safe note access...")
    
    # 完整 note
    note: ResearchNote = {
        "pack_id": "test_1",
        "source": "web",
        "ref": "test",
        "snippet": "test",
        "key_points": [],
        "evidence_quotes": [{"quote": "test", "why_relevant": "test", "location_hint": None}],
        "raw_pointer": "doc1",
        "confidence": "high",
        "media_type": "text",
        "fetched_at": datetime.now().isoformat()
    }
    
    assert safe_note_pack_id(note) == "test_1"
    assert safe_note_source(note) == "web"
    assert len(safe_note_evidence_quotes(note)) == 1
    
    # 最小 note（几乎所有字段缺失）
    minimal_note: ResearchNote = {}  # type: ignore
    
    assert safe_note_pack_id(minimal_note).startswith("note_")
    assert safe_note_source(minimal_note) == "unknown"
    assert safe_note_evidence_quotes(minimal_note) == []
    
    logger.info("✓ Safe note access test passed")


async def test_local_kb_placeholder():
    """测试本地 KB 占位接口"""
    from backend.src.app.tools.local_kb_tool import (
        local_kb_search,
        local_kb_fetch,
        local_kb_index_document
    )
    
    logger.info("Testing local KB placeholder...")
    
    try:
        await local_kb_search("test query")
        assert False, "Should have raised NotImplementedError"
    except NotImplementedError as e:
        assert "not implemented" in str(e).lower()
        logger.info("✓ local_kb_search raises NotImplementedError as expected")
    
    try:
        await local_kb_fetch("test_doc")
        assert False, "Should have raised NotImplementedError"
    except NotImplementedError as e:
        assert "not implemented" in str(e).lower()
        logger.info("✓ local_kb_fetch raises NotImplementedError as expected")
    
    try:
        await local_kb_index_document("doc1", "Test", "Content")
        assert False, "Should have raised NotImplementedError"
    except NotImplementedError as e:
        assert "not implemented" in str(e).lower()
        logger.info("✓ local_kb_index_document raises NotImplementedError as expected")
    
    logger.info("✓ Local KB placeholder test passed")


async def main():
    """运行所有冒烟测试"""
    logger.info("=" * 60)
    logger.info("Starting DeepResearch Smoke Tests")
    logger.info("=" * 60)
    
    tests = [
        ("SessionCache Serialization", test_session_cache_serialization),
        ("Text Chunking", test_chunking),
        ("Safe Note Access", test_safe_note_access),
        ("Local KB Placeholder", test_local_kb_placeholder),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            logger.info(f"\n--- Running: {test_name} ---")
            await test_func()
            passed += 1
        except Exception as e:
            logger.error(f"✗ {test_name} FAILED: {e}", exc_info=True)
            failed += 1
    
    logger.info("\n" + "=" * 60)
    logger.info(f"Test Results: {passed} passed, {failed} failed")
    logger.info("=" * 60)
    
    if failed > 0:
        exit(1)
    else:
        logger.info("✓ All smoke tests passed!")
        exit(0)


if __name__ == "__main__":
    asyncio.run(main())
