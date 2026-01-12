"""
Milvus Schema 迁移脚本

用法：
    python -m backend.src.app.scripts.migrate_schema
"""
import logging
from ..memory.milvus_client import get_milvus_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def migrate_to_multimodal_schema():
    """迁移到多模态Schema（会删除现有数据）"""
    logger.warning("⚠️  即将重建 Milvus Schema 以支持多模态RAG")
    logger.warning("⚠️  这将删除 evidence_chunks 集合中的所有现有数据！")
    
    response = input("\n确认继续？输入 'yes' 继续，其他任何输入取消: ")
    
    if response.lower() != "yes":
        logger.info("❌ 操作已取消")
        return
    
    logger.info("🔄 开始迁移...")
    
    try:
        client = get_milvus_client()
        
        # 重建集合
        client.rebuild_evidence_collection()
        
        logger.info("✅ Schema 迁移完成！")
        logger.info("\n新增字段:")
        logger.info("  - media_type (VARCHAR, 16): 'text' | 'image'")
        logger.info("  - image_url (VARCHAR, 2048): 图片URL或本地路径")
        logger.info("\n现在可以使用 ingestion.py 入库本地文件和图片了！")
    
    except Exception as e:
        logger.error(f"❌ 迁移失败: {e}")
        raise


if __name__ == "__main__":
    migrate_to_multimodal_schema()
