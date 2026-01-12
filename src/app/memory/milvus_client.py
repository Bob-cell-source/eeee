"""
Milvus Client - Connection and Collection Management
"""
import logging
from typing import Optional
from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility,
)
from ..config import get_settings

logger = logging.getLogger(__name__)


class MilvusClient:
    """Milvus database client singleton"""
    
    _instance: Optional["MilvusClient"] = None
    _connected: bool = False
    
    # Collection names
    EVIDENCE_COLLECTION = "evidence_chunks"
    REPORTS_COLLECTION = "research_reports"
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._connected:
            self.settings = get_settings()
            self._connect()
    
    def _connect(self):
        """Connect to Milvus"""
        try:
            connections.connect(
                alias="default",
                host=self.settings.milvus_host,
                port=self.settings.milvus_port,
            )
            self._connected = True
            logger.info(f"Connected to Milvus at {self.settings.milvus_host}:{self.settings.milvus_port}")
            self._ensure_collections()
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            self._connected = False
            raise
    
    def _ensure_collections(self):
        """Ensure required collections exist"""
        self._create_evidence_collection()
        self._create_reports_collection()
    
    def drop_evidence_collection(self):
        """删除evidence_chunks集合（用于schema升级）"""
        if utility.has_collection(self.EVIDENCE_COLLECTION):
            utility.drop_collection(self.EVIDENCE_COLLECTION)
            logger.warning(f"已删除集合 {self.EVIDENCE_COLLECTION}")
    
    def rebuild_evidence_collection(self):
        """重建 evidence_chunks 集合（schema 升级时使用）"""
        self.drop_evidence_collection()
        self._create_evidence_collection()
        logger.info(f"已重建集合 {self.EVIDENCE_COLLECTION}")
    
    def _create_evidence_collection(self):
        """Create evidence_chunks collection if not exists (支持多模态)"""
        if utility.has_collection(self.EVIDENCE_COLLECTION):
            logger.info(f"Collection {self.EVIDENCE_COLLECTION} already exists")
            return
        
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.settings.embedding_dim),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),  # 文本内容或图片描述(Caption)
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=32),  # upload|arxiv|github|local_file|web
            FieldSchema(name="ref", dtype=DataType.VARCHAR, max_length=512),  # doc_id/arxiv_id/repo/file_path
            FieldSchema(name="task_id", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="round_id", dtype=DataType.INT64),
            FieldSchema(name="snippet", dtype=DataType.VARCHAR, max_length=2048),
            FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=4096),  # JSON string
            FieldSchema(name="created_at", dtype=DataType.INT64),
            # 多模态字段
            FieldSchema(name="media_type", dtype=DataType.VARCHAR, max_length=16, default_value="text"),  # text|image
            FieldSchema(name="image_url", dtype=DataType.VARCHAR, max_length=2048, default_value=""),  # 图片URL或本地路径
        ]
        
        schema = CollectionSchema(fields=fields, description="Evidence chunks for RAG and research")
        collection = Collection(name=self.EVIDENCE_COLLECTION, schema=schema)
        
        # Create index for vector search
        index_params = {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
        collection.create_index(field_name="embedding", index_params=index_params)
        logger.info(f"Created collection {self.EVIDENCE_COLLECTION} with index")
    
    def _create_reports_collection(self):
        """Create research_reports collection if not exists"""
        if utility.has_collection(self.REPORTS_COLLECTION):
            logger.info(f"Collection {self.REPORTS_COLLECTION} already exists")
            return
        
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.settings.embedding_dim),
            FieldSchema(name="task_id", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="round_id", dtype=DataType.INT64),
            FieldSchema(name="summary", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="findings", dtype=DataType.VARCHAR, max_length=65535),  # JSON array
            FieldSchema(name="open_questions", dtype=DataType.VARCHAR, max_length=65535),  # JSON array
            FieldSchema(name="evidence_ids", dtype=DataType.VARCHAR, max_length=65535),  # JSON array
            FieldSchema(name="created_at", dtype=DataType.INT64),
        ]
        
        schema = CollectionSchema(fields=fields, description="Research reports and summaries")
        collection = Collection(name=self.REPORTS_COLLECTION, schema=schema)
        
        # Create index for vector search
        index_params = {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
        collection.create_index(field_name="embedding", index_params=index_params)
        logger.info(f"Created collection {self.REPORTS_COLLECTION} with index")
    
    def get_evidence_collection(self) -> Collection:
        """Get evidence collection"""
        collection = Collection(self.EVIDENCE_COLLECTION)
        collection.load()
        return collection
    
    def get_reports_collection(self) -> Collection:
        """Get reports collection"""
        collection = Collection(self.REPORTS_COLLECTION)
        collection.load()
        return collection
    
    def is_connected(self) -> bool:
        """Check if connected to Milvus"""
        try:
            utility.list_collections()
            return True
        except Exception:
            return False
    
    def disconnect(self):
        """Disconnect from Milvus"""
        try:
            connections.disconnect("default")
            self._connected = False
            logger.info("Disconnected from Milvus")
        except Exception as e:
            logger.error(f"Error disconnecting from Milvus: {e}")


def get_milvus_client() -> MilvusClient:
    """Get Milvus client instance"""
    return MilvusClient()
