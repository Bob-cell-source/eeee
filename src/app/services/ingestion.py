"""
本地知识库入库脚本 - 支持文本和图片的多模态RAG

用法:
    python -m backend.src.app.services.ingestion <file_path> [--task_id <id>]
    
示例:
    python -m backend.src.app.services.ingestion ./my_document.pdf
    python -m backend.src.app.services.ingestion ./folder/ --task_id my_research
"""
import asyncio
import logging
import time
import uuid
import argparse
import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from ..tools.file_parse_tool import get_file_parser
from ..tools.image_parse_tool import get_image_parser
from ..tools.embeddings import get_embedding_provider
from ..memory.writer import get_writer
from ..utils.hashing import generate_evidence_id, hash_text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextSplitter:
    """简单的文本分割器（可替换为 LangChain 的 RecursiveCharacterTextSplitter）"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split_text(self, text: str) -> List[str]:
        """分割文本为块"""
        if not text:
            return []
        
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = start + self.chunk_size
            chunk = text[start:end]
            
            # 尝试在句子边界分割
            if end < text_len:
                last_period = chunk.rfind('。')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)
                if break_point > self.chunk_size // 2:  # 确保不会分割得太小
                    chunk = chunk[:break_point + 1]
                    end = start + break_point + 1
            
            chunks.append(chunk.strip())
            start = end - self.chunk_overlap
        
        return [c for c in chunks if c]  # 过滤空块


async def generate_image_caption(image_url: str) -> str:
    """
    为图片生成描述（Caption）
    
    Args:
        image_url: 图片URL或data URL
    
    Returns:
        图片的详细描述文本
    """
    try:
        image_parser = get_image_parser()
        
        # 使用详细的提示词要求生成Dense Caption
        question = """请详细描述这张图片的内容，包括：
1. 主要对象和场景
2. 图表类型（如果是图表）及其展示的数据/趋势
3. 任何可见的文字或标签
4. 颜色、布局等视觉特征
5. 图片可能传达的信息或结论

请用中文详细描述（200字以内）。"""
        
        result = await image_parser.parse(
            question=question,
            image_url=image_url
        )
        
        caption = result.get("description", "")
        if result.get("extracted_text"):
            caption += f"\n\n提取的文字：{result['extracted_text']}"
        if result.get("analysis"):
            caption += f"\n\n分析：{result['analysis']}"
        
        return caption or "图片描述生成失败"
    except Exception as e:
        logger.error(f"Failed to generate caption for image: {e}")
        return f"图片描述生成失败"


async def ingest_file(
    file_path: str,
    task_id: Optional[str] = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    extract_images: bool = True,
) -> Dict[str, int]:
    """
    入库单个文件（支持文本和图片）
    
    Args:
        file_path: 文件路径
        task_id: 任务ID（可选，默认生成）
        chunk_size: 文本块大小
        chunk_overlap: 文本块重叠
        extract_images: 是否提取并入库图片
    
    Returns:
        统计信息 {text_chunks: int, images: int}
    """
    if not task_id:
        task_id = f"ingest_{uuid.uuid4().hex[:8]}"
    
    logger.info(f"开始入库文件: {file_path} (task_id={task_id})")
    
    file_parser = get_file_parser()
    embedding_provider = get_embedding_provider()
    writer = get_writer()
    splitter = TextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    # 1. 解析文件
    try:
        parse_result = await file_parser.parse(
            file=file_path,
            extract_images=extract_images,
            max_chars=500000,  # 大文件支持
        )
    except Exception as e:
        logger.error(f"文件解析失败: {e}")
        return {"text_chunks": 0, "images": 0}
    
    file_name = Path(file_path).name
    text_content = parse_result.get("text", "")
    images_base64 = parse_result.get("images_base64", [])
    
    logger.info(f"解析完成: {len(text_content)} 字符, {len(images_base64)} 张图片")
    
    chunks_to_write = []
    text_chunk_count = 0
    image_count = 0
    created_at = int(time.time())
    
    # 2. 处理文本块
    if text_content:
        text_chunks = splitter.split_text(text_content)
        logger.info(f"文本分割为 {len(text_chunks)} 块")
        
        for i, chunk_text in enumerate(text_chunks):
            chunk_id = generate_evidence_id("local_file", file_name, hash_text(chunk_text))
            embedding = await embedding_provider.embed(chunk_text)
            
            chunks_to_write.append({
                "id": chunk_id,
                "embedding": embedding,
                "text": chunk_text,
                "source": "local_file",
                "ref": file_path,
                "snippet": chunk_text[:200],
                "task_id": task_id,
                "round_id": 0,
                "metadata": json.dumps({
                    "file_name": file_name,
                    "chunk_index": i,
                    "total_chunks": len(text_chunks),
                }),
                "created_at": created_at,
                "media_type": "text",
                "image_url": "",
            })
            text_chunk_count += 1
    
    # 3. 处理图片（生成Caption并索引）
    if extract_images and images_base64:
        logger.info(f"开始处理 {len(images_base64)} 张图片...")
        
        # 并发生成Caption（使用base64数据）
        caption_tasks = []
        for img_base64 in images_base64:
            # 直接使用base64数据生成data URL
            data_url = f"data:image/png;base64,{img_base64}"
            caption_tasks.append(generate_image_caption(image_url=data_url))
        
        captions = await asyncio.gather(*caption_tasks)
        
        for i, (img_base64, caption) in enumerate(zip(images_base64, captions)):
            # 生成data URL作为image_url（前端可以直接渲染）
            data_url = f"data:image/png;base64,{img_base64}"
            img_id = generate_evidence_id("local_file", f"{file_name}_img_{i}", hash_text(caption))
            
            # 使用Caption生成embedding（统一索引策略）
            embedding = await embedding_provider.embed(caption)
            
            chunks_to_write.append({
                "id": img_id,
                "embedding": embedding,
                "text": caption,  # 存储Caption作为text
                "source": "local_file",
                "ref": file_path,
                "snippet": caption[:200],
                "task_id": task_id,
                "round_id": 0,
                "metadata": json.dumps({
                    "file_name": file_name,
                    "image_index": i,
                }),
                "created_at": created_at,
                "media_type": "image",
                "image_url": data_url,  # 存储data URL
            })
            image_count += 1
        
        logger.info(f"图片Caption生成完成: {image_count} 张")
    
    # 4. 批量写入Milvus
    if chunks_to_write:
        logger.info(f"写入 {len(chunks_to_write)} 条记录到Milvus...")
        writer.write_evidence_batch(chunks_to_write)
        logger.info(f"✅ 入库完成: {text_chunk_count} 文本块, {image_count} 图片")
    
    return {"text_chunks": text_chunk_count, "images": image_count}


async def ingest_folder(
    folder_path: str,
    task_id: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    入库整个文件夹
    
    Args:
        folder_path: 文件夹路径
        task_id: 任务ID
        **kwargs: 传递给 ingest_file 的参数
    
    Returns:
        统计信息
    """
    folder = Path(folder_path)
    if not folder.is_dir():
        raise ValueError(f"不是有效的文件夹: {folder_path}")
    
    # 支持的文件类型
    supported_extensions = [".pdf", ".txt", ".md", ".docx", ".pptx", ".html", ".htm"]
    files = [f for f in folder.rglob("*") if f.suffix.lower() in supported_extensions]
    
    logger.info(f"找到 {len(files)} 个文件待入库")
    
    total_text = 0
    total_images = 0
    
    for file_path in files:
        try:
            stats = await ingest_file(str(file_path), task_id=task_id, **kwargs)
            total_text += stats["text_chunks"]
            total_images += stats["images"]
        except Exception as e:
            logger.error(f"处理文件 {file_path} 失败: {e}")
    
    return {
        "total_files": len(files),
        "text_chunks": total_text,
        "images": total_images,
    }


async def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description="本地知识库入库工具")
    parser.add_argument("path", help="文件或文件夹路径")
    parser.add_argument("--task_id", default=None, help="任务ID（可选）")
    parser.add_argument("--chunk_size", type=int, default=1000, help="文本块大小")
    parser.add_argument("--chunk_overlap", type=int, default=200, help="文本块重叠")
    parser.add_argument("--no_images", action="store_true", help="不提取图片")
    parser.add_argument("--rebuild_schema", action="store_true", help="重建Milvus schema（会删除现有数据）")
    
    args = parser.parse_args()
    
    # 重建schema（如果需要）
    if args.rebuild_schema:
        logger.warning("⚠️ 即将重建Milvus schema，所有现有数据将被删除！")
        response = input("确认继续？(yes/no): ")
        if response.lower() == "yes":
            from ..memory.milvus_client import get_milvus_client
            client = get_milvus_client()
            client.rebuild_evidence_collection()
            logger.info("✅ Schema 重建完成")
        else:
            logger.info("操作已取消")
            return
    
    # 入库
    path = Path(args.path)
    
    if path.is_file():
        stats = await ingest_file(
            str(path),
            task_id=args.task_id,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            extract_images=not args.no_images,
        )
        print(f"\n✅ 入库完成:")
        print(f"  文本块: {stats['text_chunks']}")
        print(f"  图片: {stats['images']}")
    
    elif path.is_dir():
        stats = await ingest_folder(
            str(path),
            task_id=args.task_id,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            extract_images=not args.no_images,
        )
        print(f"\n✅ 入库完成:")
        print(f"  文件数: {stats['total_files']}")
        print(f"  文本块: {stats['text_chunks']}")
        print(f"  图片: {stats['images']}")
    
    else:
        print(f"❌ 错误: {args.path} 不是有效的文件或文件夹")


if __name__ == "__main__":
    asyncio.run(main())
