from __future__ import annotations
import os
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field
from langchain_core.tools import tool

from .github_tool import get_github_tool
from .arxiv_tool import get_arxiv_tool
from .serper_tool import get_serper_tool
from .web_visit_tool import get_web_visit_tool
from .file_parse_tool import FileParseClient
from .content_extract_tool import ContentExtractClient
from .image_parse_tool import ImageParseClient
from typing import Any

def content_to_str(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                # 常见块：{"type":"text","text":"..."} 或 {"text":"..."}
                txt = item.get("text") or item.get("content") or ""
                if txt:
                    parts.append(str(txt))
        return "\n".join([p for p in parts if p])
    return str(content)

# LLM adapter

def _require_openai():
    try:
        from langchain_openai import ChatOpenAI  # noqa
        from langchain_core.messages import HumanMessage  # noqa
    except Exception as e:
        raise RuntimeError("需要 pip install langchain-openai") from e

async def _generate_text(prompt: str) -> str:
    _require_openai()
    from langchain_openai import ChatOpenAI
    import os
    from ..config import get_settings
    
    settings = get_settings()
    # 使用环境变量配置，langchain 会自动读取 OPENAI_API_KEY/OPENAI_API_BASE
    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL") or settings.llm_model,
        temperature=0
    )
    resp = await llm.ainvoke(prompt)
    return content_to_str(resp.content)

async def _generate_vision(question: str, image_data_url: str) -> str:
    _require_openai()
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage
    import os
    from ..config import get_settings
    
    settings = get_settings()
    # 使用环境变量配置，langchain 会自动读取 OPENAI_API_KEY/OPENAI_API_BASE
    llm = ChatOpenAI(
        model=os.getenv("VISION_MODEL") or settings.vision_model,
        temperature=0
    )
    msg = HumanMessage(content=[
        {"type": "text", "text": question},
        {"type": "image_url", "image_url": {"url": image_data_url}},
    ])
    resp = await llm.ainvoke([msg])
    return content_to_str(resp.content)

_github = get_github_tool()  # 走你现有 settings/token/blacklist/max_results :contentReference[oaicite:9]{index=9}
_arxiv = get_arxiv_tool()    # 走你现有 settings/max_results :contentReference[oaicite:10]{index=10}
_serper = get_serper_tool()
_web_visit = get_web_visit_tool()
_file_parser = FileParseClient()
_content_extractor = ContentExtractClient(generate=_generate_text)
_image_parser = ImageParseClient(generate_vision=_generate_vision)






# ----------------------------
# GitHub
# ----------------------------
class GitHubRepoSearchInput(BaseModel):
    query: str = Field(..., description="GitHub 仓库搜索关键词，例如 'rag agent'")
    max_results: Optional[int] = Field(None, ge=1, le=100, description="最多返回条数（<=100）")
    sort: str = Field("stars", description="stars|forks|help-wanted-issues|updated")
    order: str = Field("desc", description="asc|desc")
    language: Optional[str] = Field(None, description="语言过滤，例如 python")

@tool("github_search_repos", args_schema=GitHubRepoSearchInput)

async def github_search_repos(
    query: str,
    max_results: Optional[int] = None,
    sort: str = "stars",
    order: str = "desc",
    language: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """在 GitHub 上按关键词搜索仓库，返回仓库基本信息列表（name/url/stars/description 等）。"""
    repos = await _github.search_repos(query=query, max_results=max_results, sort=sort, order=order, language=language)
    return [_github.repo_to_dict(r) for r in repos]


class GitHubCodeSearchInput(BaseModel):
    query: str = Field(..., description="代码搜索关键词，例如 'def build_index'")
    max_results: Optional[int] = Field(None, ge=1, le=100)
    language: Optional[str] = Field(None, description="语言过滤，例如 python")
    repo: Optional[str] = Field(None, description="限定仓库，格式 owner/repo")

@tool("github_search_code", args_schema=GitHubCodeSearchInput)
async def github_search_code(
    query: str,
    max_results: Optional[int] = None,
    language: Optional[str] = None,
    repo: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """搜索 GitHub 代码"""
    results = await _github.search_code(query=query, max_results=max_results, language=language, repo=repo)
    return [asdict(r) for r in results]


class GitHubGetRepoInput(BaseModel):
    full_name: str = Field(..., description="仓库全名，例如 owner/repo")

@tool("github_get_repo", args_schema=GitHubGetRepoInput)
async def github_get_repo(full_name: str) -> Optional[Dict[str, Any]]:
    """获取仓库详情。"""
    repo = await _github.get_repo(full_name)
    return _github.repo_to_dict(repo) if repo else None


class GitHubGetReadmeInput(BaseModel):
    full_name: str = Field(..., description="仓库全名，例如 owner/repo")

@tool("github_get_readme", args_schema=GitHubGetReadmeInput)
async def github_get_readme(full_name: str) -> Optional[str]:
    """获取 README 原文。"""
    return await _github.get_readme(full_name)


# ----------------------------
# arXiv
# ----------------------------
class ArxivSearchInput(BaseModel):
    query: str = Field(..., description="arXiv 搜索关键词（内部会拼到 all:<query>）")
    max_results: Optional[int] = Field(None, ge=1, le=50)
    sort_by: str = Field("relevance", description="relevance|lastUpdatedDate|submittedDate")
    sort_order: str = Field("descending", description="ascending|descending")

@tool("arxiv_search", args_schema=ArxivSearchInput)
async def arxiv_search(
    query: str,
    max_results: Optional[int] = None,
    sort_by: str = "relevance",
    sort_order: str = "descending",
) -> List[Dict[str, Any]]:
    """搜索 arXiv 论文。"""
    papers = await _arxiv.search(query=query, max_results=max_results, sort_by=sort_by, sort_order=sort_order)
    return [_arxiv.paper_to_dict(p) for p in papers]


class ArxivSearchByCategoryInput(BaseModel):
    category: str = Field(..., description="例如 cs.AI / cs.LG")
    query: Optional[str] = Field(None, description="可选附加关键词")
    max_results: Optional[int] = Field(None, ge=1, le=50)

@tool("arxiv_search_by_category", args_schema=ArxivSearchByCategoryInput)
async def arxiv_search_by_category(
    category: str,
    query: Optional[str] = None,
    max_results: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """按分类搜索 arXiv。"""
    papers = await _arxiv.search_by_category(category=category, query=query, max_results=max_results)
    return [_arxiv.paper_to_dict(p) for p in papers]


class ArxivGetPaperInput(BaseModel):
    arxiv_id: str = Field(..., description="例如 2301.00001 或 2301.00001v1（支持 arXiv: 前缀）")

@tool("arxiv_get_paper", args_schema=ArxivGetPaperInput)
async def arxiv_get_paper(arxiv_id: str) -> Optional[Dict[str, Any]]:
    """按 arXiv ID 获取论文详情。"""
    paper = await _arxiv.get_paper(arxiv_id)
    return _arxiv.paper_to_dict(paper) if paper else None


# ----------------------------
# serper
# ----------------------------

from typing import Union

class WebSearchInput(BaseModel):
    queries: Union[str, List[str]] = Field(..., description="单条或批量 query")
    max_results: int = Field(10, ge=1, le=100)
    country: Optional[str] = None
    locale: Optional[str] = None

@tool("web_search", args_schema=WebSearchInput)
async def web_search(
    queries: Union[str, List[str]],
    max_results: int = 10,
    country: Optional[str] = None,
    locale: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """在网络上搜索 query，返回最多 max_results 条结果的结构化字典。"""

    if isinstance(queries, str):
        queries = [queries]
    return await _serper.search_many(
        queries=queries,
        max_results=max_results,
        country=country,
        locale=locale,
    )
# ----------------------------
# web_visit
# ----------------------------

class WebVisitInput(BaseModel):
    url: str = Field(..., description="要访问的网页URL")
    max_chars: int = Field(20000, ge=1000, le=200000, description="返回正文最大字符数")
    use_jina: bool = Field(True, description="优先使用 r.jina.ai 抓取纯文本")

@tool("web_visit", args_schema=WebVisitInput)
async def web_visit(url: str, max_chars: int = 20000, use_jina: bool = True) -> Dict[str, Any]:
    """访问指定网页并提取正文与元数据。

    返回结构化字典，包含正文（截断到 `max_chars`）、标题、摘要和其他元数据。
    如果 `use_jina` 为 True，会优先使用 r.jina.ai 的纯文本抓取服务。
    """

    return await _web_visit.visit(url=url, max_chars=max_chars, use_jina=use_jina)
# ----------------------------
# content_extract
# ----------------------------

class ContentExtractInput(BaseModel):
    question: str = Field(..., description="要回答的问题/研究目标")
    document_text: str = Field(..., description="长文正文（来自 web_visit/file_parse）")
    max_quotes: int = Field(8, ge=1, le=30, description="最多抽取多少条证据引用")

@tool("content_extract", args_schema=ContentExtractInput)
async def content_extract(question: str, document_text: str, max_quotes: int = 8) -> Dict[str, Any]:
    """从长文本中抽取与 `question` 相关的证据片段并给出简要回答。

    返回结构化字典，通常包含：
    - key_points：简洁中文（<=10条）
    - evidence_quotes：最多 {max_quotes} 条，必须是原文片段（quote），并说明 why_relevant
    - location_hint：来自哪个段落标记 P1/P2...
    """

    return await _content_extractor.extract(question=question, document_text=document_text, max_quotes=max_quotes)
# ----------------------------
# file_parse
# ----------------------------

class FileParseInput(BaseModel):
    file: str = Field(..., description="本地路径或URL")
    extract_images: bool = Field(False, description="是否提取图片（PDF 支持最好）")
    max_chars: int = Field(60000, ge=5000, le=400000, description="返回文本最大字符数")
    timeout_s: int = Field(30, ge=5, le=120, description="下载/读取超时秒数")

@tool("file_parse", args_schema=FileParseInput)
async def file_parse(file: str, extract_images: bool = False, max_chars: int = 60000, timeout_s: int = 30) -> Dict[str, Any]:
    """解析本地或远程文件（支持 PDF/HTML/文本等），返回提取的正文、元数据和可选图片。

    - `file`: 本地路径或可访问 URL
    - `extract_images`: 若为 True，尝试提取并返回图片及其位置
    - `max_chars`: 返回正文的最大字符数限制
    返回示例字段：`text`, `num_pages`, `metadata`, `images`（可选）
    """

    return await _file_parser.parse(file=file, extract_images=extract_images, max_chars=max_chars, timeout_s=timeout_s)

# ----------------------------
# file_parse
# ----------------------------

class ImageParseInput(BaseModel):
    image_url: Optional[str] = Field(None, description="图片URL（与 image_path/image_base64 三选一）")
    image_path: Optional[str] = Field(None, description="本地图片路径（与 image_url/image_base64 三选一）")
    image_base64: Optional[str] = Field(None, description="图片base64（不含 data: 前缀）")
    question: str = Field("请描述图片内容；如有文字请提取；如为图表请解释结构与关键信息。", description="对图片的提问/指令")

@tool("image_parse", args_schema=ImageParseInput)
async def image_parse(
    image_url: Optional[str] = None,
    image_path: Optional[str] = None,
    image_base64: Optional[str] = None,
    question: str = "请描述图片内容；如有文字请提取；如为图表请解释结构与关键信息。",
) -> Dict[str, Any]:
    """解析图片并回答指定问题。

    支持 `image_url` / `image_path` / `image_base64` 三种输入之一。返回结构化结果，包含：
    - `description`: 对图片的整体描述
    - `extracted_text`: 提取到的任何文字（OCR）
    - `analysis`: 若为图表，则给出要点和解释
    - `confidence`: 置信度或提示信息
    """

    return await _image_parser.parse(question=question, image_url=image_url, image_path=image_path, image_base64=image_base64)


TOOLS = [
    github_search_repos,
    github_search_code,
    github_get_repo,
    github_get_readme,
    arxiv_search,
    arxiv_search_by_category,
    arxiv_get_paper,
    web_search,
    web_visit, 
    content_extract,
    file_parse,
    image_parse,
]
