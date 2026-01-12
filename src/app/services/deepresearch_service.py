"""
DeepResearch 服务 - 多轮迭代研究 (IterResearch)

该服务采用 IterResearch 方法：
1. 每轮研究都会根据核心报告和检索到的证据构建一个最小的“工作空间”。
2. ActionDecider 模块决定是继续搜索还是输出最终答案。
3. 证据存储在 Milvus 数据库中，而不是直接传递到上下文。
4. 核心报告会随着轮次迭代而不断完善。
5. 当达到最大轮数或所有待解决问题都已解决时，研究停止。
"""
import logging
import json
import uuid
from typing import Dict, Any, Optional, List, Annotated, TypedDict, Literal,cast,AsyncIterator
from datetime import datetime
from enum import Enum
import os
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from pydantic import SecretStr
from langgraph.checkpoint.memory import MemorySaver
import operator
from langchain_core.runnables import RunnableConfig

from ..tools.langchain_tool import TOOLS
from ..tools.embeddings import get_embedding_provider
from ..memory.writer import get_writer
from ..memory.retriever import get_retriever
from ..utils.hashing import generate_evidence_id, hash_text
from ..utils.tracing import (
    get_tracing_config,
    trace_node,
    trace_llm_call,
    print_workflow_diagram,
)
from ..utils.chunking import get_default_chunker, DocChunk
from .session_cache import SessionCache, RawDoc, ResearchNote, EvidenceQuote
from dotenv import load_dotenv
load_dotenv()
import os
logger = logging.getLogger(__name__)
from ..config import get_settings    
settings = get_settings()
DASHSCOPE_BASE_URL = os.getenv("LLM_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1"
import random


def make_llm(model: str, temperature: float = 0) -> ChatOpenAI:
    """构造大模型调用"""
    key = os.getenv("LLM_API_KEY")
    if not key:
        raise RuntimeError("Missing DASHSCOPE_API_KEY")
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        base_url=DASHSCOPE_BASE_URL,
        api_key=SecretStr(key),
    )


def message_to_text(content: Any) -> str:
    """拼出纯文本字符串"""
    if isinstance(content, str):
        return content
    # LangChain 有时是 list[ str | dict ]
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict):
                # 常见是 {"type": "text", "text": "..."} 之类
                parts.append(str(p.get("text", "")))
        return "".join(parts)
    return str(content)

def _extract_text_from_response(response_obj) -> str:
    """将各种响应格式规范化为纯文本字符串。 

    支持的类型包括：字符串、字节串、列表（包含字符串或类似消息的对象）、字典
    以及具有 `.content` 属性的对象。 
    同时会去除三重反引号代码块以及可选的开头“json”标识符。.
    """
    raw = getattr(response_obj, "content", response_obj)

    # list of things (messages / strings)
    if isinstance(raw, list):
        text = None
        for item in raw:
            if isinstance(item, str):
                text = item
                break
            if hasattr(item, "content") and isinstance(item.content, str):
                text = item.content
                break
        if text is None:
            try:
                text = json.dumps(raw)
            except Exception:
                text = str(raw)

    elif isinstance(raw, dict):
        # common keys that might contain textual content
        for k in ("content", "text", "message", "result"):
            if k in raw and isinstance(raw[k], str):
                text = raw[k]
                break
        else:
            try:
                text = json.dumps(raw)
            except Exception:
                text = str(raw)

    else:
        text = raw if isinstance(raw, str) else str(raw)

    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 2:
            block = parts[1]
            if block.lower().startswith("json"):
                block = block[4:].lstrip()
            text = block.strip()

    return text

class NodeCompleteEvent(TypedDict):
    type: Literal["node_complete"]
    node: str
    state: Dict[str, Any]   # ✅ node 输出 patch
    task_id: str
# ============================================================================
# EvidencePack data structure and helpers
# ============================================================================

class EvidencePack(TypedDict, total=False):
    """
    结构化证据笔记（已弱化，推荐使用 ResearchNote）
    
    保留兼容性，但新代码应使用 session_cache.ResearchNote
    """
    pack_id: str
    source: str  # "arxiv"|"github"|"web"|"local_file"
    ref: str  # arxiv_id / repo full_name / url / file_path
    url: Optional[str]
    title: Optional[str]
    snippet: str  # <= 300 chars（总体摘要）
    key_points: List[str]  # 来自 content_extract
    evidence_quotes: List[Dict[str, Any]]  # 来自 content_extract
    raw_pointer: Optional[str]  # doc_id，用于按需取原文
    confidence: str  # "high"|"medium"|"low"
    relevance: str  # one sentence
    # 多模态
    media_type: str  # "text"|"image"
    image_url: Optional[str]
    # 元信息
    fetched_at: Optional[str]  # ISO timestamp
    target_question: Optional[str]  # 对应的 open_question


def _make_pack_from_item(item: Dict[str, Any], tool: str) -> EvidencePack:
    """
    [DEPRECATED] 把外部工具返回的原始结果 item 转换成一个紧凑的 EvidencePack
    
    ⚠️ 该函数仅用于生成"线索 pack"（lead pack），真正的 EvidencePack 应该由
    content_extract 从 doc.text 生成。
    
    新代码应该使用 materialize_round_cache_node 中的 hit→doc→note 流程。
    """
    # Identify source and extract fields
    if "arxiv" in tool.lower() or "arxiv_id" in item:
        source = "arxiv"
        ref = item.get("arxiv_id") or item.get("id", "")
        title = item.get("title", "")
        snippet = (item.get("summary") or "")[:300]
        url = item.get("abs_url") or item.get("url")
    elif "github" in tool.lower() or "full_name" in item:
        source = "github"
        ref = item.get("full_name") or item.get("name", "")
        title = item.get("name") or item.get("full_name")
        snippet = (item.get("description") or "")[:300]
        url = item.get("url")
    else:
        source = tool or "web"
        ref = item.get("url") or item.get("id", "")
        title = item.get("title") or item.get("name", "")
        snippet = (item.get("snippet") or item.get("content") or "")[:300]
        url = item.get("url")
    
    # Extract key points (simple heuristic: split into sentences, take first 5)
    raw_text = (item.get("summary") or item.get("description") or snippet)[:1000]
    sentences = [s.strip() for s in raw_text.split(". ") if s.strip()]
    key_points = [(s[:120] + ("..." if len(s) > 120 else "")) for s in sentences[:5]]
    
    pack_id = generate_evidence_id(source, ref or title or url or "", hash_text(snippet))
    
    pack: EvidencePack = {
        "pack_id": pack_id,
        "source": source,
        "ref": ref,
        "url": url,
        "title": title,
        "snippet": snippet,
        "key_points": key_points,
        "relevance": "",
        "confidence": "medium",
        "raw_pointer": None,
    }
    return pack


# ============================================================================
# State Definition (TypedDict)
# ============================================================================

class ResearchState(TypedDict):
    """
    LangGraph State for DeepResearch workflow
    
    状态字段会在节点之间传递，节点可以读取和更新状态。
    Annotated 用于定义状态更新策略（add/replace）。
    """
    # 核心字段
    task_id: str
    query: str
    round_id: int

    # 任务描述
    task_spec: Dict[str, Any]

    # 中心报告
    summary: str
    findings: Annotated[List[str], operator.add]  # append模式
    open_questions: List[str]  # replace模式
    evidence_ids: Annotated[List[str], operator.add]

    # === Session Cache（新增）===
    session_cache: Dict[str, Any]  # SessionCache.to_state_dict() 序列化结果
    round_doc_ids: List[str]  # 本轮新增 doc_id（replace）
    all_notes: Annotated[List[ResearchNote], operator.add]  # 累计证据笔记
    round_notes: List[ResearchNote]  # 本轮证据笔记（replace）
    persist_queue: Annotated[List[str], operator.add]  # 待持久化的 pack_id/doc_id
    
    # Evidence packs (保留兼容，但推荐使用 all_notes/round_notes)
    evidence_packs: Annotated[List[EvidencePack], operator.add]  # new packs per round
    new_evidence_packs: List[EvidencePack]   # replace

    # Workspace 上下文
    workspace: str

    # 执行工具
    tool_queries: List[str]
    tool_results: List[Dict[str, Any]]

    # 工作流控制和决定
    next_action: Literal["continue", "answer", "stop"]
    stop_reason: Optional[str]

    # 返回给LLM的message
    messages: Annotated[List[Any], add_messages]

    # output
    final_answer: str
    citations: List[Dict[str, Any]]

    # configuration
    max_rounds: int
    max_papers: int
    max_repos: int
    top_k: int

    # Tool loop / workspace limits (for progressive disclosure)
    max_tool_steps_per_round: int  # default 3-5
    max_workspace_chars: int  # default 6000-10000
    max_packs_per_step: int  # default 3-5
    max_total_packs_per_round: int  # default 10-20
    
    # Trace for UI
    trace: Annotated[List[Dict[str, Any]], operator.add]

class ResearchStateUpdate(TypedDict, total=False):
    task_spec: Dict[str, Any]
    open_questions: List[str]
    trace: List[Dict[str, Any]]
    workspace: str
    next_action: Literal["continue", "answer", "stop"]
    stop_reason: Optional[str]
    tool_queries: List[str]
    tool_results: List[Dict[str, Any]]
    evidence_ids: List[str]
    summary: str
    findings: List[str]
    final_answer: str
    citations: List[Dict[str, Any]]
    round_id: int
    messages: List[Any]
    evidence_packs: List[EvidencePack]  # new packs per round
    new_evidence_packs: List[EvidencePack]   # replace
    # Session Cache
    session_cache: Dict[str, Any]
    round_doc_ids: List[str]
    all_notes: List[ResearchNote]
    round_notes: List[ResearchNote]
    persist_queue: List[str]

# ============================================================================
# Graph Nodes (业务逻辑)
# ============================================================================

@trace_node("parse_task")
async def parse_task_node(state: ResearchState) -> ResearchStateUpdate:
    """
    Node: 解析用户查询，提取任务规格
    
    输入: query
    输出: task_spec, open_questions
    """
    logger.info(f"[Task {state['task_id']}] Parsing task spec")
    logger.debug(f"📥 输入 query: {state['query'][:100]}...")
    
    llm = make_llm(model="qwen3-max")
    
    # 动态获取当前日期信息
    current_date = datetime.now().strftime("%Y年%m月%d日")
    current_year = datetime.now().year
    current_month = datetime.now().month
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""你是一个研究任务规划助手。当前日期：{current_date}（{current_year}年{current_month}月）。

从用户查询中智能分析并提取研究任务规格，返回 JSON 对象包含：
- topic: 主要研究主题（中文）
- specific_questions: 具体要回答的问题列表（中文）
  * 如果原问题已经足够具体，保持为单个问题
  * 如果原问题较宽泛，分解为2-5个可独立回答的子问题
  * 分解时确保子问题互不重叠、覆盖原问题的核心维度
- output_type: 期望的输出格式（summary/comparison/analysis/list/report等）
- time_range: 时间范围对象（基于query语义智能推断），包含：
  * description: 时间描述（如"过去五年"、"2023-2024"、"不限"）
  * start_year: 起始年份（数字或null）
  * end_year: 结束年份（数字或null，默认当前年份{current_year}）
  * priority: 时效性要求（"latest"=优先最新/"all"=不限时间/"specific"=特定时间段）
- constraints: 提到的任何约束或偏好（中文列表）

**时间范围推断指南**（根据query自行判断）：
- "最近"、"近期"、"当前" → 最近1-3个月，priority="latest"
- "过去N年" → start_year={current_year}-N, end_year={current_year}
- "YYYY-YYYY" → 提取具体年份范围
- "进展"、"趋势"、"发展" → 暗示需要时间跨度，默认过去3-5年
- 无时间词 → start_year=null, end_year=null, priority="all"

**任务分解指南**：
- 简单查询（如"什么是RAG"） → 不分解，保持1个问题
- 综述型查询（如"推荐系统进展"） → 分解为方法/应用/挑战等2-4个维度
- 对比型查询（如"A vs B"） → 分解为各自特点+差异对比

只返回有效的 JSON，不要其他文字。"""),
        ("human", "{query}"),
    ])
    
    
    chain = prompt | llm
    response = await chain.ainvoke({"query": state["query"]})
    
    import json
    try:
        
        raw = message_to_text(response.content)
        content = raw.strip()

        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        task_spec = json.loads(content)
    except:
        task_spec = {
            "topic": state["query"],
            "specific_questions": [state["query"]],
            "output_type": "summary",
            "time_range": {
                "description": "不限",
                "start_year": None,
                "end_year": None,
                "priority": "all"
            },
            "constraints": [],
        }
    
    return {
        "task_spec": task_spec,
        "open_questions": task_spec.get("specific_questions", [state["query"]]),
        "trace": [{
            "round": 0,
            "stage": "parse_task",
            "timestamp": datetime.now().isoformat(),
            "message": "Task specification parsed",
            "data": task_spec,
        }],
    }

@trace_node("build_workspace")
async def build_workspace_node(state: ResearchState) -> ResearchStateUpdate:
    """
    Node: 构建工作空间（重构版：multi-query recall + notes优先）
    
    策略：
    1. 优先使用 all_notes（本次会话积累）
    2. 多查询召回：query + open_questions[:3] + summary[:300]
    3. 去重合并，展示紧凑的 workspace
    
    输入: query, summary, findings, open_questions, all_notes, round_id
    输出: workspace
    """
    logger.info(f"[Task {state['task_id']}] Building workspace for round {state['round_id']}")
    logger.debug(f"📥 All notes: {len(state.get('all_notes', []))}, Findings: {len(state.get('findings', []))}")
    
    from ..services.session_cache import safe_note_pack_id
    
    embedding_provider = get_embedding_provider()
    retriever = get_retriever()
    
    # === Multi-query recall ===
    queries_to_embed = [state["query"]]
    
    # 添加前 3 个待解决问题
    open_questions = state.get("open_questions", [])
    for q in open_questions[:3]:
        if len(q) > 10:  # 过滤太短的问题
            queries_to_embed.append(q)
    
    # 添加摘要前 300 字
    summary = state.get("summary", "")
    if summary and len(summary) > 50:
        queries_to_embed.append(summary[:300])
    
    logger.debug(f"Multi-query recall with {len(queries_to_embed)} queries")
    
    # 对每个查询进行召回
    all_recalled_ids = set()
    recalled_evidence_list = []
    max_per_query = max(5, state["top_k"] // len(queries_to_embed))
    
    for query_text in queries_to_embed:
        try:
            query_embedding = await embedding_provider.embed(query_text)
            recalled = retriever.search_evidence(
                query_embedding=query_embedding,
                top_k=max_per_query,
                task_id=state["task_id"],
            )
            for ev in recalled:
                ev_id = ev.get("id", "")
                if ev_id and ev_id not in all_recalled_ids:
                    all_recalled_ids.add(ev_id)
                    recalled_evidence_list.append(ev)
        except Exception as e:
            logger.warning(f"Recall failed for query '{query_text[:50]}': {e}")
            continue
    
    logger.info(f"Multi-query recalled {len(recalled_evidence_list)} unique evidence items")
    
    # 转换 recalled evidence 为 EvidencePack（仅用于向后兼容）
    recalled_packs: List[EvidencePack] = []
    max_packs = state.get("max_total_packs_per_round", 20)
    for ev in recalled_evidence_list[:max_packs]:
        pack: EvidencePack = {
            "pack_id": ev.get("id", ""),
            "source": ev.get("source", ""),
            "ref": ev.get("ref", ""),
            "url": ev.get("metadata", {}).get("url"),
            "title": ev.get("metadata", {}).get("title"),
            "snippet": (ev.get("snippet") or "")[:300],
            "key_points": [],
            "relevance": "",
            "confidence": "medium",
            "raw_pointer": None,
        }
        recalled_packs.append(pack)

    # === 优先使用 all_notes（本轮会话积累）===
    all_notes = state.get("all_notes", [])
    
    # Merge notes + recalled packs (notes 优先)
    seen_ids = set()
    unique_items: List[Any] = []
    
    # 先添加 all_notes
    for note in all_notes:
        note_id = safe_note_pack_id(note)
        if note_id not in seen_ids:
            seen_ids.add(note_id)
            unique_items.append(("note", note))
    
    # 再添加 recalled packs（去重）
    for pack in recalled_packs:
        pack_id = pack.get("pack_id", "")
        if pack_id and pack_id not in seen_ids:
            seen_ids.add(pack_id)
            unique_items.append(("pack", pack))

    
    # === 构建 workspace（紧凑展示）===
    current_date = datetime.now().strftime("%Y年%m月%d日")
    
    workspace_parts = [
        f"## 研究问题\n{state['query']}",
        f"\n## 当前时间\n{current_date}",
        f"\n## 当前总结\n{state.get('summary') or '暂无总结。'}",
    ]
    
    # 任务规格（截断）
    try:
        ts = state.get("task_spec", {}) or {}
        task_spec_json = json.dumps(ts, ensure_ascii=False)
    except Exception:
        task_spec_json = str(state.get("task_spec", {}))
    if task_spec_json:
        task_spec_snippet = task_spec_json if len(task_spec_json) <= 1000 else task_spec_json[:1000] + " ... (已截断)"
        workspace_parts.append(f"\n## 任务规格（已截断）\n{task_spec_snippet}")
    
    if state.get("open_questions"):
        workspace_parts.append(
            f"\n## 待解决问题\n" + "\n".join(f"- {q}" for q in state["open_questions"])
        )
    
    if state.get("findings"):
        workspace_parts.append(
            f"\n## 最近发现\n" + "\n".join(f"- {f}" for f in state["findings"][-5:])
        )
    
    # 证据展示（notes 优先，紧凑格式）
    if unique_items:
        evidence_summary = "\n## 证据（紧凑）\n"
        from ..services.session_cache import safe_note_evidence_quotes
        
        for item_type, item in unique_items[:max_packs]:
            if item_type == "note":
                # ResearchNote：显示 key_points + evidence_quotes
                pack_id = safe_note_pack_id(item)
                source = item.get("source", "")
                ref = item.get("ref", "")
                snippet = item.get("snippet", "")
                key_points = item.get("key_points", [])[:3]
                media_type = item.get("media_type", "text")
                
                if media_type == 'image':
                    image_url = item.get('image_url', '')
                    evidence_summary += f"- [IMAGE] [{pack_id}] {snippet[:120]}\n"
                    evidence_summary += f"  🖼️ URL: {image_url}\n"
                else:
                    evidence_summary += f"- [{pack_id}] {source}:{ref} - {snippet[:120]}\n"
                    for kp in key_points:
                        evidence_summary += f"  * {kp}\n"
            
            elif item_type == "pack":
                # EvidencePack（向后兼容）
                p = item
                media_type = p.get('media_type', 'text')
                
                if media_type == 'image':
                    image_url = p.get('image_url', '')
                    evidence_summary += f"- [IMAGE] [{p.get('pack_id', '')}] {p.get('snippet', '')[:120]}\n"
                    evidence_summary += f"  🖼️ URL: {image_url}\n"
                else:
                    evidence_summary += f"- [{p.get('pack_id', '')}] {p.get('source', '')}:{p.get('ref', '')} - {p.get('snippet', '')[:120]}\n"
                
                for kp in p.get("key_points", [])[:3]:
                    evidence_summary += f"  * {kp}\n"
        
        workspace_parts.append(evidence_summary)
    
    workspace = "\n".join(workspace_parts)
    
    # Truncate to max_workspace_chars if needed
    max_chars = state.get("max_workspace_chars", 8000)
    if len(workspace) > max_chars:
        workspace = workspace[:max_chars] + "\n... (truncated)"
    
    return {
        "workspace": workspace,
        "trace": [{
            "round": state["round_id"],
            "stage": "build_workspace",
            "timestamp": datetime.now().isoformat(),
            "message": f"Workspace built: {len(recalled_packs)} recalled + {len(existing_packs)} existing packs",
        }],
    }

@trace_node("tool_loop")
async def tool_loop_node(state:ResearchState)-> ResearchStateUpdate:
    """工具循环节点：整合了决定行动和执行工具的功能。
    大型语言模型自主决定是否调用工具（多步循环）。
    工具输出被压缩成证据包（渐进式展示）。
    输入: workspace, open_questions, round_id, max_rounds, budget params
    输出: tool_results, evidence_packs, next_action, trace
    """
    logger.info(f"[Task {state['task_id']}] Entering tool loop for round {state['round_id']}")
    logger.debug(f"📥 Open questions: {len(state.get('open_questions', []))}, Workspace size: {len(state.get('workspace', ''))} chars")
    llm = make_llm(model="qwen3-max")
    llm_with_tools = llm.bind_tools(TOOLS)

    new_tool_results: List[Dict[str, Any]] = []
    new_packs: List[EvidencePack] = []  # Store as Dict to match state typing

    workspace = state.get("workspace", "")
    open_questions = state.get("open_questions", [])
    
    max_steps = state.get("max_tool_steps_per_round", 3)
    max_per_step = state.get("max_packs_per_step", 3)
    max_total = state.get("max_total_packs_per_round", 20)
    max_chars = state.get("max_workspace_chars", 8000)

    next_action = "continue"
    reason = None
    for step in range(max_steps):
        remaining_budget = max_total - len(new_packs)
        
        # 获取时间范围信息
        task_spec = state.get("task_spec", {})
        time_range = task_spec.get("time_range", {})
        time_desc = time_range.get("description", "不限")
        start_year = time_range.get("start_year")
        end_year = time_range.get("end_year")
        priority = time_range.get("priority", "all")
        
        time_filter_hint = ""
        if start_year and end_year:
            time_filter_hint = f"\n**时间范围：{start_year}-{end_year}年（{time_desc}）**"
            if priority == "latest":
                time_filter_hint += "\n⚠️ 优先搜索最新内容（按时间降序）"
        
        # 动态获取当前日期
        current_date = datetime.now().strftime("%Y年%m月%d日")
        
        # System prompt with budget constraints
        system_prompt = f"""你是一个研究助手，可以调用工具收集证据。当前日期：{current_date}。{time_filter_hint}

**当前工作空间（精简）：**
{workspace}

**待解决问题：**
{chr(10).join(f"{i+1}. {q}" for i, q in enumerate(open_questions))}

**预算：** 步骤 {step+1}/{max_steps}，证据包 {len(new_packs)}/{max_total}，剩余 {remaining_budget}，工作空间限制 {max_chars} 字符

**可用工具**：
- arxiv_search: 搜索学术论文
- github_search_repos: 搜索GitHub仓库
- github_search_code: 搜索代码片段
- web_search: 网络搜索（获取相关网页链接）
- web_visit: 访问具体网页并提取正文内容
- content_extract: 从长文本中提取与问题相关的证据

**工具使用规则：**
- 仅在需要外部证据回答待解决问题时调用工具
- 每次调用工具前，明确："我要解决待解决问题中的第N个问题"
- 推荐流程：web_search找到相关链接 → web_visit访问页面 → content_extract提取证据
- 对于学术问题优先使用 arxiv_search
- 对于代码/技术实现优先使用 github_search
- 对于最新资讯/博客优先使用 web_search + web_visit
- 关注高质量来源（最新论文、热门仓库、权威网站）
- 搜索时使用中文或英文关键词
{f"- 时间过滤：优先搜索 {start_year}-{end_year} 年的内容" if start_year else ""}

**输出规则（结构化）：**
- 如果将调用工具：先输出 JSON 行动计划，再直接调用
  JSON 格式：{{{{"action": "TOOL", "tool_name": "工具名", "tool_args": {{}}, "target_question_id": N, "rationale": "解释为什么调用"}}}}
- 如果不调用任何工具：只返回有效 JSON：
  {{{{"action": "STOP", "reason": "说明为什么停止"}}}}
"""
        messages = [HumanMessage(content=system_prompt)]
        response = await llm_with_tools.ainvoke(messages)
        
        # === 解析结构化 action ===
        action_plan = None
        if hasattr(response, "content") and response.content:
            try:
                # 尝试解析 JSON
                content_str = _extract_text_from_response(response)
                # 查找 JSON 块
                import re
                json_match = re.search(r'\{.*?"action".*?\}', content_str, re.DOTALL)
                if json_match:
                    action_plan = json.loads(json_match.group(0))
                    logger.debug(f"Parsed action plan: {action_plan}")
            except Exception as e:
                logger.debug(f"Failed to parse action JSON: {e}")
        
        if hasattr(response,"tool_calls") and response.tool_calls:
            # 确定目标问题（优先使用 action_plan，否则用简单正则）
            target_question = ""
            if action_plan and "target_question_id" in action_plan:
                idx = action_plan["target_question_id"] - 1
                if 0 <= idx < len(open_questions):
                    target_question = open_questions[idx]
            else:
                # Fallback: 简单解析
                if hasattr(response, "content") and response.content:
                    import re
                    match = re.search(r'(?:解决|回答|处理).*?第(\d+).*?问题', str(response.content))
                    if match and open_questions:
                        idx = int(match.group(1)) - 1
                        if 0 <= idx < len(open_questions):
                            target_question = open_questions[idx]
            
            tool_node = ToolNode(TOOLS)
            tool_messages = await tool_node.ainvoke({"messages": [response]})
            for msg in tool_messages.get("messages", []):
                    if hasattr(msg, "content"):
                        try:
                            res = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
                        except Exception:
                            res = msg.content
                        
                        tool_name = msg.name if hasattr(msg, "name") else "unknown"
                        
                        new_tool_results.append({
                            "tool": tool_name,
                            "query": getattr(response, "content", ""),
                            "result": res,
                            "target_question": target_question or open_questions[0] if open_questions else "",  # 记录目标问题
                        })
                        # [PROGRESSIVE DISCLOSURE] Convert items to compact EvidencePacks
                        if isinstance(res, list):
                            for item in res[:max_per_step]:
                                if isinstance(item, dict):
                                    pack = _make_pack_from_item(item, tool_name)
                                    new_packs.append(pack)
                                    
                                    if len(new_packs) >= max_total:
                                        break
            show_packs = new_packs[-max_total:]
            evidence_summary = "\n## 证据包（本步骤）\n"
            for p in show_packs:
                evidence_summary += f"- [{p.get('pack_id', '')}] {p.get('source', '')}:{p.get('ref', '')} - {p.get('snippet', '')[:120]}\n"
                for kp in p.get("key_points", [])[:3]:
                    evidence_summary += f"  * {kp}\n"           
            workspace = f"""## 研究问题
    {state.get('query')}

    ## 当前总结
    {state.get('summary') or '暂无总结。'}

    ## 待解决问题
    {chr(10).join(f"- {q}" for q in open_questions)}

    {evidence_summary}"""
        
            if len(workspace) > max_chars:
                workspace = workspace[:max_chars] + "\n... (truncated)"
            if len(new_packs) >= max_total:
                break
                
            continue  # Next step
        
        # No tool calls: expect JSON control response
        content = _extract_text_from_response(response)
        try:
            control = json.loads(content)
        except Exception:
            control = {}
        
        if control.get("type") == "ANSWER":
            next_action = "answer"
            reason = control.get("reason")
            break
        elif control.get("type") == "STOP":
            next_action = "stop"
            reason = control.get("reason")
            break
        # Otherwise keep looping
    
    trace_msg = f"Tool loop finished: {step+1} steps, {len(new_packs)} packs, next_action={next_action}"
    
    return {
        "tool_results": new_tool_results,
        "evidence_packs": new_packs,
        "new_evidence_packs": new_packs,    # 本轮专用（replace）
        "next_action": next_action,
        "trace": [{
            "round": state["round_id"],
            "stage": "tool_loop",
            "timestamp": datetime.now().isoformat(),
            "message": trace_msg,
            "data": {"reason": reason, "steps": step+1, "packs": len(new_packs)},
        }],
    }


@trace_node("materialize_round_cache")
async def materialize_round_cache_node(state: ResearchState) -> ResearchStateUpdate:
    """
    Node: 物化本轮 Session Cache（hit→doc→note 升级）
    
    关键功能：
    1. 将 tool_results 中的 hit 结果升级为 doc（调用 web_visit/get_readme 等）
    2. 对每个 doc 进行分块（chunking）并存入 session_docs
    3. 调用 content_extract 生成结构化证据笔记（ResearchNote）
    4. 处理图片解析链路（file_parse images → image_parse）
    5. 更新 workspace 展示
    
    输入: tool_results, session_cache, open_questions
    输出: session_cache, round_doc_ids, round_notes, all_notes, workspace
    """
    logger.info(f"[Task {state['task_id']}] Materializing round cache for round {state['round_id']}")
    logger.debug(f"📥 Tool results: {len(state.get('tool_results', []))}")
    
    from ..tools.langchain_tool import TOOLS
    from ..tools.content_extract_tool import ContentExtractClient
    from langchain_core.runnables import RunnableConfig
    
    # 初始化或恢复 SessionCache
    cache_data = state.get("session_cache", {})
    cache = SessionCache.from_state_dict(cache_data) if cache_data else SessionCache()
    
    chunker = get_default_chunker()
    embedding_provider = get_embedding_provider()
    
    # 获取当前待解决问题
    open_questions = state.get("open_questions", [state["query"]])
    goal_text = "\n".join(f"{i+1}. {q}" for i, q in enumerate(open_questions[:5]))
    
    # 准备 content_extract client
    llm = make_llm(model="qwen3-max", temperature=0)
    async def _generate(prompt: str) -> str:
        resp = await llm.ainvoke([HumanMessage(content=prompt)])
        return _extract_text_from_response(resp)
    
    content_extractor = ContentExtractClient(generate=_generate)
    
    # 准备 image_parse（如需要）
    from ..tools.image_parse_tool import get_image_parser
    image_parser = get_image_parser()
    
    round_doc_ids: List[str] = []
    round_notes: List[ResearchNote] = []
    
    # === Phase 1: 解析 tool_results，升级 hit → doc ===
    docs_to_process: List[Dict[str, Any]] = []
    
    for result in state["tool_results"]:
        tool_name = result.get("tool", "")
        target_question = result.get("target_question", "")
        data = result.get("result", [])
        
        if not isinstance(data, list):
            continue
        
        for item in data:
            if not isinstance(item, dict):
                continue
            
            doc_candidate: Optional[Dict[str, Any]] = None
            
            # Case 0: 本地知识库 hit（占位）
            if "kb_id" in item and "doc_id" in item:
                doc_candidate = {
                    "type": "kb_hit",
                    "kb_id": item["kb_id"],
                    "doc_id": item["doc_id"],
                    "title": item.get("title", ""),
                    "snippet": item.get("snippet", ""),
                    "target_question": target_question
                }
            
            # Case 1: web_search hit → 需要 web_visit
            if tool_name == "web_search" and "url" in item:
                # 标记为需要升级的 hit
                doc_candidate = {
                    "type": "web_hit",
                    "url": item["url"],
                    "title": item.get("title", ""),
                    "snippet": item.get("snippet", ""),
                    "target_question": target_question
                }
            
            # Case 2: arxiv_search hit → 使用摘要作为 doc
            elif "arxiv_id" in item:
                doc_candidate = {
                    "type": "arxiv_doc",
                    "doc_id": f"arxiv_{item['arxiv_id']}",
                    "source": "arxiv",
                    "ref": item["arxiv_id"],
                    "url": item.get("abs_url") or item.get("url"),
                    "title": item.get("title", ""),
                    "text": item.get("summary", ""),
                    "origin": "web",
                    "target_question": target_question
                }
            
            # Case 3: github_search_repos hit → 需要 get_readme
            elif "full_name" in item:
                doc_candidate = {
                    "type": "github_hit",
                    "full_name": item["full_name"],
                    "url": item.get("url", ""),
                    "title": item.get("name", ""),
                    "description": item.get("description", ""),
                    "target_question": target_question
                }
            
            # Case 4: web_visit 已经返回 doc
            elif "text" in item and "url" in item and len(item.get("text", "")) > 200:
                doc_candidate = {
                    "type": "web_doc",
                    "doc_id": f"web_{hash_text(item['url'])}",
                    "source": "web",
                    "ref": item["url"],
                    "url": item["url"],
                    "title": item.get("title", ""),
                    "text": item["text"],
                    "origin": "web",
                    "target_question": target_question
                }
            
            if doc_candidate:
                docs_to_process.append(doc_candidate)
    
    logger.info(f"Found {len(docs_to_process)} doc candidates to materialize")
    
    # === Phase 2: 升级 hit → doc（调用真实工具）===
    materialized_docs: List[RawDoc] = []
    
    # 准备工具映射
    tool_map = {tool.name: tool for tool in TOOLS}
    runnable_config = RunnableConfig(
        configurable={"task_id": state["task_id"]},
        run_name=f"materialize_tools_round_{state['round_id']}"
    )
    
    for doc_info in docs_to_process[:10]:  # 限制数量避免超时
        try:
            doc_id = ""
            doc_text = ""
            doc_source = doc_info.get("source", "web")
            doc_ref = doc_info.get("ref", "")
            doc_url = doc_info.get("url")
            doc_title = doc_info.get("title", "")
            doc_origin = doc_info.get("origin", "web")
            degraded = False
            
            if doc_info["type"] == "kb_hit":
                # 调用本地 KB fetch（占位）
                logger.debug(f"Calling local_kb_fetch for: {doc_info['doc_id']}")
                try:
                    from ..tools.local_kb_tool import local_kb_fetch
                    result = await local_kb_fetch(
                        doc_id=doc_info["doc_id"],
                        kb_id=doc_info.get("kb_id")
                    )
                    doc_id = f"kb_{doc_info['kb_id']}_{doc_info['doc_id']}"
                    doc_text = result["text"]
                    doc_source = "local_kb"
                    doc_ref = doc_info["doc_id"]
                    doc_title = result.get("title", doc_info["title"])
                    doc_origin = "local"  # 本地文档优先级高
                except NotImplementedError:
                    logger.info(f"Local KB not implemented, using snippet for {doc_info['doc_id']}")
                    doc_id = f"kb_{doc_info.get('kb_id', 'unknown')}_{doc_info['doc_id']}"
                    doc_text = f"[Local KB Hit]\n{doc_info['title']}\n\n{doc_info['snippet']}"
                    doc_source = "local_kb"
                    doc_ref = doc_info["doc_id"]
                    doc_origin = "local"
                    degraded = True
                except Exception as e:
                    logger.warning(f"local_kb_fetch failed: {e}")
                    continue
            
            elif doc_info["type"] == "web_hit":
                # 调用真实的 web_visit 工具
                logger.debug(f"Calling web_visit for: {doc_info['url']}")
                try:
                    if "web_visit" in tool_map:
                        result = await tool_map["web_visit"].ainvoke(
                            {"url": doc_info["url"]},
                            config=runnable_config
                        )
                        if isinstance(result, dict) and "text" in result:
                            doc_id = f"web_{hash_text(doc_info['url'])}"
                            doc_text = result["text"]
                            doc_source = "web"
                            doc_ref = doc_info["url"]
                            doc_title = result.get("title", doc_info["title"])
                        else:
                            raise ValueError("Invalid web_visit result")
                    else:
                        raise ValueError("web_visit tool not available")
                except Exception as e:
                    logger.warning(f"web_visit failed for {doc_info['url']}: {e}, using snippet")
                    doc_id = f"web_{hash_text(doc_info['url'])}"
                    doc_text = f"[Title] {doc_info['title']}\n\n{doc_info['snippet']}"
                    doc_source = "web"
                    doc_ref = doc_info["url"]
                    degraded = True
                
            elif doc_info["type"] == "github_hit":
                # 调用 github_get_readme 工具
                logger.debug(f"Calling github_get_readme for: {doc_info['full_name']}")
                try:
                    if "github_get_readme" in tool_map:
                        result = await tool_map["github_get_readme"].ainvoke(
                            {"repo": doc_info["full_name"]},
                            config=runnable_config
                        )
                        if isinstance(result, dict) and "content" in result:
                            doc_id = f"github_{hash_text(doc_info['full_name'])}"
                            doc_text = result["content"]
                            doc_source = "github"
                            doc_ref = doc_info["full_name"]
                            doc_title = doc_info["title"]
                        else:
                            raise ValueError("Invalid github_get_readme result")
                    else:
                        raise ValueError("github_get_readme tool not available")
                except Exception as e:
                    logger.warning(f"github_get_readme failed for {doc_info['full_name']}: {e}, using description")
                    doc_id = f"github_{hash_text(doc_info['full_name'])}"
                    doc_text = f"[Repo] {doc_info['full_name']}\n\n{doc_info['description']}"
                    doc_source = "github"
                    doc_ref = doc_info["full_name"]
                    degraded = True
                
            elif doc_info["type"] in ("arxiv_doc", "web_doc"):
                # 已经是 doc
                doc_id = doc_info["doc_id"]
                doc_text = doc_info["text"]
                doc_source = doc_info.get("source", "web")
                doc_ref = doc_info.get("ref", "")
                doc_url = doc_info.get("url")
                doc_title = doc_info.get("title", "")
            else:
                continue
            
            if not doc_text.strip():
                logger.warning(f"Empty doc text for {doc_id}, skipping")
                continue
            
            # 分块
            chunks = chunker.split_text_to_chunks(doc_text, doc_id)
            
            # 创建 RawDoc
            raw_doc = RawDoc(
                doc_id=doc_id,
                source=doc_source,
                ref=doc_ref,
                url=doc_url,
                title=doc_title,
                fetched_at=datetime.now().isoformat(),
                text=doc_text,
                chunks=chunks,
                metadata={"target_question": doc_info.get("target_question", "")},
                origin=doc_origin,
                degraded=degraded
            )
            
            cache.add_doc(raw_doc)
            round_doc_ids.append(doc_id)
            materialized_docs.append(raw_doc)
            
        except Exception as e:
            logger.error(f"Failed to materialize doc: {e}")
            continue
    
    logger.info(f"Materialized {len(materialized_docs)} docs with total {sum(len(d.chunks) for d in materialized_docs)} chunks")
    
    # === Phase 3: 对每个 doc 调用 content_extract 生成 ResearchNote ===
    for raw_doc in materialized_docs:
        try:
            # 调用 content_extract
            extract_result = await content_extractor.extract(
                question=goal_text,
                document_text=raw_doc.text[:10000],  # 限制长度
                max_quotes=5
            )
            
            if "error" in extract_result:
                logger.warning(f"content_extract failed for {raw_doc.doc_id}: {extract_result['error']}")
                # Fallback: 使用简单摘要
                key_points = [raw_doc.text[:200]]
                evidence_quotes = []
            else:
                key_points = extract_result.get("key_points", [])
                evidence_quotes = extract_result.get("evidence_quotes", [])
            
            # 生成 snippet
            snippet = " | ".join(key_points[:3])[:300]
            
            # 创建 ResearchNote
            pack_id = generate_evidence_id(raw_doc.source, raw_doc.ref, hash_text(snippet))
            note: ResearchNote = {
                "pack_id": pack_id,
                "source": raw_doc.source,
                "ref": raw_doc.ref,
                "url": raw_doc.url,
                "title": raw_doc.title,
                "snippet": snippet,
                "key_points": key_points[:10],
                "evidence_quotes": evidence_quotes[:8],
                "raw_pointer": raw_doc.doc_id,
                "confidence": "medium",
                "relevance": "",
                "media_type": "text",
                "image_url": None,
                "fetched_at": raw_doc.fetched_at,
                "target_question": raw_doc.metadata.get("target_question")
            }
            
            cache.add_note(note)
            round_notes.append(note)
            
        except Exception as e:
            logger.error(f"Failed to generate note for {raw_doc.doc_id}: {e}")
            continue
    
    logger.info(f"Generated {len(round_notes)} research notes")
    
    # === Phase 4: 处理图片（如 tool_results 中有 images）===
    # 图片解析链路：file_parse(extract_images=True) → images_base64 → image_parse
    for result in state["tool_results"]:
        data = result.get("result", [])
        if isinstance(data, dict) and "images_base64" in data:
            images_b64 = data["images_base64"][:3]  # 限制数量
            for idx, img_b64 in enumerate(images_b64):
                try:
                    data_url = f"data:image/png;base64,{img_b64}"
                    question = "请详细描述这张图片的内容，包括主要对象、文字、图表类型等。"
                    
                    img_result = await image_parser.parse(question=question, image_url=data_url)
                    
                    description = img_result.get("description", "")
                    extracted_text = img_result.get("extracted_text", "")
                    analysis = img_result.get("analysis", "")
                    
                    # 组合为文本
                    img_text = f"{description}\n\n提取文字：{extracted_text}\n\n分析：{analysis}"
                    
                    # 创建图片 doc
                    img_doc_id = f"image_{hash_text(img_b64[:100])}"
                    img_doc = RawDoc(
                        doc_id=img_doc_id,
                        source="local_file",
                        ref=f"image_{idx}",
                        url=data_url,
                        title=f"Image {idx + 1}",
                        fetched_at=datetime.now().isoformat(),
                        text=img_text,
                        chunks=[],  # 图片不分块
                        metadata={"image_index": idx},
                        origin="local"
                    )
                    
                    cache.add_doc(img_doc)
                    round_doc_ids.append(img_doc_id)
                    
                    # 创建图片 note
                    img_pack_id = generate_evidence_id("image", img_doc_id, hash_text(description))
                    img_note: ResearchNote = {
                        "pack_id": img_pack_id,
                        "source": "local_file",
                        "ref": f"image_{idx}",
                        "url": data_url,
                        "title": f"Image {idx + 1}",
                        "snippet": description[:300],
                        "key_points": [description[:200]],
                        "evidence_quotes": [],
                        "raw_pointer": img_doc_id,
                        "confidence": "medium",
                        "relevance": "",
                        "media_type": "image",
                        "image_url": data_url,
                        "fetched_at": img_doc.fetched_at,
                        "target_question": None
                    }
                    
                    cache.add_note(img_note)
                    round_notes.append(img_note)
                    
                except Exception as e:
                    logger.error(f"Failed to parse image {idx}: {e}")
                    continue
    
    # === Phase 5: 更新 workspace（只展示紧凑笔记，不包含原文）===
    current_date = datetime.now().strftime("%Y年%m月%d日")
    
    workspace_parts = [
        f"## 研究问题\n{state['query']}",
        f"\n## 当前时间\n{current_date}",
        f"\n## 当前总结\n{state.get('summary') or '暂无总结。'}",
    ]
    
    if state.get("open_questions"):
        workspace_parts.append(
            f"\n## 待解决问题\n" + "\n".join(f"{i+1}. {q}" for i, q in enumerate(state["open_questions"]))
        )
    
    if state.get("findings"):
        workspace_parts.append(
            f"\n## 最近发现\n" + "\n".join(f"- {f}" for f in state["findings"][-5:])
        )
    
    # 证据包展示（紧凑）
    all_notes = list(state.get("all_notes", [])) + round_notes
    if round_notes:
        evidence_summary = "\n## 本轮证据笔记（紧凑）\n"
        for note in round_notes[:10]:
            if note.get("media_type") == "image":
                evidence_summary += f"- [IMAGE] [{note['pack_id']}] {note['snippet'][:120]}\n"
                evidence_summary += f"  🖼️ URL: {note.get('image_url', '')[:50]}...\n"
            else:
                evidence_summary += f"- [{note['pack_id']}] {note['source']}:{note['ref']} - {note['snippet'][:120]}\n"
                for kp in note.get("key_points", [])[:2]:
                    evidence_summary += f"  * {kp[:100]}\n"
        workspace_parts.append(evidence_summary)
    
    workspace = "\n".join(workspace_parts)
    
    # 限制长度
    max_chars = state.get("max_workspace_chars", 8000)
    if len(workspace) > max_chars:
        workspace = workspace[:max_chars] + "\n... (truncated)"
    
    # 序列化 session_cache
    session_cache_dict = cache.to_state_dict()
    
    # 添加到 persist_queue
    persist_queue_additions = [note["pack_id"] for note in round_notes]
    
    return {
        "session_cache": session_cache_dict,
        "round_doc_ids": round_doc_ids,
        "round_notes": round_notes,
        "all_notes": round_notes,  # 会被 add 到累计列表
        "workspace": workspace,
        "persist_queue": persist_queue_additions,
        "trace": [{
            "round": state["round_id"],
            "stage": "materialize_round_cache",
            "timestamp": datetime.now().isoformat(),
            "message": f"Materialized {len(materialized_docs)} docs, {len(round_notes)} notes",
            "data": {"docs": len(materialized_docs), "notes": len(round_notes), "images": sum(1 for n in round_notes if n.get("media_type") == "image")}
        }]
    }


@trace_node("normalize_evidence")
async def normalize_evidence_node(state: ResearchState) -> ResearchStateUpdate:
    """
    Node: 智能提取证据并存储到 Milvus
    
    使用LLM根据当前待解决问题智能提取证据的核心信息
    输入: tool_results, open_questions
    输出: evidence_ids (追加)
    """
    logger.info(f"[Task {state['task_id']}] Normalizing evidence with LLM extraction")
    logger.debug(f"📥 Tool results count: {len(state.get('tool_results', []))}")
    
    embedding_provider = get_embedding_provider()
    writer = get_writer()
    llm = make_llm(model="qwen3-max", temperature=0)
    
    # 获取当前待解决问题用于智能提取
    open_questions = state.get("open_questions", [state["query"]])
    goal_text = "\n".join(f"- {q}" for q in open_questions[:5])
    
    new_evidence_ids = []
    chunks = []
    
    # LLM提取提示模板
    extractor_prompt = ChatPromptTemplate.from_messages([
        ("system", """请处理以下内容，并根据当前的研究目标提取关键信息。

**提取指南**：
1. **理由 (rationale)**: 为什么这段内容与目标相关？（1-2句）
2. **证据 (evidence)**: 提取原文中最核心的段落，保留原文细节（数据、结论、方法等），不要过度摘要（最多2000字符）
3. **摘要 (summary)**: 对提取的信息进行逻辑概括（3-5句，200字符内）
4. **关键点 (key_points)**: 最重要的3-5个要点（每个不超过100字符）

返回 JSON 格式：
{{
    "rationale": "...",
    "evidence": "...",
    "summary": "...",
    "key_points": ["...", "..."]
}}"""),
        ("human", """## 当前待解决问题（研究目标）
{goal}

## 原始内容
来源：{source}
标题：{title}
内容：
{content}

请提取关键信息（JSON格式）：""")
    ])
    
    for result in state["tool_results"]:
        tool_name = result.get("tool", "")
        target_question = result.get("target_question", "")  # 从tool_loop传来的目标问题
        data = result.get("result", [])
        
        if not isinstance(data, list):
            continue
        
        for item in data:
            if not isinstance(item, dict):
                continue
            
            # arXiv paper
            if "arxiv_id" in item:
                ev_id = generate_evidence_id("arxiv", item["arxiv_id"], hash_text(item.get("summary", "")))
                if ev_id in state.get("evidence_ids", []):
                    continue
                
                # LLM智能提取
                raw_content = item.get("summary", "")[:10000]
                try:
                    extract_response = await (extractor_prompt | llm).ainvoke({
                        "goal": goal_text,
                        "source": "arxiv",
                        "title": item.get("title", ""),
                        "content": raw_content,
                    })
                    extracted = json.loads(_extract_text_from_response(extract_response))
                except Exception as e:
                    logger.warning(f"LLM extraction failed: {e}, using fallback")
                    extracted = {
                        "rationale": "相关论文",
                        "evidence": raw_content[:2000],
                        "summary": raw_content[:200],
                        "key_points": [raw_content[:100]]
                    }
                
                embedding = await embedding_provider.embed(extracted["evidence"])
                chunks.append({
                    "id": ev_id,
                    "embedding": embedding,
                    "text": extracted["evidence"],
                    "source": "arxiv",
                    "ref": item["arxiv_id"],
                    "snippet": extracted["summary"],
                    "task_id": state["task_id"],
                    "round_id": state["round_id"],
                    "metadata": {
                        "title": item.get("title", ""),
                        "authors": item.get("authors", []),
                        "url": item.get("abs_url", ""),
                        "rationale": extracted["rationale"],
                        "key_points": extracted["key_points"],
                        "target_question": target_question,
                    },
                })
                new_evidence_ids.append(ev_id)
            
            # GitHub repo
            elif "full_name" in item:
                description = item.get("description", "") or item.get("name", "")
                readme = item.get("readme", "")[:5000]  # 包含README内容
                ev_id = generate_evidence_id("github", item["full_name"], hash_text(description + readme))
                if ev_id in state.get("evidence_ids", []):
                    continue
                
                # LLM智能提取
                raw_content = f"{description}\n\nREADME:\n{readme}"
                try:
                    extract_response = await (extractor_prompt | llm).ainvoke({
                        "goal": goal_text,
                        "source": "github",
                        "title": item.get("full_name", ""),
                        "content": raw_content[:10000],
                    })
                    extracted = json.loads(_extract_text_from_response(extract_response))
                except Exception as e:
                    logger.warning(f"LLM extraction failed: {e}, using fallback")
                    extracted = {
                        "rationale": "相关仓库",
                        "evidence": raw_content[:2000],
                        "summary": description[:200],
                        "key_points": [description[:100]]
                    }
                
                embedding = await embedding_provider.embed(extracted["evidence"])
                chunks.append({
                    "id": ev_id,
                    "embedding": embedding,
                    "text": extracted["evidence"],
                    "source": "github",
                    "ref": item["full_name"],
                    "snippet": extracted["summary"],
                    "task_id": state["task_id"],
                    "round_id": state["round_id"],
                    "metadata": {
                        "name": item.get("name", ""),
                        "stars": item.get("stars", 0),
                        "url": item.get("url", ""),
                        "rationale": extracted["rationale"],
                        "key_points": extracted["key_points"],
                        "target_question": target_question,
                    },
                })
                new_evidence_ids.append(ev_id)
            
            # Web页面（新增支持）
            elif "text" in item and "url" in item:
                url = item["url"]
                ev_id = generate_evidence_id("web", url, hash_text(item.get("text", "")[:1000]))
                if ev_id in state.get("evidence_ids", []):
                    continue
                
                # LLM智能提取
                raw_content = item.get("text", "")[:10000]
                try:
                    extract_response = await (extractor_prompt | llm).ainvoke({
                        "goal": goal_text,
                        "source": "web",
                        "title": item.get("title", ""),
                        "content": raw_content,
                    })
                    extracted = json.loads(_extract_text_from_response(extract_response))
                except Exception as e:
                    logger.warning(f"LLM extraction failed: {e}, using fallback")
                    extracted = {
                        "rationale": "相关网页",
                        "evidence": raw_content[:2000],
                        "summary": raw_content[:200],
                        "key_points": [raw_content[:100]]
                    }
                
                embedding = await embedding_provider.embed(extracted["evidence"])
                chunks.append({
                    "id": ev_id,
                    "embedding": embedding,
                    "text": extracted["evidence"],
                    "source": "web",
                    "ref": url,
                    "snippet": extracted["summary"],
                    "task_id": state["task_id"],
                    "round_id": state["round_id"],
                    "metadata": {
                        "title": item.get("title", ""),
                        "url": url,
                        "rationale": extracted["rationale"],
                        "key_points": extracted["key_points"],
                        "target_question": target_question,
                    },
                })
                new_evidence_ids.append(ev_id)
    
    if chunks:
        writer.write_evidence_batch(chunks)
    
    return {
        "evidence_ids": new_evidence_ids,
        "trace": [{
            "round": state["round_id"],
            "stage": "normalize_evidence",
            "timestamp": datetime.now().isoformat(),
            "message": f"Stored {len(new_evidence_ids)} new evidence items with LLM extraction",
        }],
    }


@trace_node("update_report")
async def update_report_node(state: ResearchState) -> ResearchStateUpdate:
    """
    Node: 更新中心报告（重构版：只消费 notes）
    
    输入: round_notes, all_notes, summary, findings, open_questions
    输出: summary (更新), findings (追加), open_questions (更新)
    """
    logger.info(f"[Task {state['task_id']}] Updating central report from notes")
    logger.debug(f"📥 Round notes: {len(state.get('round_notes', []))}")
    
    # 如果本轮没有新 notes，不更新
    if not state.get("round_notes"):
        logger.info("No new notes this round, skipping update_report")
        return {
            "trace": [{
                "round": state["round_id"],
                "stage": "update_report",
                "timestamp": datetime.now().isoformat(),
                "message": "No new notes to process",
            }],
        }
    
    # 构建 notes 的紧凑表示（key_points + evidence_quotes）
    round_notes = state.get("round_notes", [])[:20]  # 限制数量
    evidence_text_lines = []
    
    from ..services.session_cache import safe_note_pack_id, safe_note_source, safe_note_evidence_quotes
    
    for note in round_notes:
        pack_id = safe_note_pack_id(note)
        source = safe_note_source(note)
        ref = note.get("ref", "")
        snippet = note.get("snippet", "")
        key_points = note.get("key_points", [])[:5]
        evidence_quotes = safe_note_evidence_quotes(note)[:3]
        
        kp_text = "\n".join(f"  - {kp}" for kp in key_points)
        eq_text = "\n".join(
            f"  → [{eq.get('location_hint', '')}] {eq.get('quote', '')[:150]}"
            for eq in evidence_quotes
        )
        
        evidence_text_lines.append(
            f"[{pack_id}] {source}:{ref}\n"
            f"Snippet: {snippet}\n"
            f"Key Points:\n{kp_text}\n"
            f"Evidence Quotes:\n{eq_text}"
        )
    
    evidence_text = "\n\n".join(evidence_text_lines)
    
    llm = make_llm(model="qwen3-max")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """分析新的研究笔记并更新报告。
基于紧凑的笔记（带 pack_id 引用），提供：
1. 新发现（简洁陈述列表，必须引用 pack_id，格式：[pack_id]）
2. 哪些待解决问题现在可以回答了（列表）
3. 新出现的待解决问题（列表）
4. 整合新旧发现的更新摘要

重要：每个发现必须在方括号中引用至少一个 pack_id。

返回 JSON：
{{
    "new_findings": ["基于证据 [pack_id_123] 的发现1", "发现2 [pack_id_456]"],
    "resolved_questions": ["已回答的问题"],
    "new_open_questions": ["新出现的问题"],
    "updated_summary": "引用 [pack_id_xxx] 证据的综合摘要"
}}"""),
        ("human", """当前报告：
摘要：{summary}
最近发现：{findings}
待解决问题：{open_questions}

新笔记：
{evidence}

请提供报告更新。"""),
    ])
    
    chain = prompt | llm
    response = await chain.ainvoke({
        "summary": state.get("summary") or "None yet",
        "findings": state.get("findings", [])[-5:],
        "open_questions": state.get("open_questions", []),
        "evidence": evidence_text,
    })
    
    try:
        content = _extract_text_from_response(response)
        update = json.loads(content)
    except Exception:
        update = {"new_findings": [], "resolved_questions": [], "new_open_questions": [], "updated_summary": state.get("summary", "")}
    
    # Update open questions
    resolved = set(update.get("resolved_questions", []))
    remaining_questions = [q for q in state.get("open_questions", []) if q not in resolved]
    new_open_questions = remaining_questions + update.get("new_open_questions", [])
    
    return {
        "summary": update.get("updated_summary", state.get("summary", "")),
        "findings": update.get("new_findings", []),
        "open_questions": new_open_questions,
        "trace": [{
            "round": state["round_id"],
            "stage": "update_report",
            "timestamp": datetime.now().isoformat(),
            "message": f"Added {len(update.get('new_findings', []))} findings",
            "data": update,
        }],
    }


@trace_node("check_stop")
async def check_stop_node(state: ResearchState) -> ResearchStateUpdate:
    """
    Node: 检查停止条件（使用LLM决策）
    
    输入: round_id, max_rounds, open_questions, findings, summary
    输出: next_action, stop_reason
    """
    logger.info(f"[Task {state['task_id']}] Checking stop conditions")
    logger.debug(f"📥 Round {state['round_id']}/{state['max_rounds']}, Open questions: {len(state.get('open_questions', []))}")
    
    # Hard stop conditions
    if state["round_id"] >= state["max_rounds"]:
        return {
            "next_action": "stop",
            "stop_reason": "max_rounds_reached",
            "trace": [{
                "round": state["round_id"],
                "stage": "check_stop",
                "timestamp": datetime.now().isoformat(),
                "message": "停止：达到最大轮次",
            }],
        }
    
    # LLM-based decision
    current_date = datetime.now().strftime("%Y年%m月%d日")
    current_year = datetime.now().year
    
    llm = make_llm(model="qwen3-max")
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""决定是继续研究还是完成答案。当前日期：{current_date}（{current_year}年）

继续研究的情况：
- 存在需要更多证据的具体未解决问题
- 当前发现不足以形成全面答案
- 新证据表明有希望的研究方向

完成答案的情况：
- 主要研究问题已得到充分解答
- 最近几轮未产生重要新信息
- 收集的证据足以形成全面回应
- 已达到最大轮次

返回 JSON：
{{{{
    "action": "continue" 或 "stop",
    "reason": "解释",
    "confidence": "high" 或 "medium" 或 "low"
}}}}"""),
        ("human", """当前状态：
轮次：{{round_id}}/{{max_rounds}}
摘要：{{summary}}
发现数：{{findings_count}}
待解决问题数：{{open_questions_count}}
本轮新证据数：{{new_evidence_count}}

最近发现：
{{recent_findings}}

最近待解决问题：
{{recent_questions}}

应该继续研究还是完成答案？"""),
    ])
    
    chain = prompt | llm
    response = await chain.ainvoke({
        "round_id": state["round_id"],
        "max_rounds": state["max_rounds"],
        "summary": state.get("summary", "暂无"),
        "findings_count": len(state.get("findings", [])),
        "open_questions_count": len(state.get("open_questions", [])),
        "new_evidence_count": len(state.get("evidence_packs", [])),
        "recent_findings": "\n".join(f"- {f}" for f in state.get("findings", [])[-3:]),
        "recent_questions": "\n".join(f"- {q}" for q in state.get("open_questions", [])[:3]),
    })
    
    try:
        content = _extract_text_from_response(response)
        decision = json.loads(content)
    except Exception:
        # Default to continue on parse error
        decision = {"action": "continue", "reason": "解析错误，默认继续", "confidence": "low"}
    
    next_action = decision.get("action", "continue")
    if next_action == "stop":
        return {
            "next_action": "stop",
            "stop_reason": decision.get("reason", "LLM决定停止"),
            "trace": [{
                "round": state["round_id"],
                "stage": "check_stop",
                "timestamp": datetime.now().isoformat(),
                "message": f"停止：{decision.get('reason', 'LLM决定')}",
                "data": decision,
            }],
        }
    else:
        return {
            "trace": [{
                "round": state["round_id"],
                "stage": "check_stop",
                "timestamp": datetime.now().isoformat(),
                "message": f"继续研究：{decision.get('reason', 'LLM决定')}",
                "data": decision,
            }],
        }


@trace_node("write_answer")
async def write_final_answer_node(state: ResearchState) -> ResearchStateUpdate:
    """
    Node: 生成最终答案（重构版：摘要驱动+按需取原文）
    
    策略：
    1. 第一次调用：只给 all_notes 的摘要层（key_points + evidence_quotes）
    2. LLM 可以输出 <answer>...</answer> 或 JSON: {"type":"NEED_RAW","requests":[...]}
    3. 若 NEED_RAW：从 Session Cache 提取指定 doc_id 的 chunks，补充后再调用
    4. 最多循环 2-3 次，最终必须产出 <answer>
    
    输入: query, task_spec, summary, findings, all_notes, session_cache
    输出: final_answer, citations
    """
    logger.info(f"[Task {state['task_id']}] Writing final answer (Session Cache mode)")
    logger.debug(f"📥 Total findings: {len(state.get('findings', []))}, Notes: {len(state.get('all_notes', []))}")
    
    # 恢复 SessionCache（使用新的反序列化方法）
    cache_data = state.get("session_cache", {})
    cache = SessionCache.from_state_dict(cache_data) if cache_data else SessionCache()
            cache.add_doc(doc)
    
    all_notes = state.get("all_notes", [])
    if not all_notes:
        # Fallback：没有 notes，使用传统 Milvus 检索（不推荐）
        logger.warning("No all_notes found, falling back to Milvus retrieval")
        return await _write_answer_fallback(state)
    
    llm = make_llm(model="qwen3-max")
    
    # === Phase 1: 构建摘要层证据（不包含原文）===
    evidence_summary_lines = []
    for note in all_notes[:20]:  # 限制数量
        if note.get("media_type") == "image":
            evidence_summary_lines.append(
                f"[IMAGE] [{note['pack_id']}] {note['source']}:{note['ref']}\n"
                f"  Caption: {note['snippet']}\n"
                f"  URL: {note.get('image_url', '')}"
            )
        else:
            evidence_summary_lines.append(
                f"[{note['pack_id']}] {note['source']}:{note['ref']} - {note.get('title', '')}\n"
                f"  Snippet: {note['snippet']}"
            )
            
            # Key points
            for kp in note.get("key_points", [])[:3]:
                evidence_summary_lines.append(f"    * {kp}")
            
            # Evidence quotes（带 location_hint）
            for eq in note.get("evidence_quotes", [])[:3]:
                quote = eq.get("quote", "")[:200]
                why = eq.get("why_relevant", "")[:100]
                loc = eq.get("location_hint", "")
                evidence_summary_lines.append(f"    → Quote [{loc}]: {quote}\n      Why: {why}")
    
    evidence_summary = "\n".join(evidence_summary_lines)
    
    # === Phase 2: 多轮对话（最多 3 次）===
    max_rounds = 3
    final_answer = ""
    raw_chunks_provided = []  # 记录已提供的原文
    
    for attempt in range(max_rounds):
        logger.info(f"Write answer attempt {attempt + 1}/{max_rounds}")
        
        # 构建 prompt
        if attempt == 0:
            # 第一次：只给摘要
            system_prompt = """你是一个研究助手，正在撰写研究查询的最终答案。

**当前阶段**：你将看到证据的**摘要层**（key_points + evidence_quotes），没有完整原文。

**你的任务**：
1. 如果摘要足够回答问题：直接输出 <answer>...</answer>（完整答案，中文，引用 [pack_id]）
2. 如果需要查看原文细节：输出 JSON：
   ```json
   {
     "type": "NEED_RAW",
     "requests": [
       {"pack_id": "xxx", "reason": "为什么需要这段原文", "location_hint": "P3"}
     ]
   }
   ```

**输出规则**：
- 如果输出 <answer>：必须包含所有关键信息、引用 [pack_id]、图片用 ![desc](url)
- 如果输出 NEED_RAW：最多请求 3 个 doc 的原文片段
- 不要同时输出 <answer> 和 NEED_RAW

**图片处理**：
- 图片证据标记为 [IMAGE]，使用提供的 URL 嵌入：`![描述](url)`
- 不要编造 URL"""
            
            user_prompt = f"""研究查询：{state['query']}

任务规格：{json.dumps(state.get('task_spec', {}), ensure_ascii=False)}

研究摘要：{state.get('summary', '')}

关键发现：
{chr(10).join(f"- {f}" for f in state.get('findings', []))}

证据摘要层（Key Points + Quotes）：
{evidence_summary}

剩余待解决问题：
{chr(10).join(f"- {q}" for q in state.get('open_questions', [])) if state.get('open_questions') else "None"}

请输出 <answer>...</answer> 或 NEED_RAW JSON。"""
        
        else:
            # 后续轮次：补充原文
            system_prompt = """你是一个研究助手，继续撰写答案。

**当前阶段**：你已经看过证据摘要，现在提供了你请求的**原文片段**。

**你的任务**：
1. 结合原文片段，输出完整答案 <answer>...</answer>
2. 如果仍需更多原文（最多再请求 1 次）：输出 NEED_RAW JSON
3. **必须**在本轮或下一轮输出 <answer>，不能无限循环

**输出规则**：同上一轮"""
            
            user_prompt = f"""研究查询：{state['query']}

之前的证据摘要：
{evidence_summary[:2000]}...

你请求的原文片段：
{chr(10).join(f"[Doc {c['source']}] Location: {c['location']}\n{c['text'][:1000]}\n" for c in raw_chunks_provided[-3:])}

请输出最终答案 <answer>...</answer> 或（最后一次机会）NEED_RAW JSON。"""
        
        response = await llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        
        raw_resp = _extract_text_from_response(response)
        
        # 检查是否输出 <answer>
        import re
        answer_match = re.search(r'<answer>(.*?)</answer>', raw_resp, re.DOTALL)
        if answer_match:
            final_answer = answer_match.group(1).strip()
            logger.info(f"Got final answer on attempt {attempt + 1}")
            break
        
        # 检查是否输出 NEED_RAW
        try:
            # 尝试解析 JSON
            need_raw_match = re.search(r'\{[^}]*"type"\s*:\s*"NEED_RAW"[^}]*\}', raw_resp, re.DOTALL)
            if need_raw_match:
                need_raw = json.loads(need_raw_match.group(0))
            else:
                # 尝试完整解析
                need_raw = json.loads(raw_resp)
            
            if need_raw.get("type") == "NEED_RAW":
                requests = need_raw.get("requests", [])
                logger.info(f"LLM requested {len(requests)} raw chunks")
                
                # 提取原文
                for req in requests[:3]:  # 最多 3 个
                    pack_id = req.get("pack_id")
                    location_hint = req.get("location_hint")
                    
                    if pack_id:
                        note = cache.get_note(pack_id)
                        if note and note.get("raw_pointer"):
                            doc_id = note["raw_pointer"]
                            chunks = cache.extract_chunks_for_need_raw(
                                doc_id=doc_id,
                                location_hints=[location_hint] if location_hint else None,
                                pack_ids=[pack_id],
                                max_chunks=2,
                                max_chars=2000
                            )
                            raw_chunks_provided.extend(chunks)
                
                if not raw_chunks_provided:
                    # 没有提取到原文，强制输出答案
                    logger.warning("NEED_RAW but no chunks found, forcing answer")
                    break
                
                # 继续下一轮
                continue
            else:
                # 不是 NEED_RAW，也不是 <answer>，当作错误
                logger.warning(f"Unexpected LLM output format: {raw_resp[:200]}")
                break
        
        except json.JSONDecodeError:
            # 不是 JSON，也没有 <answer> 标签
            logger.warning(f"LLM output is neither <answer> nor NEED_RAW JSON: {raw_resp[:200]}")
            # Fallback：使用原始输出
            final_answer = raw_resp
            break
    
    # 如果循环结束还没有 answer，使用最后一次输出
    if not final_answer:
        logger.warning("Failed to get <answer> after max rounds, using last LLM output")
        final_answer = raw_resp
    
    # === Phase 3: 构建 citations（来自 all_notes）===
    citations = []
    for note in all_notes[:20]:
        citations.append({
            "source": note["source"],
            "ref": note["ref"],
            "snippet": note["snippet"],
            "pack_id": note["pack_id"],
            "url": note.get("url"),
            "title": note.get("title"),
            "media_type": note.get("media_type", "text"),
            "image_url": note.get("image_url")
        })
    
    return {
        "final_answer": final_answer,
        "citations": citations,
        "trace": [{
            "round": state["round_id"],
            "stage": "write_final_answer",
            "timestamp": datetime.now().isoformat(),
            "message": f"Generated answer with {len(citations)} citations, {len(raw_chunks_provided)} raw chunks provided",
        }],
    }


async def _write_answer_fallback(state: ResearchState) -> ResearchStateUpdate:
    """Fallback: 传统 Milvus 检索模式（不推荐）"""
    logger.warning("Using fallback write_answer mode (Milvus retrieval)")
    embedding_provider = get_embedding_provider()
    retriever = get_retriever()
    
    query_embedding = await embedding_provider.embed(state["query"])
    all_evidence = retriever.search_evidence(
        query_embedding=query_embedding,
        top_k=20,
        task_id=state["task_id"],
        enable_rerank=False,
    )
    
    evidence_lines = []
    for ev in all_evidence:
        media_type = ev.get('media_type', 'text')
        if media_type == 'image':
            evidence_lines.append(
                f"[IMAGE] [{ev['source']}:{ev['ref']}] {ev['snippet']} (URL: {ev.get('image_url', '')})"
            )
        else:
            evidence_lines.append(f"[{ev['source']}:{ev['ref']}] {ev['snippet']}")
    
    evidence_text = "\n".join(evidence_lines)
    
    llm = make_llm(model="qwen3-max")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个研究助手，正在撰写研究查询的最终答案。
基于研究发现和证据，提供全面的答案。

指南：
1. 回应原始查询的所有方面
2. 使用 [来源:引用] 格式引用来源
3. 如需要，用章节清晰组织答案
4. 全面但简洁
5. 注明任何局限性或需要进一步研究的领域
6. 使用中文回答

**图片处理：**
- 如果证据中包含图片（标记为 [IMAGE]），请使用 markdown 格式嵌入：`![描述](url)`
- **必须且只能**使用证据中提供的 URL，不要编造链接

**重要**：你必须把最终的结论包裹在 <answer> 标签中。

示例格式：
<answer>
[这里是最终答案，包含所有关键信息、引用和图片]
![图表说明](https://example.com/image.png)
</answer>"""),
        ("human", """研究查询：{query}

任务规格：{task_spec}

研究摘要：{summary}

关键发现：
{findings}

证据：
{evidence}

剩余待解决问题：
{open_questions}

请提供最终的全面答案（中文）。"""),
    ])
    
    chain = prompt | llm
    response = await chain.ainvoke({
        "query": state["query"],
        "task_spec": json.dumps(state.get("task_spec", {})),
        "summary": state.get("summary", ""),
        "findings": "\n".join(f"- {f}" for f in state.get("findings", [])),
        "evidence": evidence_text,
        "open_questions": "\n".join(f"- {q}" for q in state.get("open_questions", [])) if state.get("open_questions") else "None",
    })
    
    citations = [
        {
            "source": ev["source"],
            "ref": ev["ref"],
            "snippet": ev["snippet"],
            "metadata": ev.get("metadata", {}),
        }
        for ev in all_evidence
    ]
    
    raw_response = _extract_text_from_response(response)
    import re
    answer_match = re.search(r'<answer>(.*?)</answer>', raw_response, re.DOTALL)
    if answer_match:
        final_answer = answer_match.group(1).strip()
    else:
        final_answer = raw_response
        logger.warning("No <answer> tag found in LLM response, using full response")
    
    return {
        "final_answer": final_answer,
        "citations": citations,
        "trace": [{
            "round": state["round_id"],
            "stage": "write_final_answer",
            "timestamp": datetime.now().isoformat(),
            "message": f"Generated answer with {len(citations)} citations (fallback mode)",
        }],
    }


# ============================================================================
# Conditional Edges (路由逻辑)
# ============================================================================

def should_continue(state: ResearchState) -> str:
    """
    决定工作流下一步：继续迭代 vs 输出答案
    
    返回值对应 graph edge 的 target node name
    """
    next_action = state.get("next_action", "continue")
    
    if next_action == "answer":
        return "write_answer"
    elif next_action == "stop":
        return "write_answer"
    else:
        # 增加轮次，继续下一轮
        return "next_round"


def increment_round(state: ResearchState) -> ResearchStateUpdate:
    """Helper node: 增加轮次计数"""
    return {"round_id": state["round_id"] + 1}


@trace_node("persist_evidence")
async def persist_evidence_node(state: ResearchState) -> ResearchStateUpdate:
    """
    Node: 持久化证据到 Milvus（任务结束后）
    
    策略：
    1. 只写"可检索的笔记文本"：evidence_quotes.quote + key_points
    2. 按 chunk/quote 粒度写入（不是整篇 doc）
    3. metadata 区分 origin=local/web
    4. 去重：source+ref+hash_text(quote)
    
    输入: persist_queue, all_notes, session_cache
    输出: 无（副作用：写入 Milvus）
    """
    logger.info(f"[Task {state['task_id']}] Persisting evidence to Milvus")
    
    persist_queue = state.get("persist_queue", [])
    all_notes = state.get("all_notes", [])
    
    if not persist_queue:
        logger.info("No evidence to persist (empty queue)")
        return {
            "trace": [{
                "round": state["round_id"],
                "stage": "persist_evidence",
                "timestamp": datetime.now().isoformat(),
                "message": "No evidence to persist",
            }]
        }
    
    embedding_provider = get_embedding_provider()
    writer = get_writer()
    
    # 恢复 session_cache 获取 origin 信息
    cache = SessionCache()
    if state.get("session_cache"):
        cached_data = state["session_cache"]
        for doc_id, doc_dict in cached_data.get("docs", {}).items():
            # 简化：只记录 origin
            cache.docs[doc_id] = type('Doc', (), {'origin': doc_dict.get('origin', 'web'), 'doc_id': doc_id})()
    
    chunks_to_write = []
    written_ids = set()
    
    for pack_id in persist_queue:
        # 找到对应的 note
        note = next((n for n in all_notes if n['pack_id'] == pack_id), None)
        if not note:
            continue
        
        # 获取 origin
        raw_pointer = note.get("raw_pointer")
        origin = "web"
        if raw_pointer and raw_pointer in cache.docs:
            origin = getattr(cache.docs[raw_pointer], 'origin', 'web')
        
        # 策略1: 写入每个 evidence_quote 作为独立 chunk
        for idx, eq in enumerate(note.get("evidence_quotes", [])[:5]):
            quote = eq.get("quote", "")
            if not quote or len(quote) < 20:
                continue
            
            # 去重
            chunk_id = generate_evidence_id(note["source"], note["ref"], hash_text(quote))
            if chunk_id in written_ids:
                continue
            written_ids.add(chunk_id)
            
            # 生成 embedding
            embedding = await embedding_provider.embed(quote)
            
            # snippet = quote[:300]
            snippet = eq.get("why_relevant", "")[:200] or quote[:200]
            
            chunks_to_write.append({
                "id": chunk_id,
                "embedding": embedding,
                "text": quote,  # 完整 quote
                "source": note["source"],
                "ref": note["ref"],
                "snippet": snippet,
                "task_id": state["task_id"],
                "round_id": state["round_id"],
                "metadata": {
                    "title": note.get("title", ""),
                    "url": note.get("url", ""),
                    "pack_id": pack_id,
                    "location_hint": eq.get("location_hint", ""),
                    "why_relevant": eq.get("why_relevant", ""),
                    "origin": origin,  # 区分 local/web
                    "media_type": note.get("media_type", "text"),
                    "image_url": note.get("image_url"),
                    "target_question": note.get("target_question", ""),
                    "chunk_type": "evidence_quote"  # 标记为证据引用
                },
                "created_at": int(datetime.now().timestamp())
            })
        
        # 策略2: 写入 key_points 拼接（作为背景块，可选）
        key_points_text = " | ".join(note.get("key_points", [])[:5])
        if key_points_text and len(key_points_text) > 50:
            chunk_id = generate_evidence_id(note["source"], note["ref"], hash_text(key_points_text))
            if chunk_id not in written_ids:
                written_ids.add(chunk_id)
                
                embedding = await embedding_provider.embed(key_points_text)
                
                chunks_to_write.append({
                    "id": chunk_id,
                    "embedding": embedding,
                    "text": key_points_text,
                    "source": note["source"],
                    "ref": note["ref"],
                    "snippet": key_points_text[:300],
                    "task_id": state["task_id"],
                    "round_id": state["round_id"],
                    "metadata": {
                        "title": note.get("title", ""),
                        "url": note.get("url", ""),
                        "pack_id": pack_id,
                        "origin": origin,
                        "media_type": note.get("media_type", "text"),
                        "image_url": note.get("image_url"),
                        "target_question": note.get("target_question", ""),
                        "chunk_type": "key_points"  # 标记为关键点
                    },
                    "created_at": int(datetime.now().timestamp())
                })
    
    # 批量写入
    if chunks_to_write:
        logger.info(f"Writing {len(chunks_to_write)} chunks to Milvus (origin分布: {sum(1 for c in chunks_to_write if c['metadata'].get('origin') == 'local')} local, {sum(1 for c in chunks_to_write if c['metadata'].get('origin') == 'web')} web)")
        writer.write_evidence_batch(chunks_to_write)
    
    return {
        "trace": [{
            "round": state["round_id"],
            "stage": "persist_evidence",
            "timestamp": datetime.now().isoformat(),
            "message": f"Persisted {len(chunks_to_write)} chunks to Milvus",
            "data": {
                "total_chunks": len(chunks_to_write),
                "local_chunks": sum(1 for c in chunks_to_write if c['metadata'].get('origin') == 'local'),
                "web_chunks": sum(1 for c in chunks_to_write if c['metadata'].get('origin') == 'web')
            }
        }]
    }


# ============================================================================
# Build StateGraph
# ============================================================================

def create_research_graph() -> StateGraph:
    """
    构建 DeepResearch LangGraph 工作流（重构版）
    
    新流程结构：
    START -> parse_task -> build_workspace -> tool_loop -> materialize_round_cache
                                                                    |
                                                                    v
                                                (skip normalize_evidence，已在materialize完成)
                                                                    |
                                                                    v
                                                      update_report -> check_stop
                                                                    |
                                                                    v
                                    [continue] -> increment_round -> build_workspace (循环)
                                    [stop/answer] -> write_answer -> persist_evidence -> END
    
    关键改动：
    1. tool_loop 之后必须经过 materialize_round_cache（即使next_action=answer）
    2. materialize 完成 hit→doc→note 升级和分块
    3. write_answer 使用 Session Cache 而非 Milvus 检索
    4. persist_evidence 在最后执行（任务结束后入库）
    """
    workflow = StateGraph(ResearchState)
    
    # Add nodes
    workflow.add_node("parse_task", parse_task_node)  # type: ignore
    workflow.add_node("build_workspace", build_workspace_node)  # type: ignore
    workflow.add_node("tool_loop", tool_loop_node)  # type: ignore
    workflow.add_node("materialize_round_cache", materialize_round_cache_node)  # type: ignore [NEW]
    workflow.add_node("normalize_evidence", normalize_evidence_node)  # type: ignore [DEPRECATED, 保留兼容]
    workflow.add_node("update_report", update_report_node)  # type: ignore
    workflow.add_node("check_stop", check_stop_node)  # type: ignore
    workflow.add_node("write_answer", write_final_answer_node)  # type: ignore
    workflow.add_node("persist_evidence", persist_evidence_node)  # type: ignore [NEW]
    workflow.add_node("increment_round", increment_round)  # type: ignore

    
    # Set entry point
    workflow.set_entry_point("parse_task")
    
    # Add edges
    workflow.add_edge("parse_task", "build_workspace")
    workflow.add_edge("build_workspace", "tool_loop")
    
    # === 关键改动：tool_loop 后必须先 materialize（无论 next_action）===
    workflow.add_edge("tool_loop", "materialize_round_cache")
    
    # materialize 后根据 next_action 分支
    workflow.add_conditional_edges(
        "materialize_round_cache",
        lambda s: "write_answer" if s.get("next_action") in ("answer", "stop") else "update_report",
        {"write_answer": "write_answer", "update_report": "update_report"}
    )
    
    # 继续研究路径
    workflow.add_edge("update_report", "check_stop")
    workflow.add_conditional_edges(
        "check_stop",
        lambda state: "write_answer" if state.get("next_action") == "stop" else "increment_round",
        {
            "write_answer": "write_answer",
            "increment_round": "increment_round",
        }
    )
    workflow.add_edge("increment_round", "build_workspace")  # Loop back
    
    # === 关键改动：write_answer 后必须 persist_evidence 再 END ===
    workflow.add_edge("write_answer", "persist_evidence")
    workflow.add_edge("persist_evidence", END)
    
    return workflow
    
    return workflow


# ============================================================================
# Service Interface
# ============================================================================

class DeepResearchLangGraphService:
    """
    DeepResearch Service using LangGraph
    
    使用方式:
        service = DeepResearchLangGraphService()
        async for event in service.run_research(query="What is RAG?"):
            print(event)
    """
    
    def __init__(self, checkpointer=None):
        """
        Initialize service
        
        Args:
            checkpointer: LangGraph checkpointer for state persistence
                         (e.g., MemorySaver, SqliteSaver, PostgresSaver)
        """
        self.workflow = create_research_graph()
        self.checkpointer = checkpointer or MemorySaver()
        self.app = self.workflow.compile(checkpointer=self.checkpointer)
        
        # 初始化追踪配置
        self.tracing_config = get_tracing_config()
        logger.info(f"🔍 追踪配置: LangSmith={'✅' if self.tracing_config.langsmith_enabled else '❌'}, 本地追踪={'✅' if self.tracing_config.local_trace_enabled else '❌'}")
        
        # 打印工作流图（调试模式）
        if os.getenv("DEBUG", "false").lower() == "true":
            print_workflow_diagram()
    
    async def run_research(
        self,
        query: str,
        max_rounds: int = 5,
        max_papers: int = 10,
        max_repos: int = 5,
        top_k: int = 10,
        config: Optional[Dict[str, Any]] = None,
    )-> AsyncIterator[NodeCompleteEvent]:
        """
        运行研究任务（流式输出）
        
        Args:
            query: 研究问题
            max_rounds: 最大轮数
            max_papers: 每轮最多论文数
            max_repos: 每轮最多仓库数
            top_k: 检索 top-k
            config: LangGraph 运行配置（可包含 thread_id）
        
        Yields:
            Dict: 状态更新事件
        """
        task_id = str(uuid.uuid4())
        
        logger.info(f"🚀 开始研究任务: {task_id}")
        logger.info(f"📋 Query: {query}")
        logger.info(f"⚙️ Config: max_rounds={max_rounds}, max_papers={max_papers}, max_repos={max_repos}")
        
        # Initial state
        initial_state = {
            "task_id": task_id,
            "query": query,
            "round_id": 1,
            "task_spec": {},
            "summary": "",
            "findings": [],
            "open_questions": [],
            "evidence_ids": [],
            # Session Cache (新增)
            "session_cache": {},
            "round_doc_ids": [],
            "all_notes": [],
            "round_notes": [],
            "persist_queue": [],
            # Evidence packs (保留兼容)
            "evidence_packs": [],
            "new_evidence_packs": [],
            "workspace": "",
            "tool_queries": [],
            "tool_results": [],
            "next_action": "continue",
            "stop_reason": None,
            "messages": [],
            "final_answer": "",
            "citations": [],
            "max_rounds": max_rounds,
            "max_papers": max_papers,
            "max_repos": max_repos,
            "top_k": top_k,
            # Tool loop / workspace limits for progressive disclosure
            "max_tool_steps_per_round": 3,
            "max_workspace_chars": 8000,
            "max_packs_per_step": 3,
            "max_total_packs_per_round": 20,
            "trace": [],
        }
        
        # 使用追踪配置生成 config（包含 LangSmith 元数据）
        if config is None:
            config = self.tracing_config.get_langchain_config(task_id)
        else:
            # 合并用户提供的 config 和追踪配置
            trace_config = self.tracing_config.get_langchain_config(task_id)
            config.setdefault("metadata", {}).update(trace_config.get("metadata", {}))
        
        logger.info(f"🔍 追踪配置: {config.get('metadata', {})}")
        
        # Run graph with streaming
        async for event in self.app.astream(initial_state, cast(RunnableConfig, config)):
            for node_name, output_state in event.items():
                yield {
                "type": "node_complete",
                "node": str(node_name),
                "state": cast(Dict[str, Any], output_state),  # ✅ 不再要求 ResearchState
                "task_id": task_id,
            }
    
    async def get_state(self, task_id: str) -> Optional[ResearchState]:
        config = {"configurable": {"thread_id": task_id}}
        try:
            snap = await self.app.aget_state(cast(RunnableConfig, config))
            if not snap:
                return None
            return cast(ResearchState, snap.values)  # ✅ 让 Pylance 通过
        except Exception:
            return None



# ============================================================================
# Singleton
# ============================================================================

_service: Optional[DeepResearchLangGraphService] = None


def get_deepresearch_langgraph_service() -> DeepResearchLangGraphService:
    """Get LangGraph service instance"""
    global _service
    if _service is None:
        _service = DeepResearchLangGraphService()
    return _service


# ============================================================================
# Example Usage
# ============================================================================

async def main():
    """示例：运行 LangGraph 研究任务"""
    service = get_deepresearch_langgraph_service()
    
    async for event in service.run_research(
        query="调研一下推荐系统的最新进展",
        max_rounds=3,
    ):
        node = event.get("node")
        state = event.get("state", {})
        
        print(f"\n{'='*60}")
        print(f"Node: {node}")
        
        if node == "tool_loop":
            print(f"Next Action: {state.get('next_action')}")
            print(f"Tool Results: {len(state.get('tool_results', []))} items")
            print(f"New Packs: {len(state.get('evidence_packs', []))} packs")
        
        elif node == "update_report":
            print(f"Summary: {state.get('summary', '')[:200]}...")
            print(f"Findings: {len(state.get('findings', []))} total")
            print(f"Open Questions: {state.get('open_questions')}")
        
        elif node == "write_answer":
            print(f"Final Answer:\n{state.get('final_answer', '')[:500]}...")
            print(f"Citations: {len(state.get('citations', []))}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
