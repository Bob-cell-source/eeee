"""
工作流追踪工具 - 支持 LangSmith 和本地日志
"""
import os
import logging
import json
from typing import Any, Dict, Optional, Callable
from datetime import datetime
from functools import wraps
import asyncio

logger = logging.getLogger(__name__)


class TracingConfig:
    """追踪配置管理"""
    
    def __init__(self):
        self.langsmith_enabled = False
        self.langsmith_project = None
        self.local_trace_enabled = True
        self.trace_file = None
        self._setup()
    
    def _setup(self):
        """初始化追踪配置"""
        # 尝试启用 LangSmith
        langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
        langsmith_tracing = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
        
        if langchain_api_key and langsmith_tracing:
            try:
                # 验证 LangSmith 可用性
                import langsmith
                self.langsmith_enabled = True
                self.langsmith_project = os.getenv("LANGCHAIN_PROJECT", "deepresearch")
                logger.info(f"✅ LangSmith 追踪已启用 - 项目: {self.langsmith_project}")
            except ImportError:
                logger.warning("⚠️ LangSmith 不可用 (pip install langsmith)，使用本地追踪")
                self.langsmith_enabled = False
        else:
            logger.info("ℹ️ LangSmith 未配置，使用本地追踪")
        
        # 配置本地追踪文件
        trace_dir = os.getenv("TRACE_DIR", "logs/traces")
        os.makedirs(trace_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.trace_file = os.path.join(trace_dir, f"trace_{timestamp}.jsonl")
        logger.info(f"📝 本地追踪文件: {self.trace_file}")
    
    def get_langchain_config(self, task_id: str) -> Dict[str, Any]:
        """获取 LangChain 运行配置（包含追踪元数据）"""
        config = {
            "configurable": {"thread_id": task_id},
            "metadata": {
                "task_id": task_id,
                "timestamp": datetime.now().isoformat(),
            }
        }
        
        if self.langsmith_enabled:
            assert self.langsmith_project is not None, "langsmith_project is None"
            config["metadata"]["langsmith_project"] = self.langsmith_project

        
        return config


# 全局追踪配置实例
_tracing_config: Optional[TracingConfig] = None


def get_tracing_config() -> TracingConfig:
    """获取追踪配置单例"""
    global _tracing_config
    if _tracing_config is None:
        _tracing_config = TracingConfig()
    return _tracing_config


class LocalTracer:
    """本地追踪记录器"""
    
    def __init__(self, trace_file: str):
        self.trace_file = trace_file
        self.traces = []
    
    def log_node_execution(
        self,
        task_id: str,
        node_name: str,
        input_state: Dict[str, Any],
        output_state: Dict[str, Any],
        duration_ms: float,
        error: Optional[str] = None
    ):
        """记录节点执行"""
        trace_entry = {
            "timestamp": datetime.now().isoformat(),
            "task_id": task_id,
            "node": node_name,
            "duration_ms": duration_ms,
            "input_summary": self._summarize_state(input_state),
            "output_summary": self._summarize_state(output_state),
            "error": error,
        }
        
        # 写入文件
        try:
            with open(self.trace_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(trace_entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"写入追踪文件失败: {e}")
        
        self.traces.append(trace_entry)
    
    def log_llm_call(
        self,
        task_id: str,
        node_name: str,
        model: str,
        prompt_summary: str,
        response_summary: str,
        duration_ms: float,
        token_usage: Optional[Dict[str, int]] = None
    ):
        """记录 LLM 调用"""
        trace_entry = {
            "timestamp": datetime.now().isoformat(),
            "task_id": task_id,
            "type": "llm_call",
            "node": node_name,
            "model": model,
            "prompt_summary": prompt_summary[:500],
            "response_summary": response_summary[:500],
            "duration_ms": duration_ms,
            "token_usage": token_usage,
        }
        
        try:
            with open(self.trace_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(trace_entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"写入追踪文件失败: {e}")
    
    def _summarize_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """精简状态用于追踪"""
        summary = {}
        
        # 只保留关键字段的摘要
        key_fields = ["task_id", "query", "round_id", "next_action", "stop_reason"]
        for field in key_fields:
            if field in state:
                summary[field] = state[field]
        
        # 计数型字段
        if "evidence_packs" in state:
            summary["evidence_packs_count"] = len(state.get("evidence_packs", []))
        if "tool_results" in state:
            summary["tool_results_count"] = len(state.get("tool_results", []))
        if "findings" in state:
            summary["findings_count"] = len(state.get("findings", []))
        if "open_questions" in state:
            summary["open_questions_count"] = len(state.get("open_questions", []))
        
        # 文本摘要
        if "summary" in state and state["summary"]:
            summary["summary_preview"] = state["summary"][:200]
        if "workspace" in state and state["workspace"]:
            summary["workspace_length"] = len(state["workspace"])
        
        return summary
    
    def get_traces(self) -> list:
        """获取所有追踪记录"""
        return self.traces


def trace_node(node_name: str):
    """节点执行追踪装饰器"""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(state: Dict[str, Any], *args, **kwargs):
            config = get_tracing_config()
            task_id = state.get("task_id", "unknown")
            
            # 记录输入
            logger.info(f"🚀 [{node_name}] 开始执行 - Task: {task_id}, Round: {state.get('round_id', 0)}")
            logger.debug(f"📥 [{node_name}] 输入状态摘要: {_log_state_summary(state)}")
            
            start_time = asyncio.get_event_loop().time()
            error = None
            output_state = {}
            
            try:
                # 执行节点
                output_state = await func(state, *args, **kwargs)
                
                # 记录输出
                logger.info(f"✅ [{node_name}] 执行完成")
                logger.debug(f"📤 [{node_name}] 输出状态摘要: {_log_state_summary(output_state)}")
                
            except Exception as e:
                error = str(e)
                logger.error(f"❌ [{node_name}] 执行失败: {error}", exc_info=True)
                raise
            
            finally:
                # 计算执行时间
                duration_ms = (asyncio.get_event_loop().time() - start_time) * 1000
                
                # 本地追踪
                if config.local_trace_enabled and config.trace_file:
                    tracer = LocalTracer(config.trace_file)
                    tracer.log_node_execution(
                        task_id=task_id,
                        node_name=node_name,
                        input_state=state,
                        output_state=output_state,
                        duration_ms=duration_ms,
                        error=error
                    )
            
            return output_state
        
        return wrapper
    return decorator


def _log_state_summary(state: Dict[str, Any]) -> str:
    """生成状态摘要日志"""
    summary_parts = []
    
    if "query" in state:
        summary_parts.append(f"query='{state['query'][:50]}...'")
    if "round_id" in state:
        summary_parts.append(f"round={state['round_id']}")
    if "next_action" in state:
        summary_parts.append(f"action={state['next_action']}")
    if "evidence_packs" in state:
        summary_parts.append(f"packs={len(state.get('evidence_packs', []))}")
    if "tool_results" in state:
        summary_parts.append(f"tools={len(state.get('tool_results', []))}")
    if "findings" in state:
        summary_parts.append(f"findings={len(state.get('findings', []))}")
    
    return ", ".join(summary_parts) if summary_parts else "empty"


def trace_llm_call(node_name: str, model: str):
    """LLM 调用追踪装饰器"""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            config = get_tracing_config()
            
            # 提取 task_id（如果在参数中）
            task_id = "unknown"
            if args and isinstance(args[0], dict):
                task_id = args[0].get("task_id", "unknown")
            
            logger.info(f"🤖 [{node_name}] LLM 调用开始 - Model: {model}")
            
            start_time = asyncio.get_event_loop().time()
            
            try:
                result = await func(*args, **kwargs)
                
                duration_ms = (asyncio.get_event_loop().time() - start_time) * 1000
                logger.info(f"✅ [{node_name}] LLM 调用完成 - {duration_ms:.0f}ms")
                
                # 本地追踪
                if config.local_trace_enabled and config.trace_file:
                    tracer = LocalTracer(config.trace_file)
                    
                    # 提取 prompt 和 response 摘要
                    prompt_summary = str(args)[:200] if args else ""
                    response_summary = str(result)[:200] if result else ""
                    
                    tracer.log_llm_call(
                        task_id=task_id,
                        node_name=node_name,
                        model=model,
                        prompt_summary=prompt_summary,
                        response_summary=response_summary,
                        duration_ms=duration_ms
                    )
                
                return result
            
            except Exception as e:
                logger.error(f"❌ [{node_name}] LLM 调用失败: {e}", exc_info=True)
                raise
        
        return wrapper
    return decorator


def print_workflow_diagram():
    """打印工作流程图（用于调试和文档）"""
    diagram = """
    
╔═══════════════════════════════════════════════════════════════════════════╗
║                    DeepResearch LangGraph 工作流                          ║
╚═══════════════════════════════════════════════════════════════════════════╝

    START
      ↓
┌─────────────────┐
│  parse_task     │  → 解析查询，提取任务规格
│  (LLM: qwen3)   │     输入: query
└────────┬────────┘     输出: task_spec, open_questions
         ↓
┌─────────────────┐
│ build_workspace │  → 构建工作空间（检索 + 去重）
│  (Retriever)    │     输入: query, round_id, evidence_packs
└────────┬────────┘     输出: workspace (紧凑型，带 EvidencePack 摘要)
         ↓
┌─────────────────┐
│   tool_loop     │  → LLM 自主工具调用循环
│ (LLM + Tools)   │     输入: workspace, open_questions
└────────┬────────┘     输出: tool_results, new_evidence_packs, next_action
         ↓
    ┌────────┐
    │ 检查   │
    │action? │
    └───┬────┘
        ├─→ answer/stop ──────────────────────────┐
        │                                          ↓
        └─→ continue                       ┌─────────────────┐
             ↓                             │ write_answer    │
    ┌─────────────────┐                   │  (LLM: qwen3)   │
    │normalize_evidence│ → 存储到 Milvus  └────────┬────────┘
    │   (Milvus)      │                            ↓
    └────────┬────────┘                          END
             ↓
    ┌─────────────────┐
    │ update_report   │  → 更新中心报告
    │  (LLM: qwen3)   │     输入: new_evidence_packs, summary, findings
    └────────┬────────┘     输出: updated summary, new findings, open_questions
             ↓
    ┌─────────────────┐
    │  check_stop     │  → 检查停止条件
    │  (Logic)        │     - max_rounds? no_questions? low_evidence?
    └────────┬────────┘
             ↓
        ┌────────┐
        │ 停止?  │
        └───┬────┘
            ├─→ stop ─────────────→ write_answer → END
            │
            └─→ continue
                 ↓
        ┌─────────────────┐
        │ increment_round │
        └────────┬────────┘
                 ↓
          [循环回 build_workspace]

═══════════════════════════════════════════════════════════════════════════

关键数据流：
  • task_spec: 任务规格（主题、问题、输出格式、约束）
  • workspace: 紧凑工作空间（摘要 + 最近发现 + EvidencePack 索引）
  • evidence_packs: 渐进式展示的证据包（pack_id, snippet, key_points）
  • next_action: 流程控制（continue/answer/stop）
  • findings: 累积发现（带引用 [pack_id]）
  • open_questions: 待解决问题列表

追踪配置：
  • LangSmith: 设置 LANGCHAIN_API_KEY, LANGCHAIN_TRACING_V2=true
  • 本地追踪: logs/traces/trace_YYYYMMDD_HHMMSS.jsonl
    """
    print(diagram)


if __name__ == "__main__":
    # 测试追踪配置
    print_workflow_diagram()
    config = get_tracing_config()
    print(f"\n追踪状态:")
    print(f"  LangSmith: {'✅ 已启用' if config.langsmith_enabled else '❌ 未启用'}")
    print(f"  本地追踪: {'✅ 已启用' if config.local_trace_enabled else '❌ 未启用'}")
    print(f"  追踪文件: {config.trace_file}")
