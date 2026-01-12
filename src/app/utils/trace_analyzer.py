"""
追踪日志分析工具 - 查看和分析 DeepResearch 工作流追踪
"""
import json
import os
from typing import List, Dict, Any,Optional
from pathlib import Path
from datetime import datetime
import argparse


class TraceAnalyzer:
    """追踪日志分析器"""
    
    def __init__(self, trace_file: str):
        self.trace_file = trace_file
        self.traces = []
        self._load_traces()
    
    def _load_traces(self):
        """加载追踪日志"""
        if not os.path.exists(self.trace_file):
            print(f"❌ 追踪文件不存在: {self.trace_file}")
            return
        
        with open(self.trace_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    self.traces.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
        
        print(f"✅ 加载了 {len(self.traces)} 条追踪记录")
    
    def print_summary(self):
        """打印追踪摘要"""
        if not self.traces:
            print("📭 没有追踪记录")
            return
        
        # 按节点统计
        node_stats = {}
        llm_calls = []
        total_duration = 0
        
        for trace in self.traces:
            if trace.get("type") == "llm_call":
                llm_calls.append(trace)
            else:
                node = trace.get("node", "unknown")
                duration = trace.get("duration_ms", 0)
                
                if node not in node_stats:
                    node_stats[node] = {"count": 0, "total_duration": 0, "errors": 0}
                
                node_stats[node]["count"] += 1
                node_stats[node]["total_duration"] += duration
                total_duration += duration
                
                if trace.get("error"):
                    node_stats[node]["errors"] += 1
        
        # 打印统计
        print("\n" + "="*80)
        print("📊 节点执行统计")
        print("="*80)
        
        for node, stats in sorted(node_stats.items()):
            avg_duration = stats["total_duration"] / stats["count"] if stats["count"] > 0 else 0
            error_marker = " ⚠️" if stats["errors"] > 0 else ""
            print(f"  {node:20s} | 执行: {stats['count']:2d}次 | 平均耗时: {avg_duration:6.0f}ms{error_marker}")
        
        print(f"\n  总耗时: {total_duration:.0f}ms")
        
        if llm_calls:
            print("\n" + "="*80)
            print(f"🤖 LLM 调用统计 (共 {len(llm_calls)} 次)")
            print("="*80)
            
            total_llm_time = sum(call.get("duration_ms", 0) for call in llm_calls)
            avg_llm_time = total_llm_time / len(llm_calls) if llm_calls else 0
            
            for call in llm_calls:
                node = call.get("node", "unknown")
                model = call.get("model", "unknown")
                duration = call.get("duration_ms", 0)
                print(f"  {node:20s} | Model: {model:15s} | {duration:6.0f}ms")
            
            print(f"\n  LLM 总耗时: {total_llm_time:.0f}ms (平均 {avg_llm_time:.0f}ms/次)")
    
    def print_timeline(self):
        """打印执行时间线"""
        if not self.traces:
            return
        
        print("\n" + "="*80)
        print("⏱️ 执行时间线")
        print("="*80)
        
        node_traces = [t for t in self.traces if t.get("type") != "llm_call"]
        
        for i, trace in enumerate(node_traces, 1):
            timestamp = trace.get("timestamp", "")
            node = trace.get("node", "unknown")
            duration = trace.get("duration_ms", 0)
            error = trace.get("error")
            
            # 格式化时间戳
            try:
                dt = datetime.fromisoformat(timestamp)
                time_str = dt.strftime("%H:%M:%S")
            except:
                time_str = timestamp[:8] if len(timestamp) >= 8 else timestamp
            
            status = "❌" if error else "✅"
            print(f"  {i:2d}. [{time_str}] {status} {node:20s} ({duration:6.0f}ms)")
            
            if error:
                print(f"      错误: {error[:100]}")
            
            # 打印关键输出摘要
            output = trace.get("output_summary", {})
            if output:
                summary_parts = []
                if "next_action" in output:
                    summary_parts.append(f"action={output['next_action']}")
                if "evidence_packs_count" in output:
                    summary_parts.append(f"packs={output['evidence_packs_count']}")
                if "findings_count" in output:
                    summary_parts.append(f"findings={output['findings_count']}")
                
                if summary_parts:
                    print(f"      → {', '.join(summary_parts)}")
    
    def print_details(self, node_name: Optional[str] = None):
        """打印详细信息"""
        print("\n" + "="*80)
        print(f"🔍 详细追踪 {'- Node: ' + node_name if node_name else ''}")
        print("="*80)
        
        for trace in self.traces:
            if node_name and trace.get("node") != node_name:
                continue
            
            if trace.get("type") == "llm_call":
                print(f"\n🤖 LLM 调用 ({trace.get('node', 'unknown')})")
                print(f"  Model: {trace.get('model', 'unknown')}")
                print(f"  Duration: {trace.get('duration_ms', 0):.0f}ms")
                print(f"  Prompt: {trace.get('prompt_summary', '')[:200]}")
                print(f"  Response: {trace.get('response_summary', '')[:200]}")
            else:
                print(f"\n📦 节点执行: {trace.get('node', 'unknown')}")
                print(f"  Timestamp: {trace.get('timestamp', '')}")
                print(f"  Duration: {trace.get('duration_ms', 0):.0f}ms")
                
                input_summary = trace.get("input_summary", {})
                output_summary = trace.get("output_summary", {})
                
                if input_summary:
                    print(f"  Input: {json.dumps(input_summary, ensure_ascii=False)[:200]}")
                if output_summary:
                    print(f"  Output: {json.dumps(output_summary, ensure_ascii=False)[:200]}")
                
                if trace.get("error"):
                    print(f"  ❌ Error: {trace['error']}")


def find_latest_trace_file(trace_dir: str = "logs/traces") -> Optional[str]:
    """查找最新的追踪文件"""
    if not os.path.exists(trace_dir):
        return None
    
    trace_files = [
        os.path.join(trace_dir, f)
        for f in os.listdir(trace_dir)
        if f.startswith("trace_") and f.endswith(".jsonl")
    ]
    
    if not trace_files:
        return None
    
    # 按修改时间排序
    trace_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return trace_files[0]


def main():
    parser = argparse.ArgumentParser(description="DeepResearch 追踪日志分析工具")
    parser.add_argument("--file", "-f", help="追踪文件路径（默认使用最新的）")
    parser.add_argument("--summary", "-s", action="store_true", help="显示摘要统计")
    parser.add_argument("--timeline", "-t", action="store_true", help="显示执行时间线")
    parser.add_argument("--details", "-d", nargs="?", const=True, help="显示详细信息（可指定节点名）")
    parser.add_argument("--all", "-a", action="store_true", help="显示所有信息")
    
    args = parser.parse_args()
    
    # 查找追踪文件
    trace_file = args.file
    if not trace_file:
        trace_file = find_latest_trace_file()
        if not trace_file:
            print("❌ 未找到追踪文件，请先运行 DeepResearch 任务")
            print("提示: 追踪文件位于 logs/traces/ 目录")
            return
        print(f"📂 使用最新追踪文件: {trace_file}")
    
    # 创建分析器
    analyzer = TraceAnalyzer(trace_file)
    
    # 显示信息
    if args.all:
        analyzer.print_summary()
        analyzer.print_timeline()
        analyzer.print_details()
    else:
        if args.summary or (not args.timeline and not args.details):
            analyzer.print_summary()
        
        if args.timeline:
            analyzer.print_timeline()
        
        if args.details:
            node_name = args.details if isinstance(args.details, str) else None
            analyzer.print_details(node_name)


if __name__ == "__main__":
    main()
