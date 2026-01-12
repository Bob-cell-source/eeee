"""
Report Exporter - 导出研究报告为文件
"""
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
import json

from ..config import get_settings

logger = logging.getLogger(__name__)

class ReportExporter:
    """将研究报告导出为不同的格式"""
    
    def __init__(self):
        self.settings = get_settings()
        self.reports_dir = self.settings.reports_dir
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def export_markdown(
        self,
        task_id: str,
        title: str,
        content: str,
        citations: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        导出为MD文件
        
        Args:
            task_id: 导出的研究任务ID
            title: Report title
            content: Report content (already Markdown formatted)
            citations: List of citations
            metadata: Additional metadata
        
        Returns:
            Path to the saved report file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"report_{task_id}_{timestamp}.md"
        filepath = self.reports_dir / filename
        
        # Build report content
        report_lines = [
            f"# {title}",
            "",
            f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
            f"*Task ID: {task_id}*",
            "",
            "---",
            "",
            content,
            "",
        ]
        
        # Add citations section
        if citations:
            report_lines.extend([
                "---",
                "",
                "## References",
                "",
            ])
            
            for i, citation in enumerate(citations, 1):
                source = citation.get("source", "unknown")
                ref = citation.get("ref", "")
                snippet = citation.get("snippet", "")
                url = citation.get("url") or citation.get("metadata", {}).get("url") or ""

                title_text = (
                    citation.get("title")
                    or citation.get("metadata", {}).get("title")
                    or ref
                )
                location = citation.get("location") or citation.get("metadata", {}).get("location")
                if source == "arxiv":
                    title_text = citation.get("metadata", {}).get("title", ref)
                    url = f"https://arxiv.org/abs/{ref}"
                    report_lines.append(f"{i}. [{title_text}]({url}) - arXiv:{ref}")
                elif source == "github":
                    report_lines.append(f"{i}. [{ref}](https://github.com/{ref}) - GitHub")
                elif source == "upload":
                    doc_title = citation.get("metadata", {}).get("title", ref)
                    report_lines.append(f"{i}. {doc_title} (Uploaded Document: {ref})")
                elif source in ("web", "visit", "web_page"):
                    # web_search/web_visit 的引用
                    if url:
                        report_lines.append(f"{i}. [{title_text}]({url}) - Web")
                    else:
                        report_lines.append(f"{i}. {title_text} - Web")
                    if location:
                        report_lines.append(f"   - Location: {location}")

                elif source in ("file", "document"):
                    # file_parse 的引用
                    # ref 可以放文件名/文件ID，url 可放原始URL（如果是下载的）
                    line = f"{i}. {title_text} - File"
                    if url and url != ref:
                        line += f" ([source]({url}))"
                    report_lines.append(line)
                    if location:
                        report_lines.append(f"   - Location: {location}")

                elif source in ("image", "figure"):
                    # 图片解析的引用（至少能让报告里回溯到图片来源）
                    line = f"{i}. {title_text} - Image"
                    if url:
                        line += f" ([source]({url}))"
                    report_lines.append(line)
                    if location:
                        report_lines.append(f"   - Location: {location}")

                else:
                    report_lines.append(f"{i}. {ref}")
                
                if snippet:
                    # Truncate long snippets
                    snippet_display = snippet[:200] + "..." if len(snippet) > 200 else snippet
                    report_lines.append(f"   > {snippet_display}")
                report_lines.append("")
        
        # Add metadata section if provided
        if metadata:
            report_lines.extend([
                "---",
                "",
                "## Metadata",
                "",
                "```json",
                json.dumps(metadata, indent=2, ensure_ascii=False),
                "```",
            ])
        
        # Write file
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write("\n".join(report_lines))
            
            logger.info(f"Exported report to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Error exporting report: {e}")
            raise
    
    def export_json(
        self,
        task_id: str,
        data: Dict[str, Any],
    ) -> Path:
        """
        Export report data as JSON file
        
        Args:
            task_id: Research task ID
            data: Report data dictionary
        
        Returns:
            Path to the saved JSON file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"report_{task_id}_{timestamp}.json"
        filepath = self.reports_dir / filename
        
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Exported JSON report to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Error exporting JSON report: {e}")
            raise
    
    def get_report_path(self, filename: str) -> Optional[Path]:
        """Get full path to a report file"""
        filepath = self.reports_dir / filename
        return filepath if filepath.exists() else None
    
    def list_reports(self) -> List[Dict[str, Any]]:
        """List all saved reports"""
        reports = []
        for filepath in self.reports_dir.glob("report_*.md"):
            reports.append({
                "filename": filepath.name,
                "path": str(filepath),
                "size": filepath.stat().st_size,
                "modified": datetime.fromtimestamp(filepath.stat().st_mtime).isoformat(),
            })
        return sorted(reports, key=lambda x: x["modified"], reverse=True)


def get_exporter() -> ReportExporter:
    """Get exporter instance"""
    return ReportExporter()
