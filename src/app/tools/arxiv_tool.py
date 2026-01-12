"""
arXiv Tool - Search and retrieve papers from arXiv
"""
import logging
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import httpx
import xml.etree.ElementTree as ET

from ..config import get_settings
from .utils import validate_non_empty

logger = logging.getLogger(__name__)


@dataclass
class ArxivPaper:
    """Represents an arXiv paper"""
    arxiv_id: str
    title: str
    summary: str
    authors: List[str]
    categories: List[str]
    published: str
    updated: str
    pdf_url: str
    abs_url: str


class ArxivTool:
    """Tool for searching and retrieving arXiv papers"""
    
    BASE_URL = "https://export.arxiv.org/api/query"
    
    def __init__(self, max_results: int = 10):
        self.max_results = max_results
    
    async def search(
        self,
        query: str,
        max_results: Optional[int] = None,
        sort_by: str = "relevance",  # relevance, lastUpdatedDate, submittedDate
        sort_order: str = "descending",
    ) -> List[ArxivPaper]:
        """
        Search arXiv for papers matching the query
        
        Args:
            query: Search query (supports arXiv query syntax)
            max_results: Maximum number of results to return
            sort_by: Sort criteria
            sort_order: Sort order (ascending/descending)
        
        Returns:
            List of ArxivPaper objects
        """
        try:
            validate_non_empty(query, "query")
        except ValueError as e:
            logger.warning(f"Invalid query: {e}")
            return []
        
        max_results = max_results or self.max_results
        
        # Build query parameters
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results,
            "sortBy": sort_by,
            "sortOrder": sort_order,
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    self.BASE_URL,
                    params=params,
                    timeout=30.0,
                )
                response.raise_for_status()
                
                return self._parse_response(response.text)
        except httpx.HTTPStatusError as e:
            logger.error(f"arXiv HTTP error: {e.response.status_code} - {e}")
            return []
        except Exception as e:
            logger.error(f"arXiv search error: {e}", exc_info=True)
            return []
    
    async def search_by_category(
        self,
        category: str,
        query: Optional[str] = None,
        max_results: Optional[int] = None,
    ) -> List[ArxivPaper]:
        """
        Search arXiv by category
        
        Args:
            category: arXiv category (e.g., cs.AI, cs.LG, physics.optics)
            query: Additional search query
            max_results: Maximum number of results
        """
        search_query = f"cat:{category}"
        if query:
            search_query = f"({search_query}) AND (all:{query})"
        
        max_results = max_results or self.max_results
        
        params = {
            "search_query": search_query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    self.BASE_URL,
                    params=params,
                    timeout=30.0,
                )
                response.raise_for_status()
                
                return self._parse_response(response.text)
        except Exception as e:
            logger.error(f"arXiv category search error: {e}")
            return []
    
    async def get_paper(self, arxiv_id: str) -> Optional[ArxivPaper]:
        """
        Get a specific paper by arXiv ID
        
        Args:
            arxiv_id: The arXiv ID (e.g., 2301.00001 or 2301.00001v1)
        """
        # Clean the ID
        arxiv_id = arxiv_id.replace("arXiv:", "").strip()
        
        params = {
            "id_list": arxiv_id,
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    self.BASE_URL,
                    params=params,
                    timeout=30.0,
                )
                response.raise_for_status()
                
                papers = self._parse_response(response.text)
                return papers[0] if papers else None
        except Exception as e:
            logger.error(f"arXiv get paper error: {e}")
            return None
    
    def _parse_response(self, xml_content: str) -> List[ArxivPaper]:
        """Parse arXiv API XML response"""
        papers = []
        
        try:
            # Define namespace
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            
            root = ET.fromstring(xml_content)
            
            for entry in root.findall("atom:entry", ns):
                # Extract arXiv ID from the id URL
                id_elem = entry.find("atom:id", ns)
                if id_elem is None:
                    continue
                
                id_url = id_elem.text or ""
                arxiv_id = id_url.split("/abs/")[-1] if "/abs/" in id_url else ""
                
                # Title
                title_elem = entry.find("atom:title", ns)
                title = (title_elem.text or "").strip().replace("\n", " ") if title_elem is not None else ""
                
                # Summary/Abstract
                summary_elem = entry.find("atom:summary", ns)
                summary = (summary_elem.text or "").strip() if summary_elem is not None else ""
                
                # Authors
                authors = []
                for author in entry.findall("atom:author", ns):
                    name_elem = author.find("atom:name", ns)
                    if name_elem is not None and name_elem.text:
                        authors.append(name_elem.text)
                
                # Categories
                categories = []
                for category in entry.findall("atom:category", ns):
                    term = category.get("term")
                    if term:
                        categories.append(term)
                
                # Published date
                published_elem = entry.find("atom:published", ns)
                published = (published_elem.text or "") if published_elem is not None else ""
                
                # Updated date
                updated_elem = entry.find("atom:updated", ns)
                updated = (updated_elem.text or "") if updated_elem is not None else ""
                
                # PDF URL
                pdf_url = ""
                abs_url = ""
                for link in entry.findall("atom:link", ns):
                    if link.get("title") == "pdf":
                        pdf_url = link.get("href", "")
                    elif link.get("type") == "text/html":
                        abs_url = link.get("href", "")
                
                if not pdf_url and arxiv_id:
                    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
                if not abs_url and arxiv_id:
                    abs_url = f"https://arxiv.org/abs/{arxiv_id}"
                
                papers.append(ArxivPaper(
                    arxiv_id=arxiv_id,
                    title=title,
                    summary=summary,
                    authors=authors,
                    categories=categories,
                    published=published,
                    updated=updated,
                    pdf_url=pdf_url,
                    abs_url=abs_url,
                ))
        except Exception as e:
            logger.error(f"Error parsing arXiv response: {e}")
        
        return papers
    
    def paper_to_dict(self, paper: ArxivPaper) -> Dict[str, Any]:
        """Convert ArxivPaper to dictionary for serialization"""
        return {
            "arxiv_id": paper.arxiv_id,
            "title": paper.title,
            "summary": paper.summary,
            "authors": paper.authors,
            "categories": paper.categories,
            "published": paper.published,
            "updated": paper.updated,
            "pdf_url": paper.pdf_url,
            "abs_url": paper.abs_url,
        }


def get_arxiv_tool() -> ArxivTool:
    """Get arXiv tool instance"""
    settings = get_settings()
    return ArxivTool(max_results=settings.arxiv_max_results)
