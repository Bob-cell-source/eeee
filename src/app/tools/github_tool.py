# 这是一个github工具

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import asyncio
import httpx

from ..config import get_settings
from .utils import validate_non_empty

logger = logging.getLogger(__name__)
@dataclass
class GitHubRepo:
    """Represents a GitHub repository"""
    full_name: str  # owner/repo
    name: str
    owner: str
    description: str
    url: str
    stars: int
    forks: int
    language: str
    topics: List[str]
    readme_url: str
    updated_at: str
    created_at: str

@dataclass
class GitHubCodeResult:
    repo_full_name:str
    file_path: str
    file_name: str
    html_url: str
    snippet: str

class GitHubTool:

    API_BASE = "https://api.github.com"
    
    def __init__(self, token: Optional[str] = None, max_results: int = 10, blacklist: Optional[List[str]] = None, rate_limit: int = 5):
        self.token = token or get_settings().github_token
        self.max_results = max_results
        self.blacklist = set(blacklist or get_settings().github_blacklist.split(","))
        self._rate_limiter = asyncio.Semaphore(rate_limit)

    def _get_headers(self) -> Dict[str, str]:

        headers = {
            "Accept": "application/vnd.github.v3+json",
            "user-agent": "Research-Agent"
        }
        if self.token:
            headers["Authorization"] = f"token {self.token}"
        return headers
    async def search_repos(
            self,
        query: str,
        max_results: Optional[int] = None,
        sort: str = "stars",  # stars, forks, help-wanted-issues, updated
        order: str = "desc",
        language: Optional[str] = None,
    )-> List[GitHubRepo]:
        try:
            validate_non_empty(query, "query")
        except ValueError as e:
            logger.warning(f"Invalid query: {e}")
            return []
        
        max_results = max_results or self.max_results

        search_query = query
        if language:
            search_query += f" language:{language}"
        params = {
            "q": search_query,
            "sort": sort,
            "order": order,
            "per_page": min(max_results, 100),
        }

        try:
            async with self._rate_limiter:
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        f"{self.API_BASE}/search/repositories",
                        params=params,
                        headers=self._get_headers(),
                        timeout=30.0,
                    )
                response.raise_for_status()
                
                data = response.json()
                return self._parse_repos(data.get("items", []))
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 403:
                logger.warning("GitHub API rate limit reached")
            else:
                logger.error(f"GitHub search error: {e}")
            return []
        except Exception as e:
            logger.error(f"GitHub search error: {e}")
            return []
        
    async def search_code(
        self,
        query: str,
        max_results: Optional[int] = None,
        language: Optional[str] = None,
        repo: Optional[str] = None,
    ) -> List[GitHubCodeResult]:
        """
        search_code 的 Docstring
        
        :param self: 搜索的参数
        :param query: 搜索请求
        :type query: str
        :param max_results: 搜索的最大结果数
        :type max_results: Optional[int]
        :param language: 要搜索的语言
        :type language: Optional[str]
        :param repo: 指定repo
        :type repo: Optional[str]
        :return: 返回code列表
        :rtype: List[GitHubCodeResult]
        """
        if not self.token:
            logger.warning("GitHub code search requires authentication")
            return []
        
        max_results = max_results or self.max_results

        search_query = query
        if language:
            search_query += f"language:{language}"
        if repo:
            search_query += f"repo:{repo}"
        params = {
            "q": search_query,
            "per_page": min(max_results, 100),
        }
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.API_BASE}/search/code",
                    params=params,
                    headers=self._get_headers(),
                    timeout=30.0,
                )
                response.raise_for_status()
                
                data = response.json()
                return self._parse_code_results(data.get("items", []))
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 403:
                logger.warning("GitHub API rate limit reached")
            else:
                logger.error(f"GitHub code search error: {e}")
            return []
        except Exception as e:
            logger.error(f"GitHub code search error: {e}")
            return []
    async def get_repo(self, full_name: str) -> Optional[GitHubRepo]:
        """
        Get repository details
        
        Args:
            full_name: Repository full name (owner/repo)
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.API_BASE}/repos/{full_name}",
                    headers=self._get_headers(),
                    timeout=30.0,
                )
                response.raise_for_status()
                
                data = response.json()
                repos = self._parse_repos([data])
                return repos[0] if repos else None
        except Exception as e:
            logger.error(f"GitHub get repo error: {e}")
            return None
    
    async def get_readme(self, full_name: str) -> Optional[str]:
        """
        Get repository README content
        
        Args:
            full_name: Repository full name (owner/repo)
        """
        try:
            async with httpx.AsyncClient() as client:
                # First, get README metadata
                response = await client.get(
                    f"{self.API_BASE}/repos/{full_name}/readme",
                    headers={
                        **self._get_headers(),
                        "Accept": "application/vnd.github.v3.raw",
                    },
                    timeout=30.0,
                )
                response.raise_for_status()
                
                return response.text
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.debug(f"No README found for {full_name}")
            else:
                logger.error(f"GitHub get README error: {e}")
            return None
        except Exception as e:
            logger.error(f"GitHub get README error: {e}")
            return None
        
    def _parse_repos(self, items: List[Dict[str, Any]]) -> List[GitHubRepo]:
        """Parse repository search results"""
        repos = []
        for item in items:
            try:
                full_name = item.get("full_name", "")
                # 过滤黑名单仓库
                if full_name in self.blacklist:
                    logger.debug(f"Skipping blacklisted repo: {full_name}")
                    continue
                
                owner = item.get("owner", {})
                repos.append(GitHubRepo(
                    full_name=full_name,
                    name=item.get("name", ""),
                    owner=owner.get("login", ""),
                    description=item.get("description") or "",
                    url=item.get("html_url", ""),
                    stars=item.get("stargazers_count", 0),
                    forks=item.get("forks_count", 0),
                    language=item.get("language") or "",
                    topics=item.get("topics", []),
                    readme_url=f"https://github.com/{item.get('full_name', '')}/blob/main/README.md",
                    updated_at=item.get("updated_at", ""),
                    created_at=item.get("created_at", ""),
                ))
            except Exception as e:
                logger.error(f"Error parsing repo: {e}")
        
        return repos
    def _parse_code_results(self, items: List[Dict[str, Any]]) -> List[GitHubCodeResult]:
        """Parse code search results"""
        results = []
        for item in items:
            try:
                repo = item.get("repository", {})
                results.append(GitHubCodeResult(
                    repo_full_name=repo.get("full_name", ""),
                    file_path=item.get("path", ""),
                    file_name=item.get("name", ""),
                    html_url=item.get("html_url", ""),
                    snippet=item.get("text_matches", [{}])[0].get("fragment", "") if item.get("text_matches") else "",
                ))
            except Exception as e:
                logger.error(f"Error parsing code result: {e}")
        
        return results
    
    def repo_to_dict(self, repo: GitHubRepo) -> Dict[str, Any]:
        """Convert GitHubRepo to dictionary for serialization"""
        return {
            "full_name": repo.full_name,
            "name": repo.name,
            "owner": repo.owner,
            "description": repo.description,
            "url": repo.url,
            "stars": repo.stars,
            "forks": repo.forks,
            "language": repo.language,
            "topics": repo.topics,
            "readme_url": repo.readme_url,
            "updated_at": repo.updated_at,
            "created_at": repo.created_at,
        }
    
def get_github_tool() -> GitHubTool:
    """Get GitHub tool instance"""
    settings = get_settings()
    blacklist = []
    if settings.github_blacklist:
        blacklist = [repo.strip() for repo in settings.github_blacklist.split(",") if repo.strip()]
    return GitHubTool(
        token=settings.github_token if settings.github_token else None,
        max_results=settings.github_max_results,
        blacklist=blacklist,
        rate_limit=settings.github_rate_limit,
    )