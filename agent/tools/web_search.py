"""
Web search tool for regulatory & market context.

Lets the agent look up CMS guidelines, health plan news,
or provider organization backgrounds.
"""

import os
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool

load_dotenv()

@tool
def search_web(query: str) -> str:
    """Search the web for roster-related regulations, orgs, or news.
    
    Use this for CMS context, provider group background, or compliance info.
    """
    if not os.environ.get("TAVILY_API_KEY"):
        return "⚠️ Web search is unavailable (no API key). Check `.env`."
    
    try:
        search = TavilySearchResults(max_results=3)
        results = search.invoke(query)
        
        if not results:
            return f"Search for '{query}' returned no results."
        
        output = f"### Web Results for '{query}'\n\n"
        for i, res in enumerate(results):
            content = res.get("content", "No content").strip()
            url = res.get("url", "#")
            output += f"{i+1}. {content[:400]}... [Read more]({url})\n\n"
        return output
        
    except Exception as e:
        return f"❌ Search error: {e}"

ALL_WEB_TOOLS = [search_web]
