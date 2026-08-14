

import requests
import os

from bs4 import BeautifulSoup
from ddgs import DDGS

try:
    from googlesearch import search as _google_search
except Exception:
    _google_search = None
from typing import List, Dict, Any, Optional, Union
import numpy as np
import json

try:
    from sentence_transformers import util, SentenceTransformer
except Exception:
    pass

def search_exa(query:str,
               api_key:str = None,
               top_k = 5,
               search_type = "auto",
               **kwargs):
    """Search using Exa - the fastest and most accurate web search API for AI."""
    from exa_py import Exa
    if api_key is None:
        api_key = os.environ.get('EXA_API_KEY')
    exa = Exa(api_key)
    exa.headers["x-exa-integration"] = "npcpy"

    results = exa.search_and_contents(
        query,
        type=search_type,
        text=True
    )
    return results.results[0:top_k]

def search_searxng(query: str, num_results: int = 5, instance_url: str = None):
    """Search using SearXNG public instances."""
    instances = [instance_url] if instance_url else [
        os.environ.get('SEARXNG_URL'),
        'https://search.sapti.me',
        'https://searx.work',
        'https://search.ononoki.org',
        'https://searx.tiekoetter.com',
        'https://search.bus-hit.me',
        'https://priv.au',
    ]
    instances = [i for i in instances if i]

    for instance in instances:
        try:
            response = requests.get(
                f"{instance}/search",
                params={'q': query, 'format': 'json', 'categories': 'general'},
                headers={'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:121.0) Gecko/20100101 Firefox/121.0'},
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                results = []
                for r in data.get('results', [])[:num_results]:
                    results.append({
                        'title': r.get('title', ''),
                        'link': r.get('url', ''),
                        'content': r.get('content', '')
                    })
                if results:
                    return results
        except Exception as e:
            print(f"SearXNG {instance} failed: {e}")
            continue
    return []

def search_startpage(query: str, num_results: int = 5):
    """Search using Startpage (scraping)."""
    try:
        response = requests.post(
            'https://www.startpage.com/sp/search',
            data={'query': query, 'cat': 'web'},
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'text/html',
            },
            timeout=10
        )
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []
            for item in soup.select('.result')[:num_results]:
                title_el = item.select_one('h2') or item.select_one('a')
                link_el = item.select_one('a[href^="http"]')
                desc_el = item.select_one('p') or item.select_one('.description')
                if link_el:
                    results.append({
                        'title': title_el.get_text(strip=True) if title_el else '',
                        'link': link_el.get('href', ''),
                        'content': desc_el.get_text(strip=True) if desc_el else ''
                    })
            return results
    except Exception as e:
        print(f"Startpage search failed: {e}")
    return []

def search_brave(query: str, num_results: int = 5, api_key: str = None):
    """Search using Brave Search API."""
    if api_key is None:
        api_key = os.environ.get('BRAVE_API_KEY')
    if not api_key:
        print("Brave API key not set")
        return []

    try:
        response = requests.get(
            'https://api.search.brave.com/res/v1/web/search',
            params={'q': query, 'count': num_results},
            headers={'X-Subscription-Token': api_key, 'Accept': 'application/json'},
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            results = []
            for r in data.get('web', {}).get('results', [])[:num_results]:
                results.append({
                    'title': r.get('title', ''),
                    'link': r.get('url', ''),
                    'content': r.get('description', '')
                })
            return results
    except Exception as e:
        print(f"Brave search failed: {e}")
    return []

def search_perplexity(
    query: str,
    api_key: str = None,
    model: str = "sonar",
    max_tokens: int = 400,
    temperature: float = 0.2,
    top_p: float = 0.9,
):
    if api_key is None:
        api_key = os.environ.get("PERPLEXITY_API_KEY")
        if api_key is None:
            raise ValueError("PERPLEXITY_API_KEY not set. Set it in your environment.")
        
    
    url = "https://api.perplexity.ai/chat/completions"
    payload = {
        "model": "sonar",
        "messages": [
            {"role": "system", "content": "Be precise and concise."},
            {"role": "user", "content": query},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "return_images": False,
        "return_related_questions": False,
        "top_k": 0,
        "stream": False,
        "presence_penalty": 0,
        "frequency_penalty": 1,
        "response_format": None,
    }

    
    headers = {"Authorization": f"Bearer {api_key}", 
               "Content-Type": "application/json"}

    response = requests.post(url,
                             json=payload,
                             headers=headers)

    if response.status_code != 200:
        raise Exception(f"Perplexity API error {response.status_code}: {response.text[:200]}")

    data = response.json()
    return [data["choices"][0]["message"]["content"], data.get("citations", [])]

def search_web(
    query: str,
    num_results: int = 5,
    provider: str=None,
    api_key=None,
    perplexity_kwargs: Optional[Dict[str, Any]] = None,
) -> List:
    """
    Function Description:
        This function searches the web for information based on a query.
    Args:
        query: The search query.
    Keyword Args:
        num_results: The number of search results to retrieve.
        provider: The search engine provider to use ('perplexity' or 'duckduckgo').
    Returns:
        A list of dictionaries with 'title', 'link', and 'content' keys.
    """
    if perplexity_kwargs is None:
        perplexity_kwargs = {}
    results = []
    if provider is None:
        provider = 'startpage'

    if provider == "perplexity":
        search_result = search_perplexity(query, api_key=api_key, **perplexity_kwargs)

        return search_result

    if provider == "startpage":
        results = search_startpage(query, num_results=num_results)
        if results:
            return results
        print("Startpage failed, falling back to searxng")
        provider = 'searxng'

    if provider == "searxng":
        results = search_searxng(query, num_results=num_results)
        if results:
            return results
        print("SearXNG failed, falling back to duckduckgo")
        provider = 'duckduckgo'

    if provider == "duckduckgo":
        try:
            with DDGS() as ddgs:
                search_results = ddgs.text(query, max_results=num_results)
                results = [
                    {"title": r["title"], "link": r["href"], "content": r["body"]}
                    for r in search_results
                ]
        except Exception as e:
            print("DuckDuckGo search failed: ", e)
            results = []
    elif provider == 'exa':
        return search_exa(query, api_key=api_key, top_k=num_results)

    elif provider == 'startpage':
        results = search_startpage(query, num_results=num_results)

    elif provider == 'brave':
        results = search_brave(query, num_results=num_results, api_key=api_key)

    elif provider == 'google':
        if _google_search is None:
            print("googlesearch-python not installed, falling back to duckduckgo")
            return search_web(query, num_results=num_results, provider='duckduckgo')
        urls = list(_google_search(query, num_results=num_results))
        
        
        
        for url in urls:
            try:
                
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                }
                response = requests.get(url, headers=headers, timeout=5)
                response.raise_for_status()

                
                soup = BeautifulSoup(response.text, "html.parser")

                
                title = soup.title.string if soup.title else url

                
                content = " ".join([p.get_text() for p in soup.find_all("p")])
                content = " ".join(content.split())  

                results.append(
                    {
                        "title": title,
                        "link": url,
                        "content": (
                            content[:500] + "..." if len(content) > 500 else content
                        ),
                    }
                )

            except Exception as e:
                print(f"Error fetching {url}: {str(e)}")
                continue

    
    
    content_str = "\n".join(
        [r["content"] + "\n Citation: " + r["link"] + "\n\n\n" for r in results]
    )
    link_str = "\n".join([r["link"] + "\n" for r in results])
    return [content_str, link_str]

