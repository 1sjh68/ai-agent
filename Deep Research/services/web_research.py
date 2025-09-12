# services/web_research.py

import os
import logging
import re
import json
import socket
import asyncio
import ssl
import certifi

import aiohttp
import httplib2
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import fitz  # PyMuPDF

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google_auth_httplib2 import AuthorizedHttp
from google.oauth2.service_account import Credentials

# 从重构后的模块中导入依赖
from config import Config
from services.llm_interaction import call_ai
# 将truncate_text_for_context导入移到使用它的函数内部


def get_google_auth_http(config: Config):
    """根据配置获取用于 Google API 的认证 HTTP 对象，支持代理。
    
    参数:
    - config: 配置对象
    
    返回:
    - 认证后的 HTTP 对象
    """
    proxy_url = os.environ.get("HTTP_PROXY")
    proxy_info = None
    if proxy_url:
        parsed_proxy = urlparse(proxy_url)
        proxy_info = httplib2.ProxyInfo(
            proxy_type=httplib2.socks.PROXY_TYPE_HTTP,
            proxy_host=parsed_proxy.hostname,
            proxy_port=parsed_proxy.port,
            proxy_user=parsed_proxy.username,
            proxy_pass=parsed_proxy.password
        )
        logging.info(f"Google API 将使用代理: {parsed_proxy.hostname}:{parsed_proxy.port}")

    if config.google_service_account_path and os.path.exists(config.google_service_account_path):
        try:
            creds = Credentials.from_service_account_file(
                config.google_service_account_path, 
                scopes=['https://www.googleapis.com/auth/cse']
            )
            return AuthorizedHttp(creds, http=httplib2.Http(proxy_info=proxy_info, timeout=config.api_request_timeout_seconds))
        except Exception as e:
            logging.error(f"未能使用服务账户进行 Google API 认证，回退到默认方式。错误: {e}")
    
    return httplib2.Http(proxy_info=proxy_info, timeout=config.api_request_timeout_seconds)


def perform_search(config: Config, query: str) -> list[dict]:
    """
    执行一次 Google 自定义搜索。
    (V13.1: 增加了 API Key 与 CSE ID 配对轮换功能)
    
    参数:
    - config: 配置对象
    - query: 搜索查询字符串
    
    返回:
    - 搜索结果列表
    """
    num_keys = len(config.google_api_keys)
    num_ids = len(config.google_cse_ids)
    
    if not num_keys or not num_ids:
        logging.error("Google API 密钥列表 (GOOGLE_API_KEYS) 或 CSE ID 列表 (GOOGLE_CSE_IDS) 未配置。搜索功能已禁用。")
        return []
    
    num_pairs_to_try = min(num_keys, num_ids)

    for i in range(num_pairs_to_try):
        current_index = config.current_google_api_key_index
        
        if current_index >= num_pairs_to_try:
            current_index = 0
            config.current_google_api_key_index = 0
            
        current_key = config.google_api_keys[current_index]
        current_cse_id = config.google_cse_ids[current_index]
        
        logging.info(f"正在使用密钥/ID对 #{current_index + 1} 执行 Google 搜索: '{query}'")

        try:
            http_auth = get_google_auth_http(config)
            service = build("customsearch", "v1", developerKey=current_key, http=http_auth)
            res = service.cse().list(q=query, cx=current_cse_id, num=config.num_search_results).execute()
            items = res.get('items', [])
            logging.info(f"  使用密钥/ID对 #{current_index + 1} 搜索成功，返回了 {len(items)} 个结果。")
            return items

        except HttpError as e:
            try:
                error_details = json.loads(e.content.decode('utf-8', 'ignore'))
                reason = error_details.get('error', {}).get('errors', [{}])[0].get('reason')
            except (json.JSONDecodeError, IndexError):
                reason = None

            if e.resp.status == 429 and reason == 'rateLimitExceeded':
                logging.warning(f"  密钥/ID对 #{current_index + 1} 已达到每日配额。正在尝试切换到下一个配对...")
                config.current_google_api_key_index = (current_index + 1)
                
                if i == num_pairs_to_try - 1:
                    logging.error("所有 Google API 密钥/ID 配对的每日配额均已用尽。")
                    return []
                continue
            else:
                logging.error(f"  Google 搜索 API 发生不可恢复的 HTTP 错误: {e.content}")
                return []
        
        except Exception as e:
            logging.error(f"  Google 搜索期间发生未知错误: {e}", exc_info=True)
            return []

    logging.error("未能使用任何一个 Google API 密钥/ID 配对成功完成搜索。")
    return []


def create_intelligent_search_queries(config: Config, knowledge_gap: str, full_document_context: str) -> list[str]:
    from utils.text_processor import truncate_text_for_context
    """根据知识空白和文档上下文，生成更智能、更有效的搜索引擎查询。
    
    参数:
    - config: 配置对象
    - knowledge_gap: 知识空白描述
    - full_document_context: 完整文档上下文
    
    返回:
    - 搜索查询列表
    """
    logging.info(f"  正在为知识空白生成智能搜索查询: '{knowledge_gap[:100]}...'")
    context_summary_for_query_gen = truncate_text_for_context(config, full_document_context, 1000, "middle")

    prompt = f"""
    根据以下提供的“知识空白”和“文档上下文摘要”，生成1-3个高度具体且有效的搜索引擎查询。
    查询应简洁，并使用关键词。避免提出问题；而是制定搜索词条。
    知识空白: "{knowledge_gap}"
    文档上下文摘要: --- {context_summary_for_query_gen} ---
    生成的搜索查询 (每行一个，最多3个):
    """
    messages = [{"role": "user", "content": prompt}]
    response_content = call_ai(config, config.researcher_model_name, messages, temperature=0.2, max_tokens_output=150)

    if "AI模型调用失败" in response_content or not response_content.strip():
        return [knowledge_gap]
    queries = [q.strip() for q in response_content.splitlines() if q.strip()]
    if not queries:
        return [knowledge_gap]
        
    logging.info(f"  为知识空白“{knowledge_gap[:50]}...”生成了 {len(queries)} 个查询: {queries}")
    return queries[:config.max_queries_per_gap]


async def scrape_and_summarize_async(session: aiohttp.ClientSession, config: Config, url: str, knowledge_gap: str, specific_query: str) -> str:
    from utils.text_processor import truncate_text_for_context
    """异步地抓取单个 URL 的内容，并调用 AI 进行摘要。
    
    参数:
    - session: aiohttp 会话对象
    - config: 配置对象
    - url: 要抓取的 URL
    - knowledge_gap: 知识空白描述
    - specific_query: 特定查询字符串
    
    返回:
    - 包含 URL、查询和摘要的字符串
    """
    logging.info(f"  [异步] 抓取和总结中: {url}")
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        proxy_str = os.environ.get('HTTP_PROXY')
        ssl_context = ssl.create_default_context(cafile=certifi.where())

        async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=30), proxy=proxy_str, ssl=ssl_context) as response:
            if response.status != 200:
                logging.warning(f"  [异步] 抓取 {url} 失败，状态码：{response.status}。")
                return ""

            content_type = response.headers.get('Content-Type', '').lower()
            text_content = ""

            if 'application/pdf' in content_type:
                pdf_bytes = await response.read()
                with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
                    text_content = "".join(page.get_text() for page in doc)
            elif 'text/html' in content_type:
                html_content = await response.text()
                soup = BeautifulSoup(html_content, 'lxml')
                for element in soup(["script", "style"]):
                    element.decompose()
                text_content = soup.get_text(separator='\n', strip=True)
            
            if not text_content.strip():
                return ""

            summary_prompt = f"""请根据以下查询“{specific_query}”来总结下面的文本，以填补知识空白“{knowledge_gap}”。提取最相关的信息。
            文本内容：--- {truncate_text_for_context(config, text_content, 6000)} ---
            总结：
            """
            
            # 关键修正：在异步函数中调用同步的 call_ai 函数，必须使用 asyncio.to_thread
            summary = await asyncio.to_thread(
                call_ai, config, config.summary_model_name, [{"role": "user", "content": summary_prompt}], max_tokens_output=768
            )
            
            if "AI模型调用失败" not in summary and summary.strip():
                return f"URL: {url}\n查询: {specific_query}\n总结:\n{summary}\n"
            return ""

    except Exception as e:
        logging.error(f"  [ASYNC] 抓取和总结 {url} 时发生错误: {e}", exc_info=False)
        return ""


async def run_research_cycle_async(config: Config, knowledge_gaps: list[str], full_document_context: str) -> str:
    """为一系列知识空白执行完整的异步研究周期。
    
    参数:
    - config: 配置对象
    - knowledge_gaps: 知识空白列表
    - full_document_context: 完整文档上下文
    
    返回:
    - 研究结果摘要
    """
    if not knowledge_gaps: return ""
    
    logging.info(f"\n--- 开始为 {len(knowledge_gaps)} 个知识空白进行智能知识发现 ---")
    all_tasks = []
    final_brief_list = []

    async with aiohttp.ClientSession() as session:
        for gap_text in knowledge_gaps:
            # 将同步函数放入线程中运行，避免阻塞主事件循环
            search_queries = await asyncio.to_thread(create_intelligent_search_queries, config, gap_text, full_document_context)
            for query in search_queries:
                search_results = await asyncio.to_thread(perform_search, config, query)
                for res_item in search_results:
                    if url := res_item.get('link'):
                        all_tasks.append(scrape_and_summarize_async(session, config, url, gap_text, query))
        
        # 关键修正：任务的执行和结果处理，必须在 session 保持打开状态的 `async with` 块内完成
        if not all_tasks:
            logging.warning("--- 异步研究周期中未创建任何有效的抓取任务 ---")
            return ""

        logging.info(f"正在并发执行 {len(all_tasks)} 个抓取/摘要任务...")
        completed_briefs = await asyncio.gather(*all_tasks)
        final_brief_list = [str(b) for b in completed_briefs if b and isinstance(b, str)]

    if not final_brief_list:
        logging.warning("--- 智能研究周期未产生任何有效简报 ---")
        return ""
        
    logging.info(f"--- 知识发现完成，生成了 {len(final_brief_list)} 份简报 ---")
    return "\n\n===== 研究简报开始 =====\n\n" + "\n".join(final_brief_list) + "\n===== 研究简报结束 =====\n\n"
