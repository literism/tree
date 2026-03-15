"""
基于Playwright的Bing搜索工具
支持多query串行检索
"""

import asyncio
from playwright.async_api import async_playwright
import time
from typing import List, Dict, Optional


# 配置
LOCAL_PROXY = "http://172.27.130.33:7890"
COOKIE_FILE = "/mnt/literism/tree/data/cookies.txt"


def load_cookies(path: str) -> List[Dict]:
    """
    从 Netscape cookie file 解析 cookies
    
    参数:
    - path: cookies文件路径
    
    返回:
    - cookies列表
    """
    cookies = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("#") or line.strip() == "":
                    continue
                parts = line.strip().split("\t")
                if len(parts) >= 7:
                    cookies.append({
                        "domain": parts[0],
                        "path": parts[2],
                        "secure": parts[3].lower() == "true",
                        "name": parts[5],
                        "value": parts[6],
                    })
    except FileNotFoundError:
        print(f"警告: Cookie文件 {path} 不存在，将不使用cookies")
    return cookies


async def search_bing_single(
    query: str,
    context,
    num_results: int = 10,
    delay: float = 0,
    verbose: bool = False
) -> List[Dict[str, str]]:
    """
    搜索单个query（支持分页）
    
    参数:
    - query: 搜索关键词
    - context: playwright browser context
    - num_results: 返回结果数量
    - delay: 请求前延迟（秒）
    - verbose: 是否打印详细信息
    
    返回:
    - 搜索结果列表，每个元素包含title和url
    """
    if delay > 0:
        await asyncio.sleep(delay)
    
    all_results = []
    page = await context.new_page()
    
    # 拦截广告请求
    async def intercept(route, request):
        url = request.url
        if "doubleclick" in url or "ads" in url:
            await route.abort()
        else:
            await route.continue_()
    
    await page.route("**/*", intercept)
    
    try:
        if verbose:
            print(f"[*] 搜索: {query}")
        
        # Bing每页约10个结果，计算需要的页数
        pages_needed = (num_results + 9) // 10
        
        for page_num in range(pages_needed):
            # first参数：1, 11, 21, 31...
            first = page_num * 10 + 1
            
            # 构建Bing国际版URL（带分页）
            bing_url = (
                "https://www.bing.com/search?"
                f"q={query.replace(' ', '+')}"
                "&mkt=en-US"
                "&setlang=en"
                f"&first={first}"
            )
            
            if verbose and page_num > 0:
                print(f"    翻页到第 {page_num + 1} 页...")
            
            await page.goto(bing_url, timeout=20000)
            
            # 等待搜索结果加载
            try:
                await page.wait_for_selector("li.b_algo", timeout=20000)
            except:
                # 如果等待超时，可能是没有更多结果了
                if verbose:
                    print(f"    第 {page_num + 1} 页加载超时，停止翻页")
                break
            
            # 提取搜索结果
            page_results = await page.evaluate("""
() => {
    const items = [];
    const blocks = document.querySelectorAll("li.b_algo, div.b_title");
    
    for (const block of blocks) {
        const a = block.querySelector("h2 a");
        if (!a) continue;
        
        const title = a.innerText.trim();
        const url = a.href;
        
        if (!title) continue;
        items.push({title, url});
    }
    
    return items;
}
""")
            
            # 如果这一页没有结果，说明已经没有更多了
            if not page_results:
                if verbose:
                    print(f"    第 {page_num + 1} 页无结果，停止翻页")
                break
            
            all_results.extend(page_results)
            
            # 如果已经获取足够的结果，停止
            if len(all_results) >= num_results:
                break
            
            # 页面之间稍微延迟
            if page_num < pages_needed - 1 and len(all_results) < num_results:
                await asyncio.sleep(1)
        
        # 截取需要的数量
        all_results = all_results[:num_results]
        
        if verbose:
            print(f"    ✓ 找到 {len(all_results)} 个结果")
        
        return all_results
        
    except Exception as e:
        if verbose:
            print(f"    ✗ 搜索失败: {e}")
        return all_results  # 返回已获取的结果
    finally:
        await page.close()


async def search_bing_batch(
    queries: List[str],
    num_results: int = 10,
    delay_between_queries: float = 2.0,
    cookie_file: Optional[str] = None,
    proxy: Optional[str] = None,
    headless: bool = True,
    verbose: bool = False
) -> Dict[str, List[Dict[str, str]]]:
    """
    批量搜索多个queries（串行）
    
    参数:
    - queries: 搜索关键词列表
    - num_results: 每个query返回结果数量
    - delay_between_queries: query之间的延迟（秒）
    - cookie_file: cookies文件路径（可选）
    - proxy: 代理服务器（可选）
    - headless: 是否无头模式
    - verbose: 是否打印详细信息
    
    返回:
    - 字典，键为query，值为搜索结果列表
    """
    results_dict = {}
    
    # 加载cookies
    cookies = []
    if cookie_file:
        cookies = load_cookies(cookie_file)
    
    async with async_playwright() as p:
        # 启动浏览器
        browser = await p.chromium.launch(
            headless=headless,
            args=["--disable-blink-features=AutomationControlled"]
        )
        
        # 创建context配置
        context_options = {
            "locale": "en-US",
            "user_agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "extra_http_headers": {
                "sec-ch-ua": '"Google Chrome";v="120", "Chromium";v="120", "Not A(Brand";v="99"',
                "sec-ch-ua-mobile": "?0",
                "sec-ch-ua-platform": "Windows",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Upgrade-Insecure-Requests": "1",
                "Accept-Language": "en-US,en;q=0.9",
            }
        }
        
        # 添加代理
        if proxy:
            context_options["proxy"] = {"server": proxy}
        
        context = await browser.new_context(**context_options)
        
        # 加载cookies
        if cookies:
            await context.add_cookies(cookies)
        
        # 串行搜索每个query
        for i, query in enumerate(queries):
            delay = delay_between_queries if i > 0 else 0
            results = await search_bing_single(query, context, num_results, delay, verbose)
            results_dict[query] = results
        
        await browser.close()
    
    return results_dict


def search_bing(
    queries: List[str],
    num_results: int = 10,
    delay: float = 2.0,
    cookie_file: str = COOKIE_FILE,
    proxy: str = LOCAL_PROXY,
    headless: bool = True,
    verbose: bool = False
) -> Dict[str, List[Dict[str, str]]]:
    """
    同步版本的Bing搜索（便捷函数）
    
    参数:
    - queries: 搜索关键词列表（单个字符串或列表）
    - num_results: 每个query返回结果数量，默认10
    - delay: query之间的延迟（秒），默认2.0
    - cookie_file: cookies文件路径，默认"/mnt/literism/tree/data/cookies.txt"
    - proxy: 代理服务器，默认LOCAL_PROXY
    - headless: 是否无头模式，默认True
    - verbose: 是否打印详细信息，默认False
    
    返回:
    - 字典，键为query，值为搜索结果列表
      每个结果是字典: {'title': 标题, 'url': URL}
    
    示例:
    >>> results = search_bing(["Python tutorial", "Machine learning"])
    >>> for query, items in results.items():
    ...     for item in items:
    ...         print(item['title'], item['url'])
    """
    # 如果传入单个字符串，转换为列表
    if isinstance(queries, str):
        queries = [queries]
    
    return asyncio.run(
        search_bing_batch(
            queries=queries,
            num_results=num_results,
            delay_between_queries=delay,
            cookie_file=cookie_file,
            proxy=proxy,
            headless=headless,
            verbose=verbose
        )
    )


def print_results(results_dict: Dict[str, List[Dict[str, str]]]):
    """
    格式化打印搜索结果
    
    参数:
    - results_dict: search_bing返回的结果字典
    """
    print("\n" + "=" * 80)
    print("搜索结果汇总")
    print("=" * 80)
    
    for query, results in results_dict.items():
        print(f"\n【{query}】- {len(results)} 个结果")
        print("-" * 80)
        
        for i, r in enumerate(results, 1):
            print(f"{i}. {r['title']}")
            print(f"   {r['url']}")
        
        print()


if __name__ == "__main__":
    import sys
    
    # 测试用例
    if len(sys.argv) > 1:
        # 从命令行参数获取queries
        test_queries = sys.argv[1:]
    else:
        # 默认测试queries
        test_queries = [
            "apple inc. - history",
            "google company overview",
            "microsoft corporation"
        ]
    
    print("=" * 80)
    print("Bing搜索测试")
    print("=" * 80)
    print(f"将搜索 {len(test_queries)} 个queries")
    print(f"每个query之间延迟 2 秒")
    print()
    
    # 执行搜索（verbose=True显示进度）
    results = search_bing(
        queries=test_queries,
        num_results=5,
        delay=2.0,
        headless=True,
        verbose=True
    )
    
    # 打印结果
    print_results(results)

