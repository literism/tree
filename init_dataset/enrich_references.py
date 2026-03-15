"""
丰富引用内容
1. 为没有URL的引用搜索URL
2. 使用Playwright爬取所有引用URL的内容
3. 提取文字信息并过滤失败的引用
"""

import json
import asyncio
import trafilatura
from pathlib import Path
from typing import Dict, Optional
from playwright.async_api import async_playwright
from bing_search_playwright import search_bing
from expand_references import extract_real_url


# 配置
INPUT_FILE = "/mnt/literism/tree/data/wikipedia_references_searched.json"
OUTPUT_FILE = "/mnt/literism/tree/data/wikipedia_references_enriched.json"

INITIAL_SEARCH_RESULTS = 3
MAX_SEARCH_RESULTS = 15
REQUEST_TIMEOUT = 15
DELAY_BETWEEN_REQUESTS = 0.0
MIN_CONTENT_LENGTH = 300
MAX_CONCURRENT_REQUESTS = 5  # 最大并发请求数

LOCAL_PROXY = "http://172.27.130.33:7890"
COOKIE_FILE = "/mnt/literism/tree/data/cookies.txt"


def is_url(text: str) -> bool:
    """判断文本是否是URL"""
    if not text:
        return False
    return text.startswith('http://') or text.startswith('https://')


def is_file_download_url(url: str) -> bool:
    """判断URL是否是文件下载地址"""
    file_extensions = [
        '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
        '.zip', '.rar', '.tar', '.gz', '.7z',
        '.mp3', '.mp4', '.avi', '.mov', '.wmv',
        '.jpg', '.jpeg', '.png', '.gif', '.svg',
        '.exe', '.dmg', '.apk',
        '.xml', '.csv'
    ]
    
    url_lower = url.lower()
    for ext in file_extensions:
        if url_lower.endswith(ext) or f'{ext}?' in url_lower or f'{ext}#' in url_lower:
            return True
    
    return False


def search_url_for_text(query: str, verbose: bool = False) -> Optional[str]:
    """为文本搜索URL"""
    if verbose:
        print(f"      搜索URL...")
    
    num_results = INITIAL_SEARCH_RESULTS
    
    while num_results <= MAX_SEARCH_RESULTS:
        try:
            results = search_bing(
                queries=[query],
                num_results=num_results,
                delay=0,
                verbose=False
            )
            
            if query in results and results[query]:
                for result in results[query]:
                    url = extract_real_url(result['url'])
                    if is_url(url) and not is_file_download_url(url):
                        if verbose:
                            print(f"        ✓ URL: {url[:60]}...")
                        return url
            
            if num_results >= MAX_SEARCH_RESULTS:
                break
            
            if verbose:
                print(f"        ! 扩大搜索到{num_results + 5}个...")
            
            num_results += 5
            
        except Exception as e:
            if verbose:
                print(f"        ✗ 搜索错误: {e}")
            break
    
    if verbose:
        print(f"        ✗ 未找到有效URL")
    
    return None


async def fetch_page_content_async(url: str, context, verbose: bool = False) -> Optional[str]:
    """使用Playwright爬取网页并提取文字内容"""
    page = await context.new_page()
    
    try:
        await page.goto(url, timeout=REQUEST_TIMEOUT * 1000, wait_until='domcontentloaded')
        html = await page.content()
        
        extracted = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=False
        )
        
        if extracted and len(extracted.strip()) > 0:
            if verbose:
                print(f"        ✓ 提取 {len(extracted)} 字符")
            return extracted
        else:
            if verbose:
                print(f"        ! 提取内容为空")
            return None
            
    except Exception as e:
        if verbose:
            print(f"        ✗ 爬取失败: {str(e)[:40]}")
        return None
    finally:
        await page.close()


async def process_single_reference_async(
    ref_key: str,
    ref_value: str,
    context,
    semaphore: asyncio.Semaphore,
    verbose: bool = False
) -> tuple:
    """处理单个引用（异步，带并发控制）"""
    async with semaphore:  # 控制并发数
        result = {
            'original': ref_value,
            'url': None,
            'search_query': None,
            'content': None,
            'failed': None
        }
        
        # 步骤1：确保有URL
        if is_url(ref_value):
            result['url'] = ref_value
        else:
            result['search_query'] = ref_value
            url = search_url_for_text(ref_value, verbose)
            if url:
                result['url'] = url
            else:
                result['failed'] = 'no_url_found'
                return ref_key, result
        
        # 检查是否是文件下载地址
        if is_file_download_url(result['url']):
            if verbose:
                print(f"        ! 文件下载地址，跳过")
            result['failed'] = 'file_download'
            return ref_key, result
        
        # 步骤2：爬取内容
        if result['url']:
            if DELAY_BETWEEN_REQUESTS > 0:
                await asyncio.sleep(DELAY_BETWEEN_REQUESTS)
            content = await fetch_page_content_async(result['url'], context, verbose)
            
            if content and len(content.strip()) > MIN_CONTENT_LENGTH:
                result['content'] = content
            elif content and len(content.strip()) > 0:
                result['failed'] = 'too_short'
            else:
                result['failed'] = 'empty_content' if content is not None else 'fetch_failed'
        
        return ref_key, result


def load_cookies(path):
    """加载cookies文件"""
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
        pass
    return cookies


async def process_all_topics_async(data, enriched_data, processed_topics_set):
    """异步处理所有topics"""
    
    cookies = load_cookies(COOKIE_FILE)
    
    processed_topics = len(processed_topics_set)
    total_topics = len(data)
    newly_processed = 0
    
    stats = {
        'search_success': 0,
        'search_failed': 0,
        'fetch_success': 0,
        'fetch_failed': 0,
        'file_download': 0,
        'empty_content': 0,
        'too_short': 0
    }
    
    # 启动浏览器
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=["--disable-blink-features=AutomationControlled"]
        )
        
        context_options = {
            "locale": "en-US",
            "user_agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "extra_http_headers": {
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            },
            "proxy": {"server": LOCAL_PROXY}
        }
        
        context = await browser.new_context(**context_options)
        
        if cookies:
            await context.add_cookies(cookies)
        
        # 处理每个topic
        for topic_key, topic_data in data.items():
            if topic_key in processed_topics_set:
                continue
            
            processed_topics += 1
            newly_processed += 1
            topic = topic_data.get('topic', topic_key)
            
            refs = topic_data.get('references', {})
            total_refs = len(refs)
            enriched_refs = {}
            
            # 该topic的统计
            topic_stats = {
                'success': 0,
                'failed': 0,
                'search_failed': 0,
                'file_download': 0,
                'empty_content': 0,
                'fetch_failed': 0,
                'too_short': 0
            }
            
            print(f"\n[{processed_topics}/{total_topics}] {topic} ({total_refs} 个引用)")
            
            # 创建信号量控制并发
            semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
            
            # 创建所有任务
            tasks = []
            for ref_key, ref_value in refs.items():
                task = process_single_reference_async(ref_key, ref_value, context, semaphore, verbose=False)
                tasks.append(task)
            
            # 并发处理并实时更新进度
            completed = 0
            for coro in asyncio.as_completed(tasks):
                ref_key, result = await coro
                completed += 1
                
                # 统计
                if result.get('failed'):
                    topic_stats['failed'] += 1
                    
                    if result['failed'] == 'no_url_found':
                        stats['search_failed'] += 1
                        topic_stats['search_failed'] += 1
                    elif result['failed'] == 'file_download':
                        stats['file_download'] += 1
                        topic_stats['file_download'] += 1
                    elif result['failed'] == 'empty_content':
                        stats['empty_content'] += 1
                        topic_stats['empty_content'] += 1
                    elif result['failed'] == 'too_short':
                        stats['too_short'] += 1
                        topic_stats['too_short'] += 1
                    elif result['failed'] == 'fetch_failed':
                        stats['fetch_failed'] += 1
                        topic_stats['fetch_failed'] += 1
                else:
                    topic_stats['success'] += 1
                    
                    if result['search_query']:
                        stats['search_success'] += 1
                    if result['content']:
                        stats['fetch_success'] += 1
                
                enriched_refs[ref_key] = result
                
                # 在一行中更新进度
                progress_msg = (
                    f"  进度: {completed}/{total_refs} | "
                    f"✓ 成功: {topic_stats['success']} | "
                    f"✗ 失败: {topic_stats['failed']}"
                )
                print(f"\r{progress_msg}", end='', flush=True)
            
            # 保存该topic
            enriched_data[topic_key] = {
                'topic': topic_data.get('topic'),
                'category': topic_data.get('category'),
                'pageid': topic_data.get('pageid'),
                'references': enriched_refs
            }
            
            # 换行并输出该topic的详细统计
            print()  # 换行
            success_rate = (topic_stats['success'] / total_refs * 100) if total_refs > 0 else 0
            print(f"  完成: ✓ {topic_stats['success']}/{total_refs} ({success_rate:.1f}%) | ✗ {topic_stats['failed']}/{total_refs}", end='')
            
            if topic_stats['failed'] > 0:
                fail_details = []
                if topic_stats['search_failed'] > 0:
                    fail_details.append(f"无URL:{topic_stats['search_failed']}")
                if topic_stats['file_download'] > 0:
                    fail_details.append(f"文件:{topic_stats['file_download']}")
                if topic_stats['empty_content'] > 0:
                    fail_details.append(f"空:{topic_stats['empty_content']}")
                if topic_stats['too_short'] > 0:
                    fail_details.append(f"短:{topic_stats['too_short']}")
                if topic_stats['fetch_failed'] > 0:
                    fail_details.append(f"失败:{topic_stats['fetch_failed']}")
                print(f" ({', '.join(fail_details)})", end='')
            
            # 立即保存
            try:
                Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
                with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                    json.dump(enriched_data, f, ensure_ascii=False, indent=2)
                print(f" | 💾 已保存")
            except Exception as save_error:
                print(f" | ! 保存失败: {save_error}")
        
        await browser.close()
    
    return stats, newly_processed


def main():
    """主函数"""
    print("=" * 80)
    print("丰富引用内容")
    print("=" * 80)
    
    # 1. 读取输入文件
    print("\n[1] 加载数据...")
    
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"  ✓ 加载了 {len(data)} 个topics")
    except FileNotFoundError:
        print(f"  ✗ 文件不存在: {INPUT_FILE}")
        return
    except Exception as e:
        print(f"  ✗ 加载失败: {e}")
        return
    
    # 统计引用
    total_refs = 0
    url_refs = 0
    text_refs = 0
    
    for topic_key, topic_data in data.items():
        refs = topic_data.get('references', {})
        total_refs += len(refs)
        for ref_key, ref_value in refs.items():
            if is_url(ref_value):
                url_refs += 1
            else:
                text_refs += 1
    
    print(f"  ✓ 总引用数: {total_refs}")
    print(f"    - 已有URL: {url_refs}")
    print(f"    - 只有文本: {text_refs}")
    
    # 断点续传
    enriched_data = {}
    processed_topics_set = set()
    
    if Path(OUTPUT_FILE).exists():
        try:
            with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
                enriched_data = json.load(f)
            processed_topics_set = set(enriched_data.keys())
            print(f"  ✓ 发现已处理的数据: {len(processed_topics_set)} 个topics")
            print(f"  → 将跳过已处理的，继续处理剩余 {len(data) - len(processed_topics_set)} 个")
        except Exception as e:
            print(f"  ! 读取输出文件失败，将重新开始: {e}")
            enriched_data = {}
            processed_topics_set = set()
    
    # 2. 处理引用
    print("\n[2] 处理引用...")
    remaining = len(data) - len(processed_topics_set)
    print(f"  总共: {len(data)} 个topics")
    print(f"  已处理: {len(processed_topics_set)} 个")
    print(f"  剩余: {remaining} 个")
    print(f"  配置: 初始搜索{INITIAL_SEARCH_RESULTS}个，最多{MAX_SEARCH_RESULTS}个")
    print(f"  延迟: {DELAY_BETWEEN_REQUESTS}秒/请求")
    print()
    
    # 异步处理
    stats, newly_processed = asyncio.run(
        process_all_topics_async(data, enriched_data, processed_topics_set)
    )
    
    # 3. 最终确认
    print("[3] 最终确认...")
    print(f"  ✓ 所有数据已保存")
    print(f"  ✓ 输出文件: {OUTPUT_FILE}")
    
    # 4. 统计信息
    print("\n" + "=" * 80)
    print("处理完成！")
    print("=" * 80)
    print(f"总topics数: {len(data)}")
    print(f"  - 之前已处理: {len(processed_topics_set)}")
    print(f"  - 本次新处理: {newly_processed}")
    print(f"\n总引用数: {total_refs}")
    print(f"  - 已有URL: {url_refs}")
    print(f"  - 只有文本: {text_refs}")
    print(f"\nURL搜索:")
    print(f"  - 搜索成功: {stats['search_success']}")
    print(f"  - 搜索失败: {stats['search_failed']}")
    print(f"\n内容爬取:")
    print(f"  - 爬取成功: {stats['fetch_success']}")
    print(f"  - 爬取失败: {stats['fetch_failed']}")
    print(f"  - 文件下载: {stats['file_download']}")
    print(f"  - 内容为空: {stats['empty_content']}")
    print(f"  - 内容过短: {stats['too_short']}")
    print(f"\n输出文件: {OUTPUT_FILE}")
    print("=" * 80)
    
    if newly_processed == 0:
        print("\n提示: 所有topics都已处理完成。如需重新处理，请删除输出文件。")


if __name__ == "__main__":
    main()

