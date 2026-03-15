import mwparserfromhell
import xml.etree.ElementTree as ET
from multiprocessing import Pool
import json
import sys
import time
import os

# ----------------- 配置 -----------------
INPUT_FILE = "/mnt/literism/data/wiki_dataset/full_article/enwiki-latest-pages-articles.xml"   # 已解压 XML
OUTPUT_FILE = "/mnt/literism/data/wiki_dataset/wiki_intro.jsonl"
NUM_PROCESSES = 16
MAX_PAGES = -1   # 设置最大处理页面数，-1 表示处理全部

# 断点续传配置
RESUME_MODE = False  # True=从断点继续（追加模式），False=从头开始（覆盖模式）

# 性能和安全配置
MAX_CONTENT_SIZE = 5 * 1024 * 1024  # 单个页面最大 5MB，超过则跳过
PROGRESS_INTERVAL = 100              # 每处理多少页显示一次进度
MIN_INTRO_LENGTH = 50                # 最小介绍段落长度

# ----------------------------------------

def is_real_article(text):
    """判断文章是否为真实内容页"""
    if text is None:
        return False
    t = text.strip()
    if len(t) < 200:
        return False
    if t.lower().startswith("#redirect"):
        return False
    return True

def clean_unicode(text):
    """清理无效的 Unicode 字符（特别是孤立的代理对字符）"""
    if text is None:
        return ""
    try:
        return text.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
    except:
        return ""

def extract_intro_text(wikitext):
    """提取页面开头到第一个标题之前的内容，保留文字但去除图片、链接、表格等格式"""
    try:
        # 检查内容大小，避免处理过大的内容
        if len(wikitext) > MAX_CONTENT_SIZE:
            return ""
        
        # 找到第一个标题的位置（== ... ==）
        lines = wikitext.split('\n')
        first_heading_idx = None
        
        for idx, line in enumerate(lines):
            line = line.strip()
            # 检测标题行（至少两个等号开头和结尾）
            if line.startswith('==') and line.endswith('==') and len(line) >= 4:
                first_heading_idx = idx
                break
        
        # 提取开头部分的文本
        if first_heading_idx is not None:
            intro_lines = lines[:first_heading_idx]
        else:
            # 如果没有标题，取前100行或全部内容（以较小者为准）
            intro_lines = lines[:min(len(lines), 100)]
        
        intro_wikitext = '\n'.join(intro_lines)
        
        # 使用 mwparserfromhell 解析 wikitext
        code = mwparserfromhell.parse(intro_wikitext)
        
        # 移除模板（通常包含信息框、引用等）
        for template in code.filter_templates():
            try:
                code.remove(template)
            except:
                pass
        
        # 移除 HTML 标签（但保留标签内的文本）
        for tag in code.filter_tags():
            try:
                # 保留标签内的文本内容
                if hasattr(tag, 'contents') and tag.contents:
                    code.replace(tag, str(tag.contents))
                else:
                    code.remove(tag)
            except:
                pass
        
        # 移除注释
        for comment in code.filter_comments():
            try:
                code.remove(comment)
            except:
                pass
        
        # 处理链接：保留链接文字
        for wikilink in code.filter_wikilinks():
            try:
                # 获取链接显示的文本（如果有的话）
                if wikilink.text:
                    link_text = str(wikilink.text)
                else:
                    # 如果没有显示文本，使用链接目标
                    link_text = str(wikilink.title)
                
                # 用纯文本替换链接
                code.replace(wikilink, link_text)
            except:
                pass
        
        # 处理外部链接
        for extlink in code.filter_external_links():
            try:
                if extlink.title:
                    link_text = str(extlink.title)
                else:
                    link_text = str(extlink.url)
                code.replace(extlink, link_text)
            except:
                pass
        
        # 转换为纯文本
        # strip_code() 会自动处理加粗（'''text'''）、斜体（''text''）等格式
        # 保留文本内容，去除格式标记
        text = code.strip_code()
        
        # 清理无效的 Unicode 字符
        text = clean_unicode(text)
        
        # 清理多余空行和空格
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        text = '\n'.join(lines)
        
        # 移除特殊标记（如引用标记）
        text = text.replace('[', '').replace(']', '')
        
        return text.strip()
    
    except Exception as e:
        # 静默处理错误，返回空字符串
        return ""

def process_page_text(page_id, page_title, wikitext):
    """处理单个页面，提取开头介绍段落"""
    try:
        # 快速检查
        if not is_real_article(wikitext):
            return None
        
        # 检查内容大小
        if wikitext and len(wikitext) > MAX_CONTENT_SIZE:
            return None  # 跳过超大页面
        
        # 提取开头介绍文本
        intro_text = extract_intro_text(wikitext)
        
        # 检查介绍段落长度
        if not intro_text or len(intro_text) < MIN_INTRO_LENGTH:
            return None  # 跳过没有有效介绍的页面
        
        # 清理标题中的无效字符
        page_title = clean_unicode(page_title) if page_title else ""
        
        return {
            "id": page_id,
            "url": f"https://en.wikipedia.org/wiki/{page_title.replace(' ', '_')}",
            "title": page_title,
            "intro": intro_text
        }
    
    except Exception as e:
        # 如果处理失败，跳过该页面
        return None

def load_processed_ids(output_file):
    """加载已处理的页面ID"""
    processed_ids = set()
    if os.path.exists(output_file):
        print(f"正在加载已处理的页面ID从 {output_file}...")
        try:
            with open(output_file, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        if 'id' in data:
                            processed_ids.add(data['id'])
                    except:
                        continue
            print(f"已加载 {len(processed_ids)} 个已处理的页面ID")
        except Exception as e:
            print(f"加载已处理ID时出错: {e}")
    else:
        print(f"输出文件不存在，将从头开始处理")
    return processed_ids

def worker(item):
    """Worker 进程处理函数，带异常保护"""
    try:
        page_id, page_title, wikitext = item
        return process_page_text(page_id, page_title, wikitext)
    except Exception as e:
        # 捕获所有异常，避免worker崩溃导致进程池卡住
        return None

def page_generator(xml_file, processed_ids=None, resume_mode=False):
    """生成 (id, title, text) 元组，可选择跳过已处理的页面"""
    if processed_ids is None:
        processed_ids = set()
    
    skipped_count = 0
    total_count = 0
    
    for event, elem in ET.iterparse(xml_file, events=("end",)):
        if elem.tag.endswith("page"):
            title_elem = elem.find("./{*}title")
            pid_elem = elem.find("./{*}id")
            text_elem = elem.find("./{*}revision/{*}text")

            title = title_elem.text if title_elem is not None else None
            pid = pid_elem.text if pid_elem is not None else None
            text = text_elem.text if text_elem is not None else None

            total_count += 1
            
            # 如果启用断点续传，在主进程中过滤已处理的页面
            if resume_mode and pid in processed_ids:
                skipped_count += 1
                if skipped_count % 5000 == 0:
                    print(f"\r正在跳过已处理页面: {skipped_count}/{total_count} ({skipped_count*100//total_count}%)", end='', flush=True)
            else:
                if skipped_count > 0 and resume_mode:
                    # 完成跳过，打印最终统计
                    print(f"\r已跳过 {skipped_count} 个已处理的页面，开始处理新页面...")
                    skipped_count = -1  # 标记已打印，避免重复
                yield (pid, title, text)
            
            elem.clear()

def main():
    start_time = time.time()
    
    print("=" * 80)
    print("Wikipedia 文章开头介绍提取工具")
    print("=" * 80)
    print(f"输入文件: {INPUT_FILE}")
    print(f"输出文件: {OUTPUT_FILE}")
    print(f"进程数: {NUM_PROCESSES}")
    print(f"最大页面数: {MAX_PAGES if MAX_PAGES > 0 else '全部'}")
    print(f"断点续传: {'开启' if RESUME_MODE else '关闭'}")
    print("=" * 80)
    
    # 根据RESUME_MODE决定是否加载已处理的页面ID
    processed_ids = set()
    already_processed_count = 0
    file_mode = "w"
    
    if RESUME_MODE:
        processed_ids = load_processed_ids(OUTPUT_FILE)
        already_processed_count = len(processed_ids)
        file_mode = "a" if os.path.exists(OUTPUT_FILE) and already_processed_count > 0 else "w"
        
        if already_processed_count > 0:
            print(f"\n⚠ 断点续传模式：将跳过 {already_processed_count} 个已处理的页面")
            print(f"⚠ 这可能需要较长时间（需要遍历XML直到找到新页面）")
            print(f"⚠ 如果想从头开始，请设置 RESUME_MODE = False\n")
        else:
            print("\n未找到已处理的记录，将从头开始处理\n")
    else:
        if os.path.exists(OUTPUT_FILE):
            print(f"\n⚠ 输出文件已存在，将被覆盖！")
            print(f"⚠ 如果想继续之前的进度，请设置 RESUME_MODE = True\n")
    
    # 将已处理的ID传递给生成器
    gen = page_generator(INPUT_FILE, processed_ids, resume_mode=RESUME_MODE)
    
    # 创建进程池
    pool = Pool(NUM_PROCESSES)

    processed = 0
    written = 0
    skipped = 0

    print(f"{'继续' if file_mode == 'a' else '开始'}处理文章...\n")
    
    with open(OUTPUT_FILE, file_mode, encoding="utf-8", errors='ignore') as fout:
        try:
            # 使用更小的 chunksize 避免长时间卡在某个chunk上
            for result in pool.imap_unordered(worker, gen, chunksize=1):
                processed += 1

                if result:
                    try:
                        fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                        fout.flush()
                        written += 1
                    except Exception as e:
                        # 如果写入失败，跳过该条目
                        skipped += 1
                else:
                    skipped += 1

                # 进度显示
                if processed % PROGRESS_INTERVAL == 0 or processed <= 10:
                    elapsed = time.time() - start_time
                    speed = processed / elapsed if elapsed > 0 else 0
                    success_rate = (written / processed * 100) if processed > 0 else 0
                    sys.stdout.write(
                        f"\r处理: {processed} | 成功: {written} ({success_rate:.1f}%) | 跳过: {skipped} | 速度: {speed:.1f} p/s     "
                    )
                    sys.stdout.flush()

                if MAX_PAGES > 0 and processed >= MAX_PAGES:
                    print(f"\n\n已达到 MAX_PAGES={MAX_PAGES}，停止处理。")
                    break
        except KeyboardInterrupt:
            print("\n\n用户中断，正在关闭进程池...")
        finally:
            pool.terminate()
            pool.join()

    print("\n" + "=" * 80)
    print("处理完成！")
    if RESUME_MODE and already_processed_count > 0:
        print(f"原有记录: {already_processed_count} 篇")
        print(f"本次处理: {processed} 篇 (成功: {written}, 跳过: {skipped})")
        print(f"总计: {already_processed_count + written} 篇")
    else:
        print(f"本次处理: {processed} 篇 (成功: {written}, 跳过: {skipped})")
    print(f"输出文件: {OUTPUT_FILE}")
    print("=" * 80)

if __name__ == "__main__":
    main()

