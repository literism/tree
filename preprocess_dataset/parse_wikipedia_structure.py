"""
Wikipedia 页面解析程序
解析Wikipedia页面的wikitext，提取文章结构和引用链接
"""

import json
import os
import re
from pathlib import Path
import mwparserfromhell
from collections import defaultdict

# ========== 配置 ==========
INPUT_DIR = "/mnt/literism/tree/data/wikipedia_pages"
STRUCTURE_OUTPUT = "/mnt/literism/tree/data/wikipedia_structures.json"
REFERENCES_OUTPUT = "/mnt/literism/tree/data/wikipedia_references.json"

# 需要跳过的标题
SKIP_TITLES = {
    'see also', 'see', 'notes', 'sources', 'external links',
    'further reading', 'bibliography', 'explanatory notes'
}

# 是否保存段落内容（如果为True，会在每个节点中保存content字段）
SAVE_CONTENT = True
# ==========================


def extract_references_from_section(wikitext):
    """
    从References部分提取所有引用，建立name -> URL/文本的映射
    返回: {ref_name: url或文本}
    """
    ref_dict = {}
    
    # 找到References部分
    lines = wikitext.split('\n')
    in_references = False
    ref_start = None
    ref_end = None
    ref_level = None
    
    for i, line in enumerate(lines):
        line_s = line.strip()
        if line_s.startswith('==') and 'reference' in line_s.lower():
            in_references = True
            ref_start = i
            # 计算level
            level = 0
            for c in line_s:
                if c == '=':
                    level += 1
                else:
                    break
            ref_level = level
        elif in_references and line_s.startswith('=='):
            level = 0
            for c in line_s:
                if c == '=':
                    level += 1
                else:
                    break
            if level <= ref_level:
                ref_end = i
                break
    
    if ref_start is None:
        return ref_dict
    
    if ref_end is None:
        ref_end = len(lines)
    
    # References部分的内容
    ref_content = '\n'.join(lines[ref_start:ref_end])
    
    # 在References部分查找所有有name属性的ref标签
    # 匹配完整的ref标签
    ref_pattern = r'<ref(?:\s+[^>]*?)?>.*?</ref>'
    
    for match in re.finditer(ref_pattern, ref_content, re.IGNORECASE | re.DOTALL):
        ref_text = match.group(0)
        
        # 提取name属性
        name_match = re.search(r'name\s*=\s*["\']?([^"\'>\s]+)["\']?', ref_text, re.IGNORECASE)
        if not name_match:
            continue
        
        ref_name = name_match.group(1)
        
        # 提取内容
        content_match = re.search(r'<ref[^>]*>(.*?)</ref>', ref_text, re.IGNORECASE | re.DOTALL)
        content = content_match.group(1) if content_match else ''
        
        # 提取URL（取第一个）
        urls = re.findall(r'https?://[^\s\[\]<>"|\']+', content)
        
        if urls:
            # 使用第一个URL
            ref_dict[ref_name] = urls[0]
        else:
            # 使用文本内容
            clean_text = re.sub(r'\{\{[^\}]+\}\}', '', content)
            clean_text = re.sub(r'<[^>]+>', '', clean_text)
            clean_text = re.sub(r'\[\[([^\]|]+\|)?([^\]]+)\]\]', r'\2', clean_text)
            clean_text = clean_text.strip()
            ref_dict[ref_name] = clean_text
    
    return ref_dict


def build_reference_mapping(wikitext, ref_dict):
    """
    为wikitext中的所有ref标签建立位置到ref_key的映射
    
    参数:
    - wikitext: 完整的wikitext
    - ref_dict: 从References部分得到的{ref_name: url/文本}字典
    
    返回: (ref_position_map, updated_ref_dict)
    - ref_position_map: {ref在文本中的位置: ref_key}
    - updated_ref_dict: 更新后的字典（可能包含新的匿名引用）
    """
    ref_position_map = {}
    url_to_key = {}  # URL -> ref_key的反向映射
    anon_counter = 1
    
    # 建立URL到key的反向映射
    for key, value in ref_dict.items():
        if value.startswith('http://') or value.startswith('https://'):
            url_to_key[value] = key
    
    # 查找所有ref标签
    self_closing_pattern = r'<ref\s+[^>]*?/>'
    full_pattern = r'<ref(?:\s+[^>]*?)?>.*?</ref>'
    
    all_refs = []
    for match in re.finditer(self_closing_pattern, wikitext, re.IGNORECASE | re.DOTALL):
        all_refs.append((match.start(), match.group(0), True))
    
    for match in re.finditer(full_pattern, wikitext, re.IGNORECASE | re.DOTALL):
        all_refs.append((match.start(), match.group(0), False))
    
    # 按位置排序并去重
    all_refs.sort(key=lambda x: (x[0], x[2]))
    unique_refs = []
    last_pos = -1
    for pos, text, is_self_closing in all_refs:
        if pos != last_pos:
            unique_refs.append((pos, text, is_self_closing))
            last_pos = pos
    
    # 处理每个ref标签
    for ref_start, ref_text, is_self_closing in unique_refs:
        # 提取name属性
        name_match = re.search(r'name\s*=\s*["\']?([^"\'>\s]+)["\']?', ref_text, re.IGNORECASE)
        ref_name = name_match.group(1) if name_match else None
        
        # 提取内容中的URL
        content_match = re.search(r'<ref[^>]*>(.*?)</ref>', ref_text, re.IGNORECASE | re.DOTALL)
        content = content_match.group(1) if content_match else ''
        urls = re.findall(r'https?://[^\s\[\]<>"|\']+', content)
        
        ref_key = None
        
        # 情况1：引用中直接包含URL
        if urls:
            first_url = urls[0]
            if first_url in url_to_key:
                # URL已存在，使用已有的key
                ref_key = url_to_key[first_url]
            else:
                # 新URL，创建新的匿名key
                ref_key = f"anon_{anon_counter}"
                anon_counter += 1
                ref_dict[ref_key] = first_url
                url_to_key[first_url] = ref_key
        
        # 情况2：引用中不包含URL但有name属性
        elif ref_name:
            if ref_name in ref_dict:
                # name在字典中，使用这个name
                ref_key = ref_name
            # 如果name不在字典中，跳过（ref_key保持None）
        
        # 情况3：既没有URL也没有name，或name不在字典中
        # 跳过，不记录
        
        if ref_key:
            ref_position_map[ref_start] = ref_key
    
    return ref_position_map, ref_dict


def extract_citations_from_text(text, ref_position_map, text_offset=0):
    """
    从文本中提取引用的ref_key列表
    
    参数:
    - text: 要搜索的文本
    - ref_position_map: 全局位置 -> ref_key的映射
    - text_offset: 该文本在完整wikitext中的起始位置
    
    返回: ref_key列表（如 ["smith2020", "anon_1", "jones2019"]）
    """
    if not text:
        return []
    
    citations = []
    
    # 匹配所有ref标签
    self_closing_pattern = r'<ref\s+[^>]*?/>'
    full_pattern = r'<ref(?:\s+[^>]*?)?>.*?</ref>'
    
    # 收集所有匹配的位置
    all_positions = set()
    
    for match in re.finditer(self_closing_pattern, str(text), re.IGNORECASE | re.DOTALL):
        ref_start_global = text_offset + match.start()
        all_positions.add(ref_start_global)
    
    for match in re.finditer(full_pattern, str(text), re.IGNORECASE | re.DOTALL):
        ref_start_global = text_offset + match.start()
        all_positions.add(ref_start_global)
    
    # 查找对应的ref_key
    for pos in sorted(all_positions):
        if pos in ref_position_map:
            ref_key = ref_position_map[pos]
            if ref_key not in citations:
                citations.append(ref_key)
    
    return citations


def parse_references_section(wikicode, wikitext):
    """
    解析References部分，提取引用ID和对应的链接/文本
    返回: {引用ID: {'text': 文本, 'urls': [url列表]}}
    """
    references = {}
    ref_counter = 1
    
    # 方法1：解析reflist和references列表
    # 查找References部分（通常在== References ==标题下）
    lines = wikitext.split('\n')
    in_references = False
    references_content = []
    
    for line in lines:
        line_stripped = line.strip()
        
        # 检测References标题
        if line_stripped.startswith('==') and 'reference' in line_stripped.lower():
            in_references = True
            continue
        
        # 检测下一个标题（结束References部分）
        if in_references and line_stripped.startswith('=='):
            break
        
        if in_references:
            references_content.append(line)
    
    # 从references内容中提取列表项
    references_text = '\n'.join(references_content)
    
    # 匹配有序列表项（通常是 # 或 * 开头）
    list_items = re.findall(r'^[#*]\s*(.+)$', references_text, re.MULTILINE)
    
    for i, item in enumerate(list_items, 1):
        # 提取URL
        urls = re.findall(r'https?://[^\s\[\]<>"|\']+', item)
        
        # 清理文本（移除wikitext标记）
        clean_text = re.sub(r'\[\[([^\]|]+\|)?([^\]]+)\]\]', r'\2', item)  # 移除链接保留文本
        clean_text = re.sub(r"'''?([^']+)'''?", r'\1', clean_text)  # 移除加粗/斜体
        clean_text = re.sub(r'<[^>]+>', '', clean_text)  # 移除HTML标签
        clean_text = clean_text.strip()
        
        ref_id = str(i)
        references[ref_id] = {
            'text': clean_text,
            'urls': urls
        }
    
    # 方法2：查找所有的ref标签（用于内联引用）
    for tag in wikicode.filter_tags():
        if tag.tag.lower() == 'ref':
            # 提取ref的内容
            content = tag.contents
            if not content:
                continue
            
            content_str = str(content).strip()
            
            # 提取URL
            urls = re.findall(r'https?://[^\s\[\]<>"|\']+', content_str)
            
            # 尝试提取name属性作为ID
            ref_name = None
            if hasattr(tag, 'attributes') and tag.attributes:
                try:
                    # mwparserfromhell 的 attributes 是一个类似字典的对象
                    for attr in tag.attributes:
                        attr_name = str(attr.name).strip().lower()
                        if attr_name == 'name':
                            ref_name = str(attr.value).strip().strip('"\'')
                            break
                except (AttributeError, ValueError):
                    # 如果访问失败，尝试其他方法
                    try:
                        if hasattr(tag, 'get'):
                            ref_name = tag.get('name')
                            if ref_name:
                                ref_name = str(ref_name).strip().strip('"\'')
                    except:
                        pass
            
            # 清理文本
            clean_text = re.sub(r'\[\[([^\]|]+\|)?([^\]]+)\]\]', r'\2', content_str)
            clean_text = re.sub(r"'''?([^']+)'''?", r'\1', clean_text)
            clean_text = re.sub(r'<[^>]+>', '', clean_text)
            clean_text = re.sub(r'\{\{[^\}]+\}\}', '', clean_text)  # 移除模板
            clean_text = clean_text.strip()
            
            # 如果有name，使用name作为ID，否则使用数字
            if ref_name and ref_name not in references:
                references[ref_name] = {
                    'text': clean_text,
                    'urls': urls
                }
            elif clean_text or urls:  # 只有当有内容时才添加
                # 使用数字ID，避免冲突
                while str(ref_counter) in references:
                    ref_counter += 1
                references[str(ref_counter)] = {
                    'text': clean_text,
                    'urls': urls
                }
                ref_counter += 1
    
    # 方法3：查找cite模板
    for template in wikicode.filter_templates():
        template_name = str(template.name).strip().lower()
        if template_name.startswith('cite'):
            # 提取URL参数
            urls = []
            text_parts = []
            
            for param in template.params:
                param_name = str(param.name).strip().lower()
                param_value = str(param.value).strip()
                
                if param_name in ['url', 'chapter-url', 'archive-url', 'website']:
                    if param_value:
                        urls.append(param_value)
                elif param_name in ['title', 'work', 'publisher', 'author', 'last', 'first']:
                    if param_value:
                        text_parts.append(param_value)
            
            # 生成文本描述
            text = ', '.join(filter(None, text_parts))
            
            if text or urls:  # 只有当有内容时才添加
                # 使用数字ID
                while str(ref_counter) in references:
                    ref_counter += 1
                references[str(ref_counter)] = {
                    'text': text,
                    'urls': urls
                }
                ref_counter += 1
    
    return references


def extract_section_structure(wikitext, save_content=False):
    """
    提取文章的标题结构和每个部分的引用
    
    参数:
    - wikitext: Wikipedia的wikitext源码
    - save_content: 是否保存段落内容（默认False，只提取引用）
    
    返回: {
        'structure': 层级结构,
        'references': 引用字典（ref_key: url/文本）
    }
    """
    try:
        # 步骤1：从References部分提取引用字典
        ref_dict = extract_references_from_section(wikitext)
        
        # 步骤2：为所有ref标签建立位置映射
        ref_position_map, ref_dict = build_reference_mapping(wikitext, ref_dict)
        
        # 分割成行来处理标题
        lines = wikitext.split('\n')
        
        # 提取所有标题及其位置
        headings = []
        for line_idx, line in enumerate(lines):
            line = line.strip()
            if line.startswith('==') and line.endswith('==') and len(line) >= 4:
                # 计算标题级别（单边等号数）
                # == Title == -> 2个等号 -> level 2
                # === Subtitle === -> 3个等号 -> level 3  
                # ==== Subsubtitle ==== -> 4个等号 -> level 4
                level = 0
                for char in line:
                    if char == '=':
                        level += 1
                    else:
                        break
                # level现在就是单边的等号数，直接使用作为层级
                # 注意：Wikipedia中 == 是主章节（level 2），=== 是子章节（level 3）
                
                # 提取标题文本
                title = line.strip('=').strip()
                
                # 检查是否是需要跳过的标题
                title_lower = title.lower()
                if title_lower in SKIP_TITLES or title_lower in ['references', 'reference']:
                    continue
                
                headings.append({
                    'title': title,
                    'level': level,
                    'line_idx': line_idx
                })
        
        if not headings:
            return {'structure': [], 'references': ref_dict}
        
        # 创建行号到字符偏移量的映射
        line_to_offset = [0]  # 第0行的偏移是0
        for line in lines:
            # +1 是为了包含换行符
            line_to_offset.append(line_to_offset[-1] + len(line) + 1)
        
        # 为每个标题提取内容和引用
        for i, heading in enumerate(headings):
            start_line = heading['line_idx'] + 1
            
            # 找到下一个标题的位置
            if i + 1 < len(headings):
                end_line = headings[i + 1]['line_idx']
            else:
                end_line = len(lines)
            
            # 提取该部分的文本
            section_text = '\n'.join(lines[start_line:end_line])
            
            # 找到该部分的第一个子标题位置
            first_subsection_line = None
            for j in range(i + 1, len(headings)):
                if headings[j]['level'] > heading['level']:
                    first_subsection_line = headings[j]['line_idx']
                    break
                elif headings[j]['level'] <= heading['level']:
                    break
            
            # 如果有子标题，提取overview部分
            if first_subsection_line is not None:
                overview_end = first_subsection_line
                overview_text = '\n'.join(lines[start_line:overview_end])
                # 计算overview在wikitext中的偏移量
                overview_offset = line_to_offset[start_line]
                overview_citations = extract_citations_from_text(overview_text, ref_position_map, overview_offset)
                
                # 移除空行和wikitext标记后检查是否有实际内容
                cleaned_overview = overview_text.strip()
                # 移除常见的wikitext标记
                cleaned_overview = re.sub(r'\{\{[^}]+\}\}', '', cleaned_overview)  # 移除模板
                cleaned_overview = re.sub(r'<[^>]+>', '', cleaned_overview)  # 移除HTML标签
                # 移除文件/图片链接
                cleaned_overview = re.sub(r'\[\[File:[^\]]+\]\]', '', cleaned_overview, flags=re.IGNORECASE)
                cleaned_overview = re.sub(r'\[\[Image:[^\]]+\]\]', '', cleaned_overview, flags=re.IGNORECASE)
                # 处理普通链接，保留显示文字
                cleaned_overview = re.sub(r'\[\[([^\]|]+\|)?([^\]]+)\]\]', r'\2', cleaned_overview)
                # 移除粗体和斜体标记
                cleaned_overview = re.sub(r"'{2,}", '', cleaned_overview)
                # 清理多余空白
                cleaned_overview = re.sub(r'\s+', ' ', cleaned_overview)
                cleaned_overview = cleaned_overview.strip()
                
                # 只有当有实际文本内容或引用时才创建overview
                heading['has_overview'] = len(cleaned_overview) > 20 or len(overview_citations) > 0
                heading['overview_citations'] = overview_citations if heading['has_overview'] else []
                # 如果需要保存内容，保存清理后的文本
                heading['overview_content'] = cleaned_overview if save_content and heading['has_overview'] else None
            else:
                heading['has_overview'] = False
                heading['overview_citations'] = []
                heading['overview_content'] = None
            
            # 提取该部分的所有引用（不包括子部分）
            if first_subsection_line:
                section_only_text = '\n'.join(lines[start_line:first_subsection_line])
            else:
                section_only_text = section_text
            
            # 计算section在wikitext中的偏移量
            section_offset = line_to_offset[start_line]
            citations = extract_citations_from_text(section_only_text, ref_position_map, section_offset)
            heading['citations'] = citations
            heading['subsections_start'] = first_subsection_line
            
            # 如果需要保存内容，清理并保存该部分的文本
            if save_content:
                cleaned_section = section_only_text.strip()
                # 移除常见的wikitext标记
                cleaned_section = re.sub(r'\{\{[^}]+\}\}', '', cleaned_section)  # 移除模板
                cleaned_section = re.sub(r'<[^>]+>', '', cleaned_section)  # 移除HTML标签
                # 移除文件/图片链接
                cleaned_section = re.sub(r'\[\[File:[^\]]+\]\]', '', cleaned_section, flags=re.IGNORECASE)
                cleaned_section = re.sub(r'\[\[Image:[^\]]+\]\]', '', cleaned_section, flags=re.IGNORECASE)
                # 处理普通链接，保留显示文字
                cleaned_section = re.sub(r'\[\[([^\]|]+\|)?([^\]]+)\]\]', r'\2', cleaned_section)
                # 移除粗体和斜体标记
                cleaned_section = re.sub(r"'{2,}", '', cleaned_section)
                # 清理多余空白
                cleaned_section = re.sub(r'\s+', ' ', cleaned_section)
                cleaned_section = cleaned_section.strip()
                heading['section_content'] = cleaned_section
            else:
                heading['section_content'] = None
        
        # 构建树形结构
        def build_tree(headings, start_idx=0, parent_level=0):
            """递归构建树形结构"""
            result = []
            i = start_idx
            
            while i < len(headings):
                heading = headings[i]
                
                # 如果当前标题的级别小于等于父级，返回
                if heading['level'] <= parent_level:
                    break
                
                # 如果是直接子节点
                if heading['level'] == parent_level + 1:
                    node = {
                        'title': heading['title'],
                        'level': heading['level'],
                        'citations': heading['citations'],
                        'children': []
                    }
                    
                    # 如果需要保存内容且有内容，添加content字段
                    if save_content and heading.get('section_content'):
                        node['content'] = heading['section_content']
                    
                    # 如果有overview，添加overview节点
                    if heading.get('has_overview', False):
                        overview_node = {
                            'title': 'overview',
                            'level': heading['level'] + 1,
                            'citations': heading['overview_citations'],
                            'children': []
                        }
                        # 如果需要保存内容且有内容，添加content字段
                        if save_content and heading.get('overview_content'):
                            overview_node['content'] = heading['overview_content']
                        node['children'].append(overview_node)
                    
                    # 递归处理子节点
                    children, next_i = build_tree(headings, i + 1, heading['level'])
                    node['children'].extend(children)
                    
                    result.append(node)
                    i = next_i
                else:
                    i += 1
            
            return result, i
        
        # 从第一级标题开始构建
        # Wikipedia通常从 == Title == (level 2) 开始，所以parent_level设为1
        structure, _ = build_tree(headings, 0, 1)
        
        # 为每个节点收集所有子节点的引用（父节点应包含所有子部分的引用）
        def collect_all_citations(node):
            """递归收集节点及其所有子节点的引用"""
            all_citations = set(node.get('citations', []))
            
            for child in node.get('children', []):
                child_citations = collect_all_citations(child)
                all_citations.update(child_citations)
            
            # 更新节点的citations为包含所有子节点的引用
            node['citations'] = sorted(list(all_citations), key=lambda x: int(x) if x.isdigit() else 999)
            
            return all_citations
        
        # 对每个顶级节点应用
        for node in structure:
            collect_all_citations(node)
        
        return {
            'structure': structure,
            'references': ref_dict
        }
        
    except Exception as e:
        import traceback
        print(f"  解析错误: {e}")
        print("  详细错误信息：")
        traceback.print_exc()
        return {'structure': [], 'references': {}}


def process_single_file(category, wikitext_file, metadata_file, save_content=False):
    """处理单个Wikipedia页面文件"""
    # 读取元数据
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    topic = metadata['topic']
    
    # 读取wikitext
    with open(wikitext_file, 'r', encoding='utf-8') as f:
        wikitext = f.read()
    
    # 解析结构
    result = extract_section_structure(wikitext, save_content=save_content)
    
    # 组装输出
    output = {
        'topic': topic,
        'category': category,
        'pageid': metadata.get('pageid'),
        'url': metadata.get('url'),
        'structure': result['structure'],
        'references': result['references']
    }
    
    return output


def process_all_pages(input_dir, structure_output, references_output, save_content=False):
    """处理所有爬取的页面"""
    input_path = Path(input_dir)
    
    # 统计
    total = 0
    processed = 0
    failed = 0
    
    # 收集所有结构和references
    all_structures = {}
    all_references = {}
    
    print("=" * 80)
    print("开始解析Wikipedia页面...")
    if save_content:
        print("【模式】: 保存段落内容")
    else:
        print("【模式】: 仅提取引用")
    print("=" * 80)
    
    # 遍历所有类别
    for category_dir in input_path.iterdir():
        if not category_dir.is_dir():
            continue
        
        category = category_dir.name
        print(f"\n[{category}] 开始处理...")
        
        # 查找所有.txt文件
        txt_files = list(category_dir.glob("*.txt"))
        total += len(txt_files)
        
        for txt_file in txt_files:
            json_file = txt_file.with_suffix('.json')
            
            if not json_file.exists():
                print(f"  ✗ 缺少元数据文件: {json_file.name}")
                failed += 1
                continue
            
            try:
                # 处理文件
                result = process_single_file(category, txt_file, json_file, save_content=save_content)
                
                # 收集结构
                topic_key = f"{category}:{result['topic']}"
                all_structures[topic_key] = {
                    'topic': result['topic'],
                    'category': result['category'],
                    'pageid': result['pageid'],
                    'url': result['url'],
                    'structure': result['structure']
                }
                
                # 收集references
                all_references[topic_key] = {
                    'topic': result['topic'],
                    'category': result['category'],
                    'pageid': result['pageid'],
                    'references': result['references']
                }
                
                processed += 1
                print(f"  ✓ {result['topic']}")
                
            except Exception as e:
                print(f"  ✗ 处理失败 {txt_file.name}: {e}")
                failed += 1
    
    # 保存所有结构到一个文件
    print(f"\n保存所有结构到 {structure_output}...")
    with open(structure_output, 'w', encoding='utf-8') as f:
        json.dump(all_structures, f, ensure_ascii=False, indent=2)
    
    # 保存所有references到一个文件
    print(f"保存所有references到 {references_output}...")
    with open(references_output, 'w', encoding='utf-8') as f:
        json.dump(all_references, f, ensure_ascii=False, indent=2)
    
    # 打印统计
    print("\n" + "=" * 80)
    print("解析完成！")
    print(f"总计: {total} 个页面")
    print(f"成功: {processed}")
    print(f"失败: {failed}")
    print(f"结构输出文件: {structure_output}")
    print(f"引用输出文件: {references_output}")
    print("=" * 80)


def main():
    print("=" * 80)
    print("Wikipedia 页面结构解析程序")
    print("=" * 80)
    
    process_all_pages(INPUT_DIR, STRUCTURE_OUTPUT, REFERENCES_OUTPUT, save_content=SAVE_CONTENT)


if __name__ == "__main__":
    main()

