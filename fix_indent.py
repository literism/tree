#!/usr/bin/env python3
"""修复prepare_dataset.py的缩进问题"""

# 读取文件
with open('summary_based_classifier/prepare_dataset.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 使用autopep8修复缩进
try:
    import autopep8
    fixed = autopep8.fix_code(content, options={'aggressive': 2})
    
    with open('summary_based_classifier/prepare_dataset.py', 'w', encoding='utf-8') as f:
        f.write(fixed)
    
    print("✓ 使用autopep8修复完成")
except ImportError:
    print("autopep8未安装，尝试手动修复...")
    
    # 手动修复关键的缩进问题
    lines = content.split('\n')
    
    # 这些行需要特定的缩进
    fixes = {
        212: ('if isinstance(current_summary_data, dict):', 24),
        218: ('else:', 28),  # if explanation的else
        220: ('else:', 24),  # if isinstance的else
        254: ('if isinstance(child_summary_data, dict):', 24),
        265: ('else:', 24),
        276: ('all_child_summaries.append(child_summary)', 24),
        281: ('continue', 24),
        284: ('correct_indices = []', 20),
        285: ('for title in output_titles:', 20),
        300: ('continue', 24),
    }
    
    for line_num, (expected_start, indent) in fixes.items():
        if line_num - 1 < len(lines):
            line = lines[line_num - 1]
            if line.strip().startswith(expected_start.split('(')[0]):
                lines[line_num - 1] = ' ' * indent + line.strip()
                print(f"Fixed line {line_num}")
    
    with open('summary_based_classifier/prepare_dataset.py', 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print("✓ 手动修复完成")

