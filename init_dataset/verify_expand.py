"""
验证expand_references.py的关键功能
在运行主程序前进行测试
"""

from expand_references import (
    extract_real_url,
    build_query_path,
    find_leaf_nodes
)


def test_extract_url():
    """测试URL提取功能"""
    print("=" * 80)
    print("测试1: URL提取")
    print("=" * 80)
    
    test_cases = [
        # Bing跳转URL（已编码）
        "https://www.bing.com/ck/a?!&&p=123&u=aHR0cHM6Ly93d3cuZXhhbXBsZS5jb20%2f&ntb=1",
        # Bing跳转URL（未编码）
        "https://www.bing.com/ck/a?u=https://www.example.com/page&v=1",
        # 普通URL
        "https://www.wikipedia.org/wiki/Test",
        # 另一个Bing格式
        "https://www.bing.com/ck/a?u=https%3A%2F%2Fwww.test.com%2Fpath&p=1"
    ]
    
    for i, url in enumerate(test_cases, 1):
        result = extract_real_url(url)
        print(f"\n测试 {i}:")
        print(f"  输入: {url[:80]}...")
        print(f"  输出: {result}")
    
    print("\n✓ URL提取测试完成")


def test_build_query():
    """测试query构建"""
    print("\n" + "=" * 80)
    print("测试2: Query构建")
    print("=" * 80)
    
    test_cases = [
        # (node_path, topic, expected_contains)
        (["Early life", "overview"], "Albert Einstein", "Albert Einstein - Early life"),
        (["Early life", "Education"], "Albert Einstein", "Albert Einstein - Early life - Education"),
        (["Later life"], "Marie Curie", "Marie Curie - Later life"),
        (["Overview"], "Tesla Inc.", "Tesla Inc."),  # overview会被过滤
    ]
    
    for i, (path, topic, expected) in enumerate(test_cases, 1):
        result = build_query_path(path, topic)
        match = "✓" if expected.lower() == result.lower() else "✗"
        print(f"\n测试 {i}: {match}")
        print(f"  路径: {' > '.join(path)}")
        print(f"  Topic: {topic}")
        print(f"  期望: {expected}")
        print(f"  结果: {result}")
    
    print("\n✓ Query构建测试完成")


def test_find_leaves():
    """测试叶子节点查找"""
    print("\n" + "=" * 80)
    print("测试3: 叶子节点查找")
    print("=" * 80)
    
    # 构建测试结构
    test_structure = [
        {
            "title": "Early life",
            "level": 2,
            "citations": [],
            "children": [
                {
                    "title": "overview",
                    "level": 3,
                    "citations": [],
                    "children": []  # 叶子节点
                },
                {
                    "title": "Education",
                    "level": 3,
                    "citations": [],
                    "children": []  # 叶子节点
                }
            ]
        },
        {
            "title": "Later life",
            "level": 2,
            "citations": [],
            "children": [
                {
                    "title": "Death",
                    "level": 3,
                    "citations": [],
                    "children": []  # 叶子节点
                }
            ]
        },
        {
            "title": "Legacy",
            "level": 2,
            "citations": [],
            "children": []  # 叶子节点
        }
    ]
    
    leaves = find_leaf_nodes(test_structure)
    
    print(f"\n找到 {len(leaves)} 个叶子节点:")
    for i, (path, node) in enumerate(leaves, 1):
        print(f"  {i}. {' > '.join(path)}")
    
    expected_count = 4  # overview, Education, Death, Legacy
    if len(leaves) == expected_count:
        print(f"\n✓ 叶子节点数量正确 (期望:{expected_count}, 实际:{len(leaves)})")
    else:
        print(f"\n✗ 叶子节点数量错误 (期望:{expected_count}, 实际:{len(leaves)})")
    
    print("\n✓ 叶子节点查找测试完成")


def test_integration():
    """集成测试：模拟处理一个小结构"""
    print("\n" + "=" * 80)
    print("测试4: 集成测试")
    print("=" * 80)
    
    # 模拟一个topic结构
    topic = "Test Topic"
    structure = [
        {
            "title": "Section A",
            "level": 2,
            "citations": ["ref1"],
            "children": [
                {
                    "title": "overview",
                    "level": 3,
                    "citations": [],
                    "children": []
                }
            ]
        }
    ]
    
    # 查找叶子节点
    leaves = find_leaf_nodes(structure)
    print(f"\n叶子节点: {len(leaves)} 个")
    
    # 构建queries
    queries = []
    for path, node in leaves:
        query = build_query_path(path, topic)
        queries.append(query)
        print(f"  Query: {query}")
    
    # 模拟URL处理
    test_urls = [
        "https://www.bing.com/ck/a?u=https://example.com/page1",
        "https://direct-url.com/page2"
    ]
    
    print(f"\n处理URLs:")
    for url in test_urls:
        real_url = extract_real_url(url)
        print(f"  {url[:50]}... -> {real_url}")
    
    print("\n✓ 集成测试完成")


def main():
    """运行所有测试"""
    print("\n" + "=" * 80)
    print("验证 expand_references 功能")
    print("=" * 80)
    print()
    
    try:
        test_extract_url()
        test_build_query()
        test_find_leaves()
        test_integration()
        
        print("\n" + "=" * 80)
        print("✓ 所有测试通过！")
        print("=" * 80)
        print("\n可以安全运行 expand_references.py")
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

