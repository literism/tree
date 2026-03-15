"""
Wikipedia数据处理流水线主程序
整合爬虫和解析两个步骤
"""

import sys
import os

def main():
    print("=" * 80)
    print("Wikipedia 数据处理流水线")
    print("=" * 80)
    print("\n这个流水线包含两个步骤：")
    print("1. 爬取Wikipedia页面")
    print("2. 解析页面结构和引用")
    print()
    
    choice = input("请选择要执行的步骤 (1/2/all): ").strip().lower()
    
    if choice in ['1', 'all']:
        print("\n" + "=" * 80)
        print("步骤 1: 爬取Wikipedia页面")
        print("=" * 80)
        from crawl_wikipedia_topics import main as crawl_main
        try:
            crawl_main()
        except Exception as e:
            print(f"爬取过程出错: {e}")
            if choice == 'all':
                print("继续执行下一步...")
            else:
                return
    
    if choice in ['2', 'all']:
        print("\n" + "=" * 80)
        print("步骤 2: 解析页面结构")
        print("=" * 80)
        from parse_wikipedia_structure import main as parse_main
        try:
            parse_main()
        except Exception as e:
            print(f"解析过程出错: {e}")
            return
    
    if choice not in ['1', '2', 'all']:
        print("无效的选择，请输入 1、2 或 all")
        return
    
    print("\n" + "=" * 80)
    print("流水线执行完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()

