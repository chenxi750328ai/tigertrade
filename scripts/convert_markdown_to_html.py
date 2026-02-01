#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将Markdown周报转换为HTML格式
"""
import sys
import os
import re
from datetime import datetime

sys.path.insert(0, '/home/cx/tigertrade')

try:
    import markdown
    from markdown.extensions import codehilite, tables, toc
except ImportError:
    print("安装markdown库...")
    os.system("pip install markdown markdown-extensions")
    import markdown
    from markdown.extensions import codehilite, tables, toc

def markdown_to_html(md_file, html_file):
    """将Markdown文件转换为HTML"""
    with open(md_file, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # 配置Markdown扩展
    extensions = [
        'codehilite',
        'tables',
        'toc',
        'fenced_code',
        'nl2br',
    ]
    
    # 转换为HTML
    html_body = markdown.markdown(
        md_content,
        extensions=extensions,
        extension_configs={
            'codehilite': {
                'css_class': 'highlight',
                'use_pygments': False
            },
            'toc': {
                'permalink': True,
                'baselevel': 2
            }
        }
    )
    
    # 生成完整的HTML文档
    html_template = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TigerTrade周报：架构重构与执行器模块化</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", "Helvetica Neue", Helvetica, Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            background-color: white;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            margin-top: 0;
        }}
        h2 {{
            color: #34495e;
            border-bottom: 2px solid #ecf0f1;
            padding-bottom: 8px;
            margin-top: 30px;
        }}
        h3 {{
            color: #555;
            margin-top: 25px;
        }}
        h4 {{
            color: #666;
            margin-top: 20px;
        }}
        code {{
            background-color: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: "Courier New", monospace;
            font-size: 0.9em;
        }}
        pre {{
            background-color: #2d2d2d;
            color: #f8f8f2;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
            line-height: 1.5;
        }}
        pre code {{
            background-color: transparent;
            padding: 0;
            color: inherit;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #3498db;
            color: white;
            font-weight: bold;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        blockquote {{
            border-left: 4px solid #3498db;
            margin: 20px 0;
            padding: 10px 20px;
            background-color: #f8f9fa;
            color: #555;
        }}
        ul, ol {{
            margin: 15px 0;
            padding-left: 30px;
        }}
        li {{
            margin: 8px 0;
        }}
        a {{
            color: #3498db;
            text-decoration: none;
        }}
        a:hover {{
            text-decoration: underline;
        }}
        .meta {{
            background-color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
            font-size: 0.9em;
            color: #555;
        }}
        .meta strong {{
            color: #2c3e50;
        }}
        .toc {{
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 5px;
            margin: 20px 0;
        }}
        .toc ul {{
            list-style-type: none;
            padding-left: 0;
        }}
        .toc li {{
            margin: 5px 0;
        }}
        .toc a {{
            color: #34495e;
        }}
        .highlight {{
            background-color: #fff3cd;
            padding: 2px 4px;
            border-radius: 3px;
        }}
        .success {{
            color: #27ae60;
        }}
        .warning {{
            color: #f39c12;
        }}
        .error {{
            color: #e74c3c;
        }}
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 2px solid #ecf0f1;
            text-align: center;
            color: #7f8c8d;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        {html_body}
        <div class="footer">
            <p><strong>本文档由AI Agent自主完成</strong></p>
            <p>实际工作时间：约12小时（AI Agent自主工作，人类通过自然语言指令控制）</p>
            <p>生成时间：{datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}</p>
        </div>
    </div>
</body>
</html>"""
    
    # 保存HTML文件
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_template)
    
    print(f"✅ HTML文件已生成: {html_file}")

if __name__ == '__main__':
    import sys
    
    # 支持命令行参数指定文件
    if len(sys.argv) > 1:
        md_file = sys.argv[1]
    else:
        md_file = '/home/cx/tigertrade/docs/WEEKLY_REPORT_2026_WEEK05.md'
    
    # 自动生成HTML文件名
    if md_file.endswith('.md'):
        html_file = md_file[:-3] + '.html'
    else:
        html_file = md_file + '.html'
    
    if not os.path.exists(md_file):
        print(f"❌ Markdown文件不存在: {md_file}")
        sys.exit(1)
    
    markdown_to_html(md_file, html_file)
    print(f"📄 文件大小: {os.path.getsize(html_file) / 1024:.2f} KB")
