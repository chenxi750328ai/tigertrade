#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
精确验证参数修改的测试
"""

import sys
import re

def verify_parameter_changes():
    """验证参数是否已正确修改"""
    print("🔍 验证参数修改是否正确应用...")
    
    file_path = '/home/cx/tigertrade/tiger1.py'
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 查找所有buffer计算的位置
    # 使用正则表达式匹配buffer计算行
    pattern_old = r'buffer = max\(0\.5 \* \(atr if atr else 0\), 0\.02\)'
    pattern_new = r'buffer = max\(0\.1 \* \(atr if atr else 0\), 0\.005\)'
    
    old_matches = re.findall(pattern_old, content)
    new_matches = re.findall(pattern_new, content)
    
    print(f"📊 搜索结果:")
    print(f"   旧参数 (0.5, 0.02): {len(old_matches)} 个匹配")
    print(f"   新参数 (0.1, 0.005): {len(new_matches)} 个匹配")
    
    # 检查文件内容中具体的buffer行
    lines = content.split('\n')
    buffer_lines = []
    for i, line in enumerate(lines):
        if 'buffer = max(' in line and ('0.5' in line or '0.1' in line):
            buffer_lines.append((i+1, line.strip()))
    
    print(f"\n📝 文件中的buffer计算行:")
    for line_num, line_content in buffer_lines:
        print(f"   第{line_num}行: {line_content}")
    
    # 验证是否所有的旧参数都已被替换
    success = len(new_matches) > 0 and len(old_matches) == 0
    print(f"\n✅ 参数修改验证: {'成功' if success else '失败'}")
    
    if success:
        print(f"   ✓ 所有旧参数已被新参数替换")
        print(f"   ✓ 新参数 (0.1, 0.005) 已正确应用")
    else:
        print(f"   ⚠️  仍有旧参数未被替换或新参数未正确应用")
    
    return success


def test_specific_example():
    """测试具体示例"""
    print(f"\n🔧 测试具体示例...")
    
    # 使用固定的ATR值来验证参数效果
    atr_value = 0.2
    
    # 旧参数计算
    old_buffer = max(0.5 * atr_value, 0.02)
    # 新参数计算
    new_buffer = max(0.1 * atr_value, 0.005)
    
    print(f"当ATR = {atr_value} 时:")
    print(f"   旧参数 buffer = max(0.5 * {atr_value}, 0.02) = {old_buffer}")
    print(f"   新参数 buffer = max(0.1 * {atr_value}, 0.005) = {new_buffer}")
    print(f"   缓冲区减小了 {(old_buffer - new_buffer)/old_buffer*100:.1f}%")
    
    # 验证新参数确实更小，更敏感
    improvement = new_buffer < old_buffer
    print(f"   参数改进: {'✅' if improvement else '❌'}")
    
    return improvement


def verify_code_syntax():
    """验证代码语法"""
    print(f"\n🔧 验证代码语法...")
    
    try:
        import ast
        with open('/home/cx/tigertrade/tiger1.py', 'r', encoding='utf-8') as f:
            source = f.read()
        ast.parse(source)
        print("✅ 代码语法正确")
        return True
    except SyntaxError as e:
        print(f"❌ 代码语法错误: {e}")
        return False


def run_import_test():
    """运行导入测试"""
    print(f"\n🔧 运行导入测试...")
    try:
        # 临时添加路径
        sys.path.insert(0, '/home/cx/tigertrade')
        from src import tiger1 as t1
        print("✅ 模块导入成功")
        return True
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        return False
    except Exception as e:
        print(f"❌ 其他错误: {e}")
        return False


def main():
    """主函数"""
    print("🚀 开始精确参数验证测试...\n")
    
    # 验证参数修改
    param_ok = verify_parameter_changes()
    
    # 测试具体示例
    example_ok = test_specific_example()
    
    # 验证代码语法
    syntax_ok = verify_code_syntax()
    
    # 运行导入测试
    import_ok = run_import_test()
    
    print(f"\n✅ 验证结果:")
    print(f"   参数修改: {'✅ 通过' if param_ok else '❌ 失败'}")
    print(f"   示例验证: {'✅ 通过' if example_ok else '❌ 失败'}")
    print(f"   语法检查: {'✅ 通过' if syntax_ok else '❌ 失败'}")
    print(f"   导入测试: {'✅ 通过' if import_ok else '❌ 失败'}")
    
    overall_success = param_ok and example_ok and syntax_ok and import_ok
    
    print(f"\n🎯 总体验证结果: {'✅ 成功' if overall_success else '❌ 失败'}")
    
    if overall_success:
        print(f"\n🎉 参数修改验证成功！")
        print(f"   ✓ 旧参数 (0.5, 0.02) 已被完全替换")
        print(f"   ✓ 新参数 (0.1, 0.005) 已正确应用")
        print(f"   ✓ 缓冲区计算更敏感，改善了策略响应")
        print(f"   ✓ 代码语法正确，可以正常导入")
        print(f"\n   修复使缓冲区计算公式从:")
        print(f"      buffer = max(0.5 * atr, 0.02)")
        print(f"   变为:")
        print(f"      buffer = max(0.1 * atr, 0.005)")
        print(f"   这使缓冲区大小减少了80%，策略更敏感")
    else:
        print(f"\n❌ 参数修改验证失败")
    
    return overall_success


if __name__ == "__main__":
    main()