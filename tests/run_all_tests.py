#!/usr/bin/env python3
"""
全套测试运行脚本
运行所有单元测试和集成测试
"""

import unittest
import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

def run_all_tests():
    """运行所有测试"""
    print("="*60)
    print("🧪 博弈论项目 - 全套测试")
    print("="*60)
    
    # 查找所有测试文件
    test_dir = Path(__file__).parent
    test_files = [
        'test_config',
        'test_market_environment', 
        'test_ai_models',
        'test_experiments'
    ]
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 加载所有测试
    for test_file in test_files:
        try:
            module = __import__(test_file)
            suite.addTests(loader.loadTestsFromModule(module))
            print(f"✅ 已加载测试模块: {test_file}")
        except ImportError as e:
            print(f"❌ 无法加载测试模块 {test_file}: {e}")
        except Exception as e:
            print(f"⚠️  加载测试模块 {test_file} 时出现错误: {e}")
    
    # 运行测试
    print("\n" + "="*60)
    print("🚀 开始运行测试...")
    print("="*60)
    
    runner = unittest.TextTestRunner(
        verbosity=2,
        stream=sys.stdout,
        descriptions=True,
        failfast=False
    )
    
    result = runner.run(suite)
    
    # 输出测试结果总结
    print("\n" + "="*60)
    print("📊 测试结果总结")
    print("="*60)
    
    print(f"总测试数: {result.testsRun}")
    print(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    
    if result.failures:
        print("\n❌ 失败的测试:")
        for test, traceback in result.failures:
            print(f"  - {test}")
    
    if result.errors:
        print("\n💥 错误的测试:")
        for test, traceback in result.errors:
            print(f"  - {test}")
    
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    print(f"\n📈 成功率: {success_rate:.1f}%")
    
    if result.wasSuccessful():
        print("\n🎉 所有测试通过！")
        return True
    else:
        print("\n⚠️  部分测试失败，请检查代码。")
        return False


def run_quick_test():
    """运行快速测试（仅关键功能）"""
    print("⚡ 运行快速测试...")
    
    from validate_model import (
        test_basic_imports,
        test_configuration,
        test_ai_models,
        run_mini_experiment
    )
    
    tests = [
        ("基本导入", test_basic_imports),
        ("配置功能", test_configuration),
        ("AI模型", test_ai_models),
        ("迷你实验", run_mini_experiment)
    ]
    
    passed = 0
    for test_name, test_func in tests:
        try:
            if test_func():
                print(f"  ✅ {test_name} - 通过")
                passed += 1
            else:
                print(f"  ❌ {test_name} - 失败")
        except Exception as e:
            print(f"  💥 {test_name} - 错误: {e}")
    
    print(f"\n快速测试结果: {passed}/{len(tests)} 通过")
    return passed == len(tests)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='运行测试套件')
    parser.add_argument('--quick', '-q', action='store_true',
                       help='仅运行快速测试')
    parser.add_argument('--module', '-m', 
                       help='运行特定测试模块')
    
    args = parser.parse_args()
    
    if args.quick:
        success = run_quick_test()
    elif args.module:
        # 运行特定模块测试
        try:
            loader = unittest.TestLoader()
            suite = loader.loadTestsFromName(args.module)
            runner = unittest.TextTestRunner(verbosity=2)
            result = runner.run(suite)
            success = result.wasSuccessful()
        except Exception as e:
            print(f"无法运行模块 {args.module}: {e}")
            success = False
    else:
        success = run_all_tests()
    
    sys.exit(0 if success else 1) 