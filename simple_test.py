#!/usr/bin/env python3
"""
简单测试脚本
"""

import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

print("=== 测试开始 ===")

try:
    print("1. 测试配置导入...")
    from config.game_config import GameConfig
    config = GameConfig()
    print("   ✓ 配置导入成功")
    
    print("2. 测试市场环境导入...")
    from core.market_environment import MarketEnvironment, LocationType, TimePeriod
    print("   ✓ 市场环境导入成功")
    
    print("3. 测试市场环境初始化...")
    market = MarketEnvironment(config)
    print("   ✓ 市场环境初始化成功")
    
    print("4. 测试订单生成...")
    orders = market.generate_orders(30.0)  # 生成30分钟的订单
    print(f"   ✓ 生成 {len(orders)} 个订单")
    
    if orders:
        first_order = orders[0]
        print(f"   第一个订单: 价格={first_order.price:.2f}元, "
              f"位置={first_order.location_type.value}, "
              f"时间段={first_order.time_period.value}")
    
    print("5. 测试司机策略处理...")
    driver_strategies = {'司机A': 25.0, '司机B': 30.0}
    market.apply_competition_effects(driver_strategies)
    print(f"   ✓ 竞争效应应用成功，供给水平: {market.current_state.market_supply_level:.2f}")
    
    print("6. 测试订单分配...")
    driver_orders = market.process_driver_decisions(orders, driver_strategies)
    
    total_accepted = 0
    for driver, accepted_orders in driver_orders.items():
        count = len(accepted_orders)
        total_accepted += count
        print(f"   {driver}: {count} 个订单")
    
    acceptance_rate = total_accepted / len(orders) if orders else 0
    print(f"   ✓ 总接单率: {acceptance_rate:.1%}")
    
    print("7. 测试收益计算...")
    revenues = market.calculate_driver_revenues(driver_orders, driver_strategies)
    for driver, revenue in revenues.items():
        print(f"   {driver}: {revenue:.2f}元")
    
    print("8. 测试市场摘要...")
    summary = market.get_market_summary()
    print(f"   当前时间: {summary['current_time']:.1f}点")
    print(f"   时间段: {summary['time_period']}")
    print(f"   订单率: {summary['order_rate']:.2f}")
    print(f"   接单率: {summary['acceptance_rate']:.1%}")
    
    print("\n🎉 所有测试通过！市场环境模拟器工作正常。")
    
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    import traceback
    traceback.print_exc()
    
except Exception as e:
    print(f"❌ 运行错误: {e}")
    import traceback
    traceback.print_exc()

print("\n=== 测试结束 ===")