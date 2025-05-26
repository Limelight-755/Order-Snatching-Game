#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
市场环境模拟器测试脚本 - 无特殊字符版本
"""

import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

print("=== 市场环境模拟器测试 ===")

try:
    print("1. 导入配置模块...")
    from config.game_config import GameConfig
    config = GameConfig()
    print("   配置模块导入成功")
    
    print("2. 导入市场环境模块...")
    from core.market_environment import MarketEnvironment, LocationType, TimePeriod
    print("   市场环境模块导入成功")
    
    print("3. 初始化市场环境...")
    market = MarketEnvironment(config)
    print("   市场环境初始化成功")
    
    print("4. 测试订单生成...")
    orders = market.generate_orders(30.0)  # 生成30分钟的订单
    print(f"   生成订单数量: {len(orders)}")
    
    if orders:
        first_order = orders[0]
        print(f"   第一个订单详情:")
        print(f"     价格: {first_order.price:.2f}元")
        print(f"     位置类型: {first_order.location_type.value}")
        print(f"     时间段: {first_order.time_period.value}")
        print(f"     等待时限: {first_order.waiting_time_limit:.1f}分钟")
    
    print("5. 测试竞争效应...")
    driver_strategies = {'司机A': 25.0, '司机B': 30.0}
    market.apply_competition_effects(driver_strategies)
    print(f"   市场供给水平: {market.current_state.market_supply_level:.2f}")
    print(f"   竞争激烈程度: {market.current_state.competition_intensity:.2f}")
    
    print("6. 测试订单分配...")
    driver_orders = market.process_driver_decisions(orders, driver_strategies)
    
    total_accepted = 0
    for driver, accepted_orders in driver_orders.items():
        count = len(accepted_orders)
        total_accepted += count
        print(f"   {driver}: 接受 {count} 个订单")
    
    acceptance_rate = total_accepted / len(orders) if orders else 0
    print(f"   总接单率: {acceptance_rate:.1%}")
    
    print("7. 测试收益计算...")
    revenues = market.calculate_driver_revenues(driver_orders, driver_strategies)
    for driver, revenue in revenues.items():
        print(f"   {driver}: 收益 {revenue:.2f}元")
    
    print("8. 测试时间变化...")
    initial_time = market.current_state.current_time
    market.update_market_time(2.0)  # 时间前进2小时
    print(f"   时间从 {initial_time:.1f}点 变化到 {market.current_state.current_time:.1f}点")
    print(f"   当前时间段: {market.current_state.time_period.value}")
    print(f"   当前订单率: {market.current_state.order_rate:.2f}")
    
    print("9. 测试市场摘要...")
    summary = market.get_market_summary()
    print(f"   总订单数: {summary['total_orders']}")
    print(f"   接单率: {summary['acceptance_rate']:.1%}")
    print(f"   总收益: {summary['total_revenue']:.2f}元")
    print(f"   平均订单价格: {summary['avg_order_price']:.2f}元")
    
    print("10. 测试市场冲击...")
    original_rate = market.current_state.order_rate
    market.apply_market_shock('demand_surge', magnitude=2.0, duration=3)
    new_rate = market.current_state.order_rate
    print(f"    冲击前订单率: {original_rate:.2f}")
    print(f"    冲击后订单率: {new_rate:.2f}")
    print(f"    冲击倍数: {new_rate/original_rate:.2f}")
    
    print("\n全部测试通过！市场环境模拟器工作正常。")
    
except ImportError as e:
    print(f"导入错误: {e}")
    import traceback
    traceback.print_exc()
    
except Exception as e:
    print(f"运行错误: {e}")
    import traceback
    traceback.print_exc()

print("\n=== 测试结束 ===")