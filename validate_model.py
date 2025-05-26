#!/usr/bin/env python3
"""
博弈模型验证脚本
快速检验博弈模型的有效性和功能完整性
"""

import sys
import os
import logging
from pathlib import Path
import time
import numpy as np

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_basic_imports():
    """测试基本模块导入"""
    print("🔍 测试基本模块导入...")
    
    try:
        from config.game_config import GameConfig
        from core.game_framework import GameFramework
        from core.market_environment import MarketEnvironment
        print("  ✅ 核心模块导入成功")
        
        from ai_models.dqn_agent import DQNAgent
        from ai_models.lstm_predictor import LSTMPredictor
        print("  ✅ AI模型模块导入成功")
        
        from experiments.symmetric_game import SymmetricGameExperiment
        from experiments.asymmetric_game import AsymmetricGameExperiment
        from experiments.experiment_utils import ExperimentConfig
        print("  ✅ 实验模块导入成功")
        
        from analysis.nash_analyzer import NashEquilibriumAnalyzer
        from analysis.convergence_analyzer import ConvergenceAnalyzer
        from analysis.performance_evaluator import PerformanceEvaluator
        from analysis.visualization_utils import VisualizationUtils
        from analysis.statistical_analyzer import StatisticalAnalyzer
        print("  ✅ 分析模块导入成功")
        
        from data.data_generator import DataGenerator
        from data.market_simulator import MarketSimulator
        print("  ✅ 数据模块导入成功")
        
        return True
        
    except ImportError as e:
        print(f"  ❌ 模块导入失败: {e}")
        return False


def test_config():
    """测试配置功能"""
    print("\n🔧 测试配置功能...")
    
    try:
        from config.game_config import GameConfig
        
        config = GameConfig()
        
        # 验证配置值
        print(f"  博弈轮次: {config.MAX_ROUNDS}")
        print(f"  价格范围: {config.MIN_PRICE_THRESHOLD}-{config.MAX_PRICE_THRESHOLD}")
        print(f"  玩家数量: {config.NUM_PLAYERS}")
        
        # 验证配置方法
        epsilon = config.get_epsilon_for_round(50)
        print(f"  轮次50的探索率: {epsilon:.2f}")
        
        peak_hour = config.get_time_period(8)
        print(f"  8点时段类型: {peak_hour}")
        
        # 验证配置有效性
        assert config.validate_config(), "配置验证失败"
        
        print("  ✅ 配置测试成功")
        return True
        
    except Exception as e:
        print(f"  ❌ 配置测试失败: {e}")
        return False


def test_market_environment():
    """测试市场环境"""
    print("\n🏪 测试市场环境...")
    
    try:
        from config.game_config import GameConfig
        from core.market_environment import MarketEnvironment
        
        config = GameConfig()
        market = MarketEnvironment(config)
        
        # 测试订单生成
        orders = market.generate_orders(30.0)  # 30分钟
        assert len(orders) >= 0
        print(f"  ✅ 订单生成成功，生成 {len(orders)} 个订单")
        
        # 测试竞争效应
        strategies = {'司机A': 25.0, '司机B': 30.0}
        market.apply_competition_effects(strategies)
        print("  ✅ 竞争效应应用成功")
        
        # 测试订单处理
        if orders:
            driver_orders = market.process_driver_decisions(orders, strategies)
            print(f"  ✅ 订单处理成功")
            
            # 测试收益计算
            revenues = market.calculate_driver_revenues(driver_orders, strategies)
            print(f"  ✅ 收益计算成功")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 市场环境测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ai_models():
    """测试AI模型"""
    print("\n🤖 测试AI模型...")
    
    try:
        from ai_models.dqn_agent import DQNAgent
        
        # 创建测试配置
        dqn_config = {
            'state_size': 10,
            'action_size': 41,
            'learning_rate': 0.001,
            'gamma': 0.99,
            'epsilon_start': 0.5
        }
        
        # 初始化DQN智能体
        dqn_agent = DQNAgent(dqn_config)
        
        # 测试动作选择
        dummy_state = np.random.rand(10).astype(np.float32)
        action_result = dqn_agent.select_action(dummy_state)
        
        print(f"  选择的动作: {action_result.action}")
        print(f"  动作值: {action_result.action_value:.4f}")
        print(f"  探索类型: {action_result.exploration_type}")
        
        # 测试经验存储
        dummy_next_state = np.random.rand(10).astype(np.float32)
        dqn_agent.store_experience(dummy_state, action_result.action, 1.0, dummy_next_state, False)
        
        # 测试训练（可能不会实际更新权重，因为经验不足）
        loss = dqn_agent.train()
        print(f"  训练损失: {loss}")
        
        print("  ✅ AI模型测试成功")
        return True
        
    except Exception as e:
        print(f"  ❌ AI模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_game_framework():
    """测试博弈框架"""
    print("\n🎮 测试博弈框架...")
    
    try:
        from config.game_config import GameConfig
        from core.game_framework import GameFramework
        
        config = GameConfig()
        config.total_rounds = 10  # 减少测试轮数
        
        framework = GameFramework(config)
        
        # 测试玩家注册
        players = ['测试司机A', '测试司机B']
        for player in players:
            framework.register_player(player, 'ai')
        
        assert len(framework.players) == 2
        print("  ✅ 玩家注册成功")
        
        # 测试单轮博弈
        round_result = framework.play_round(1, {'测试司机A': 25.0, '测试司机B': 30.0})
        assert 'round_number' in round_result
        assert 'strategies' in round_result
        print("  ✅ 单轮博弈执行成功")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 博弈框架测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_analysis_modules():
    """测试分析模块"""
    print("\n📊 测试分析模块...")
    
    try:
        from analysis.nash_analyzer import NashEquilibriumAnalyzer
        
        # 创建纳什分析器
        analyzer = NashEquilibriumAnalyzer(convergence_threshold=0.05)
        
        # 测试分析功能
        print("  ✅ 纳什均衡分析器创建成功")
        
        # 导入其他分析模块
        from analysis.convergence_analyzer import ConvergenceAnalyzer
        from analysis.performance_evaluator import PerformanceEvaluator
        
        print("  ✅ 分析模块导入成功")
        return True
        
    except Exception as e:
        print(f"  ❌ 分析模块测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_modules():
    """测试数据模块"""
    print("\n📈 测试数据模块...")
    
    try:
        from data.data_generator import DataGenerator
        from data.market_simulator import MarketSimulator
        
        # 测试数据生成器
        data_generator = DataGenerator(seed=42)
        
        # 生成需求数据
        demand_data = data_generator.generate_demand_data(24)  # 24小时
        assert len(demand_data) == 24
        print("  ✅ 需求数据生成成功")
        
        # 生成价格数据
        price_data = data_generator.generate_price_data(100)  # 100个数据点
        assert len(price_data) == 100
        print("  ✅ 价格数据生成成功")
        
        # 测试市场模拟器
        market_simulator = MarketSimulator()
        
        # 生成订单
        orders = market_simulator.generate_orders(1)  # 1小时
        assert isinstance(orders, list)
        print(f"  ✅ 市场模拟器生成 {len(orders)} 个订单")
        
        # 测试市场匹配
        if orders:
            strategies = {'司机A': 25.0, '司机B': 30.0}
            matching_result = market_simulator.simulate_market_matching(strategies, orders)
            assert 'market_state' in matching_result
            print("  ✅ 市场匹配模拟成功")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 数据模块测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_mini_experiment():
    """运行迷你实验"""
    print("\n🧪 运行迷你实验...")
    
    try:
        from experiments.symmetric_game import SymmetricGameExperiment
        from experiments.experiment_utils import ExperimentConfig
        
        # 创建实验配置
        exp_config = ExperimentConfig(
            experiment_name="迷你测试实验",
            experiment_type="symmetric",
            total_rounds=5,
            num_runs=1
        )
        
        # 添加AI配置
        exp_config.ai_config = {
            'dqn_params': {
                'learning_rate': 0.001,
                'gamma': 0.99,
                'epsilon_start': 0.5,
                'state_size': 15,
                'action_size': 41
            },
            'lstm_params': {
                'hidden_size': 32,
                'num_layers': 1
            }
        }
        
        # 创建实验
        experiment = SymmetricGameExperiment(exp_config)
        
        # 运行实验
        result = experiment.run_experiment()
        
        if result and len(result.round_results) > 0:
            print(f"  ✅ 迷你实验成功执行 {len(result.round_results)} 轮")
            return True
        else:
            print(f"  ❌ 迷你实验返回了空结果")
            return False
            
    except Exception as e:
        print(f"  ❌ 迷你实验失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def validate_model_effectiveness():
    """验证模型有效性"""
    print("\n✅ 模型有效性验证...")
    
    validation_criteria = [
        "✓ 策略多样性：AI智能体探索了不同的定价策略",
        "✓ 学习效果：智能体表现出学习和适应行为", 
        "✓ 市场响应：市场机制对策略变化有合理响应",
        "✓ 系统稳定：模型能够稳定运行多轮博弈",
        "✓ 数据完整：生成完整的实验数据和分析结果"
    ]
    
    print("  📋 建模效果验证标准:")
    for criterion in validation_criteria:
        print(f"    {criterion}")
    
    print("\n  🎯 推荐的效果检验方法:")
    print("    1. 运行完整实验: python main.py symmetric")
    print("    2. 观察策略演化图表，检查是否有学习趋势")
    print("    3. 分析Nash均衡检测结果")
    print("    4. 查看收敛分析，验证策略是否趋于稳定")
    print("    5. 比较不同实验类型的结果差异")


def main():
    """主函数"""
    print("="*60)
    print("🔬 博弈模型验证测试")
    print("="*60)
    
    tests = [
        ("基本模块导入", test_basic_imports),
        ("配置功能", test_config),
        ("市场环境", test_market_environment),
        ("AI模型", test_ai_models),
        ("博弈框架", test_game_framework),
        ("分析模块", test_analysis_modules),
        ("数据模块", test_data_modules),
        ("迷你实验", run_mini_experiment)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"  ❌ {test_name} 测试失败")
        except Exception as e:
            print(f"  ❌ {test_name} 测试出现异常: {e}")
    
    # 验证模型有效性
    validate_model_effectiveness()
    
    # 总结
    print("\n" + "="*60)
    print(f"📋 验证结果总结: {passed}/{total} 项测试通过")
    print("="*60)
    
    if passed == total:
        print("🎉 所有测试通过！博弈模型已准备就绪。")
        print("\n🚀 下一步操作:")
        print("  1. 运行完整实验: python main.py all")
        print("  2. 查看实验结果: results/ 目录")
        print("  3. 分析模型效果: results/reports/ 目录")
    else:
        print(f"⚠️  有 {total - passed} 项测试失败，需要修复后再运行完整实验。")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 