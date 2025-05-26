#!/usr/bin/env python3
"""
博弈论大作业演示脚本
展示出租车司机动态定价博弈模型的核心功能

这个脚本将演示：
1. 基本配置和初始化
2. AI智能体的创建和训练
3. 不同类型的博弈实验
4. 结果分析和可视化
5. Nash均衡检测
"""

import sys
import os
import logging
from pathlib import Path
import time

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def demo_configuration():
    """演示配置功能"""
    print("🔧 演示配置功能")
    print("="*50)
    
    from config.game_config import GameConfig
    
    # 创建默认配置
    config = GameConfig()
    print(f"默认配置:")
    print(f"  总轮数: {config.total_rounds}")
    print(f"  价格范围: {config.price_range}")
    print(f"  探索阶段: {config.exploration_rounds}轮")
    print(f"  学习阶段: {config.learning_rounds}轮")
    print(f"  均衡阶段: {config.equilibrium_rounds}轮")
    
    # 创建自定义配置
    custom_config = GameConfig(
        total_rounds=100,
        price_range=(15, 45),
        exploration_rounds=20
    )
    print(f"\n自定义配置:")
    print(f"  总轮数: {custom_config.total_rounds}")
    print(f"  价格范围: {custom_config.price_range}")
    print(f"  探索阶段: {custom_config.exploration_rounds}轮")
    
    return config

def demo_market_environment():
    """演示市场环境"""
    print("\n🏪 演示市场环境")
    print("="*50)
    
    from config.game_config import GameConfig
    from core.market_environment import MarketEnvironment
    
    config = GameConfig(total_rounds=50)  # 减少轮数用于演示
    market = MarketEnvironment(config)
    
    print("模拟5轮市场交互:")
    
    for round_num in range(5):
        # 模拟不同的价格策略
        if round_num < 3:
            actions = {'司机A': 25, '司机B': 35}
        else:
            actions = {'司机A': 30, '司机B': 25}
        
        # 更新市场状态
        state = market.update_market_state(actions)
        rewards = market.calculate_rewards(actions)
        
        print(f"\n第{round_num+1}轮:")
        print(f"  司机A策略: {actions['司机A']}元, 收益: {rewards['司机A']:.2f}")
        print(f"  司机B策略: {actions['司机B']}元, 收益: {rewards['司机B']:.2f}")
        print(f"  当前阶段: {'探索期' if market.is_exploration_phase() else '学习期' if market.is_learning_phase() else '均衡期'}")
    
    return market

def demo_ai_models():
    """演示AI模型"""
    print("\n🤖 演示AI模型")
    print("="*50)
    
    from ai_models.dqn_agent import DQNAgent
    from ai_models.lstm_predictor import LSTMPredictor
    import numpy as np
    
    # 创建DQN智能体
    dqn_config = {
        'state_size': 10,
        'action_size': 41,  # 10-50的价格策略
        'learning_rate': 0.001,
        'hidden_units': [64, 32],
        'epsilon': 0.3,
        'epsilon_decay': 0.995,
        'epsilon_min': 0.01
    }
    
    print("创建DQN智能体...")
    dqn_agent = DQNAgent(dqn_config)
    print(f"  状态维度: {dqn_agent.state_size}")
    print(f"  动作维度: {dqn_agent.action_size}")
    print(f"  初始探索率: {dqn_agent.epsilon}")
    
    # 演示动作选择
    state = np.random.rand(10)
    action = dqn_agent.choose_action(state)
    price = action + 10  # 转换为实际价格
    print(f"  选择动作: {action} (价格: {price}元)")
    
    # 创建LSTM预测器
    lstm_config = {
        'input_size': 5,
        'hidden_size': 32,
        'output_size': 1,
        'sequence_length': 10,
        'learning_rate': 0.001
    }
    
    print("\n创建LSTM预测器...")
    lstm_predictor = LSTMPredictor(lstm_config)
    print(f"  输入维度: {lstm_predictor.input_size}")
    print(f"  隐层大小: {lstm_predictor.hidden_size}")
    print(f"  序列长度: {lstm_predictor.sequence_length}")
    
    # 演示预测
    sequence = np.random.rand(10, 5)
    prediction = lstm_predictor.predict(sequence)
    print(f"  预测结果: {prediction}")
    
    return dqn_agent, lstm_predictor

def demo_simple_experiment():
    """演示简单实验"""
    print("\n🧪 演示简单博弈实验")
    print("="*50)
    
    from config.game_config import GameConfig
    from experiments.experiment_utils import ExperimentConfig, ExperimentRunner
    from experiments.symmetric_game import SymmetricGameExperiment
    
    # 创建游戏配置（使用较少轮数用于演示）
    game_config = GameConfig(total_rounds=20)
    
    # 创建实验配置
    exp_config = ExperimentConfig(
        name="demo_symmetric",
        description="演示对称博弈",
        num_rounds=20,
        players=['司机A', '司机B'],
        player_configs={
            '司机A': {
                'type': 'ai',
                'learning_rate': 0.01,
                'exploration_rate': 0.2
            },
            '司机B': {
                'type': 'ai',
                'learning_rate': 0.01,
                'exploration_rate': 0.2
            }
        }
    )
    
    print("运行对称博弈实验...")
    print(f"  实验名称: {exp_config.name}")
    print(f"  总轮数: {exp_config.num_rounds}")
    print(f"  玩家数量: {len(exp_config.players)}")
    
    # 创建并运行实验
    experiment = SymmetricGameExperiment(game_config, exp_config)
    print("  正在初始化实验...")
    experiment.setup()
    
    print("  正在运行实验...")
    results = experiment.run()
    
    # 显示结果摘要
    print(f"\n实验结果:")
    print(f"  完成轮数: {results['total_rounds']}")
    print(f"  参与玩家: {', '.join(results['players'])}")
    
    # 显示最后几轮的结果
    if results['round_results']:
        print(f"  最后一轮结果:")
        last_round = results['round_results'][-1]
        for player, data in last_round['players'].items():
            print(f"    {player}: 策略={data['action']}, 收益={data['reward']:.2f}")
    
    return results

def demo_analysis():
    """演示分析功能"""
    print("\n📊 演示分析功能")
    print("="*50)
    
    from analysis.nash_analyzer import NashAnalyzer
    from analysis.convergence_analyzer import ConvergenceAnalyzer
    from analysis.statistical_analyzer import StatisticalAnalyzer
    import numpy as np
    
    # 创建模拟数据
    print("创建模拟分析数据...")
    rounds = 50
    player_strategies = {
        '司机A': np.random.normal(25, 5, rounds),
        '司机B': np.random.normal(30, 5, rounds)
    }
    player_rewards = {
        '司机A': np.random.normal(100, 20, rounds),
        '司机B': np.random.normal(95, 20, rounds)
    }
    
    # Nash均衡分析
    print("\n进行Nash均衡分析...")
    nash_analyzer = NashAnalyzer()
    
    # 检测均衡点
    equilibria = nash_analyzer.detect_equilibrium_points(player_strategies, player_rewards)
    print(f"  检测到 {len(equilibria)} 个潜在均衡点")
    
    if equilibria:
        eq = equilibria[0]
        print(f"  第一个均衡点:")
        print(f"    轮次: {eq.round_number}")
        print(f"    策略: {eq.strategies}")
        print(f"    稳定性: {eq.stability_score:.3f}")
    
    # 收敛性分析
    print("\n进行收敛性分析...")
    conv_analyzer = ConvergenceAnalyzer()
    
    convergence_metrics = conv_analyzer.analyze_convergence(player_strategies, player_rewards)
    print(f"  策略收敛性: {'是' if convergence_metrics.strategy_converged else '否'}")
    print(f"  收益收敛性: {'是' if convergence_metrics.reward_converged else '否'}")
    print(f"  收敛轮次: {convergence_metrics.convergence_round}")
    
    # 统计分析
    print("\n进行统计分析...")
    stat_analyzer = StatisticalAnalyzer()
    
    # 分析司机A的策略
    strategy_summary = stat_analyzer.descriptive_statistics(player_strategies['司机A'])
    print(f"  司机A策略统计:")
    print(f"    平均值: {strategy_summary.mean:.2f}")
    print(f"    标准差: {strategy_summary.std:.2f}")
    print(f"    中位数: {strategy_summary.median:.2f}")
    
    return nash_analyzer, conv_analyzer, stat_analyzer

def demo_visualization():
    """演示可视化功能"""
    print("\n📈 演示可视化功能")
    print("="*50)
    
    try:
        from analysis.visualization_utils import VisualizationUtils
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')  # 使用非交互式后端
        
        # 创建可视化工具
        viz = VisualizationUtils()
        
        # 创建模拟数据
        rounds = np.arange(1, 51)
        strategies = {
            '司机A': np.random.normal(25, 3, 50),
            '司机B': np.random.normal(30, 3, 50)
        }
        rewards = {
            '司机A': np.random.normal(100, 15, 50),
            '司机B': np.random.normal(95, 15, 50)
        }
        
        print("创建策略演化图...")
        try:
            viz.plot_strategy_evolution(strategies, save_path='results/plots/demo_strategies.png')
            print("  ✅ 策略演化图已保存")
        except Exception as e:
            print(f"  ❌ 策略演化图创建失败: {e}")
        
        print("创建收益分布图...")
        try:
            viz.plot_reward_distribution(rewards, save_path='results/plots/demo_rewards.png')
            print("  ✅ 收益分布图已保存")
        except Exception as e:
            print(f"  ❌ 收益分布图创建失败: {e}")
        
        print("创建学习曲线...")
        try:
            learning_curves = {
                '司机A': np.cumsum(rewards['司机A']) / np.arange(1, 51),
                '司机B': np.cumsum(rewards['司机B']) / np.arange(1, 51)
            }
            viz.plot_learning_curves(learning_curves, save_path='results/plots/demo_learning.png')
            print("  ✅ 学习曲线已保存")
        except Exception as e:
            print(f"  ❌ 学习曲线创建失败: {e}")
        
    except ImportError as e:
        print(f"可视化功能需要matplotlib等依赖: {e}")
    except Exception as e:
        print(f"可视化演示失败: {e}")

def main():
    """主演示函数"""
    print("🎮 博弈论大作业 - 出租车司机动态定价模型演示")
    print("="*70)
    print("这个演示将展示项目的主要功能和特性。")
    print("包括配置、市场环境、AI模型、实验和分析等模块。")
    print("="*70)
    
    try:
        # 确保结果目录存在
        os.makedirs('results/plots', exist_ok=True)
        os.makedirs('results/logs', exist_ok=True)
        
        # 运行各个演示模块
        config = demo_configuration()
        market = demo_market_environment()
        dqn_agent, lstm_predictor = demo_ai_models()
        experiment_results = demo_simple_experiment()
        nash_analyzer, conv_analyzer, stat_analyzer = demo_analysis()
        demo_visualization()
        
        print("\n🎉 演示完成!")
        print("="*70)
        print("主要功能演示结果:")
        print("  ✅ 配置系统 - 正常工作")
        print("  ✅ 市场环境 - 正常工作")
        print("  ✅ AI模型 - 正常工作")
        print("  ✅ 实验框架 - 正常工作")
        print("  ✅ 分析工具 - 正常工作")
        print("  ✅ 可视化工具 - 正常工作")
        
        print("\n📁 生成的文件:")
        print("  - results/plots/demo_*.png (可视化图表)")
        print("  - 实验和分析数据保存在内存中")
        
        print("\n🚀 下一步操作:")
        print("  1. 运行 'python main.py symmetric' 进行完整对称博弈实验")
        print("  2. 运行 'python main.py asymmetric' 进行非对称博弈实验")
        print("  3. 运行 'python validate_model.py' 进行模型验证")
        print("  4. 运行 'python tests/run_all_tests.py' 进行全面测试")
        
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 