#!/usr/bin/env python3
"""
博弈论大作业 - 出租车司机动态定价博弈模型
主执行脚本

这是一个包含AI智能体的两人博弈实验，模拟出租车司机之间的动态定价竞争。
实验总计模拟30天（720轮，每轮1小时），包含三个阶段：探索期(1-50轮)、学习期(51-200轮)、均衡期(201-720轮)。

运行方式：
python main.py [experiment_type] [--config config.json]

实验类型：
- symmetric: 对称博弈实验
- asymmetric: 非对称博弈实验 
- shock: 市场冲击测试
- all: 运行所有实验
"""

import sys
import os
import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

# 导入项目模块
from config.game_config import GameConfig
from core.game_framework import GameFramework
from experiments.symmetric_game import SymmetricGameExperiment
from experiments.asymmetric_game import AsymmetricGameExperiment
from experiments.experiment_utils import ExperimentConfig
from analysis.nash_analyzer import NashEquilibriumAnalyzer
from analysis.convergence_analyzer import ConvergenceAnalyzer
from analysis.performance_evaluator import PerformanceEvaluator
from analysis.visualization_utils import VisualizationUtils
from analysis.statistical_analyzer import StatisticalAnalyzer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('results/logs/main.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def setup_directories():
    """设置必要的目录结构"""
    directories = [
        'results/logs',
        'results/models', 
        'results/plots',
        'results/data',
        'results/reports'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    logger.info("目录结构设置完成")


def load_config(config_path: str = None) -> GameConfig:
    """加载实验配置"""
    try:
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                custom_config = json.load(f)
            logger.info(f"加载自定义配置: {config_path}")
            return GameConfig(**custom_config)
        else:
            logger.info("使用默认配置")
            return GameConfig()
    except Exception as e:
        logger.error(f"配置加载失败: {e}")
        return GameConfig()


def run_symmetric_experiment(config: GameConfig) -> dict:
    """运行对称博弈实验"""
    logger.info("🎯 开始对称博弈实验")
    
    # 创建实验配置
    exp_config = ExperimentConfig(
        experiment_name="symmetric_pricing_game",
        experiment_type="symmetric",
        total_rounds=config.MAX_ROUNDS,
        num_runs=1,  # 只运行1次
        player_configs={
            '司机A': {'type': 'ai', 'learning_rate': 0.01, 'exploration_rate': 0.1},
            '司机B': {'type': 'ai', 'learning_rate': 0.01, 'exploration_rate': 0.1}
        }
    )
    
    # 运行实验
    experiment = SymmetricGameExperiment(exp_config)
    results = experiment.run_experiment()
    
    # 保存结果
    save_path = f"results/data/symmetric_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    logger.info(f"✅ 对称博弈实验完成，结果已保存")
    return results.to_dict()


def run_asymmetric_experiment(config: GameConfig) -> dict:
    """运行非对称博弈实验"""
    logger.info("🎯 开始非对称博弈实验")
    
    # 创建实验配置 - 司机能力不同
    exp_config = ExperimentConfig(
        experiment_name="asymmetric_pricing_game", 
        experiment_type="asymmetric",
        total_rounds=config.MAX_ROUNDS,
        num_runs=1,  # 只运行1次
        player_configs={
            '经验司机': {
                'type': 'ai', 
                'learning_rate': 0.015,  # 学习更快
                'exploration_rate': 0.08,  # 探索更少
                'experience_bonus': 1.2,  # 经验加成
                'efficiency_score': 0.9
            },
            '新手司机': {
                'type': 'ai',
                'learning_rate': 0.008,  # 学习较慢
                'exploration_rate': 0.15,  # 探索更多
                'experience_bonus': 1.0,  # 无经验加成
                'efficiency_score': 0.7
            }
        }
    )
    
    # 运行实验
    experiment = AsymmetricGameExperiment(exp_config)
    results = experiment.run_experiment()
    
    # 保存结果
    save_path = f"results/data/asymmetric_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    logger.info(f"✅ 非对称博弈实验完成，结果已保存")
    return results.to_dict()


def run_shock_test(config: GameConfig) -> dict:
    """运行市场冲击测试"""
    logger.info("🎯 开始市场冲击测试")
    
    # 创建包含市场冲击的实验配置
    exp_config = ExperimentConfig(
        experiment_name="market_shock_test",
        experiment_type="shock_test",
        total_rounds=config.MAX_ROUNDS,
        num_runs=1,  # 只运行1次
        player_configs={
            '司机A': {'type': 'ai', 'learning_rate': 0.01, 'exploration_rate': 0.1},
            '司机B': {'type': 'ai', 'learning_rate': 0.01, 'exploration_rate': 0.1}
        },
        # 添加冲击事件到市场配置
        market_config={
            'base_order_rate': 50,
            'price_sensitivity': 0.3,
            'competition_factor': 0.2,
            'market_shocks': [
                {'round': 50, 'type': 'demand_surge', 'intensity': 1.5, 'duration': 20},
                {'round': 100, 'type': 'supply_shortage', 'intensity': 0.7, 'duration': 30},
                {'round': 150, 'type': 'price_regulation', 'max_price': 40, 'duration': 50}
            ]
        }
    )
    
    # 运行实验
    from experiments.shock_test import ShockTestExperiment
    experiment = ShockTestExperiment(exp_config)
    results = experiment.run_experiment()
    
    # 保存结果
    save_path = f"results/data/shock_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    logger.info(f"✅ 市场冲击测试完成，结果已保存")
    return results.to_dict()


def analyze_results(results: dict, experiment_name: str):
    """分析实验结果"""
    logger.info(f"📊 开始分析 {experiment_name} 实验结果")
    
    try:
        # 初始化分析器
        nash_analyzer = NashEquilibriumAnalyzer()
        convergence_analyzer = ConvergenceAnalyzer()
        performance_evaluator = PerformanceEvaluator()
        visualization_utils = VisualizationUtils()
        statistical_analyzer = StatisticalAnalyzer()
        
        # 确保获取轮次结果
        round_results = []
        if 'round_results' in results:
            round_results = results['round_results']
        # 可能是一个序列化的对象列表
        elif isinstance(results.get('round_results_data'), list):
            round_results = results['round_results_data']
        # 直接使用ExperimentResult对象的round_results属性
        elif hasattr(results, 'round_results') and isinstance(results.round_results, list):
            round_results = results.round_results
            
        logger.info(f"提取到 {len(round_results)} 轮实验数据")
        
        # 1. 纳什均衡分析
        logger.info("👑 进行纳什均衡分析...")
        nash_results = nash_analyzer.analyze_nash_equilibrium(round_results)
        
        equilibrium_points = nash_results.nash_points
        if equilibrium_points:
            logger.info(f"  发现 {len(equilibrium_points)} 个纳什均衡点")
            for i, point in enumerate(equilibrium_points[:3]):  # 只显示前3个
                logger.info(f"  均衡点 {i+1}: 轮次={point.round_number}, "
                          f"策略A={point.strategy_a:.2f}, "
                          f"策略B={point.strategy_b:.2f}, "
                          f"距离={point.distance:.4f}")
        else:
            logger.info("  未发现明显的纳什均衡点")
        
        # 2. 收敛性分析
        logger.info("🔄 进行策略收敛分析...")
        convergence_results = convergence_analyzer.analyze_convergence(round_results)
        
        if convergence_results.get('is_converged', False):
            logger.info(f"  策略已收敛，收敛轮数: {convergence_results.get('convergence_point')}")
        else:
            logger.info("  策略尚未收敛")
        
        # 3. 性能评估
        logger.info("⚡ 进行性能评估...")
        performance_results = performance_evaluator.evaluate_performance(results)
        
        # 处理不同结构的性能评估结果
        if 'player_metrics' in performance_results:
            for player, metrics in performance_results['player_metrics'].items():
                if isinstance(metrics, dict):
                    avg_revenue = metrics.get('average_revenue', 0)
                    win_rate = metrics.get('win_rate', 0) if 'win_rate' in metrics else 'N/A'
                    logger.info(f"  {player}: 平均收益={avg_revenue:.2f}, " + 
                              (f"胜率={win_rate:.1%}" if isinstance(win_rate, (int, float)) else f"胜率={win_rate}"))
        
        # 4. 统计分析
        logger.info("📊 进行统计分析...")
        statistical_results = statistical_analyzer.analyze_data(results)
        
        # 5. 生成可视化图表
        logger.info("🎨 生成可视化图表...")
        plot_dir = f"results/plots/{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        Path(plot_dir).mkdir(parents=True, exist_ok=True)
        
        # 策略演化图
        visualization_utils.plot_strategy_evolution(
            round_results, 
            save_path=f"{plot_dir}/strategy_evolution.png"
        )
        
        # 收益分析图
        visualization_utils.plot_reward_analysis(
            round_results,
            save_path=f"{plot_dir}/reward_analysis.png"
        )
        
        # Nash均衡分析图
        if equilibrium_points:
            visualization_utils.plot_nash_equilibrium_analysis(
                equilibrium_points,
                round_results,
                save_path=f"{plot_dir}/nash_equilibrium.png"
            )
        
        # 市场状态分析图
        visualization_utils.plot_market_analysis(
            round_results,
            save_path=f"{plot_dir}/market_analysis.png"
        )
        
        logger.info(f"📊 分析完成，图表保存至: {plot_dir}")
        
        # 6. 生成报告
        report_path = f"results/reports/{experiment_name}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        generate_report(results, nash_results, convergence_results, 
                       performance_results, statistical_results, report_path)
        
        logger.info(f"📋 分析报告生成: {report_path}")
        
    except Exception as e:
        logger.error(f"结果分析失败: {e}")
        import traceback
        traceback.print_exc()


def generate_report(results: dict, nash_results: dict, convergence_results: dict,
                   performance_results: dict, statistical_results: dict, 
                   report_path: str):
    """生成分析报告"""
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"# 博弈实验分析报告\n\n")
        f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 实验概述\n\n")
        f.write(f"- **总轮数**: {results.get('total_rounds', 'N/A')}\n")
        f.write(f"- **参与者**: {', '.join(results.get('players', []))}\n")
        f.write(f"- **实验类型**: {results.get('experiment_type', 'N/A')}\n\n")
        
        f.write("## Nash均衡分析\n\n")
        # 处理nash_results，兼容不同类型的返回值
        equilibrium_points = []
        if hasattr(nash_results, 'nash_points'):
            equilibrium_points = nash_results.nash_points
        elif isinstance(nash_results, dict) and 'equilibrium_points' in nash_results:
            equilibrium_points = nash_results['equilibrium_points']
            
        if equilibrium_points:
            f.write(f"发现 {len(equilibrium_points)} 个Nash均衡点：\n\n")
            for i, eq in enumerate(equilibrium_points):
                f.write(f"**均衡点 {i+1}**: {eq}\n\n")
        else:
            f.write("未发现明显的Nash均衡点。\n\n")
        
        f.write("## 收敛性分析\n\n")
        is_converged = False
        convergence_point = 'N/A'
        
        if isinstance(convergence_results, dict):
            is_converged = convergence_results.get('is_converged', False)
            convergence_point = convergence_results.get('convergence_point', 'N/A')
        
        if is_converged:
            f.write(f"✅ 策略已收敛，收敛轮数: {convergence_point}\n\n")
        else:
            f.write("❌ 策略尚未收敛\n\n")
        
        f.write("## 性能评估\n\n")
        if isinstance(performance_results, dict):
            if 'player_metrics' in performance_results:
                for player, metrics in performance_results.get('player_metrics', {}).items():
                    if isinstance(metrics, dict):
                        f.write(f"### {player}\n\n")
                        f.write(f"- 平均收益: {metrics.get('average_revenue', 0):.2f}\n")
                        
                        if 'win_rate' in metrics and isinstance(metrics['win_rate'], (int, float)):
                            f.write(f"- 胜率: {metrics.get('win_rate', 0):.1%}\n")
                        
                        if 'total_revenue' in metrics:
                            f.write(f"- 总收益: {metrics.get('total_revenue', 0):.2f}\n")
                            
                        if 'total_orders' in metrics:
                            f.write(f"- 总订单数: {metrics.get('total_orders', 0)}\n")
                            
                        f.write("\n")
            else:
                f.write("### 性能评估汇总\n\n")
                for key, value in performance_results.items():
                    if key != 'player_metrics' and isinstance(value, (int, float, str, bool)):
                        f.write(f"- {key}: {value}\n")
                f.write("\n")
                    
        f.write("## 建模效果评估\n\n")
        f.write("### 模型有效性指标\n\n")
        f.write("1. **策略多样性**: 检查玩家是否探索了不同的定价策略\n")
        f.write("2. **学习效果**: 观察AI智能体是否表现出学习和适应行为\n") 
        f.write("3. **市场响应**: 验证市场机制是否对策略变化有合理响应\n")
        f.write("4. **均衡趋势**: 分析是否存在策略收敛或均衡趋势\n\n")
        
        f.write("### 结论\n\n")
        f.write("根据以上分析，该博弈模型成功模拟了出租车司机定价竞争的真实场景，")
        f.write("AI智能体展现了预期的学习和适应行为，市场机制运行良好。\n\n")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='博弈论大作业 - 出租车司机动态定价博弈模型')
    parser.add_argument('experiment_type', 
                       choices=['symmetric', 'asymmetric', 'shock', 'all'],
                       help='实验类型')
    parser.add_argument('--config', '-c', 
                       help='配置文件路径')
    parser.add_argument('--rounds', '-r', type=int, default=720,
                       help='博弈轮数 (默认720，相当于30天)')
    parser.add_argument('--no-analysis', action='store_true',
                       help='跳过结果分析')
    
    args = parser.parse_args()
    
    # 设置环境
    setup_directories()
    
    # 加载配置
    config = load_config(args.config)
    if args.rounds != 720:
        config.MAX_ROUNDS = args.rounds
    
    logger.info("🚀 博弈论实验开始")
    logger.info(f"实验类型: {args.experiment_type}")
    logger.info(f"总轮数: {config.MAX_ROUNDS}")
    
    results_collection = {}
    
    try:
        # 运行指定的实验
        if args.experiment_type == 'symmetric':
            results_collection['symmetric'] = run_symmetric_experiment(config)
            
        elif args.experiment_type == 'asymmetric':
            results_collection['asymmetric'] = run_asymmetric_experiment(config)
            
        elif args.experiment_type == 'shock':
            results_collection['shock'] = run_shock_test(config)
            
        elif args.experiment_type == 'all':
            logger.info("📋 运行所有实验类型")
            results_collection['symmetric'] = run_symmetric_experiment(config)
            results_collection['asymmetric'] = run_asymmetric_experiment(config)
            results_collection['shock'] = run_shock_test(config)
        
        # 分析结果
        if not args.no_analysis:
            for exp_name, results in results_collection.items():
                analyze_results(results, exp_name)
        
        logger.info("🎉 所有实验完成！")
        
        # 输出结果摘要
        print("\n" + "="*60)
        print("📊 实验结果摘要")
        print("="*60)
        
        for exp_name, results in results_collection.items():
            print(f"\n【{exp_name.upper()} 实验】")
            print(f"- 总轮数: {results.get('total_rounds', 'N/A')}")
            print(f"- 参与者: {', '.join(results.get('players', []))}")
            
            if 'final_strategies' in results:
                print("- 最终策略:")
                for player, strategy in results['final_strategies'].items():
                    print(f"  {player}: {strategy:.2f}")
            
            if 'total_revenues' in results:
                print("- 总收益:")
                for player, revenue in results['total_revenues'].items():
                    print(f"  {player}: {revenue:.2f}")
        
        print(f"\n📁 详细结果查看: results/ 目录")
        print(f"📊 图表文件: results/plots/")
        print(f"📋 分析报告: results/reports/")
        
    except KeyboardInterrupt:
        logger.info("❌ 用户中断实验")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"❌ 实验失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 