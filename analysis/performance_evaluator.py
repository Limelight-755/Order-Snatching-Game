"""
性能评估器
全面评估模型和策略的性能表现
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class PlayerPerformance:
    """玩家性能指标"""
    total_reward: float
    average_reward: float
    reward_variance: float
    strategy_stability: float
    learning_rate: float
    adaptation_speed: float
    consistency_score: float


@dataclass
class ComparisonMetrics:
    """对比性能指标"""
    reward_gap: float  # 奖励差距
    strategy_correlation: float  # 策略相关性
    mutual_adaptation: float  # 相互适应程度
    competitive_balance: float  # 竞争平衡性
    efficiency_ratio: float  # 效率比率


@dataclass
class PerformanceReport:
    """性能评估报告"""
    player_a_performance: PlayerPerformance
    player_b_performance: PlayerPerformance
    comparison_metrics: ComparisonMetrics
    overall_system_performance: Dict[str, float]
    performance_trends: Dict[str, List[float]]
    recommendations: List[str]


class PerformanceEvaluator:
    """
    性能评估器
    提供全面的性能评估和对比分析功能
    """
    
    def __init__(self, evaluation_window: int = 100):
        """
        初始化性能评估器
        
        Args:
            evaluation_window: 评估窗口大小
        """
        self.evaluation_window = evaluation_window
        
        logger.info(f"性能评估器初始化: 评估窗口={evaluation_window}")
    
    def evaluate_performance(self, results) -> Dict:
        """
        全面评估性能
        
        Args:
            results: 轮次结果(可以是对象列表或字典)
            
        Returns:
            性能评估报告
        """
        logger.info("开始性能评估...")
        
        # 检查输入类型，处理不同格式的结果数据
        round_results = []
        if isinstance(results, dict):
            # 如果是字典格式，尝试从中提取round_results
            if 'round_results' in results:
                round_results_data = results.get('round_results', [])
                
                # 检查轮次结果是否为对象或字典列表
                if round_results_data and isinstance(round_results_data[0], dict):
                    # 是字典列表，需要手动提取数据
                    strategies_a = [r.get('player_a_strategy', 0) for r in round_results_data]
                    strategies_b = [r.get('player_b_strategy', 0) for r in round_results_data]
                    rewards_a = [r.get('player_a_revenue', 0) for r in round_results_data]
                    rewards_b = [r.get('player_b_revenue', 0) for r in round_results_data]
                    
                    logger.info(f"从字典数据提取了 {len(strategies_a)} 轮结果数据")
                    
                    # 创建结果字典
                    return self._create_performance_results_dict(
                        strategies_a, strategies_b, rewards_a, rewards_b, results
                    )
                elif isinstance(round_results_data, list):
                    try:
                        # 尝试直接提取数据
                        if len(round_results_data) > 0:
                            # 初始化空列表
                            strategies_a = []
                            strategies_b = []
                            rewards_a = []
                            rewards_b = []
                            
                            # 根据不同数据类型提取属性
                            for r in round_results_data:
                                if hasattr(r, 'player_a_strategy'):
                                    # 是对象格式
                                    strategies_a.append(r.player_a_strategy)
                                    strategies_b.append(r.player_b_strategy)
                                    rewards_a.append(r.player_a_revenue)
                                    rewards_b.append(r.player_b_revenue)
                                elif isinstance(r, dict):
                                    # 是字典格式
                                    strategies_a.append(r.get('player_a_strategy', 0))
                                    strategies_b.append(r.get('player_b_strategy', 0))
                                    rewards_a.append(r.get('player_a_revenue', 0))
                                    rewards_b.append(r.get('player_b_revenue', 0))
                                elif isinstance(r, str):
                                    # 如果是字符串，可能是序列化后的数据，无法直接使用
                                    logger.warning("轮次结果为字符串格式，无法解析详细数据")
                                    break
                            
                            if strategies_a:
                                logger.info(f"从对象/字典列表提取了 {len(strategies_a)} 轮结果数据")
                                return self._create_performance_results_dict(
                                    strategies_a, strategies_b, rewards_a, rewards_b, results
                                )
                    except Exception as e:
                        logger.warning(f"处理轮次结果时出错: {e}")
            
            # 如果没有提取成功，尝试从汇总数据创建简化结果
            return self._create_simplified_performance_results(results)
        else:
            # 原始实现，处理对象列表
            try:
                strategies_a = [r.player_a_strategy for r in results]
                strategies_b = [r.player_b_strategy for r in results]
                rewards_a = [r.player_a_reward for r in results]
                rewards_b = [r.player_b_reward for r in results]
                
                logger.info(f"从对象列表提取了 {len(strategies_a)} 轮结果数据")
                
                return self._create_performance_results_dict(
                    strategies_a, strategies_b, rewards_a, rewards_b, {'round_results': results}
                )
            except (AttributeError, TypeError) as e:
                logger.warning(f"处理对象列表时出错: {e}")
                return self._create_simplified_performance_results({'round_results': results})
    
    def _create_performance_results_dict(self, strategies_a, strategies_b, 
                                        rewards_a, rewards_b, full_results):
        """创建性能结果字典"""
        # 计算玩家指标
        player_metrics = {
            'player_a': {
                'total_revenue': sum(rewards_a),
                'average_revenue': np.mean(rewards_a) if len(rewards_a) > 0 else 0,
                'revenue_variance': np.var(rewards_a) if len(rewards_a) > 1 else 0,
                'strategy_stability': self._calculate_strategy_stability(strategies_a),
                'win_rate': self._calculate_win_rate(rewards_a, rewards_b)
            },
            'player_b': {
                'total_revenue': sum(rewards_b),
                'average_revenue': np.mean(rewards_b) if len(rewards_b) > 0 else 0,
                'revenue_variance': np.var(rewards_b) if len(rewards_b) > 1 else 0,
                'strategy_stability': self._calculate_strategy_stability(strategies_b),
                'win_rate': self._calculate_win_rate(rewards_b, rewards_a)
            }
        }
        
        # 对比指标
        comparison_metrics = {
            'strategy_correlation': self._calculate_correlation(strategies_a, strategies_b),
            'revenue_correlation': self._calculate_correlation(rewards_a, rewards_b),
            'competitive_balance': self._calculate_competitive_balance(rewards_a, rewards_b)
        }
        
        # 系统整体表现
        system_performance = self._extract_system_performance(full_results)
        
        return {
            'player_metrics': player_metrics,
            'comparison_metrics': comparison_metrics,
            'system_performance': system_performance
        }
    
    def _create_simplified_performance_results(self, results):
        """从汇总数据创建简化的性能结果"""
        logger.info("创建简化的性能结果...")
        
        # 尝试从汇总数据中提取信息
        total_revenues = results.get('total_revenues', {})
        final_strategies = results.get('final_strategies', {})
        total_rounds = results.get('total_rounds', 0)
        
        player_a_id = next((key for key in total_revenues.keys() if 'a' in key.lower()), 'player_a')
        player_b_id = next((key for key in total_revenues.keys() if 'b' in key.lower()), 'player_b')
        
        player_metrics = {
            player_a_id: {
                'total_revenue': total_revenues.get(player_a_id, 0),
                'average_revenue': total_revenues.get(player_a_id, 0) / total_rounds if total_rounds > 0 else 0,
                'final_strategy': final_strategies.get(player_a_id, 0)
            },
            player_b_id: {
                'total_revenue': total_revenues.get(player_b_id, 0),
                'average_revenue': total_revenues.get(player_b_id, 0) / total_rounds if total_rounds > 0 else 0,
                'final_strategy': final_strategies.get(player_b_id, 0)
            }
        }
        
        # 简化的对比数据
        comparison_metrics = {
            'revenue_gap': abs(player_metrics[player_a_id]['total_revenue'] - 
                             player_metrics[player_b_id]['total_revenue']),
            'strategy_gap': abs(player_metrics[player_a_id].get('final_strategy', 0) - 
                              player_metrics[player_b_id].get('final_strategy', 0))
        }
        
        return {
            'player_metrics': player_metrics,
            'comparison_metrics': comparison_metrics,
            'data_quality': 'simplified'  # 标记数据质量
        }
    
    def _calculate_strategy_stability(self, strategies):
        """计算策略稳定性"""
        if len(strategies) < 2:
            return 1.0
        
        changes = [abs(strategies[i] - strategies[i-1]) for i in range(1, len(strategies))]
        avg_change = np.mean(changes)
        return 1.0 / (1.0 + avg_change)
    
    def _calculate_win_rate(self, rewards_a, rewards_b):
        """计算胜率"""
        if not rewards_a or not rewards_b:
            return 0.0
        
        wins = sum(1 for a, b in zip(rewards_a, rewards_b) if a > b)
        ties = sum(1 for a, b in zip(rewards_a, rewards_b) if a == b)
        total = len(rewards_a)
        
        # 胜利计1分，平局计0.5分
        return (wins + 0.5 * ties) / total if total > 0 else 0
    
    def _calculate_correlation(self, series_a, series_b):
        """计算两个序列的相关性"""
        if len(series_a) < 2 or len(series_b) < 2:
            return 0.0
        
        try:
            corr = np.corrcoef(series_a, series_b)[0, 1]
            return 0.0 if np.isnan(corr) else corr
        except Exception:
            return 0.0
    
    def _calculate_competitive_balance(self, rewards_a, rewards_b):
        """计算竞争平衡度"""
        if not rewards_a or not rewards_b:
            return 1.0
            
        total_a = sum(rewards_a)
        total_b = sum(rewards_b)
        
        if total_a == 0 and total_b == 0:
            return 1.0  # 完全平衡
            
        max_rev = max(total_a, total_b)
        min_rev = min(total_a, total_b)
        
        return min_rev / max_rev if max_rev > 0 else 1.0
    
    def _extract_system_performance(self, results):
        """提取系统整体表现数据"""
        system_performance = {}
        
        # 从结果中提取可能存在的系统指标
        if 'convergence_round' in results:
            system_performance['convergence_round'] = results['convergence_round']
            
        if 'nash_equilibrium_found' in results:
            system_performance['nash_equilibrium_found'] = results['nash_equilibrium_found']
        
        return system_performance
    
    def _evaluate_player_performance(self, strategies: List[float], 
                                   rewards: List[float], 
                                   player_name: str) -> PlayerPerformance:
        """评估单个玩家的性能"""
        # 基本指标
        total_reward = sum(rewards)
        average_reward = np.mean(rewards)
        reward_variance = np.var(rewards)
        
        # 策略稳定性（基于策略变化幅度）
        strategy_changes = [abs(strategies[i] - strategies[i-1]) 
                          for i in range(1, len(strategies))]
        strategy_stability = 1.0 / (1.0 + np.mean(strategy_changes))
        
        # 学习率（奖励改进速度）
        learning_rate = self._calculate_learning_rate(rewards)
        
        # 适应速度（策略调整响应速度）
        adaptation_speed = self._calculate_adaptation_speed(strategies, rewards)
        
        # 一致性评分（性能的稳定性）
        consistency_score = self._calculate_consistency_score(rewards)
        
        logger.debug(f"{player_name} 性能评估完成")
        
        return PlayerPerformance(
            total_reward=total_reward,
            average_reward=average_reward,
            reward_variance=reward_variance,
            strategy_stability=strategy_stability,
            learning_rate=learning_rate,
            adaptation_speed=adaptation_speed,
            consistency_score=consistency_score
        )
    
    def _calculate_learning_rate(self, rewards: List[float]) -> float:
        """计算学习率"""
        if len(rewards) < 4:
            return 0.0
        
        # 比较前后两半的平均奖励
        mid_point = len(rewards) // 2
        first_half_avg = np.mean(rewards[:mid_point])
        second_half_avg = np.mean(rewards[mid_point:])
        
        # 标准化学习率
        total_range = max(rewards) - min(rewards)
        if total_range == 0:
            return 0.0
        
        improvement = (second_half_avg - first_half_avg) / total_range
        return max(0.0, min(1.0, improvement + 0.5))  # 归一化到0-1
    
    def _calculate_adaptation_speed(self, strategies: List[float], 
                                  rewards: List[float]) -> float:
        """计算适应速度"""
        if len(strategies) < self.evaluation_window:
            return 0.0
        
        # 计算策略调整与奖励变化的相关性
        strategy_changes = np.diff(strategies)
        reward_changes = np.diff(rewards)
        
        if len(strategy_changes) < 2:
            return 0.0
        
        # 计算响应速度（策略对奖励变化的响应程度）
        correlation = np.corrcoef(strategy_changes[:-1], reward_changes[1:])[0, 1]
        correlation = 0.0 if np.isnan(correlation) else abs(correlation)
        
        return correlation
    
    def _calculate_consistency_score(self, rewards: List[float]) -> float:
        """计算一致性评分"""
        if len(rewards) < self.evaluation_window:
            window_size = len(rewards) // 4 if len(rewards) >= 4 else len(rewards)
        else:
            window_size = self.evaluation_window // 4
        
        if window_size < 2:
            return 0.0
        
        # 计算滑动窗口内的方差
        variances = []
        for i in range(window_size, len(rewards)):
            window_variance = np.var(rewards[i-window_size:i])
            variances.append(window_variance)
        
        if not variances:
            return 0.0
        
        # 一致性为方差稳定性的度量
        variance_stability = 1.0 / (1.0 + np.var(variances))
        return min(1.0, variance_stability)
    
    def _calculate_comparison_metrics(self, strategies_a: List[float], 
                                    strategies_b: List[float],
                                    rewards_a: List[float], 
                                    rewards_b: List[float]) -> ComparisonMetrics:
        """计算对比指标"""
        # 奖励差距
        avg_reward_a = np.mean(rewards_a)
        avg_reward_b = np.mean(rewards_b)
        reward_gap = abs(avg_reward_a - avg_reward_b) / max(avg_reward_a, avg_reward_b, 1e-8)
        
        # 策略相关性
        strategy_correlation = np.corrcoef(strategies_a, strategies_b)[0, 1]
        strategy_correlation = 0.0 if np.isnan(strategy_correlation) else abs(strategy_correlation)
        
        # 相互适应程度
        mutual_adaptation = self._calculate_mutual_adaptation(strategies_a, strategies_b)
        
        # 竞争平衡性
        competitive_balance = self._calculate_competitive_balance(rewards_a, rewards_b)
        
        # 效率比率
        efficiency_ratio = min(avg_reward_a, avg_reward_b) / max(avg_reward_a, avg_reward_b, 1e-8)
        
        return ComparisonMetrics(
            reward_gap=reward_gap,
            strategy_correlation=strategy_correlation,
            mutual_adaptation=mutual_adaptation,
            competitive_balance=competitive_balance,
            efficiency_ratio=efficiency_ratio
        )
    
    def _calculate_mutual_adaptation(self, strategies_a: List[float], 
                                   strategies_b: List[float]) -> float:
        """计算相互适应程度"""
        if len(strategies_a) < 4:
            return 0.0
        
        # 计算策略变化的相关性
        changes_a = np.diff(strategies_a)
        changes_b = np.diff(strategies_b)
        
        if len(changes_a) < 2:
            return 0.0
        
        # 滞后相关性分析
        correlations = []
        max_lag = min(10, len(changes_a) // 4)
        
        for lag in range(max_lag):
            if len(changes_a) > lag and len(changes_b) > lag:
                corr = np.corrcoef(changes_a[lag:], changes_b[:-lag] if lag > 0 else changes_b)[0, 1]
                if not np.isnan(corr):
                    correlations.append(abs(corr))
        
        return max(correlations) if correlations else 0.0
    
    def _calculate_competitive_balance(self, rewards_a: List[float], 
                                     rewards_b: List[float]) -> float:
        """计算竞争平衡性"""
        # 计算累积奖励随时间的变化
        cumulative_a = np.cumsum(rewards_a)
        cumulative_b = np.cumsum(rewards_b)
        
        # 计算领先优势的变化频率
        differences = cumulative_a - cumulative_b
        sign_changes = 0
        
        for i in range(1, len(differences)):
            if np.sign(differences[i]) != np.sign(differences[i-1]):
                sign_changes += 1
        
        # 平衡性评分（更多的领先权变化意味着更平衡）
        balance_score = sign_changes / (len(differences) - 1) if len(differences) > 1 else 0
        return min(1.0, balance_score * 2)  # 归一化
    
    def _evaluate_system_performance(self, round_results) -> Dict[str, float]:
        """评估系统整体性能"""
        performance = {}
        
        # 系统效率（总奖励）
        total_rewards_a = sum(r.player_a_reward for r in round_results)
        total_rewards_b = sum(r.player_b_reward for r in round_results)
        performance['system_efficiency'] = (total_rewards_a + total_rewards_b) / len(round_results)
        
        # 系统稳定性（纳什距离稳定性）
        nash_distances = []
        for r in round_results:
            if hasattr(r, 'nash_distance') and r.nash_distance is not None:
                nash_distances.append(r.nash_distance)
        
        if nash_distances:
            final_stability = 1.0 / (1.0 + np.var(nash_distances[-50:]))
            performance['system_stability'] = final_stability
        else:
            performance['system_stability'] = 0.0
        
        # 学习收敛性
        strategies_a = [r.player_a_strategy for r in round_results]
        strategies_b = [r.player_b_strategy for r in round_results]
        
        if len(strategies_a) >= 100:
            final_var_a = np.var(strategies_a[-50:])
            final_var_b = np.var(strategies_b[-50:])
            convergence_score = 1.0 / (1.0 + final_var_a + final_var_b)
            performance['learning_convergence'] = convergence_score
        else:
            performance['learning_convergence'] = 0.0
        
        # 探索-利用平衡
        strategy_entropy = self._calculate_strategy_entropy(strategies_a + strategies_b)
        performance['exploration_exploitation_balance'] = strategy_entropy
        
        return performance
    
    def _calculate_strategy_entropy(self, strategies: List[float]) -> float:
        """计算策略熵（多样性）"""
        if not strategies:
            return 0.0
        
        # 将策略值离散化为区间
        min_val, max_val = min(strategies), max(strategies)
        if max_val == min_val:
            return 0.0
        
        bins = 20
        hist, _ = np.histogram(strategies, bins=bins, range=(min_val, max_val))
        
        # 计算概率分布
        total = sum(hist)
        if total == 0:
            return 0.0
        
        probs = [h / total for h in hist if h > 0]
        
        # 计算熵
        entropy = -sum(p * np.log2(p) for p in probs)
        max_entropy = np.log2(bins)
        
        return entropy / max_entropy if max_entropy > 0 else 0
    
    def _analyze_performance_trends(self, round_results) -> Dict[str, List[float]]:
        """分析性能趋势"""
        trends = defaultdict(list)
        
        window_size = min(50, len(round_results) // 10)
        if window_size < 5:
            return dict(trends)
        
        # 滑动窗口分析
        for i in range(window_size, len(round_results)):
            window_results = round_results[i-window_size:i]
            
            # 平均奖励趋势
            avg_reward_a = np.mean([r.player_a_reward for r in window_results])
            avg_reward_b = np.mean([r.player_b_reward for r in window_results])
            trends['avg_reward_a'].append(avg_reward_a)
            trends['avg_reward_b'].append(avg_reward_b)
            
            # 策略稳定性趋势
            strategies_a = [r.player_a_strategy for r in window_results]
            strategies_b = [r.player_b_strategy for r in window_results]
            stability_a = 1.0 / (1.0 + np.var(strategies_a))
            stability_b = 1.0 / (1.0 + np.var(strategies_b))
            trends['stability_a'].append(stability_a)
            trends['stability_b'].append(stability_b)
            
            # 竞争强度趋势
            competition_intensity = np.var([r.player_a_reward - r.player_b_reward for r in window_results])
            trends['competition_intensity'].append(competition_intensity)
        
        return dict(trends)
    
    def _generate_recommendations(self, player_a_perf: PlayerPerformance,
                                player_b_perf: PlayerPerformance,
                                comparison_metrics: ComparisonMetrics,
                                system_performance: Dict[str, float]) -> List[str]:
        """生成性能改进建议"""
        recommendations = []
        
        # 学习率建议
        if player_a_perf.learning_rate < 0.3:
            recommendations.append("建议优化玩家A的学习算法参数，提高学习效率")
        if player_b_perf.learning_rate < 0.3:
            recommendations.append("建议优化玩家B的学习算法参数，提高学习效率")
        
        # 稳定性建议
        if player_a_perf.strategy_stability < 0.5:
            recommendations.append("玩家A策略不够稳定，建议调整探索-利用平衡参数")
        if player_b_perf.strategy_stability < 0.5:
            recommendations.append("玩家B策略不够稳定，建议调整探索-利用平衡参数")
        
        # 竞争平衡建议
        if comparison_metrics.competitive_balance < 0.3:
            recommendations.append("游戏竞争不够平衡，建议调整初始条件或规则参数")
        
        # 系统效率建议
        if system_performance.get('system_efficiency', 0) < 50:
            recommendations.append("系统整体效率较低，建议优化奖励机制设计")
        
        # 收敛性建议
        if system_performance.get('learning_convergence', 0) < 0.5:
            recommendations.append("学习收敛性不佳，建议调整网络结构或训练参数")
        
        # 多样性建议
        if system_performance.get('exploration_exploitation_balance', 0) < 0.3:
            recommendations.append("策略多样性不足，建议增加探索机制或调整epsilon参数")
        
        if not recommendations:
            recommendations.append("系统性能表现良好，可以考虑进一步优化细节参数")
        
        return recommendations
    
    def plot_performance_comparison(self, performance_report: PerformanceReport, 
                                  save_path: Optional[str] = None):
        """
        绘制性能对比图
        
        Args:
            performance_report: 性能评估报告
            save_path: 保存路径
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 性能雷达图
        self._plot_performance_radar(axes[0, 0], performance_report)
        
        # 奖励对比
        self._plot_reward_comparison(axes[0, 1], performance_report)
        
        # 稳定性对比
        self._plot_stability_comparison(axes[0, 2], performance_report)
        
        # 性能趋势
        self._plot_performance_trends(axes[1, 0], performance_report)
        
        # 系统性能
        self._plot_system_performance(axes[1, 1], performance_report)
        
        # 建议展示
        self._plot_recommendations(axes[1, 2], performance_report)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"性能对比图已保存到: {save_path}")
        
        plt.show()
    
    def _plot_performance_radar(self, ax, report: PerformanceReport):
        """绘制性能雷达图"""
        categories = ['学习率', '适应速度', '策略稳定性', '一致性', '总奖励(归一化)']
        
        # 归一化数据
        max_reward = max(report.player_a_performance.total_reward, 
                        report.player_b_performance.total_reward, 1)
        
        values_a = [
            report.player_a_performance.learning_rate,
            report.player_a_performance.adaptation_speed,
            report.player_a_performance.strategy_stability,
            report.player_a_performance.consistency_score,
            report.player_a_performance.total_reward / max_reward
        ]
        
        values_b = [
            report.player_b_performance.learning_rate,
            report.player_b_performance.adaptation_speed,
            report.player_b_performance.strategy_stability,
            report.player_b_performance.consistency_score,
            report.player_b_performance.total_reward / max_reward
        ]
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        values_a += values_a[:1]  # 闭合图形
        values_b += values_b[:1]
        angles += angles[:1]
        
        ax.plot(angles, values_a, 'o-', linewidth=2, label='玩家A')
        ax.fill(angles, values_a, alpha=0.25)
        ax.plot(angles, values_b, 'o-', linewidth=2, label='玩家B')
        ax.fill(angles, values_b, alpha=0.25)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title('性能雷达图')
        ax.legend()
        ax.grid(True)
    
    def _plot_reward_comparison(self, ax, report: PerformanceReport):
        """绘制奖励对比"""
        players = ['玩家A', '玩家B']
        total_rewards = [
            report.player_a_performance.total_reward,
            report.player_b_performance.total_reward
        ]
        avg_rewards = [
            report.player_a_performance.average_reward,
            report.player_b_performance.average_reward
        ]
        
        x = np.arange(len(players))
        width = 0.35
        
        ax.bar(x - width/2, total_rewards, width, label='总奖励', alpha=0.8)
        ax2 = ax.twinx()
        ax2.bar(x + width/2, avg_rewards, width, label='平均奖励', alpha=0.8, color='orange')
        
        ax.set_xlabel('玩家')
        ax.set_ylabel('总奖励')
        ax2.set_ylabel('平均奖励')
        ax.set_title('奖励对比')
        ax.set_xticks(x)
        ax.set_xticklabels(players)
        
        # 合并图例
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    def _plot_stability_comparison(self, ax, report: PerformanceReport):
        """绘制稳定性对比"""
        metrics = ['策略稳定性', '一致性评分', '学习率', '适应速度']
        values_a = [
            report.player_a_performance.strategy_stability,
            report.player_a_performance.consistency_score,
            report.player_a_performance.learning_rate,
            report.player_a_performance.adaptation_speed
        ]
        values_b = [
            report.player_b_performance.strategy_stability,
            report.player_b_performance.consistency_score,
            report.player_b_performance.learning_rate,
            report.player_b_performance.adaptation_speed
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        ax.bar(x - width/2, values_a, width, label='玩家A', alpha=0.8)
        ax.bar(x + width/2, values_b, width, label='玩家B', alpha=0.8)
        
        ax.set_xlabel('指标')
        ax.set_ylabel('评分')
        ax.set_title('稳定性指标对比')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 1)
    
    def _plot_performance_trends(self, ax, report: PerformanceReport):
        """绘制性能趋势"""
        trends = report.performance_trends
        if not trends:
            ax.text(0.5, 0.5, '无趋势数据', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('性能趋势')
            return
        
        rounds = range(len(list(trends.values())[0]))
        
        if 'avg_reward_a' in trends:
            ax.plot(rounds, trends['avg_reward_a'], label='玩家A平均奖励', alpha=0.8)
        if 'avg_reward_b' in trends:
            ax.plot(rounds, trends['avg_reward_b'], label='玩家B平均奖励', alpha=0.8)
        
        ax.set_xlabel('时间窗口')
        ax.set_ylabel('平均奖励')
        ax.set_title('奖励趋势分析')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_system_performance(self, ax, report: PerformanceReport):
        """绘制系统性能"""
        metrics = list(report.overall_system_performance.keys())
        values = list(report.overall_system_performance.values())
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(metrics)))
        bars = ax.bar(metrics, values, color=colors, alpha=0.8)
        
        ax.set_xlabel('系统指标')
        ax.set_ylabel('评分')
        ax.set_title('系统整体性能')
        ax.tick_params(axis='x', rotation=45)
        
        # 添加数值标签
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')
    
    def _plot_recommendations(self, ax, report: PerformanceReport):
        """显示建议"""
        ax.axis('off')
        recommendations = report.recommendations
        
        if not recommendations:
            ax.text(0.5, 0.5, '暂无建议', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
        else:
            text = "性能优化建议:\n\n"
            for i, rec in enumerate(recommendations[:5], 1):  # 最多显示5条
                text += f"{i}. {rec}\n\n"
            
            ax.text(0.05, 0.95, text, ha='left', va='top', transform=ax.transAxes,
                   fontsize=10, wrap=True, bbox=dict(boxstyle="round,pad=0.3", 
                   facecolor="lightblue", alpha=0.7))
        
        ax.set_title('优化建议')
    
    def generate_performance_report(self, performance_report: PerformanceReport) -> str:
        """
        生成性能评估文本报告
        
        Args:
            performance_report: 性能评估结果
            
        Returns:
            报告文本
        """
        report = []
        report.append("="*60)
        report.append("性能评估报告")
        report.append("="*60)
        
        # 玩家A性能
        perf_a = performance_report.player_a_performance
        report.append("\n玩家A性能指标:")
        report.append(f"  总奖励: {perf_a.total_reward:.2f}")
        report.append(f"  平均奖励: {perf_a.average_reward:.4f}")
        report.append(f"  奖励方差: {perf_a.reward_variance:.6f}")
        report.append(f"  策略稳定性: {perf_a.strategy_stability:.3f}")
        report.append(f"  学习率: {perf_a.learning_rate:.3f}")
        report.append(f"  适应速度: {perf_a.adaptation_speed:.3f}")
        report.append(f"  一致性评分: {perf_a.consistency_score:.3f}")
        
        # 玩家B性能
        perf_b = performance_report.player_b_performance
        report.append("\n玩家B性能指标:")
        report.append(f"  总奖励: {perf_b.total_reward:.2f}")
        report.append(f"  平均奖励: {perf_b.average_reward:.4f}")
        report.append(f"  奖励方差: {perf_b.reward_variance:.6f}")
        report.append(f"  策略稳定性: {perf_b.strategy_stability:.3f}")
        report.append(f"  学习率: {perf_b.learning_rate:.3f}")
        report.append(f"  适应速度: {perf_b.adaptation_speed:.3f}")
        report.append(f"  一致性评分: {perf_b.consistency_score:.3f}")
        
        # 对比指标
        comp = performance_report.comparison_metrics
        report.append("\n对比分析:")
        report.append(f"  奖励差距: {comp.reward_gap:.3f}")
        report.append(f"  策略相关性: {comp.strategy_correlation:.3f}")
        report.append(f"  相互适应程度: {comp.mutual_adaptation:.3f}")
        report.append(f"  竞争平衡性: {comp.competitive_balance:.3f}")
        report.append(f"  效率比率: {comp.efficiency_ratio:.3f}")
        
        # 系统性能
        report.append("\n系统整体性能:")
        for metric, value in performance_report.overall_system_performance.items():
            report.append(f"  {metric}: {value:.3f}")
        
        # 优化建议
        report.append("\n优化建议:")
        for i, rec in enumerate(performance_report.recommendations, 1):
            report.append(f"  {i}. {rec}")
        
        report.append("="*60)
        
        return "\n".join(report) 