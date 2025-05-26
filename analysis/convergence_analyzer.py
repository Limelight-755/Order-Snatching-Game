"""
收敛分析器
分析博弈过程中的策略收敛性和学习曲线
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
from scipy import stats
from scipy.signal import savgol_filter

logger = logging.getLogger(__name__)


@dataclass
class ConvergenceMetrics:
    """收敛度量结果"""
    convergence_rate: float  # 收敛速度
    stability_score: float  # 稳定性评分
    learning_efficiency: float  # 学习效率
    final_variance: float  # 最终方差
    convergence_round: Optional[int]  # 收敛轮次
    oscillation_frequency: float  # 震荡频率
    trend_direction: str  # 趋势方向 ('increasing', 'decreasing', 'stable')


@dataclass
class LearningPhaseAnalysis:
    """学习阶段分析结果"""
    exploration_phase: Tuple[int, int]  # 探索阶段 (开始, 结束)
    learning_phase: Tuple[int, int]  # 学习阶段
    equilibrium_phase: Tuple[int, int]  # 均衡阶段
    phase_performance: Dict[str, Dict[str, float]]  # 各阶段性能指标


class ConvergenceAnalyzer:
    """
    收敛分析器
    分析策略收敛过程和学习效率
    """
    
    def __init__(self, window_size: int = 50, smoothing_window: int = 21):
        """
        初始化收敛分析器
        
        Args:
            window_size: 分析窗口大小
            smoothing_window: 平滑窗口大小
        """
        self.window_size = window_size
        self.smoothing_window = smoothing_window
        
        logger.info(f"收敛分析器初始化: 窗口={window_size}, 平滑窗口={smoothing_window}")
    
    def analyze_convergence(self, round_results) -> Dict[str, ConvergenceMetrics]:
        """
        分析策略收敛性
        
        Args:
            round_results: 轮次结果列表，可以是RoundResult对象列表或字典列表
            
        Returns:
            收敛分析结果字典
        """
        logger.info(f"开始分析收敛性，共 {len(round_results)} 轮数据")
        
        # 提取策略数据 - 兼容对象和字典格式
        strategies_a = []
        strategies_b = []
        rewards_a = []
        rewards_b = []
        
        for r in round_results:
            if isinstance(r, dict):
                # 字典格式
                strategies_a.append(r.get('player_a_strategy'))
                strategies_b.append(r.get('player_b_strategy'))
                rewards_a.append(r.get('player_a_revenue'))
                rewards_b.append(r.get('player_b_revenue'))
            else:
                # 对象格式
                strategies_a.append(r.player_a_strategy)
                strategies_b.append(r.player_b_strategy)
                rewards_a.append(r.player_a_revenue)
                rewards_b.append(r.player_b_revenue)
        
        results = {}
        
        # 分析玩家A策略收敛
        results['player_a_strategy'] = self._analyze_series_convergence(
            strategies_a, "玩家A策略"
        )
        
        # 分析玩家B策略收敛
        results['player_b_strategy'] = self._analyze_series_convergence(
            strategies_b, "玩家B策略"
        )
        
        # 分析玩家A奖励收敛
        results['player_a_reward'] = self._analyze_series_convergence(
            rewards_a, "玩家A奖励"
        )
        
        # 分析玩家B奖励收敛
        results['player_b_reward'] = self._analyze_series_convergence(
            rewards_b, "玩家B奖励"
        )
        
        # 如果有纳什距离数据，也进行分析
        nash_distances = []
        for r in round_results:
            if isinstance(r, dict):
                if 'nash_distance' in r and r['nash_distance'] is not None:
                    nash_dist = r['nash_distance']
                    # 处理不同类型的nash_distance
                    if isinstance(nash_dist, (int, float)):
                        nash_distances.append(float(nash_dist))
                    elif isinstance(nash_dist, tuple) and nash_dist and isinstance(nash_dist[0], (int, float)):
                        nash_distances.append(float(nash_dist[0]))
                    # 忽略其他类型
            else:
                if hasattr(r, 'nash_distance') and r.nash_distance is not None:
                    nash_dist = r.nash_distance
                    # 处理不同类型的nash_distance
                    if isinstance(nash_dist, (int, float)):
                        nash_distances.append(float(nash_dist))
                    elif isinstance(nash_dist, tuple) and nash_dist and isinstance(nash_dist[0], (int, float)):
                        nash_distances.append(float(nash_dist[0]))
                    # 忽略其他类型
        
        if nash_distances:
            results['nash_distance'] = self._analyze_series_convergence(
                nash_distances, "纳什距离"
            )
        else:
            logger.warning("没有找到纳什距离数据")
        
        logger.info("收敛性分析完成")
        return results
    
    def _analyze_series_convergence(self, series: List[float], 
                                  series_name: str) -> ConvergenceMetrics:
        """分析单个序列的收敛性"""
        if len(series) < self.window_size:
            logger.warning(f"{series_name} 数据量不足，无法进行收敛分析")
            return self._create_empty_metrics()
        
        # 数据平滑
        if len(series) >= self.smoothing_window:
            smoothed = savgol_filter(series, self.smoothing_window, 3)
        else:
            smoothed = np.array(series)
        
        # 计算收敛速度（基于方差减少率）
        convergence_rate = self._calculate_convergence_rate(smoothed)
        
        # 计算稳定性评分
        stability_score = self._calculate_stability_score(smoothed)
        
        # 计算学习效率
        learning_efficiency = self._calculate_learning_efficiency(smoothed)
        
        # 计算最终方差
        final_portion = smoothed[-self.window_size:]
        final_variance = np.var(final_portion)
        
        # 检测收敛轮次
        convergence_round = self._detect_convergence_round(smoothed)
        
        # 计算震荡频率
        oscillation_frequency = self._calculate_oscillation_frequency(smoothed)
        
        # 确定趋势方向
        trend_direction = self._determine_trend_direction(smoothed)
        
        return ConvergenceMetrics(
            convergence_rate=convergence_rate,
            stability_score=stability_score,
            learning_efficiency=learning_efficiency,
            final_variance=final_variance,
            convergence_round=convergence_round,
            oscillation_frequency=oscillation_frequency,
            trend_direction=trend_direction
        )
    
    def _calculate_convergence_rate(self, series: np.ndarray) -> float:
        """计算收敛速度"""
        # 计算滑动窗口方差
        variances = []
        for i in range(self.window_size, len(series)):
            window = series[i-self.window_size:i]
            variances.append(np.var(window))
        
        if len(variances) < 2:
            return 0.0
        
        # 计算方差减少的斜率
        x = np.arange(len(variances))
        slope, _, r_value, _, _ = stats.linregress(x, variances)
        
        # 收敛速度为负斜率的归一化值
        convergence_rate = max(0, -slope) / (np.var(series) + 1e-8)
        return min(1.0, convergence_rate)
    
    def _calculate_stability_score(self, series: np.ndarray) -> float:
        """计算稳定性评分"""
        # 计算最后一段的方差
        final_portion = series[-self.window_size:]
        final_variance = np.var(final_portion)
        
        # 计算整体方差
        total_variance = np.var(series)
        
        # 稳定性评分
        if total_variance == 0:
            return 1.0
        
        stability_score = 1.0 - (final_variance / total_variance)
        return max(0.0, min(1.0, stability_score))
    
    def _calculate_learning_efficiency(self, series: np.ndarray) -> float:
        """计算学习效率"""
        # 比较前半部分和后半部分的改进
        mid_point = len(series) // 2
        first_half_mean = np.mean(series[:mid_point])
        second_half_mean = np.mean(series[mid_point:])
        
        # 计算改进程度
        total_range = np.max(series) - np.min(series)
        if total_range == 0:
            return 0.5
        
        improvement = abs(second_half_mean - first_half_mean) / total_range
        return min(1.0, improvement)
    
    def _detect_convergence_round(self, series: np.ndarray) -> Optional[int]:
        """检测收敛轮次"""
        # 计算滑动标准差
        rolling_std = []
        for i in range(self.window_size, len(series)):
            window = series[i-self.window_size:i]
            rolling_std.append(np.std(window))
        
        if not rolling_std:
            return None
        
        # 找到标准差稳定在较低水平的点
        threshold = np.mean(rolling_std) * 0.1  # 10%阈值
        
        for i, std in enumerate(rolling_std):
            if std <= threshold:
                # 检查后续几个点是否也保持稳定
                stable_count = 0
                for j in range(i, min(i + 10, len(rolling_std))):
                    if rolling_std[j] <= threshold * 1.5:
                        stable_count += 1
                
                if stable_count >= 5:  # 至少5个点保持稳定
                    return i + self.window_size
        
        return None
    
    def _calculate_oscillation_frequency(self, series: np.ndarray) -> float:
        """计算震荡频率"""
        # 计算一阶差分的符号变化次数
        diff = np.diff(series)
        sign_changes = 0
        
        for i in range(1, len(diff)):
            if np.sign(diff[i]) != np.sign(diff[i-1]) and np.sign(diff[i]) != 0:
                sign_changes += 1
        
        # 归一化频率
        frequency = sign_changes / len(diff) if len(diff) > 0 else 0
        return frequency
    
    def _determine_trend_direction(self, series: np.ndarray) -> str:
        """确定趋势方向"""
        # 使用线性回归确定总体趋势
        x = np.arange(len(series))
        slope, _, r_value, _, _ = stats.linregress(x, series)
        
        # 根据斜率和相关性确定趋势
        if abs(r_value) < 0.1:  # 相关性很低
            return 'stable'
        elif slope > 0:
            return 'increasing'
        else:
            return 'decreasing'
    
    def _create_empty_metrics(self) -> ConvergenceMetrics:
        """创建空的度量结果"""
        return ConvergenceMetrics(
            convergence_rate=0.0,
            stability_score=0.0,
            learning_efficiency=0.0,
            final_variance=0.0,
            convergence_round=None,
            oscillation_frequency=0.0,
            trend_direction='stable'
        )
    
    def analyze_learning_phases(self, round_results) -> LearningPhaseAnalysis:
        """
        分析学习阶段
        
        Args:
            round_results: 轮次结果列表
            
        Returns:
            学习阶段分析结果
        """
        total_rounds = len(round_results)
        
        # 根据轮次划分阶段（基于技术报告的设定）
        exploration_end = min(50, total_rounds // 10)
        learning_end = min(200, total_rounds * 4 // 10)
        
        exploration_phase = (1, exploration_end)
        learning_phase = (exploration_end + 1, learning_end)
        equilibrium_phase = (learning_end + 1, total_rounds)
        
        # 计算各阶段性能
        phase_performance = {}
        
        for phase_name, (start, end) in [
            ('exploration', exploration_phase),
            ('learning', learning_phase), 
            ('equilibrium', equilibrium_phase)
        ]:
            phase_results = round_results[start-1:end]
            if phase_results:
                performance = self._calculate_phase_performance(phase_results)
                phase_performance[phase_name] = performance
        
        return LearningPhaseAnalysis(
            exploration_phase=exploration_phase,
            learning_phase=learning_phase,
            equilibrium_phase=equilibrium_phase,
            phase_performance=phase_performance
        )
    
    def _calculate_phase_performance(self, phase_results) -> Dict[str, float]:
        """计算阶段性能指标"""
        if not phase_results:
            return {}
        
        strategies_a = [r.player_a_strategy for r in phase_results]
        strategies_b = [r.player_b_strategy for r in phase_results]
        rewards_a = [r.player_a_reward for r in phase_results]
        rewards_b = [r.player_b_reward for r in phase_results]
        
        return {
            'avg_strategy_a': np.mean(strategies_a),
            'avg_strategy_b': np.mean(strategies_b),
            'avg_reward_a': np.mean(rewards_a),
            'avg_reward_b': np.mean(rewards_b),
            'strategy_variance_a': np.var(strategies_a),
            'strategy_variance_b': np.var(strategies_b),
            'reward_variance_a': np.var(rewards_a),
            'reward_variance_b': np.var(rewards_b),
            'strategy_stability': 1.0 / (1.0 + np.var(strategies_a) + np.var(strategies_b))
        }
    
    def plot_convergence_analysis(self, round_results, save_path: Optional[str] = None):
        """
        绘制收敛分析图
        
        Args:
            round_results: 轮次结果列表
            save_path: 保存路径
        """
        strategies_a = [r.player_a_strategy for r in round_results]
        strategies_b = [r.player_b_strategy for r in round_results]
        rewards_a = [r.player_a_reward for r in round_results]
        rewards_b = [r.player_b_reward for r in round_results]
        rounds = [r.round_number for r in round_results]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 策略收敛图
        axes[0, 0].plot(rounds, strategies_a, label='玩家A策略', alpha=0.7)
        axes[0, 0].plot(rounds, strategies_b, label='玩家B策略', alpha=0.7)
        axes[0, 0].set_xlabel('轮次')
        axes[0, 0].set_ylabel('策略值')
        axes[0, 0].set_title('策略收敛过程')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 奖励收敛图
        axes[0, 1].plot(rounds, rewards_a, label='玩家A奖励', alpha=0.7)
        axes[0, 1].plot(rounds, rewards_b, label='玩家B奖励', alpha=0.7)
        axes[0, 1].set_xlabel('轮次')
        axes[0, 1].set_ylabel('奖励值')
        axes[0, 1].set_title('奖励收敛过程')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 滑动方差图
        window_size = min(20, len(rounds) // 10)
        if window_size >= 2:
            rolling_var_a = []
            rolling_var_b = []
            rolling_rounds = []
            
            for i in range(window_size, len(strategies_a)):
                var_a = np.var(strategies_a[i-window_size:i])
                var_b = np.var(strategies_b[i-window_size:i])
                rolling_var_a.append(var_a)
                rolling_var_b.append(var_b)
                rolling_rounds.append(rounds[i])
            
            axes[1, 0].plot(rolling_rounds, rolling_var_a, label='玩家A策略方差')
            axes[1, 0].plot(rolling_rounds, rolling_var_b, label='玩家B策略方差')
            axes[1, 0].set_xlabel('轮次')
            axes[1, 0].set_ylabel('滑动方差')
            axes[1, 0].set_title('策略稳定性分析')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # 学习阶段分析
        phase_analysis = self.analyze_learning_phases(round_results)
        phases = ['exploration', 'learning', 'equilibrium']
        avg_rewards_a = []
        avg_rewards_b = []
        
        for phase in phases:
            if phase in phase_analysis.phase_performance:
                perf = phase_analysis.phase_performance[phase]
                avg_rewards_a.append(perf.get('avg_reward_a', 0))
                avg_rewards_b.append(perf.get('avg_reward_b', 0))
            else:
                avg_rewards_a.append(0)
                avg_rewards_b.append(0)
        
        x = np.arange(len(phases))
        width = 0.35
        
        axes[1, 1].bar(x - width/2, avg_rewards_a, width, label='玩家A平均奖励')
        axes[1, 1].bar(x + width/2, avg_rewards_b, width, label='玩家B平均奖励')
        axes[1, 1].set_xlabel('学习阶段')
        axes[1, 1].set_ylabel('平均奖励')
        axes[1, 1].set_title('各阶段性能对比')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(['探索期', '学习期', '均衡期'])
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"收敛分析图已保存到: {save_path}")
        
        plt.show()
    
    def generate_convergence_report(self, convergence_results: Dict[str, ConvergenceMetrics],
                                  phase_analysis: LearningPhaseAnalysis) -> str:
        """
        生成收敛分析报告
        
        Args:
            convergence_results: 收敛分析结果
            phase_analysis: 学习阶段分析结果
            
        Returns:
            报告文本
        """
        report = []
        report.append("="*50)
        report.append("收敛性分析报告")
        report.append("="*50)
        
        # 收敛性总结
        report.append("\n收敛性分析:")
        for metric_name, metrics in convergence_results.items():
            report.append(f"\n{metric_name}:")
            report.append(f"  收敛速度: {metrics.convergence_rate:.3f}")
            report.append(f"  稳定性评分: {metrics.stability_score:.3f}")
            report.append(f"  学习效率: {metrics.learning_efficiency:.3f}")
            report.append(f"  最终方差: {metrics.final_variance:.6f}")
            report.append(f"  收敛轮次: {metrics.convergence_round or '未收敛'}")
            report.append(f"  震荡频率: {metrics.oscillation_frequency:.3f}")
            report.append(f"  趋势方向: {metrics.trend_direction}")
        
        # 学习阶段分析
        report.append(f"\n学习阶段分析:")
        report.append(f"探索阶段: 轮次 {phase_analysis.exploration_phase[0]}-{phase_analysis.exploration_phase[1]}")
        report.append(f"学习阶段: 轮次 {phase_analysis.learning_phase[0]}-{phase_analysis.learning_phase[1]}")
        report.append(f"均衡阶段: 轮次 {phase_analysis.equilibrium_phase[0]}-{phase_analysis.equilibrium_phase[1]}")
        
        # 各阶段性能对比
        for phase_name, performance in phase_analysis.phase_performance.items():
            report.append(f"\n{phase_name}阶段性能:")
            for metric, value in performance.items():
                report.append(f"  {metric}: {value:.4f}")
        
        report.append("="*50)
        
        return "\n".join(report) 