"""
纳什均衡分析器
检测和分析博弈过程中的纳什均衡点
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
import os
import matplotlib.font_manager as fm

logger = logging.getLogger(__name__)

# 设置中文字体支持
def setup_chinese_font():
    """设置中文字体支持"""
    try:
        # 使用指定的系统字体文件路径
        font_path = r"C:\Windows\Fonts\STZHONGS.TTF"
        
        if os.path.exists(font_path):
            # 注册字体
            font_prop = fm.FontProperties(fname=font_path)
            plt.rcParams['font.family'] = font_prop.get_name()
            plt.rcParams['font.sans-serif'] = [font_prop.get_name()] + plt.rcParams.get('font.sans-serif', [])
            plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
            logger.info(f"已成功配置中文字体: {font_path}")
            return True
        else:
            logger.warning(f"找不到指定的字体文件: {font_path}")
            # 回退到常见中文字体名称
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi', 'FangSong', 'Arial Unicode MS']
            plt.rcParams['axes.unicode_minus'] = False
            return False
            
    except Exception as e:
        logger.error(f"配置中文字体失败: {e}")
        
        # 尝试其他备用字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi', 'FangSong', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
        
        return False

# 初始化设置中文字体
chinese_font_configured = setup_chinese_font()


@dataclass
class NashPoint:
    """纳什均衡点数据类"""
    round_number: int
    strategy_a: float
    strategy_b: float
    distance: float
    stability_score: float
    is_approximate: bool = False
    
    def __str__(self):
        return f"纳什点(轮次={self.round_number}, 策略A={self.strategy_a:.2f}, 策略B={self.strategy_b:.2f}, 距离={self.distance:.4f})"


@dataclass
class NashAnalysisResult:
    """纳什均衡分析结果"""
    total_rounds: int
    nash_points: List[NashPoint]
    convergence_rounds: List[int]
    final_nash_distance: float
    stability_periods: List[Tuple[int, int]]  # (开始轮次, 结束轮次)
    average_distance: float
    convergence_quality: float  # 0-1评分


class NashEquilibriumAnalyzer:
    """
    纳什均衡分析器
    提供纳什均衡的检测、分析和可视化功能
    """
    
    def __init__(self, convergence_threshold: float = 0.05, stability_window: int = 20):
        """
        初始化纳什均衡分析器
        
        Args:
            convergence_threshold: 收敛阈值
            stability_window: 稳定性检测窗口大小
        """
        self.convergence_threshold = convergence_threshold
        self.stability_window = stability_window
        
        logger.info(f"纳什均衡分析器初始化: 阈值={convergence_threshold}, 窗口={stability_window}")
    
    def analyze_nash_equilibrium(self, round_results) -> NashAnalysisResult:
        """
        分析实验结果中的纳什均衡
        
        Args:
            round_results: 轮次结果列表，可以是RoundResult对象列表或字典列表
            
        Returns:
            纳什均衡分析结果
        """
        logger.info(f"开始分析纳什均衡，共 {len(round_results)} 轮数据")
        
        # 提取纳什距离数据
        nash_distances = []
        strategies_a = []
        strategies_b = []
        rounds = []
        
        for result in round_results:
            if isinstance(result, dict):
                # 字典格式
                if 'nash_distance' in result and result['nash_distance'] is not None:
                    nash_dist = result['nash_distance']
                    # 处理distance可能是元组或字典的情况
                    if isinstance(nash_dist, tuple):
                        nash_distances.append(nash_dist[0] if nash_dist else float('inf'))
                    elif isinstance(nash_dist, dict):
                        # 如果是字典，尝试获取第一个值
                        try:
                            first_value = next(iter(nash_dist.values()))
                            if isinstance(first_value, (int, float)):
                                nash_distances.append(float(first_value))
                            else:
                                continue
                        except (StopIteration, TypeError):
                            continue
                    elif isinstance(nash_dist, (int, float)):
                        nash_distances.append(float(nash_dist))
                    else:
                        # 跳过不支持的类型
                        continue
                    
                    strategies_a.append(result['player_a_strategy'])
                    strategies_b.append(result['player_b_strategy'])
                    rounds.append(result['round_number'])
            else:
                # 对象格式
                if hasattr(result, 'nash_distance') and result.nash_distance is not None:
                    nash_dist = result.nash_distance
                    # 处理distance可能是元组或字典的情况
                    if isinstance(nash_dist, tuple):
                        nash_distances.append(nash_dist[0] if nash_dist else float('inf'))
                    elif isinstance(nash_dist, dict):
                        # 如果是字典，尝试获取第一个值
                        try:
                            first_value = next(iter(nash_dist.values()))
                            if isinstance(first_value, (int, float)):
                                nash_distances.append(float(first_value))
                            else:
                                continue
                        except (StopIteration, TypeError):
                            continue
                    elif isinstance(nash_dist, (int, float)):
                        nash_distances.append(float(nash_dist))
                    else:
                        # 跳过不支持的类型
                        continue
                    
                    strategies_a.append(result.player_a_strategy)
                    strategies_b.append(result.player_b_strategy)
                    rounds.append(result.round_number)
        
        if not nash_distances:
            logger.warning("没有找到纳什距离数据")
            return self._create_empty_result(len(round_results))
        
        # 确保nash_distances中只有数字类型
        nash_distances = [float(d) for d in nash_distances if isinstance(d, (int, float))]
        
        if not nash_distances:
            logger.warning("处理后没有有效的纳什距离数据")
            return self._create_empty_result(len(round_results))
        
        # 检测纳什均衡点
        nash_points = self._detect_nash_points(rounds, strategies_a, strategies_b, nash_distances)
        
        # 检测收敛轮次
        convergence_rounds = self._detect_convergence_rounds(rounds, nash_distances)
        
        # 检测稳定期
        stability_periods = self._detect_stability_periods(rounds, nash_distances)
        
        # 计算统计指标
        final_nash_distance = nash_distances[-1] if nash_distances else float('inf')
        average_distance = np.mean(nash_distances)
        convergence_quality = self._calculate_convergence_quality(nash_distances)
        
        result = NashAnalysisResult(
            total_rounds=len(round_results),
            nash_points=nash_points,
            convergence_rounds=convergence_rounds,
            final_nash_distance=final_nash_distance,
            stability_periods=stability_periods,
            average_distance=average_distance,
            convergence_quality=convergence_quality
        )
        
        logger.info(f"纳什均衡分析完成: 找到 {len(nash_points)} 个均衡点, "
                   f"收敛质量={convergence_quality:.3f}")
        
        return result
    
    def _detect_nash_points(self, rounds: List[int], strategies_a: List[float], 
                          strategies_b: List[float], distances: List[float]) -> List[NashPoint]:
        """检测纳什均衡点"""
        nash_points = []
        
        for i, (round_num, strategy_a, strategy_b, distance) in enumerate(
            zip(rounds, strategies_a, strategies_b, distances)
        ):
            # 处理distance可能是元组的情况
            if isinstance(distance, tuple):
                distance_value = distance[0] if distance else float('inf')
            else:
                distance_value = distance
                
            if distance_value <= self.convergence_threshold:
                # 计算稳定性评分
                stability_score = self._calculate_stability_score(i, distances)
                
                nash_point = NashPoint(
                    round_number=round_num,
                    strategy_a=strategy_a,
                    strategy_b=strategy_b,
                    distance=distance_value,
                    stability_score=stability_score,
                    is_approximate=(distance_value > self.convergence_threshold * 0.5)
                )
                
                nash_points.append(nash_point)
        
        return nash_points
    
    def _calculate_stability_score(self, index: int, distances: List[float]) -> float:
        """计算稳定性评分"""
        # 获取前后窗口内的距离
        start_idx = max(0, index - self.stability_window // 2)
        end_idx = min(len(distances), index + self.stability_window // 2 + 1)
        
        window_distances = distances[start_idx:end_idx]
        
        if len(window_distances) < 2:
            return 0.0
        
        # 处理window_distances列表中可能包含元组的情况
        processed_window = []
        for d in window_distances:
            if isinstance(d, tuple):
                processed_window.append(d[0] if d else float('inf'))
            else:
                processed_window.append(d)
        
        # 计算方差的倒数作为稳定性评分
        variance = np.var(processed_window)
        stability_score = 1.0 / (1.0 + variance * 100)  # 归一化到0-1
        
        return stability_score
    
    def _detect_convergence_rounds(self, rounds: List[int], distances: List[float]) -> List[int]:
        """检测收敛轮次"""
        convergence_rounds = []
        
        for i, (round_num, distance) in enumerate(zip(rounds, distances)):
            # 处理distance可能是元组的情况
            if isinstance(distance, tuple):
                distance_value = distance[0] if distance else float('inf')
            else:
                distance_value = distance
                
            if distance_value <= self.convergence_threshold:
                # 检查是否是首次收敛或从发散状态重新收敛
                if i == 0:
                    convergence_rounds.append(round_num)
                elif isinstance(distances[i-1], tuple):
                    prev_distance = distances[i-1][0] if distances[i-1] else float('inf')
                    if prev_distance > self.convergence_threshold:
                        convergence_rounds.append(round_num)
                elif distances[i-1] > self.convergence_threshold:
                    convergence_rounds.append(round_num)
        
        return convergence_rounds
    
    def _detect_stability_periods(self, rounds: List[int], distances: List[float]) -> List[Tuple[int, int]]:
        """检测稳定期"""
        stability_periods = []
        current_period_start = None
        
        for i, (round_num, distance) in enumerate(zip(rounds, distances)):
            # 处理distance可能是元组的情况
            if isinstance(distance, tuple):
                distance_value = distance[0] if distance else float('inf')
            else:
                distance_value = distance
                
            if distance_value <= self.convergence_threshold:
                if current_period_start is None:
                    current_period_start = round_num
            else:
                if current_period_start is not None:
                    # 检查稳定期长度
                    if round_num - current_period_start >= self.stability_window:
                        stability_periods.append((current_period_start, round_num - 1))
                    current_period_start = None
        
        # 处理最后一个稳定期
        if current_period_start is not None:
            last_round = rounds[-1]
            if last_round - current_period_start >= self.stability_window:
                stability_periods.append((current_period_start, last_round))
        
        return stability_periods
    
    def _calculate_convergence_quality(self, distances: List[float]) -> float:
        """计算收敛质量评分"""
        if not distances:
            return 0.0
        
        # 处理distances列表中可能包含元组的情况
        processed_distances = []
        for d in distances:
            if isinstance(d, tuple):
                processed_distances.append(d[0] if d else float('inf'))
            else:
                processed_distances.append(d)
        
        # 计算收敛轮次比例
        converged_count = sum(1 for d in processed_distances if d <= self.convergence_threshold)
        convergence_ratio = converged_count / len(processed_distances)
        
        # 计算距离质量（距离越小质量越高）
        final_portion = processed_distances[-len(processed_distances)//4:] if len(processed_distances) >= 4 else processed_distances
        distance_quality = 1.0 - min(1.0, np.mean(final_portion) / self.convergence_threshold)
        
        # 计算稳定性质量
        stability_quality = 1.0 - min(1.0, np.std(final_portion) / self.convergence_threshold)
        
        # 综合评分
        quality_score = (convergence_ratio * 0.4 + distance_quality * 0.4 + stability_quality * 0.2)
        
        return max(0.0, min(1.0, quality_score))
    
    def _create_empty_result(self, total_rounds: int) -> NashAnalysisResult:
        """创建空的分析结果"""
        return NashAnalysisResult(
            total_rounds=total_rounds,
            nash_points=[],
            convergence_rounds=[],
            final_nash_distance=float('inf'),
            stability_periods=[],
            average_distance=float('inf'),
            convergence_quality=0.0
        )
    
    def plot_nash_convergence(self, round_results, save_path: Optional[str] = None):
        """
        绘制纳什均衡收敛图
        
        Args:
            round_results: 轮次结果列表，可以是RoundResult对象列表或字典列表
            save_path: 保存路径
        """
        # 提取数据
        rounds = []
        distances = []
        
        for r in round_results:
            if isinstance(r, dict):
                if 'nash_distance' in r and r['nash_distance'] is not None:
                    rounds.append(r['round_number'])
                    nash_dist = r['nash_distance']
                    # 处理distance可能是元组的情况
                    if isinstance(nash_dist, tuple):
                        distances.append(nash_dist[0] if nash_dist else float('inf'))
                    else:
                        distances.append(nash_dist)
            else:
                if hasattr(r, 'nash_distance') and r.nash_distance is not None:
                    rounds.append(r.round_number)
                    nash_dist = r.nash_distance
                    # 处理distance可能是元组的情况
                    if isinstance(nash_dist, tuple):
                        distances.append(nash_dist[0] if nash_dist else float('inf'))
                    else:
                        distances.append(nash_dist)
        
        if not distances:
            logger.warning("没有纳什距离数据可绘制")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # 纳什距离演化
        ax1.plot(rounds, distances, label='纳什距离', alpha=0.7, color='blue')
        ax1.axhline(y=self.convergence_threshold, color='red', linestyle='--', 
                   label=f'收敛阈值 ({self.convergence_threshold})')
        ax1.set_xlabel('轮次')
        ax1.set_ylabel('纳什距离')
        ax1.set_title('纳什均衡收敛过程')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 移动平均
        window_size = min(20, len(distances))
        if window_size > 1:
            ma_distances = np.convolve(distances, np.ones(window_size)/window_size, mode='valid')
            ma_rounds = rounds[window_size-1:]
            
            ax2.plot(ma_rounds, ma_distances, label=f'{window_size}轮移动平均', 
                    color='green', linewidth=2)
            ax2.axhline(y=self.convergence_threshold, color='red', linestyle='--', 
                       label=f'收敛阈值 ({self.convergence_threshold})')
            ax2.set_xlabel('轮次')
            ax2.set_ylabel('平均纳什距离')
            ax2.set_title('纳什距离移动平均')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"纳什收敛图已保存至: {save_path}")
        
        plt.close()
    
    def plot_strategy_space(self, round_results, save_path: Optional[str] = None):
        """
        绘制策略空间图
        
        Args:
            round_results: 轮次结果列表，可以是RoundResult对象列表或字典列表
            save_path: 保存路径
        """
        # 提取数据
        strategies_a = []
        strategies_b = []
        rewards_a = []
        rewards_b = []
        rounds = []
        
        for r in round_results:
            if isinstance(r, dict):
                strategies_a.append(r['player_a_strategy'])
                strategies_b.append(r['player_b_strategy'])
                rewards_a.append(r['player_a_revenue'])
                rewards_b.append(r['player_b_revenue'])
                rounds.append(r['round_number'])
            else:
                strategies_a.append(r.player_a_strategy)
                strategies_b.append(r.player_b_strategy)
                rewards_a.append(r.player_a_revenue)
                rewards_b.append(r.player_b_revenue)
                rounds.append(r.round_number)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # 策略空间
        scatter = ax1.scatter(strategies_a, strategies_b, c=rounds, cmap='viridis', 
                            alpha=0.7, s=50)
        ax1.set_xlabel('玩家A策略')
        ax1.set_ylabel('玩家B策略')
        ax1.set_title('策略空间轨迹')
        ax1.grid(True, alpha=0.3)
        
        cbar = fig.colorbar(scatter, ax=ax1)
        cbar.set_label('轮次')
        
        # 绘制策略向量场
        try:
            self._plot_strategy_vector_field(ax2, strategies_a, strategies_b, rewards_a, rewards_b)
        except Exception as e:
            logger.warning(f"绘制策略向量场失败: {e}")
            ax2.set_title('策略梯度场 (绘制失败)')
            ax2.set_xlabel('玩家A策略')
            ax2.set_ylabel('玩家B策略')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"策略空间图已保存至: {save_path}")
        
        plt.close()
    
    def generate_nash_report(self, analysis_result: NashAnalysisResult) -> str:
        """
        生成纳什均衡分析报告
        
        Args:
            analysis_result: 分析结果
            
        Returns:
            报告文本
        """
        report = []
        report.append("="*50)
        report.append("纳什均衡分析报告")
        report.append("="*50)
        
        # 基本信息
        report.append(f"\n实验总轮次: {analysis_result.total_rounds}")
        report.append(f"最终纳什距离: {analysis_result.final_nash_distance:.6f}")
        report.append(f"平均纳什距离: {analysis_result.average_distance:.6f}")
        report.append(f"收敛质量评分: {analysis_result.convergence_quality:.3f}/1.0")
        
        # 均衡点信息
        report.append(f"\n检测到 {len(analysis_result.nash_points)} 个纳什均衡点:")
        for point in analysis_result.nash_points[:5]:  # 显示前5个
            report.append(f"  {point}")
        if len(analysis_result.nash_points) > 5:
            report.append(f"  ... 还有 {len(analysis_result.nash_points) - 5} 个均衡点")
        
        # 收敛信息
        report.append(f"\n收敛轮次: {analysis_result.convergence_rounds}")
        
        # 稳定期信息
        report.append(f"\n检测到 {len(analysis_result.stability_periods)} 个稳定期:")
        for start, end in analysis_result.stability_periods:
            duration = end - start + 1
            report.append(f"  轮次 {start}-{end} (持续 {duration} 轮)")
        
        # 收敛质量评估
        report.append(f"\n收敛质量评估:")
        if analysis_result.convergence_quality >= 0.8:
            report.append("  ✓ 优秀 - 快速收敛且稳定")
        elif analysis_result.convergence_quality >= 0.6:
            report.append("  ✓ 良好 - 较好的收敛性能")
        elif analysis_result.convergence_quality >= 0.4:
            report.append("  ⚠ 一般 - 收敛但不够稳定")
        else:
            report.append("  ✗ 较差 - 收敛困难或不稳定")
        
        report.append("="*50)
        
        return "\n".join(report) 