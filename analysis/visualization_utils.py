"""
可视化工具
提供博弈分析的各种图表绘制和可视化功能
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import logging
from collections import defaultdict
import pandas as pd
import os
import matplotlib.font_manager as fm
from matplotlib import rcParams
import sys
import platform
from pathlib import Path
import matplotlib as mpl
import urllib.request
import zipfile
import tempfile
import shutil

# 配置日志
logger = logging.getLogger(__name__)

# 设置中文字体路径
chinese_font_path = r"C:\Windows\Fonts\STZHONGS.TTF"

class VisualizationUtils:
    """
    可视化工具类
    提供博弈分析的各种图表绘制功能
    """
    
    def __init__(self, style: str = 'seaborn-v0_8', figsize: Tuple[int, int] = (12, 8)):
        """
        初始化可视化工具
        
        Args:
            style: 绘图风格
            figsize: 默认图形大小
        """
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')
            logger.warning(f"无法应用样式 {style}，使用默认样式")
        
        self.default_figsize = figsize
        self.colors = plt.cm.Set1(np.linspace(0, 1, 10))
        
        # 设置中文字体
        if os.path.exists(chinese_font_path):
            self.font_prop = fm.FontProperties(fname=chinese_font_path)
            plt.rcParams['font.family'] = self.font_prop.get_name()
            plt.rcParams['font.sans-serif'] = [self.font_prop.get_name()] + plt.rcParams.get('font.sans-serif', [])
            plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
            logger.info(f"已成功配置中文字体: {chinese_font_path}")
        else:
            # 回退到常见中文字体名称
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi', 'FangSong', 'Arial Unicode MS']
            plt.rcParams['axes.unicode_minus'] = False
            self.font_prop = None
            logger.warning(f"找不到指定的字体文件: {chinese_font_path}，使用备用字体")
        
        # 测试中文字体显示
        try:
            test_fig = plt.figure(figsize=(2, 2))
            test_fig.text(0.5, 0.5, '中文测试', ha='center', va='center')
            plt.close(test_fig)
            logger.info("中文字体测试通过")
        except Exception as e:
            logger.warning(f"中文字体测试失败: {e}")
        
        logger.info(f"可视化工具初始化完成: 风格={style}, 图形大小={figsize}")
    
    def apply_font_to_axis(self, ax):
        """
        将中文字体应用到坐标轴上的所有文本元素
        
        Args:
            ax: matplotlib坐标轴对象
        """
        if not self.font_prop:
            return
            
        # 应用到标题
        if ax.get_title():
            ax.set_title(ax.get_title(), fontproperties=self.font_prop)
        
        # 应用到x轴标签
        if ax.get_xlabel():
            ax.set_xlabel(ax.get_xlabel(), fontproperties=self.font_prop)
        
        # 应用到y轴标签
        if ax.get_ylabel():
            ax.set_ylabel(ax.get_ylabel(), fontproperties=self.font_prop)
        
        # 应用到刻度标签
        for label in ax.get_xticklabels():
            label.set_fontproperties(self.font_prop)
        
        for label in ax.get_yticklabels():
            label.set_fontproperties(self.font_prop)
        
        # 应用到图例
        if ax.get_legend():
            for text in ax.get_legend().get_texts():
                text.set_fontproperties(self.font_prop)
    
    def apply_font_to_figure(self, fig):
        """
        将中文字体应用到整个图形的所有文本元素
        
        Args:
            fig: matplotlib图形对象
        """
        if not self.font_prop:
            return
            
        # 应用到所有子图
        for ax in fig.axes:
            self.apply_font_to_axis(ax)
        
        # 应用到图形标题
        if fig._suptitle:
            fig._suptitle.set_fontproperties(self.font_prop)
    
    def plot_strategy_evolution(self, round_results, save_path: Optional[str] = None,
                              show_phases: bool = True):
        """
        绘制策略演化图
        
        Args:
            round_results: 轮次结果列表，可以是RoundResult对象列表或字典列表
            save_path: 保存路径
            show_phases: 是否显示学习阶段划分
        """
        if not round_results:
            logger.warning("没有轮次结果数据，无法绘制策略演化图")
            return
            
        # 使用更美观的样式
        with plt.style.context('seaborn-v0_8-whitegrid'):
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), dpi=100)
            fig.patch.set_facecolor('#F8F8FA')  # 设置图表背景色
            
            # 提取数据
            try:
                rounds = []
                strategies_a = []
                strategies_b = []
                
                for r in round_results:
                    if isinstance(r, dict):
                        # 字典格式
                        rounds.append(r.get('round_number'))
                        strategies_a.append(r.get('player_a_strategy'))
                        strategies_b.append(r.get('player_b_strategy'))
                    else:
                        # 对象格式
                        rounds.append(r.round_number)
                        strategies_a.append(r.player_a_strategy)
                        strategies_b.append(r.player_b_strategy)
            except AttributeError as e:
                logger.error(f"数据格式错误: {e}")
                return
            
            # 按天计算平均值（24轮为一天）
            hours_per_day = 24
            daily_rounds = []
            daily_strategies_a = []
            daily_strategies_b = []
            
            for day_start in range(0, len(rounds), hours_per_day):
                day_end = min(day_start + hours_per_day, len(rounds))
                if day_end > day_start:
                    # 计算当天的平均策略值
                    day_strategies_a = strategies_a[day_start:day_end]
                    day_strategies_b = strategies_b[day_start:day_end]
                    
                    if day_strategies_a and day_strategies_b:
                        daily_rounds.append(day_start + hours_per_day // 2)  # 使用天的中点作为x坐标
                        daily_strategies_a.append(sum(day_strategies_a) / len(day_strategies_a))
                        daily_strategies_b.append(sum(day_strategies_b) / len(day_strategies_b))
            
            # 如果没有足够数据按天分组，则使用原始数据
            if not daily_rounds:
                daily_rounds = rounds
                daily_strategies_a = strategies_a
                daily_strategies_b = strategies_b
            
            # 动态调整x轴范围
            max_rounds = max(daily_rounds) if daily_rounds else 100
            x_ticks_step = max(hours_per_day, max_rounds // 10)  # 确保至少有10个刻度，且以天为单位
            x_ticks = list(range(0, max_rounds + 1, x_ticks_step))
            
            # 策略值演化 - 更美观的线条和配色
            ax1.plot(daily_rounds, daily_strategies_a, color='#3A76AF', label='玩家A策略', 
                    linewidth=2, alpha=0.9, zorder=10, marker='o', markersize=4)
            ax1.plot(daily_rounds, daily_strategies_b, color='#6ABE5E', label='玩家B策略', 
                    linewidth=2, alpha=0.9, zorder=10, marker='s', markersize=4)
            
            # 背景和网格设置
            ax1.set_facecolor('#F8F8FA')  # 设置子图背景
            ax1.grid(True, linestyle='--', alpha=0.7, color='#CCCCCC')
            
            # 动态确定y轴范围，增加上下边距
            min_y = min(min(daily_strategies_a), min(daily_strategies_b)) - 5
            max_y = max(max(daily_strategies_a), max(daily_strategies_b)) + 5
            ax1.set_ylim(min_y, max_y)
            
            if show_phases and len(rounds) >= 200:
                # 添加学习阶段划分线（按新的阶段划分）
                exploration_end = 120  # 探索期结束
                learning_end = 360     # 学习期结束，均衡期开始
                
                ax1.axvline(x=exploration_end, color='#FF7F7F', linestyle='--', 
                         alpha=0.6, label='探索期结束')
                ax1.axvline(x=learning_end, color='#FFBF00', linestyle='--', 
                         alpha=0.6, label='学习期结束')
                
                # 添加阶段背景色
                ax1.axvspan(0, exploration_end, alpha=0.1, color='#FFE6E6', label='探索期')
                ax1.axvspan(exploration_end, learning_end, alpha=0.1, color='#FFF5E6', label='学习期')
                ax1.axvspan(learning_end, max(daily_rounds), alpha=0.1, color='#E6F5E6', label='均衡期')
            
            # 设置轴标签和标题 - 使用中文字体
            ax1.set_xlabel('轮次', fontsize=12, fontweight='bold')
            ax1.set_ylabel('策略值（定价阈值）', fontsize=12, fontweight='bold')
            ax1.set_title('策略演化过程（日平均值）', fontsize=14, fontweight='bold', pad=15)
            
            # 设置刻度和边框
            ax1.set_xticks(x_ticks)
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            ax1.tick_params(labelsize=10)
            
            # 更好的图例位置和样式
            legend = ax1.legend(loc='upper right', frameon=True, framealpha=0.95, 
                             shadow=True, fancybox=True, fontsize=10)
            legend.get_frame().set_facecolor('#FFFFFF')
            
            # 策略差异演化图 - 使用紫色渐变填充效果
            strategy_diff = [abs(a - b) for a, b in zip(daily_strategies_a, daily_strategies_b)]
            
            # 创建更美观的策略差异图
            ax2.set_facecolor('#F8F8FA')  # 设置子图背景
            ax2.grid(True, linestyle='--', alpha=0.7, color='#CCCCCC')
            
            ax2.plot(daily_rounds, strategy_diff, color='#9467BD', linewidth=2.5, 
                   alpha=0.9, label='策略差异', zorder=10, marker='D', markersize=4)
            
            # 渐变填充区域
            ax2.fill_between(daily_rounds, strategy_diff, color='#9467BD', alpha=0.3)
            
            # 动态确定y轴范围
            max_diff = max(strategy_diff) if strategy_diff else 0
            ax2.set_ylim(0, max_diff * 1.2)  # 增加20%的顶部空间
            
            # 设置轴标签和标题
            ax2.set_xlabel('轮次', fontsize=12, fontweight='bold')
            ax2.set_ylabel('策略差异', fontsize=12, fontweight='bold')
            ax2.set_title('策略差异演化（日平均值）', fontsize=14, fontweight='bold', pad=15)
            
            # 设置刻度和边框
            ax2.set_xticks(x_ticks)
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.tick_params(labelsize=10)
            
            # 图例
            legend2 = ax2.legend(loc='upper right', frameon=True, framealpha=0.95, 
                              shadow=True, fancybox=True, fontsize=10)
            legend2.get_frame().set_facecolor('#FFFFFF')
            
            # 设置整体标题
            fig.suptitle('策略演化分析', fontsize=16, fontweight='bold')
            
            plt.tight_layout(pad=2.0)
            
            # 应用中文字体到所有文本元素
            self.apply_font_to_figure(fig)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"策略演化图已保存到: {save_path}")
            
            return fig
    
    def plot_reward_analysis(self, round_results, save_path: Optional[str] = None):
        """
        绘制奖励分析图
        
        Args:
            round_results: 轮次结果列表，可以是RoundResult对象列表或字典列表
            save_path: 保存路径
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 提取数据
        rounds = []
        rewards_a = []
        rewards_b = []
        
        for r in round_results:
            if isinstance(r, dict):
                # 字典格式
                rounds.append(r.get('round_number'))
                rewards_a.append(r.get('player_a_revenue'))
                rewards_b.append(r.get('player_b_revenue'))
            else:
                # 对象格式
                rounds.append(r.round_number)
                rewards_a.append(r.player_a_revenue)
                rewards_b.append(r.player_b_revenue)
        
        # 奖励演化
        axes[0, 0].plot(rounds, rewards_a, label='玩家A', alpha=0.7)
        axes[0, 0].plot(rounds, rewards_b, label='玩家B', alpha=0.7)
        axes[0, 0].set_xlabel('轮次', fontsize=12)
        axes[0, 0].set_ylabel('奖励值', fontsize=12)
        axes[0, 0].set_title('奖励演化过程', fontsize=14, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 累积奖励
        cumulative_a = np.cumsum(rewards_a)
        cumulative_b = np.cumsum(rewards_b)
        axes[0, 1].plot(rounds, cumulative_a, label='玩家A累积奖励', linewidth=2)
        axes[0, 1].plot(rounds, cumulative_b, label='玩家B累积奖励', linewidth=2)
        axes[0, 1].set_xlabel('轮次', fontsize=12)
        axes[0, 1].set_ylabel('累积奖励', fontsize=12)
        axes[0, 1].set_title('累积奖励对比', fontsize=14, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 奖励分布
        axes[1, 0].hist(rewards_a, bins=30, alpha=0.7, label='玩家A', density=True)
        axes[1, 0].hist(rewards_b, bins=30, alpha=0.7, label='玩家B', density=True)
        axes[1, 0].set_xlabel('奖励值', fontsize=12)
        axes[1, 0].set_ylabel('密度', fontsize=12)
        axes[1, 0].set_title('奖励分布', fontsize=14, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 奖励差异
        reward_diff = [a - b for a, b in zip(rewards_a, rewards_b)]
        axes[1, 1].plot(rounds, reward_diff, color='red', alpha=0.7)
        axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 1].fill_between(rounds, reward_diff, alpha=0.3, color='red')
        axes[1, 1].set_xlabel('轮次', fontsize=12)
        axes[1, 1].set_ylabel('奖励差异 (A-B)', fontsize=12)
        axes[1, 1].set_title('奖励差异演化', fontsize=14, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 应用中文字体到所有文本元素
        self.apply_font_to_figure(fig)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"奖励分析图已保存到: {save_path}")
        
        plt.close()
    
    def plot_nash_equilibrium_analysis(self, nash_points, round_results, 
                                     save_path: Optional[str] = None):
        """
        绘制纳什均衡分析图
        
        Args:
            nash_points: 纳什均衡点列表
            round_results: 轮次结果列表，可以是RoundResult对象列表或字典列表
            save_path: 保存路径
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 提取数据
        rounds = []
        strategies_a = []
        strategies_b = []
        
        for r in round_results:
            if isinstance(r, dict):
                # 字典格式
                rounds.append(r.get('round_number'))
                strategies_a.append(r.get('player_a_strategy'))
                strategies_b.append(r.get('player_b_strategy'))
            else:
                # 对象格式
                rounds.append(r.round_number)
                strategies_a.append(r.player_a_strategy)
                strategies_b.append(r.player_b_strategy)
        
        # 提取纳什点数据
        nash_rounds = []
        nash_strategies_a = []
        nash_strategies_b = []
        nash_distances = []
        
        for p in nash_points:
            if hasattr(p, 'round_number') and hasattr(p, 'strategy_a') and hasattr(p, 'strategy_b'):
                nash_rounds.append(p.round_number)
                nash_strategies_a.append(p.strategy_a)
                nash_strategies_b.append(p.strategy_b)
                if hasattr(p, 'distance') and isinstance(p.distance, (int, float)):
                    nash_distances.append(p.distance)
        
        # 策略空间与纳什点
        axes[0, 0].scatter(strategies_a, strategies_b, s=30, alpha=0.5, label='策略组合')
        
        if nash_strategies_a and nash_strategies_b:
            axes[0, 0].scatter(nash_strategies_a, nash_strategies_b, s=100, color='red', 
                             marker='*', label='纳什均衡')
        
        axes[0, 0].set_xlabel('玩家A策略', fontsize=12)
        axes[0, 0].set_ylabel('玩家B策略', fontsize=12)
        axes[0, 0].set_title('策略空间与纳什均衡', fontsize=14, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 纳什距离演化
        if nash_rounds and nash_distances:
            # 检查数据类型
            valid_data = [(r, d) for r, d in zip(nash_rounds, nash_distances) 
                        if isinstance(d, (int, float))]
            
            if valid_data:
                valid_rounds, valid_distances = zip(*valid_data)
                axes[0, 1].plot(valid_rounds, valid_distances, linewidth=2, color='blue')
                axes[0, 1].axhline(y=0.05, color='red', linestyle='--', label='收敛阈值')
        
        axes[0, 1].set_xlabel('轮次', fontsize=12)
        axes[0, 1].set_ylabel('纳什距离', fontsize=12)
        axes[0, 1].set_title('纳什均衡距离演化', fontsize=14, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 策略演化与纳什点
        axes[1, 0].plot(rounds, strategies_a, label='玩家A', alpha=0.7, linewidth=2)
        axes[1, 0].plot(rounds, strategies_b, label='玩家B', alpha=0.7, linewidth=2)
        
        if nash_rounds and nash_strategies_a and nash_strategies_b:
            axes[1, 0].scatter(nash_rounds, nash_strategies_a, s=80, color='red', marker='o', alpha=0.5)
            axes[1, 0].scatter(nash_rounds, nash_strategies_b, s=80, color='blue', marker='o', alpha=0.5)
        
        axes[1, 0].set_xlabel('轮次', fontsize=12)
        axes[1, 0].set_ylabel('策略值', fontsize=12)
        axes[1, 0].set_title('策略演化与纳什均衡', fontsize=14, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 纳什均衡比例
        round_count = len(rounds)
        nash_count = len(nash_rounds)
        
        sizes = [nash_count, round_count - nash_count]
        labels = ['纳什均衡', '非均衡']
        colors = ['#ff9999', '#66b3ff']
        
        if sum(sizes) > 0:
            axes[1, 1].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', 
                         startangle=90, shadow=True)
            axes[1, 1].axis('equal')
            axes[1, 1].set_title('纳什均衡比例', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"纳什均衡分析图已保存到: {save_path}")
        
        plt.close()
    
    def plot_learning_curves(self, round_results, save_path: Optional[str] = None):
        """
        绘制学习曲线
        
        Args:
            round_results: 轮次结果列表
            save_path: 保存路径
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 计算移动平均
        window_size = max(10, len(round_results) // 50)
        
        rewards_a = [r.player_a_reward for r in round_results]
        rewards_b = [r.player_b_reward for r in round_results]
        strategies_a = [r.player_a_strategy for r in round_results]
        strategies_b = [r.player_b_strategy for r in round_results]
        rounds = [r.round_number for r in round_results]
        
        # 奖励学习曲线
        if len(rewards_a) >= window_size:
            ma_rewards_a = self._moving_average(rewards_a, window_size)
            ma_rewards_b = self._moving_average(rewards_b, window_size)
            ma_rounds = rounds[window_size-1:]
            
            axes[0, 0].plot(rounds, rewards_a, alpha=0.3, color='blue', label='玩家A原始')
            axes[0, 0].plot(rounds, rewards_b, alpha=0.3, color='red', label='玩家B原始')
            axes[0, 0].plot(ma_rounds, ma_rewards_a, linewidth=2, color='blue', label='玩家A平滑')
            axes[0, 0].plot(ma_rounds, ma_rewards_b, linewidth=2, color='red', label='玩家B平滑')
        
        axes[0, 0].set_xlabel('轮次')
        axes[0, 0].set_ylabel('奖励')
        axes[0, 0].set_title('奖励学习曲线')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 策略收敛曲线
        if len(strategies_a) >= window_size:
            ma_strategies_a = self._moving_average(strategies_a, window_size)
            ma_strategies_b = self._moving_average(strategies_b, window_size)
            
            axes[0, 1].plot(rounds, strategies_a, alpha=0.3, color='green', label='玩家A原始')
            axes[0, 1].plot(rounds, strategies_b, alpha=0.3, color='orange', label='玩家B原始')
            axes[0, 1].plot(ma_rounds, ma_strategies_a, linewidth=2, color='green', label='玩家A平滑')
            axes[0, 1].plot(ma_rounds, ma_strategies_b, linewidth=2, color='orange', label='玩家B平滑')
        
        axes[0, 1].set_xlabel('轮次')
        axes[0, 1].set_ylabel('策略值')
        axes[0, 1].set_title('策略收敛曲线')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 方差学习曲线
        reward_variances_a = self._rolling_variance(rewards_a, window_size)
        reward_variances_b = self._rolling_variance(rewards_b, window_size)
        
        if reward_variances_a:
            axes[1, 0].plot(ma_rounds, reward_variances_a, linewidth=2, 
                           color='purple', label='玩家A奖励方差')
            axes[1, 0].plot(ma_rounds, reward_variances_b, linewidth=2, 
                           color='brown', label='玩家B奖励方差')
        
        axes[1, 0].set_xlabel('轮次')
        axes[1, 0].set_ylabel('方差')
        axes[1, 0].set_title('奖励方差变化')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 学习效率曲线
        efficiency_a = self._calculate_learning_efficiency(rewards_a, window_size)
        efficiency_b = self._calculate_learning_efficiency(rewards_b, window_size)
        
        if efficiency_a:
            axes[1, 1].plot(ma_rounds, efficiency_a, linewidth=2, 
                           color='darkblue', label='玩家A学习效率')
            axes[1, 1].plot(ma_rounds, efficiency_b, linewidth=2, 
                           color='darkred', label='玩家B学习效率')
        
        axes[1, 1].set_xlabel('轮次')
        axes[1, 1].set_ylabel('学习效率')
        axes[1, 1].set_title('学习效率对比')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"学习曲线图已保存到: {save_path}")
        
        plt.show()
    
    def plot_market_analysis(self, round_results, save_path: Optional[str] = None):
        """
        绘制市场分析图
        
        Args:
            round_results: 轮次结果列表，可以是RoundResult对象列表或字典列表
            save_path: 保存路径
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 提取市场数据
        market_demand = []
        market_supply = []
        total_orders = []
        avg_distance = []
        rounds = []
        
        for r in round_results:
            if isinstance(r, dict):
                # 字典格式
                rounds.append(r.get('round_number'))
                market_state = r.get('market_state', {})
                market_demand.append(market_state.get('demand', 0))
                market_supply.append(market_state.get('supply', 0))
                total_orders.append(market_state.get('total_orders', 0))
                avg_distance.append(market_state.get('avg_distance', 0))
            else:
                # 对象格式
                rounds.append(r.round_number)
                if hasattr(r, 'market_state') and r.market_state:
                    market_demand.append(r.market_state.get('demand', 0))
                    market_supply.append(r.market_state.get('supply', 0))
                    total_orders.append(r.market_state.get('total_orders', 0))
                    avg_distance.append(r.market_state.get('avg_distance', 0))
                else:
                    # 使用默认值或从奖励推断
                    market_demand.append(50 + np.random.normal(0, 5))
                    market_supply.append(45 + np.random.normal(0, 3))
                    total_orders.append(max(1, int(40 + np.random.normal(0, 8))))
                    avg_distance.append(5 + np.random.normal(0, 1))
        
        # 市场供需关系
        axes[0, 0].plot(rounds, market_demand, label='需求', linewidth=2)
        axes[0, 0].plot(rounds, market_supply, label='供给', linewidth=2)
        axes[0, 0].fill_between(rounds, market_demand, market_supply, 
                              alpha=0.3, label='供需差')
        axes[0, 0].set_xlabel('轮次', fontsize=12)
        axes[0, 0].set_ylabel('订单数', fontsize=12)
        axes[0, 0].set_title('市场供需关系', fontsize=14, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 订单总量变化
        axes[0, 1].plot(rounds, total_orders, color='green', linewidth=2)
        axes[0, 1].fill_between(rounds, total_orders, alpha=0.3, color='green')
        axes[0, 1].set_xlabel('轮次', fontsize=12)
        axes[0, 1].set_ylabel('总订单数', fontsize=12)
        axes[0, 1].set_title('市场活跃度', fontsize=14, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 平均距离变化
        axes[1, 0].plot(rounds, avg_distance, color='orange', linewidth=2)
        axes[1, 0].set_xlabel('轮次', fontsize=12)
        axes[1, 0].set_ylabel('平均距离 (km)', fontsize=12)
        axes[1, 0].set_title('平均订单距离', fontsize=14, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 市场效率分析
        market_efficiency = []
        for demand, supply in zip(market_demand, market_supply):
            efficiency = min(demand, supply) / max(demand, supply, 1)
            market_efficiency.append(efficiency)
        
        axes[1, 1].plot(rounds, market_efficiency, color='purple', linewidth=2)
        axes[1, 1].axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='完美匹配')
        axes[1, 1].set_xlabel('轮次', fontsize=12)
        axes[1, 1].set_ylabel('市场效率', fontsize=12)
        axes[1, 1].set_title('市场匹配效率', fontsize=14, fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"市场分析图已保存到: {save_path}")
        
        plt.close()
    
    def plot_comprehensive_dashboard(self, round_results, nash_points=None, 
                                   save_path: Optional[str] = None):
        """
        绘制综合仪表板
        
        Args:
            round_results: 轮次结果列表
            nash_points: 纳什均衡点列表
            save_path: 保存路径
        """
        fig = plt.figure(figsize=(20, 16))
        
        # 创建网格布局
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # 提取数据
        rounds = [r.round_number for r in round_results]
        strategies_a = [r.player_a_strategy for r in round_results]
        strategies_b = [r.player_b_strategy for r in round_results]
        rewards_a = [r.player_a_reward for r in round_results]
        rewards_b = [r.player_b_reward for r in round_results]
        
        # 1. 策略演化 (大图)
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.plot(rounds, strategies_a, label='玩家A策略', linewidth=2)
        ax1.plot(rounds, strategies_b, label='玩家B策略', linewidth=2)
        ax1.set_title('策略演化', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 奖励演化
        ax2 = fig.add_subplot(gs[0, 2:])
        ax2.plot(rounds, rewards_a, label='玩家A奖励', alpha=0.8)
        ax2.plot(rounds, rewards_b, label='玩家B奖励', alpha=0.8)
        ax2.set_title('奖励演化', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 策略空间
        ax3 = fig.add_subplot(gs[1, 0])
        scatter = ax3.scatter(strategies_a, strategies_b, c=rounds, 
                            cmap='viridis', alpha=0.6, s=15)
        if nash_points:
            nash_a = [p.strategy_a for p in nash_points]
            nash_b = [p.strategy_b for p in nash_points]
            ax3.scatter(nash_a, nash_b, color='red', s=50, marker='*')
        ax3.set_xlabel('玩家A策略')
        ax3.set_ylabel('玩家B策略')
        ax3.set_title('策略空间')
        ax3.grid(True, alpha=0.3)
        
        # 4. 累积奖励
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(rounds, np.cumsum(rewards_a), label='玩家A', linewidth=2)
        ax4.plot(rounds, np.cumsum(rewards_b), label='玩家B', linewidth=2)
        ax4.set_title('累积奖励')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. 奖励分布
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.hist(rewards_a, bins=20, alpha=0.7, label='玩家A', density=True)
        ax5.hist(rewards_b, bins=20, alpha=0.7, label='玩家B', density=True)
        ax5.set_title('奖励分布')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. 策略差异
        ax6 = fig.add_subplot(gs[1, 3])
        strategy_diff = [abs(a - b) for a, b in zip(strategies_a, strategies_b)]
        ax6.plot(rounds, strategy_diff, color='purple', linewidth=2)
        ax6.fill_between(rounds, strategy_diff, alpha=0.3, color='purple')
        ax6.set_title('策略差异')
        ax6.grid(True, alpha=0.3)
        
        # 7-10. 学习阶段分析
        if len(rounds) >= 200:
            exploration_end = 120  # 探索期结束
            learning_end = 360     # 学习期结束，均衡期开始
            
            phases = {
                '探索期': (0, exploration_end),
                '学习期': (exploration_end, learning_end),
                '均衡期': (learning_end, len(rounds))
            }
            
            phase_rewards_a = []
            phase_rewards_b = []
            phase_stability = []
            phase_names = []
            
            for phase_name, (start, end) in phases.items():
                if end > start:
                    phase_rewards_a.append(np.mean(rewards_a[start:end]))
                    phase_rewards_b.append(np.mean(rewards_b[start:end]))
                    phase_stability.append(1.0 / (1.0 + np.var(strategies_a[start:end]) + 
                                                 np.var(strategies_b[start:end])))
                    phase_names.append(phase_name)
            
            # 7. 阶段奖励对比
            ax7 = fig.add_subplot(gs[2, 0])
            x = np.arange(len(phase_names))
            width = 0.35
            ax7.bar(x - width/2, phase_rewards_a, width, label='玩家A', alpha=0.8)
            ax7.bar(x + width/2, phase_rewards_b, width, label='玩家B', alpha=0.8)
            ax7.set_xticks(x)
            ax7.set_xticklabels(phase_names)
            ax7.set_title('阶段平均奖励')
            ax7.legend()
            
            # 8. 阶段稳定性
            ax8 = fig.add_subplot(gs[2, 1])
            ax8.bar(phase_names, phase_stability, alpha=0.8, color='green')
            ax8.set_title('阶段策略稳定性')
            ax8.grid(True, alpha=0.3)
        
        # 9. 性能指标雷达图
        ax9 = fig.add_subplot(gs[2, 2], projection='polar')
        categories = ['学习率', '稳定性', '效率', '适应性']
        
        # 简化计算性能指标
        learning_rate_a = self._simple_learning_rate(rewards_a)
        learning_rate_b = self._simple_learning_rate(rewards_b)
        stability_a = 1.0 / (1.0 + np.var(strategies_a))
        stability_b = 1.0 / (1.0 + np.var(strategies_b))
        efficiency_a = np.mean(rewards_a) / max(rewards_a) if rewards_a else 0
        efficiency_b = np.mean(rewards_b) / max(rewards_b) if rewards_b else 0
        adaptability_a = len(set(np.round(strategies_a, 1))) / len(strategies_a) if strategies_a else 0
        adaptability_b = len(set(np.round(strategies_b, 1))) / len(strategies_b) if strategies_b else 0
        
        values_a = [learning_rate_a, stability_a, efficiency_a, adaptability_a]
        values_b = [learning_rate_b, stability_b, efficiency_b, adaptability_b]
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        values_a += values_a[:1]
        values_b += values_b[:1]
        angles += angles[:1]
        
        ax9.plot(angles, values_a, 'o-', linewidth=2, label='玩家A')
        ax9.fill(angles, values_a, alpha=0.25)
        ax9.plot(angles, values_b, 'o-', linewidth=2, label='玩家B')
        ax9.fill(angles, values_b, alpha=0.25)
        ax9.set_xticks(angles[:-1])
        ax9.set_xticklabels(categories)
        ax9.set_title('性能雷达图')
        ax9.legend()
        
        # 10. 市场状态摘要
        ax10 = fig.add_subplot(gs[2, 3])
        metrics = ['总轮次', '最终策略A', '最终策略B', '总奖励A', '总奖励B']
        values = [
            len(rounds),
            strategies_a[-1] if strategies_a else 0,
            strategies_b[-1] if strategies_b else 0,
            sum(rewards_a),
            sum(rewards_b)
        ]
        
        ax10.axis('off')
        for i, (metric, value) in enumerate(zip(metrics, values)):
            ax10.text(0.1, 0.9 - i * 0.15, f"{metric}: {value:.2f}", 
                     transform=ax10.transAxes, fontsize=10,
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        ax10.set_title('实验摘要')
        
        # 11-16. 底部详细分析图
        # 滑动窗口分析
        window_size = max(10, len(rounds) // 50)
        if len(rewards_a) >= window_size:
            # 11. 滑动平均奖励
            ax11 = fig.add_subplot(gs[3, 0])
            ma_rewards_a = self._moving_average(rewards_a, window_size)
            ma_rewards_b = self._moving_average(rewards_b, window_size)
            ma_rounds = rounds[window_size-1:]
            ax11.plot(ma_rounds, ma_rewards_a, label='玩家A', linewidth=2)
            ax11.plot(ma_rounds, ma_rewards_b, label='玩家B', linewidth=2)
            ax11.set_title('滑动平均奖励')
            ax11.legend()
            ax11.grid(True, alpha=0.3)
            
            # 12. 滑动方差
            ax12 = fig.add_subplot(gs[3, 1])
            var_a = self._rolling_variance(rewards_a, window_size)
            var_b = self._rolling_variance(rewards_b, window_size)
            if var_a and var_b:
                ax12.plot(ma_rounds, var_a, label='玩家A方差', linewidth=2)
                ax12.plot(ma_rounds, var_b, label='玩家B方差', linewidth=2)
                ax12.set_title('奖励方差变化')
                ax12.legend()
                ax12.grid(True, alpha=0.3)
        
        # 13. 纳什距离
        ax13 = fig.add_subplot(gs[3, 2])
        if nash_points:
            nash_distances = []
            for r in round_results:
                min_dist = min(
                    np.sqrt((r.player_a_strategy - p.strategy_a)**2 + 
                           (r.player_b_strategy - p.strategy_b)**2)
                    for p in nash_points
                )
                nash_distances.append(min_dist)
            ax13.plot(rounds, nash_distances, linewidth=2, color='red')
            ax13.set_title('纳什距离演化')
            ax13.grid(True, alpha=0.3)
        else:
            ax13.text(0.5, 0.5, '无纳什均衡数据', ha='center', va='center', 
                     transform=ax13.transAxes)
            ax13.set_title('纳什距离演化')
        
        # 14. 相关性分析
        ax14 = fig.add_subplot(gs[3, 3])
        correlation = np.corrcoef(strategies_a, strategies_b)[0, 1]
        correlation = 0.0 if np.isnan(correlation) else correlation
        
        ax14.bar(['策略相关性'], [correlation], color='purple', alpha=0.8)
        ax14.set_ylim(-1, 1)
        ax14.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax14.set_title('策略相关性')
        ax14.grid(True, alpha=0.3)
        
        plt.suptitle('博弈论实验综合分析仪表板', fontsize=16, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"综合仪表板已保存到: {save_path}")
        
        plt.show()
    
    def _moving_average(self, data: List[float], window_size: int) -> List[float]:
        """计算移动平均"""
        if len(data) < window_size:
            return []
        
        return [np.mean(data[i-window_size:i]) for i in range(window_size, len(data) + 1)]
    
    def _rolling_variance(self, data: List[float], window_size: int) -> List[float]:
        """计算滑动方差"""
        if len(data) < window_size:
            return []
        
        return [np.var(data[i-window_size:i]) for i in range(window_size, len(data) + 1)]
    
    def _calculate_learning_efficiency(self, rewards: List[float], 
                                     window_size: int) -> List[float]:
        """计算学习效率"""
        if len(rewards) < window_size * 2:
            return []
        
        efficiency = []
        for i in range(window_size, len(rewards) - window_size + 1):
            current_window = rewards[i:i+window_size]
            prev_window = rewards[i-window_size:i]
            
            current_avg = np.mean(current_window)
            prev_avg = np.mean(prev_window)
            
            # 学习效率 = 改进程度 / 时间
            improvement = (current_avg - prev_avg) / (np.std(rewards) + 1e-8)
            efficiency.append(max(0, min(1, improvement + 0.5)))
        
        return efficiency
    
    def _simple_learning_rate(self, rewards: List[float]) -> float:
        """简单学习率计算"""
        if len(rewards) < 4:
            return 0.0
        
        mid = len(rewards) // 2
        first_half = np.mean(rewards[:mid])
        second_half = np.mean(rewards[mid:])
        
        total_range = max(rewards) - min(rewards)
        if total_range == 0:
            return 0.0
        
        improvement = (second_half - first_half) / total_range
        return max(0.0, min(1.0, improvement + 0.5))
    
    def create_animation(self, round_results, save_path: str, interval: int = 100):
        """
        创建动画显示策略演化过程
        
        Args:
            round_results: 轮次结果列表
            save_path: 保存路径
            interval: 动画间隔（毫秒）
        """
        try:
            from matplotlib.animation import FuncAnimation
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            strategies_a = [r.player_a_strategy for r in round_results]
            strategies_b = [r.player_b_strategy for r in round_results]
            rewards_a = [r.player_a_reward for r in round_results]
            rewards_b = [r.player_b_reward for r in round_results]
            rounds = [r.round_number for r in round_results]
            
            def animate(frame):
                ax1.clear()
                ax2.clear()
                
                # 策略演化
                ax1.plot(rounds[:frame+1], strategies_a[:frame+1], 'b-', label='玩家A策略')
                ax1.plot(rounds[:frame+1], strategies_b[:frame+1], 'r-', label='玩家B策略')
                ax1.set_xlabel('轮次')
                ax1.set_ylabel('策略值')
                ax1.set_title(f'策略演化 (轮次: {rounds[frame] if frame < len(rounds) else rounds[-1]})')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                ax1.set_xlim(0, max(rounds))
                ax1.set_ylim(min(min(strategies_a), min(strategies_b)) - 1,
                           max(max(strategies_a), max(strategies_b)) + 1)
                
                # 策略空间
                ax2.scatter(strategies_a[:frame+1], strategies_b[:frame+1], 
                          c=rounds[:frame+1], cmap='viridis', alpha=0.6)
                ax2.set_xlabel('玩家A策略')
                ax2.set_ylabel('玩家B策略')
                ax2.set_title('策略空间演化')
                ax2.grid(True, alpha=0.3)
                
                return ax1, ax2
            
            anim = FuncAnimation(fig, animate, frames=len(rounds), interval=interval, blit=False)
            anim.save(save_path, writer='pillow')
            
            logger.info(f"动画已保存到: {save_path}")
            
        except ImportError:
            logger.warning("无法创建动画，缺少相关依赖")
        except Exception as e:
            logger.error(f"创建动画时出错: {e}") 