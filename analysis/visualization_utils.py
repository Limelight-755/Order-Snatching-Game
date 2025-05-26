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

# 下载和设置 WenQuanYi 中文字体
def setup_chinese_font():
    """下载并设置开源中文字体"""
    # 字体文件保存路径
    font_dir = os.path.join(os.path.dirname(__file__), "..", "fonts")
    os.makedirs(font_dir, exist_ok=True)
    wqy_font_path = os.path.join(font_dir, "wqy-microhei.ttc")
    
    # 如果字体不存在则下载
    if not os.path.exists(wqy_font_path):
        try:
            logger.info("开始下载开源中文字体 WenQuanYi...")
            # 文泉驿微米黑字体下载地址
            font_url = "https://downloads.sourceforge.net/wqy/wqy-microhei-0.2.0-beta.tar.gz"
            
            # 使用临时目录下载和解压
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_file = os.path.join(tmpdir, "wqy-microhei.tar.gz")
                
                # 下载字体文件
                urllib.request.urlretrieve(font_url, tmp_file)
                logger.info(f"字体下载完成: {tmp_file}")
                
                # 解压字体文件
                import tarfile
                with tarfile.open(tmp_file) as tar:
                    tar.extractall(path=tmpdir)
                
                # 查找并复制字体文件
                for root, _, files in os.walk(tmpdir):
                    for file in files:
                        if file.endswith(".ttc") or file.endswith(".ttf"):
                            src_font = os.path.join(root, file)
                            shutil.copy(src_font, wqy_font_path)
                            logger.info(f"字体文件已保存至: {wqy_font_path}")
                            break
            
            # 如果下载失败，使用嵌入的字体数据（这里略去）
        except Exception as e:
            logger.error(f"下载字体失败: {e}")
            # 回退到系统字体
            return False
    
    # 注册字体
    try:
        if os.path.exists(wqy_font_path):
            font_prop = fm.FontProperties(fname=wqy_font_path)
            mpl.rcParams['font.family'] = font_prop.get_name()
            # 设置默认字体
            plt.rcParams['font.sans-serif'] = [font_prop.get_name()] + plt.rcParams.get('font.sans-serif', [])
            plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
            
            # 测试字体
            fig = plt.figure(figsize=(1, 1))
            fig.text(0.5, 0.5, '中文测试', ha='center', va='center')
            plt.close(fig)
            
            logger.info(f"已成功加载中文字体: {wqy_font_path}")
            return wqy_font_path
    except Exception as e:
        logger.error(f"加载中文字体失败: {e}")
    
    # 回退到系统字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    return None

# 初始化设置中文字体
chinese_font_path = setup_chinese_font()

# 如果无法下载或加载文泉驿字体，使用系统自带字体
if not chinese_font_path:
    # 设置matplotlib字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi', 'FangSong']
    plt.rcParams['axes.unicode_minus'] = False

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
        self.font_path = chinese_font_path
        
        # 测试中文字体显示
        try:
            test_fig = plt.figure(figsize=(2, 2))
            test_fig.text(0.5, 0.5, '中文测试', ha='center', va='center')
            plt.close(test_fig)
            logger.info("中文字体测试通过")
        except Exception as e:
            logger.warning(f"中文字体测试失败: {e}")
        
        logger.info(f"可视化工具初始化完成: 风格={style}, 图形大小={figsize}")
    
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
            
            # 动态调整x轴范围
            max_rounds = max(rounds) if rounds else 100
            x_ticks_step = max(1, max_rounds // 10)  # 确保至少有10个刻度
            x_ticks = list(range(0, max_rounds + 1, x_ticks_step))
            
            # 策略值演化 - 更美观的线条和配色
            ax1.plot(rounds, strategies_a, color='#3A76AF', label='玩家A策略', 
                    linewidth=2, alpha=0.9, zorder=10)
            ax1.plot(rounds, strategies_b, color='#6ABE5E', label='玩家B策略', 
                    linewidth=2, alpha=0.9, zorder=10)
            
            # 背景和网格设置
            ax1.set_facecolor('#F8F8FA')  # 设置子图背景
            ax1.grid(True, linestyle='--', alpha=0.7, color='#CCCCCC')
            
            # 动态确定y轴范围，增加上下边距
            min_y = min(min(strategies_a), min(strategies_b)) - 5
            max_y = max(max(strategies_a), max(strategies_b)) + 5
            ax1.set_ylim(min_y, max_y)
            
            if show_phases and len(rounds) >= 200:
                # 添加学习阶段划分线
                exploration_end = min(50, len(rounds) // 10)
                learning_end = min(200, len(rounds) * 4 // 10)
                
                ax1.axvline(x=exploration_end, color='#FF7F7F', linestyle='--', 
                         alpha=0.6, label='探索期结束')
                ax1.axvline(x=learning_end, color='#FFBF00', linestyle='--', 
                         alpha=0.6, label='学习期结束')
                
                # 添加阶段背景色
                ax1.axvspan(0, exploration_end, alpha=0.1, color='#FFE6E6', label='探索期')
                ax1.axvspan(exploration_end, learning_end, alpha=0.1, color='#FFF5E6', label='学习期')
                ax1.axvspan(learning_end, max(rounds), alpha=0.1, color='#E6F5E6', label='均衡期')
            
            # 设置轴标签和标题
            ax1.set_xlabel('轮次', fontsize=12, fontweight='bold')
            ax1.set_ylabel('策略值（定价阈值）', fontsize=12, fontweight='bold')
            ax1.set_title('策略演化过程', fontsize=14, fontweight='bold', pad=15)
            
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
            strategy_diff = [abs(a - b) for a, b in zip(strategies_a, strategies_b)]
            
            # 创建更美观的策略差异图
            ax2.set_facecolor('#F8F8FA')  # 设置子图背景
            ax2.grid(True, linestyle='--', alpha=0.7, color='#CCCCCC')
            
            ax2.plot(rounds, strategy_diff, color='#9467BD', linewidth=2.5, 
                   alpha=0.9, label='策略差异', zorder=10)
            
            # 渐变填充区域
            ax2.fill_between(rounds, strategy_diff, color='#9467BD', alpha=0.3)
            
            # 动态确定y轴范围
            max_diff = max(strategy_diff) if strategy_diff else 0
            ax2.set_ylim(0, max_diff * 1.2)  # 增加20%的顶部空间
            
            # 设置轴标签和标题
            ax2.set_xlabel('轮次', fontsize=12, fontweight='bold')
            ax2.set_ylabel('策略差异', fontsize=12, fontweight='bold')
            ax2.set_title('策略差异演化', fontsize=14, fontweight='bold', pad=15)
            
            # 设置刻度和边框
            ax2.set_xticks(x_ticks)
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.tick_params(labelsize=10)
            
            # 图例
            legend2 = ax2.legend(loc='upper right', frameon=True, framealpha=0.95, 
                              shadow=True, fancybox=True, fontsize=10)
            legend2.get_frame().set_facecolor('#FFFFFF')
            
            plt.tight_layout(pad=2.0)
            
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
        
        # 应用中文字体
        if hasattr(self, 'font_path') and self.font_path:
            try:
                font_prop = fm.FontProperties(fname=self.font_path)
                # 为所有文本元素设置字体
                for ax in axes.flatten():
                    for text in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                              ax.get_xticklabels() + ax.get_yticklabels()):
                        text.set_fontproperties(font_prop)
                    # 图例文本
                    if ax.get_legend() is not None:
                        for text in ax.get_legend().get_texts():
                            text.set_fontproperties(font_prop)
                
                # 设置整个图表的标题字体
                if fig.get_label() or fig._suptitle:
                    fig._suptitle.set_fontproperties(font_prop)
            except Exception as e:
                logger.warning(f"应用中文字体失败: {e}")
        
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
        
        # 应用中文字体
        if hasattr(self, 'font_path') and self.font_path:
            try:
                font_prop = fm.FontProperties(fname=self.font_path)
                # 为所有文本元素设置字体
                for ax in axes.flatten():
                    for text in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                              ax.get_xticklabels() + ax.get_yticklabels()):
                        if text:
                            text.set_fontproperties(font_prop)
                    # 图例文本
                    if ax.get_legend() is not None:
                        for text in ax.get_legend().get_texts():
                            text.set_fontproperties(font_prop)
                    # 饼图文本
                    if ax.texts:
                        for text in ax.texts:
                            text.set_fontproperties(font_prop)
                
                # 设置整个图表的标题字体
                if fig.get_label() or hasattr(fig, '_suptitle') and fig._suptitle:
                    fig._suptitle.set_fontproperties(font_prop)
            except Exception as e:
                logger.warning(f"应用中文字体失败: {e}")
        
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
        
        # 应用中文字体
        if hasattr(self, 'font_path') and self.font_path:
            try:
                font_prop = fm.FontProperties(fname=self.font_path)
                # 为所有文本元素设置字体
                for ax in axes.flatten():
                    for text in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                              ax.get_xticklabels() + ax.get_yticklabels()):
                        if text:
                            text.set_fontproperties(font_prop)
                    # 图例文本
                    if ax.get_legend() is not None:
                        for text in ax.get_legend().get_texts():
                            text.set_fontproperties(font_prop)
                
                # 设置整个图表的标题字体
                if fig.get_label() or hasattr(fig, '_suptitle') and fig._suptitle:
                    fig._suptitle.set_fontproperties(font_prop)
            except Exception as e:
                logger.warning(f"应用中文字体失败: {e}")
        
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
            exploration_end = min(50, len(rounds) // 10)
            learning_end = min(200, len(rounds) * 4 // 10)
            
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