"""
对称博弈实验
两个相同能力的司机在相同的市场条件下进行博弈
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging
import os
import matplotlib.font_manager as fm

from experiments.experiment_utils import (
    ExperimentConfig, ExperimentResult, RoundResult,
    ExperimentLogger, DataCollector
)
from core.game_framework import GameFramework
from core.market_environment import MarketEnvironment
from ai_models.dqn_agent import DQNAgent
from ai_models.lstm_predictor import StrategyPredictor
from config.game_config import GameConfig

logger = logging.getLogger(__name__)

# 中文字体路径
chinese_font_path = r"C:\Windows\Fonts\STZHONGS.TTF"


class SymmetricGameExperiment:
    """
    对称博弈实验类
    实现两个相同能力玩家的博弈实验
    """
    
    def __init__(self, config: ExperimentConfig):
        """
        初始化对称博弈实验
        
        Args:
            config: 实验配置
        """
        self.config = config
        self.experiment_logger = ExperimentLogger(config.experiment_name)
        self.data_collector = DataCollector()
        
        # 初始化游戏组件
        self.game_config = GameConfig()
        self.market_env = MarketEnvironment(self.game_config)
        self.game_framework = GameFramework(self.game_config)
        
        # 初始化AI智能体
        self._initialize_agents()
        
        # 实验状态
        self.current_round = 0
        self.experiment_results = []
        
        logger.info(f"对称博弈实验初始化完成: {config.experiment_name}")
    
    def _initialize_agents(self):
        """初始化AI智能体"""
        # 获取AI配置参数
        dqn_params = self.config.ai_config.get('dqn_params', {})
        
        # 创建两个相同的DQN智能体
        player_a_config = dqn_params.copy()
        player_a_config['player_id'] = 'player_a'
        self.player_a = DQNAgent(player_a_config)
        
        player_b_config = dqn_params.copy()
        player_b_config['player_id'] = 'player_b'
        self.player_b = DQNAgent(player_b_config)
        
        # 创建策略预测器
        lstm_params = self.config.ai_config.get('lstm_params', {})
        lstm_params['input_size'] = 10  # 历史策略特征
        
        self.predictor_a = StrategyPredictor(lstm_params)
        self.predictor_b = StrategyPredictor(lstm_params)
        
        logger.info("AI智能体初始化完成")
    
    def run_experiment(self) -> ExperimentResult:
        """
        运行完整的对称博弈实验
        
        Returns:
            实验结果
        """
        start_time = datetime.now()
        
        # 创建实验结果对象
        result = ExperimentResult(
            experiment_config=self.config,
            start_time=start_time
        )
        
        self.experiment_logger.log_experiment_start(self.config)
        
        try:
            # 执行多轮实验
            for run_id in range(self.config.num_runs):
                logger.info(f"开始第 {run_id + 1}/{self.config.num_runs} 轮实验")
                
                # 重置环境和智能体
                self._reset_experiment()
                
                # 运行单轮实验
                run_results = self._run_single_experiment()
                
                # 收集结果
                result.round_results.extend(run_results)
                
                logger.info(f"完成第 {run_id + 1} 轮实验，共 {len(run_results)} 个轮次")
            
            # 计算汇总统计
            result.calculate_summary_statistics()
            
            # 记录实验结束
            end_time = datetime.now()
            result.end_time = end_time
            result.total_duration = (end_time - start_time).total_seconds()
            
            self.experiment_logger.log_experiment_end(result)
            
            # 保存结果
            self._save_results(result)
            
            return result
            
        except Exception as e:
            logger.error(f"实验执行出错: {e}")
            raise
    
    def _reset_experiment(self):
        """重置实验环境"""
        # 重置市场环境
        self.market_env.reset()
        
        # 重置智能体
        self.player_a.reset()
        self.player_b.reset()
        
        # 重置轮次计数
        self.current_round = 0
        
        logger.debug("实验环境已重置")
    
    def _run_single_experiment(self) -> List[RoundResult]:
        """
        运行单次实验（500轮博弈）
        
        Returns:
            轮次结果列表
        """
        round_results = []
        
        # 初始化历史数据
        strategy_history_a = []
        strategy_history_b = []
        
        for round_num in range(1, self.config.total_rounds + 1):
            self.current_round = round_num
            
            # 更新智能体轮次信息
            self.player_a.update_round(round_num)
            self.player_b.update_round(round_num)
            
            # 获取当前市场状态
            market_state = self.market_env.get_current_state()
            
            # 准备个人状态信息
            personal_state_a = {
                'recent_revenue': getattr(self, '_last_revenue_a', 0),
                'acceptance_rate': 0.7 + np.random.normal(0, 0.1),  # 模拟接单率
                'waiting_time': np.random.exponential(10),  # 等待时间
                'round_number': round_num,
                'history_features': [0.5, 0.5]  # 简化的历史特征
            }
            
            personal_state_b = {
                'recent_revenue': getattr(self, '_last_revenue_b', 0),
                'acceptance_rate': 0.7 + np.random.normal(0, 0.1),  # 模拟接单率
                'waiting_time': np.random.exponential(10),  # 等待时间
                'round_number': round_num,
                'history_features': [0.5, 0.5]  # 简化的历史特征
            }
            
            # 准备状态向量
            state_vector_a = self._prepare_state_vector(market_state, strategy_history_b)
            state_vector_b = self._prepare_state_vector(market_state, strategy_history_a)
            
            # 智能体决策
            # 玩家A决策
            state_a = self.player_a.preprocess_state(market_state, personal_state_a)
            # 传递对手上一轮策略信息（用于收敛学习）
            opponent_strategy_a = getattr(self, '_last_strategy_b', None)
            action_result_a = self.player_a.select_action(state_a, training=True, 
                                                        opponent_strategy=opponent_strategy_a)
            strategy_a = self.player_a.action_to_strategy(action_result_a.action)
            
            # 玩家B决策  
            state_b = self.player_b.preprocess_state(market_state, personal_state_b)
            # 传递对手上一轮策略信息（用于收敛学习）
            opponent_strategy_b = getattr(self, '_last_strategy_a', None)
            action_result_b = self.player_b.select_action(state_b, training=True,
                                                        opponent_strategy=opponent_strategy_b)
            strategy_b = self.player_b.action_to_strategy(action_result_b.action)
            
            # 执行博弈轮次
            revenue_a, revenue_b = self._execute_round(strategy_a, strategy_b, market_state)
            
            # 计算奖励
            if round_num > 1:
                # 玩家A的奖励计算
                personal_state_a.update({
                    'current_strategy': strategy_a,
                    'round_number': round_num,
                    'strategy_change': abs(strategy_a - getattr(self, '_last_strategy_a', strategy_a))
                })
                reward_a = self.player_a.calculate_reward(
                    revenue_a, getattr(self, '_last_revenue_a', 0), market_state, personal_state_a,
                    opponent_strategy=getattr(self, '_last_strategy_b', None),
                    opponent_revenue=revenue_b  # 传递对手收益信息
                )
                
                # 玩家B的奖励计算  
                personal_state_b.update({
                    'current_strategy': strategy_b,
                    'round_number': round_num,
                    'strategy_change': abs(strategy_b - getattr(self, '_last_strategy_b', strategy_b))
                })
                reward_b = self.player_b.calculate_reward(
                    revenue_b, getattr(self, '_last_revenue_b', 0), market_state, personal_state_b,
                    opponent_strategy=getattr(self, '_last_strategy_a', None),
                    opponent_revenue=revenue_a  # 传递对手收益信息
                )
                
                # 存储经验到智能体记忆中
                if hasattr(self, '_last_state_a') and hasattr(self, '_last_action_a'):
                    self.player_a.store_experience(
                        self._last_state_a, self._last_action_a, reward_a, state_a, False
                    )
                    
                if hasattr(self, '_last_state_b') and hasattr(self, '_last_action_b'):
                    self.player_b.store_experience(
                        self._last_state_b, self._last_action_b, reward_b, state_b, False
                    )
            
            # 保存当前轮次信息用于下一轮
            self._last_strategy_a = strategy_a
            self._last_strategy_b = strategy_b
            self._last_state_a = state_a
            self._last_state_b = state_b
            self._last_action_a = action_result_a.action
            self._last_action_b = action_result_b.action
            
            # 保存收益信息
            self._last_revenue_a = revenue_a
            self._last_revenue_b = revenue_b
            
            # 训练智能体
            if round_num > 1 and round_num % 10 == 0:  # 每10轮训练一次
                self.player_a.train()
                self.player_b.train()
            
            # 更新历史记录
            strategy_history_a.append(strategy_a)
            strategy_history_b.append(strategy_b)
            
            # 保持历史长度
            if len(strategy_history_a) > 20:
                strategy_history_a = strategy_history_a[-20:]
                strategy_history_b = strategy_history_b[-20:]
            
            # 检测纳什均衡
            nash_distance = self.game_framework.check_nash_equilibrium(
                strategies={
                    'player_a': strategy_a,
                    'player_b': strategy_b
                },
                payoffs={
                    'player_a': revenue_a,
                    'player_b': revenue_b
                }
            )[1]['max_beneficial_deviation']  # 返回第二个元素中的最大有利偏差
            
            # 创建轮次结果
            round_result = RoundResult(
                round_number=round_num,
                player_a_strategy=strategy_a,
                player_b_strategy=strategy_b,
                player_a_revenue=revenue_a,
                player_b_revenue=revenue_b,
                market_state=market_state,
                nash_distance=nash_distance
            )
            
            round_results.append(round_result)
            
            # 收集数据
            extra_data = {
                'action_a': action_result_a.action,
                'action_b': action_result_b.action,
            }
            
            # 添加奖励信息（如果存在）
            if round_num > 1:
                extra_data.update({
                    'reward_a': reward_a,
                    'reward_b': reward_b
                })
            
            self.data_collector.collect_round_data(round_result, extra_data)
            
            # 日志记录
            if round_num % 50 == 0:
                self.experiment_logger.log_round_result(round_result)
                logger.info(f"轮次 {round_num}: 策略A={strategy_a:.2f}, 策略B={strategy_b:.2f}")
            
            # 阶段转换检测
            self._check_phase_transition(round_num)
        
        return round_results
    
    def _prepare_state_vector(self, market_state: Dict, opponent_history: List[float]) -> np.ndarray:
        """
        准备状态向量
        
        Args:
            market_state: 市场状态
            opponent_history: 对手策略历史
            
        Returns:
            状态向量
        """
        # 基础市场特征
        features = [
            market_state.get('order_rate', 0) / 100.0,  # 归一化订单率
            market_state.get('avg_price', 30) / 50.0,   # 归一化平均价格
            market_state.get('competition_level', 0.5), # 竞争水平
            market_state.get('time_factor', 1.0),       # 时间因子
            market_state.get('location_factor', 1.0),   # 地理因子
        ]
        
        # 对手策略历史特征（最近10轮）
        recent_history = opponent_history[-10:] if opponent_history else []
        
        # 填充到固定长度
        while len(recent_history) < 10:
            recent_history.insert(0, 30.0)  # 默认策略值
        
        # 归一化对手历史
        normalized_history = [(x - 10) / 40.0 for x in recent_history]
        
        features.extend(normalized_history)
        
        return np.array(features, dtype=np.float32)
    
    def _execute_round(self, strategy_a: float, strategy_b: float, market_state: Dict) -> Tuple[float, float]:
        """
        执行一轮博弈
        
        Args:
            strategy_a: 玩家A策略
            strategy_b: 玩家B策略
            market_state: 市场状态
            
        Returns:
            (玩家A收益, 玩家B收益)
        """
        # 构建司机策略字典
        driver_strategies = {
            'player_a': strategy_a,
            'player_b': strategy_b
        }
        
        # 生成订单
        orders = self.market_env.generate_orders(duration_minutes=30.0)
        
        # 处理订单分配
        driver_orders = self.market_env.process_driver_decisions(orders, driver_strategies)
        
        # 计算收益
        revenues = self.market_env.calculate_driver_revenues(driver_orders, driver_strategies)
        
        # 更新市场状态
        self.market_env.update_market_time()
        
        return revenues['player_a'], revenues['player_b']
    
    def _calculate_reward(self, own_revenue: float, opponent_revenue: float, strategy: float) -> float:
        """
        计算智能体奖励
        
        Args:
            own_revenue: 自己的收益
            opponent_revenue: 对手的收益
            strategy: 自己的策略
            
        Returns:
            奖励值
        """
        # 基础收益奖励
        revenue_reward = own_revenue / 100.0
        
        # 相对收益奖励
        relative_reward = (own_revenue - opponent_revenue) / 100.0
        
        # 策略合理性奖励（避免极端策略）
        strategy_penalty = 0
        if strategy < 15 or strategy > 45:
            strategy_penalty = -0.1
        
        total_reward = revenue_reward * 0.7 + relative_reward * 0.2 + strategy_penalty
        
        return total_reward
    
    def _check_phase_transition(self, round_num: int):
        """检查阶段转换"""
        if round_num == 120:
            self.experiment_logger.log_phase_transition("学习", round_num)
        elif round_num == 360:
            self.experiment_logger.log_phase_transition("均衡", round_num)
    
    def _save_results(self, result: ExperimentResult):
        """保存实验结果"""
        # 保存JSON结果
        result_path = f"results/symmetric_game_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        result.save_to_json(result_path)
        
        # 保存数据CSV
        data_path = f"results/symmetric_game_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self.data_collector.export_to_csv(data_path)
        
        # 生成分析图表
        self._generate_plots(result)
        
        logger.info(f"实验结果已保存到: {result_path}")
    
    def _generate_plots(self, result: ExperimentResult):
        """生成分析图表"""
        try:
            # 策略演化图
            self._plot_strategy_evolution(result)
            
            # 收益对比图
            self._plot_revenue_comparison(result)
            
            # 纳什均衡距离图
            self._plot_nash_distance(result)
            
        except Exception as e:
            logger.warning(f"生成图表时出错: {e}")
    
    def _plot_strategy_evolution(self, result: ExperimentResult):
        """绘制策略演化图"""
        rounds = [r.round_number for r in result.round_results]
        strategies_a = [r.player_a_strategy for r in result.round_results]
        strategies_b = [r.player_b_strategy for r in result.round_results]
        
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
        
        # 直接设置中文字体
        font_prop = None
        if os.path.exists(chinese_font_path):
                font_prop = fm.FontProperties(fname=chinese_font_path)
        
        plt.figure(figsize=(12, 6))
        plt.plot(daily_rounds, daily_strategies_a, label='玩家A策略', alpha=0.7, marker='o', markersize=4)
        plt.plot(daily_rounds, daily_strategies_b, label='玩家B策略', alpha=0.7, marker='s', markersize=4)
        
        # 添加阶段分割线（按天调整）
        plt.axvline(x=120, color='red', linestyle='--', alpha=0.5, label='学习阶段开始')
        plt.axvline(x=360, color='green', linestyle='--', alpha=0.5, label='均衡阶段开始')
        
        plt.xlabel('轮次', fontsize=12)
        plt.ylabel('策略值（元）', fontsize=12)
        plt.title('对称博弈策略演化（日平均值）', fontsize=14, fontweight='bold')
        
        # 应用字体到所有文本元素
        if font_prop:
            for text in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
                text.set_fontproperties(font_prop)
            plt.gca().xaxis.label.set_fontproperties(font_prop)
            plt.gca().yaxis.label.set_fontproperties(font_prop)
            plt.gca().title.set_fontproperties(font_prop)
        
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 创建目录
        os.makedirs(os.path.dirname("results/plots/"), exist_ok=True)
        
        plot_path = f"results/plots/symmetric_strategy_evolution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_path)
        plt.close()
    
    def _plot_revenue_comparison(self, result: ExperimentResult):
        """绘制收益对比图"""
        rounds = [r.round_number for r in result.round_results]
        revenues_a = [r.player_a_revenue for r in result.round_results]
        revenues_b = [r.player_b_revenue for r in result.round_results]
        
        # 直接设置中文字体
        font_prop = None
        if os.path.exists(chinese_font_path):
                font_prop = fm.FontProperties(fname=chinese_font_path)
        
        # 创建图形和子图布局
        fig = plt.figure(figsize=(15, 10))
        
        # 将轮次转换为天
        # 每24小时为1天，收益聚合为每天总收益
        total_rounds = len(rounds)
        days = [r // 24 + 1 for r in rounds]  # 轮次转换为天数（从1开始）
        unique_days = sorted(list(set(days)))
        max_days = max(unique_days)
        
        # 计算每天的总收益
        daily_revenues_a = {}
        daily_revenues_b = {}
        
        for day in unique_days:
            daily_revenues_a[day] = 0
            daily_revenues_b[day] = 0
        
        for i, day in enumerate(days):
            daily_revenues_a[day] += revenues_a[i]
            daily_revenues_b[day] += revenues_b[i]
        
        # 转换为列表形式
        days_list = list(unique_days)
        daily_rev_a_list = [daily_revenues_a[day] for day in days_list]
        daily_rev_b_list = [daily_revenues_b[day] for day in days_list]
        
        # 排除最后一天的异常数据（如果存在超过30天的数据）
        if len(days_list) > 30:
            days_list = days_list[:30]  # 只取前30天
            daily_rev_a_list = daily_rev_a_list[:30]
            daily_rev_b_list = daily_rev_b_list[:30]
            max_days = 30
        
        # 进一步检查和清理异常数据：如果最后一天的收益异常低（可能是实验结束导致的）
        if len(days_list) > 1:
            # 计算倒数第二天的平均收益作为参考
            second_last_avg = (daily_rev_a_list[-2] + daily_rev_b_list[-2]) / 2
            last_day_avg = (daily_rev_a_list[-1] + daily_rev_b_list[-1]) / 2
            
            # 如果最后一天的收益明显异常（低于倒数第二天的20%），则排除
            if last_day_avg < second_last_avg * 0.2:
                days_list = days_list[:-1]
                daily_rev_a_list = daily_rev_a_list[:-1]
                daily_rev_b_list = daily_rev_b_list[:-1]
                max_days = len(days_list)
        
        # 计算每天收益差值的绝对值（先计算每天总收益，再取差值绝对值）
        daily_abs_diff_list = [abs(a - b) for a, b in zip(daily_rev_a_list, daily_rev_b_list)]
        
        # 创建上下两个子图
        ax_top = plt.subplot(2, 1, 1)
        ax_bottom = plt.subplot(2, 1, 2, sharex=ax_top)
        
        # 上部分：收益曲线
        ax_top.plot(days_list, daily_rev_a_list, label='玩家A收益', alpha=0.8, color='#3A76AF', linewidth=1.5)
        ax_top.plot(days_list, daily_rev_b_list, label='玩家B收益', alpha=0.8, color='#6ABE5E', linewidth=1.5)
        ax_top.set_ylabel('日收益（元/天）', fontsize=12)
        ax_top.set_title(f'对称博弈日收益对比 ({max_days}天)', fontsize=14, fontweight='bold')
        ax_top.grid(True, alpha=0.3)
        ax_top.legend(loc='upper right')
        
        # 添加阶段分割线（根据实际天数调整）
        if max_days >= 5:  # 只有当有足够天数时才显示阶段线
            learning_day = 120 // 24 + 1  # 学习阶段开始的天数
            equilibrium_day = 360 // 24 + 1  # 均衡阶段开始的天数
            
            if learning_day <= max_days:
                ax_top.axvline(x=learning_day, color='red', linestyle='--', alpha=0.5, label='学习阶段')
            if equilibrium_day <= max_days:
                ax_top.axvline(x=equilibrium_day, color='green', linestyle='--', alpha=0.5, label='均衡阶段')
        
        # 下部分：收益差值绝对值
        ax_bottom.plot(days_list, daily_abs_diff_list, label='收益差值绝对值', color='#9467BD', alpha=0.8, linewidth=1.5)
        ax_bottom.fill_between(days_list, daily_abs_diff_list, alpha=0.3, color='#9467BD')
        ax_bottom.set_xlabel('天数', fontsize=12)
        ax_bottom.set_ylabel('收益差值绝对值（元/天）', fontsize=12)
        ax_bottom.grid(True, alpha=0.3)
        ax_bottom.legend(loc='upper right')
        
        # 设置x轴刻度为整数天
        if max_days <= 10:
            tick_step = 1
        elif max_days <= 30:
            tick_step = 2
        else:
            tick_step = 5
        ax_top.set_xticks(list(range(1, max_days+1, tick_step)))
        
        # 应用字体到所有文本元素
        if font_prop:
            for ax in [ax_top, ax_bottom]:
                for text in ax.get_xticklabels() + ax.get_yticklabels():
                    text.set_fontproperties(font_prop)
                ax.xaxis.label.set_fontproperties(font_prop)
                ax.yaxis.label.set_fontproperties(font_prop)
                ax.title.set_fontproperties(font_prop)
                if ax.get_legend():
                    for text in ax.get_legend().get_texts():
                        text.set_fontproperties(font_prop)
        
        plt.tight_layout()
        plot_path = f"results/plots/symmetric_revenue_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_path, dpi=300)
        plt.close()
    
    def _plot_nash_distance(self, result: ExperimentResult):
        """绘制纳什均衡距离图"""
        rounds = [r.round_number for r in result.round_results if r.nash_distance is not None]
        distances = [r.nash_distance for r in result.round_results if r.nash_distance is not None]
        
        if not distances:
            return
        
        # 直接设置中文字体
        font_prop = None
        if os.path.exists(chinese_font_path):
                font_prop = fm.FontProperties(fname=chinese_font_path)
        
        plt.figure(figsize=(12, 6))
        plt.plot(rounds, distances, label='纳什均衡距离', color='purple', alpha=0.7)
        plt.axhline(y=0.05, color='red', linestyle='--', alpha=0.5, label='收敛阈值')
        
        plt.xlabel('轮次', fontsize=12)
        plt.ylabel('纳什均衡距离', fontsize=12)
        plt.title('纳什均衡收敛分析', fontsize=14, fontweight='bold')
        
        # 应用字体到所有文本元素
        if font_prop:
            for text in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
                text.set_fontproperties(font_prop)
            plt.gca().xaxis.label.set_fontproperties(font_prop)
            plt.gca().yaxis.label.set_fontproperties(font_prop)
            plt.gca().title.set_fontproperties(font_prop)
            if plt.gca().get_legend():
                for text in plt.gca().get_legend().get_texts():
                    text.set_fontproperties(font_prop)
        
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plot_path = f"results/plots/symmetric_nash_distance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_path)
        plt.close()


def create_symmetric_experiment(total_rounds: int = 500, num_runs: int = 1) -> SymmetricGameExperiment:
    """
    创建对称博弈实验的便捷函数
    
    Args:
        total_rounds: 总轮次数
        num_runs: 重复实验次数
        
    Returns:
        对称博弈实验实例
    """
    config = ExperimentConfig(
        experiment_name="对称博弈实验",
        experiment_type="symmetric",
        total_rounds=total_rounds,
        num_runs=num_runs,
        player_configs={
            'player_a': {'type': 'dqn', 'learning_enabled': True, 'initial_strategy': 30},
            'player_b': {'type': 'dqn', 'learning_enabled': True, 'initial_strategy': 30}
        },
        ai_config={
            'dqn_params': {
                'learning_rate': 0.001,
                'epsilon': 1.0,
                'epsilon_decay': 0.995,
                'epsilon_min': 0.05,
                'memory_size': 10000,
                'batch_size': 32
            },
            'lstm_params': {
                'hidden_size': 64,
                'num_layers': 2,
                'learning_rate': 0.001
            }
        }
    )
    
    return SymmetricGameExperiment(config) 