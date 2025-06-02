"""
非对称博弈实验
两个不同能力或初始条件的司机进行博弈
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


class AsymmetricGameExperiment:
    """
    非对称博弈实验类
    实现两个不同能力玩家的博弈实验
    """
    
    def __init__(self, config: ExperimentConfig):
        """
        初始化非对称博弈实验
        
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
        
        # 初始化AI智能体（非对称配置）
        self._initialize_asymmetric_agents()
        
        # 实验状态
        self.current_round = 0
        self.experiment_results = []
        
        logger.info(f"非对称博弈实验初始化完成: {config.experiment_name}")
    
    def _initialize_asymmetric_agents(self):
        """初始化非对称AI智能体"""
        # 状态空间和动作空间
        state_size = 15
        action_size = 41
        
        # 获取玩家配置
        player_a_config = self.config.player_configs.get('player_a', {})
        player_b_config = self.config.player_configs.get('player_b', {})
        
        # 创建玩家A（可能是新手或保守型）
        dqn_params_a = self.config.ai_config.get('dqn_params', {}).copy()
        dqn_params_a.update({
            'learning_rate': player_a_config.get('learning_rate', 0.001),
            'epsilon': player_a_config.get('initial_epsilon', 1.0),
            'buffer_size': player_a_config.get('memory_size', 5000),  # 较小的经验池
            'state_size': state_size,
            'action_size': action_size,
            'player_id': 'player_a'
        })
        
        self.player_a = DQNAgent(
            config=dqn_params_a
        )
        
        # 创建玩家B（可能是经验丰富或激进型）
        dqn_params_b = self.config.ai_config.get('dqn_params', {}).copy()
        dqn_params_b.update({
            'learning_rate': player_b_config.get('learning_rate', 0.002),
            'epsilon': player_b_config.get('initial_epsilon', 0.7),
            'buffer_size': player_b_config.get('memory_size', 15000),  # 较大的经验池
            'state_size': state_size,
            'action_size': action_size,
            'player_id': 'player_b'
        })
        
        self.player_b = DQNAgent(
            config=dqn_params_b
        )
        
        # 设置不同的市场优势
        self.player_a_advantage = player_a_config.get('market_advantage', 1.0)
        self.player_b_advantage = player_b_config.get('market_advantage', 1.0)
        
        # 创建策略预测器（不同的预测能力）
        lstm_params_a = self.config.ai_config.get('lstm_params', {}).copy()
        lstm_params_a['hidden_size'] = player_a_config.get('predictor_capacity', 32)
        lstm_params_a['input_features'] = 10
        
        lstm_params_b = self.config.ai_config.get('lstm_params', {}).copy()
        lstm_params_b['hidden_size'] = player_b_config.get('predictor_capacity', 64)
        lstm_params_b['input_features'] = 10
        
        self.predictor_a = StrategyPredictor(config=lstm_params_a)
        self.predictor_b = StrategyPredictor(config=lstm_params_b)
        
        logger.info(f"非对称智能体初始化: 玩家A优势={self.player_a_advantage}, 玩家B优势={self.player_b_advantage}")
    
    def run_experiment(self) -> ExperimentResult:
        """
        运行完整的非对称博弈实验
        
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
                logger.info(f"开始第 {run_id + 1}/{self.config.num_runs} 轮非对称实验")
                
                # 重置环境和智能体
                self._reset_experiment()
                
                # 运行单轮实验
                run_results = self._run_single_experiment()
                
                # 收集结果
                result.round_results.extend(run_results)
                
                logger.info(f"完成第 {run_id + 1} 轮实验，共 {len(run_results)} 个轮次")
            
            # 计算汇总统计
            result.calculate_summary_statistics()
            
            # 分析非对称特性
            self._analyze_asymmetric_characteristics(result)
            
            # 记录实验结束
            end_time = datetime.now()
            result.end_time = end_time
            result.total_duration = (end_time - start_time).total_seconds()
            
            self.experiment_logger.log_experiment_end(result)
            
            # 保存结果
            self._save_results(result)
            
            return result
            
        except Exception as e:
            logger.error(f"非对称实验执行出错: {e}")
            raise
    
    def _reset_experiment(self):
        """重置实验环境"""
        # 重置市场环境
        self.market_env.reset()
        
        # 重置智能体（保持非对称特性）
        self.player_a.reset()
        self.player_b.reset()
        
        # 重新设置不同的初始参数
        player_a_config = self.config.player_configs.get('player_a', {})
        player_b_config = self.config.player_configs.get('player_b', {})
        
        # 设置不同的探索率
        self.player_a.epsilon = player_a_config.get('initial_epsilon', 1.0)
        self.player_b.epsilon = player_b_config.get('initial_epsilon', 0.7)
        
        # 重置轮次计数
        self.current_round = 0
        
        logger.debug("非对称实验环境已重置")
    
    def _run_single_experiment(self) -> List[RoundResult]:
        """
        运行单次非对称实验
        
        Returns:
            轮次结果列表
        """
        round_results = []
        
        # 初始化历史数据
        strategy_history_a = []
        strategy_history_b = []
        
        # 性能追踪
        performance_gap = []
        
        for round_num in range(1, self.config.total_rounds + 1):
            self.current_round = round_num
            
            # 更新智能体轮次信息
            self.player_a.update_round(round_num)
            self.player_b.update_round(round_num)
            
            # 获取当前市场状态
            market_state = self.market_env.get_current_state()
            
            # 准备个人状态信息（考虑非对称特性）
            personal_state_a = {  # 新手司机状态
                'recent_revenue': getattr(self, '_last_revenue_a', 0),
                'acceptance_rate': 0.6 + np.random.normal(0, 0.15),  # 新手接单率较低且波动大
                'waiting_time': np.random.exponential(15),  # 新手等待时间较长
                'round_number': round_num,
                'history_features': [0.3, 0.4]  # 新手历史表现较差
            }
            
            personal_state_b = {  # 经验司机状态
                'recent_revenue': getattr(self, '_last_revenue_b', 0),
                'acceptance_rate': 0.8 + np.random.normal(0, 0.05),  # 经验司机接单率高且稳定
                'waiting_time': np.random.exponential(8),  # 经验司机等待时间较短
                'round_number': round_num,
                'history_features': [0.7, 0.8]  # 经验司机历史表现较好
            }
            
            # 准备状态向量（考虑信息不对称）
            state_vector_a = self._prepare_asymmetric_state_vector(
                market_state, strategy_history_b, 'player_a'
            )
            state_vector_b = self._prepare_asymmetric_state_vector(
                market_state, strategy_history_a, 'player_b'
            )
            
            # 智能体决策
            # 玩家A决策（新手司机）
            state_a = self.player_a.preprocess_state(market_state, personal_state_a)
            # 传递对手上一轮策略信息（用于收敛学习）
            opponent_strategy_a = getattr(self, '_last_strategy_b', None)
            action_result_a = self.player_a.select_action(state_a, training=True,
                                                        opponent_strategy=opponent_strategy_a)
            strategy_a = self.player_a.action_to_strategy(action_result_a.action)
            
            # 玩家B决策（经验司机）
            state_b = self.player_b.preprocess_state(market_state, personal_state_b)
            # 传递对手上一轮策略信息（用于收敛学习）
            opponent_strategy_b = getattr(self, '_last_strategy_a', None)
            action_result_b = self.player_b.select_action(state_b, training=True,
                                                        opponent_strategy=opponent_strategy_b)
            strategy_b = self.player_b.action_to_strategy(action_result_b.action)
            
            # 应用初始策略偏好
            strategy_a = self._apply_strategy_bias(strategy_a, 'player_a')
            strategy_b = self._apply_strategy_bias(strategy_b, 'player_b')
            
            # 执行博弈轮次（考虑市场优势）
            revenue_a, revenue_b = self._execute_asymmetric_round(
                strategy_a, strategy_b, market_state
            )
            
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
            self._last_revenue_a = revenue_a
            self._last_revenue_b = revenue_b
            
            # 训练智能体（不同频率）
            if round_num > 1:
                # 玩家A每15轮训练一次（新手）
                if round_num % 15 == 0:
                    self.player_a.train()
                    
                # 玩家B每8轮训练一次（经验丰富）
                if round_num % 8 == 0:
                    self.player_b.train()
            
            # 更新历史记录
            strategy_history_a.append(strategy_a)
            strategy_history_b.append(strategy_b)
            
            # 保持历史长度
            if len(strategy_history_a) > 20:
                strategy_history_a = strategy_history_a[-20:]
                strategy_history_b = strategy_history_b[-20:]
            
            # 计算性能差距
            gap = revenue_b - revenue_a
            performance_gap.append(gap)
            
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
            )
            
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
                'performance_gap': gap,
                'player_a_epsilon': self.player_a.epsilon,
                'player_b_epsilon': self.player_b.epsilon
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
                avg_gap = np.mean(performance_gap[-50:])
                logger.info(f"轮次 {round_num}: 策略A={strategy_a:.2f}, 策略B={strategy_b:.2f}, "
                           f"性能差距={avg_gap:.2f}")
            
            # 阶段转换检测
            self._check_phase_transition(round_num)
        
        return round_results
    
    def _prepare_asymmetric_state_vector(self, market_state: Dict, 
                                       opponent_history: List[float], 
                                       player_id: str) -> np.ndarray:
        """
        准备非对称状态向量
        
        Args:
            market_state: 市场状态
            opponent_history: 对手策略历史
            player_id: 玩家ID
            
        Returns:
            状态向量
        """
        # 基础市场特征
        features = [
            market_state.get('order_rate', 0) / 100.0,
            market_state.get('avg_price', 30) / 50.0,
            market_state.get('competition_level', 0.5),
            market_state.get('time_factor', 1.0),
            market_state.get('location_factor', 1.0),
        ]
        
        # 根据玩家能力调整信息质量
        if player_id == 'player_a':
            # 玩家A信息获取能力较弱，添加噪声
            noise_level = 0.1
            features = [f + np.random.normal(0, noise_level) for f in features]
        
        # 对手策略历史特征
        recent_history = opponent_history[-10:] if opponent_history else []
        
        # 填充到固定长度
        while len(recent_history) < 10:
            recent_history.insert(0, 30.0)
        
        # 根据玩家预测能力调整历史信息
        if player_id == 'player_a':
            # 玩家A只能看到较短的历史
            recent_history = recent_history[-5:] + [30.0] * 5
        
        # 归一化对手历史
        normalized_history = [(x - 10) / 40.0 for x in recent_history]
        
        features.extend(normalized_history)
        
        return np.array(features, dtype=np.float32)
    
    def _apply_strategy_bias(self, strategy: float, player_id: str) -> float:
        """
        应用策略偏好
        
        Args:
            strategy: 原始策略
            player_id: 玩家ID
            
        Returns:
            调整后的策略
        """
        player_config = self.config.player_configs.get(player_id, {})
        bias = player_config.get('strategy_bias', 0)
        
        # 应用偏好并确保在合理范围内
        adjusted_strategy = strategy + bias
        return max(10, min(50, adjusted_strategy))
    
    def _execute_asymmetric_round(self, strategy_a: float, strategy_b: float, 
                                market_state: Dict) -> Tuple[float, float]:
        """
        执行非对称博弈轮次
        
        Args:
            strategy_a: 玩家A策略
            strategy_b: 玩家B策略
            market_state: 市场状态
            
        Returns:
            (玩家A收益, 玩家B收益)
        """
        # 生成虚拟订单
        orders = self.market_env.generate_orders()
        
        # 处理司机决策
        strategies = {
            'player_a': strategy_a,
            'player_b': strategy_b
        }
        
        # 司机接单情况
        driver_orders = self.market_env.process_driver_decisions(orders, strategies)
        
        # 计算收益
        revenues = self.market_env.calculate_driver_revenues(driver_orders, strategies)
        
        revenue_a = revenues.get('player_a', 0.0)
        revenue_b = revenues.get('player_b', 0.0)
        
        # 应用市场优势
        revenue_a *= self.player_a_advantage
        revenue_b *= self.player_b_advantage
        
        # 更新市场状态
        self.market_env.update_market_time()
        
        return revenue_a, revenue_b
    
    def _calculate_asymmetric_reward(self, own_revenue: float, opponent_revenue: float, 
                                   strategy: float, player_id: str) -> float:
        """
        计算非对称奖励函数
        
        Args:
            own_revenue: 自己的收益
            opponent_revenue: 对手的收益
            strategy: 自己的策略
            player_id: 玩家ID
            
        Returns:
            奖励值
        """
        player_config = self.config.player_configs.get(player_id, {})
        
        # 基础收益奖励
        revenue_reward = own_revenue / 100.0
        
        # 相对收益奖励
        relative_reward = (own_revenue - opponent_revenue) / 100.0
        
        # 策略合理性奖励
        strategy_penalty = 0
        if strategy < 15 or strategy > 45:
            strategy_penalty = -0.1
        
        # 不同的奖励权重
        if player_id == 'player_a':
            # 玩家A更关注绝对收益
            total_reward = revenue_reward * 0.8 + relative_reward * 0.1 + strategy_penalty
        else:
            # 玩家B更关注相对收益
            total_reward = revenue_reward * 0.5 + relative_reward * 0.4 + strategy_penalty
        
        # 风险偏好调整
        risk_preference = player_config.get('risk_preference', 0.0)
        if risk_preference != 0:
            # 添加风险调整项
            risk_adjustment = risk_preference * abs(strategy - 30) / 20.0
            total_reward += risk_adjustment
        
        return total_reward
    
    def _analyze_asymmetric_characteristics(self, result: ExperimentResult):
        """分析非对称特性"""
        if not result.round_results:
            return
        
        # 计算性能差距演化
        performance_gaps = []
        for r in result.round_results:
            gap = r.player_b_revenue - r.player_a_revenue
            performance_gaps.append(gap)
        
        # 计算策略收敛性
        final_window = result.round_results[-50:]
        strategy_stability_a = 1.0 / (1.0 + np.std([r.player_a_strategy for r in final_window]))
        strategy_stability_b = 1.0 / (1.0 + np.std([r.player_b_strategy for r in final_window]))
        
        # 添加非对称分析到结果
        asymmetric_analysis = {
            'performance_gap_evolution': performance_gaps,
            'final_performance_gap': np.mean(performance_gaps[-50:]),
            'performance_gap_std': np.std(performance_gaps[-50:]),
            'strategy_stability_difference': abs(strategy_stability_a - strategy_stability_b),
            'convergence_speed_ratio': self._calculate_convergence_speed_ratio(result),
            'learning_efficiency_ratio': self._calculate_learning_efficiency_ratio(result)
        }
        
        # 添加到实验结果
        if not hasattr(result, 'asymmetric_analysis'):
            result.asymmetric_analysis = asymmetric_analysis
    
    def _calculate_convergence_speed_ratio(self, result: ExperimentResult) -> float:
        """计算收敛速度比"""
        # 简化的收敛速度计算
        return 1.0  # 暂时返回默认值
    
    def _calculate_learning_efficiency_ratio(self, result: ExperimentResult) -> float:
        """计算学习效率比"""
        # 简化的学习效率计算
        return 1.0  # 暂时返回默认值
    
    def _check_phase_transition(self, round_num: int):
        """检查阶段转换"""
        if round_num == 120:
            self.experiment_logger.log_phase_transition("学习", round_num)
        elif round_num == 360:
            self.experiment_logger.log_phase_transition("均衡", round_num)
    
    def _save_results(self, result: ExperimentResult):
        """保存实验结果"""
        # 保存JSON结果
        result_path = f"results/asymmetric_game_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        result.save_to_json(result_path)
        
        # 保存数据CSV
        data_path = f"results/asymmetric_game_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self.data_collector.export_to_csv(data_path)
        
        # 生成分析图表
        self._generate_asymmetric_plots(result)
        
        logger.info(f"非对称实验结果已保存到: {result_path}")
    
    def _generate_asymmetric_plots(self, result: ExperimentResult):
        """生成非对称分析图表"""
        try:
            # 策略演化对比图
            self._plot_asymmetric_strategy_evolution(result)
            
            # 收益差距演化图
            self._plot_performance_gap_evolution(result)
            
            # 学习曲线对比图
            self._plot_learning_curves(result)
            
        except Exception as e:
            logger.warning(f"生成非对称图表时出错: {e}")
    
    def _plot_asymmetric_strategy_evolution(self, result: ExperimentResult):
        """绘制非对称策略演化图"""
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
        
        plt.figure(figsize=(12, 8))
        
        # 策略演化
        plt.subplot(2, 1, 1)
        plt.plot(daily_rounds, daily_strategies_a, label='玩家A策略（弱势）', alpha=0.7, color='blue', marker='o', markersize=4)
        plt.plot(daily_rounds, daily_strategies_b, label='玩家B策略（强势）', alpha=0.7, color='red', marker='s', markersize=4)
        
        # 添加阶段分割线（按天调整）
        plt.axvline(x=120, color='gray', linestyle='--', alpha=0.5)
        plt.axvline(x=360, color='gray', linestyle='--', alpha=0.5)
        
        plt.xlabel('轮次')
        plt.ylabel('策略值（元）')
        plt.title('非对称博弈策略演化（日平均值）')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 策略差异
        plt.subplot(2, 1, 2)
        strategy_diff = [b - a for a, b in zip(daily_strategies_a, daily_strategies_b)]
        plt.plot(daily_rounds, strategy_diff, label='策略差异（B-A）', alpha=0.7, color='green', marker='D', markersize=4)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        plt.xlabel('轮次')
        plt.ylabel('策略差异（元）')
        plt.title('策略差异演化（日平均值）')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = f"results/plots/asymmetric_strategy_evolution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_path)
        plt.close()
    
    def _plot_performance_gap_evolution(self, result: ExperimentResult):
        """绘制性能差距演化图"""
        rounds = [r.round_number for r in result.round_results]
        performance_gaps = [r.player_b_revenue - r.player_a_revenue for r in result.round_results]
        
        # 计算移动平均
        window_size = 20
        moving_avg = []
        for i in range(len(performance_gaps)):
            start_idx = max(0, i - window_size + 1)
            avg = np.mean(performance_gaps[start_idx:i+1])
            moving_avg.append(avg)
        
        plt.figure(figsize=(12, 6))
        plt.plot(rounds, performance_gaps, alpha=0.3, color='blue', label='实际差距')
        plt.plot(rounds, moving_avg, color='red', linewidth=2, label='移动平均')
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        plt.xlabel('轮次')
        plt.ylabel('收益差距（元）')
        plt.title('非对称博弈收益差距演化')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plot_path = f"results/plots/asymmetric_performance_gap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_path)
        plt.close()
    
    def _plot_learning_curves(self, result: ExperimentResult):
        """绘制收益曲线对比图"""
        rounds = [r.round_number for r in result.round_results]
        revenues_a = [r.player_a_revenue for r in result.round_results]
        revenues_b = [r.player_b_revenue for r in result.round_results]
        
        # 设置字体
        font_prop = None
        font_path = r"C:\Windows\Fonts\STZHONGS.TTF"
        if os.path.exists(font_path):
            font_prop = fm.FontProperties(fname=font_path)
        
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
        
        # 定义新的配色方案 (将RGB值转换为matplotlib可识别的格式)
        color_a = (250/255, 170/255, 137/255)  # 橙红色
        color_b = (255/255, 220/255, 126/255)  # 黄色
        color_diff = (126/255, 208/255, 248/255)  # 蓝色
        
        # 上部分：收益曲线
        ax_top.plot(days_list, daily_rev_a_list, label='玩家A收益(弱势)', alpha=0.8, color=color_a, linewidth=1.5)
        ax_top.plot(days_list, daily_rev_b_list, label='玩家B收益(强势)', alpha=0.8, color=color_b, linewidth=1.5)
        ax_top.set_ylabel('日收益（元/天）', fontsize=12)
        ax_top.set_title(f'非对称博弈日收益对比 ({max_days}天)', fontsize=14, fontweight='bold')
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
        ax_bottom.plot(days_list, daily_abs_diff_list, label='收益差值绝对值', color=color_diff, alpha=0.8, linewidth=1.5)
        ax_bottom.fill_between(days_list, daily_abs_diff_list, alpha=0.3, color=color_diff)
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
        plot_path = f"results/plots/asymmetric_revenue_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_path, dpi=300)
        plt.close()


def create_asymmetric_experiment(total_rounds: int = 500, num_runs: int = 1,
                               asymmetry_type: str = 'experience') -> AsymmetricGameExperiment:
    """
    创建非对称博弈实验的便捷函数
    
    Args:
        total_rounds: 总轮次数
        num_runs: 重复实验次数
        asymmetry_type: 非对称类型 ('experience', 'information', 'market_power')
        
    Returns:
        非对称博弈实验实例
    """
    if asymmetry_type == 'experience':
        # 经验非对称：新手vs老手
        player_configs = {
            'player_a': {
                'type': 'dqn', 'learning_enabled': True,
                'initial_strategy': 25, 'strategy_bias': -2,
                'learning_rate': 0.0005, 'initial_epsilon': 1.0,
                'memory_size': 5000, 'predictor_capacity': 32,
                'market_advantage': 0.9, 'risk_preference': -0.1
            },
            'player_b': {
                'type': 'dqn', 'learning_enabled': True,
                'initial_strategy': 35, 'strategy_bias': 2,
                'learning_rate': 0.002, 'initial_epsilon': 0.6,
                'memory_size': 15000, 'predictor_capacity': 64,
                'market_advantage': 1.1, 'risk_preference': 0.1
            }
        }
    elif asymmetry_type == 'information':
        # 信息非对称：信息优势vs劣势
        player_configs = {
            'player_a': {
                'type': 'dqn', 'learning_enabled': True,
                'initial_strategy': 30, 'strategy_bias': 0,
                'learning_rate': 0.001, 'initial_epsilon': 0.8,
                'memory_size': 8000, 'predictor_capacity': 32,
                'market_advantage': 0.95, 'risk_preference': 0
            },
            'player_b': {
                'type': 'dqn', 'learning_enabled': True,
                'initial_strategy': 30, 'strategy_bias': 0,
                'learning_rate': 0.001, 'initial_epsilon': 0.8,
                'memory_size': 8000, 'predictor_capacity': 80,
                'market_advantage': 1.05, 'risk_preference': 0
            }
        }
    else:  # market_power
        # 市场力量非对称：大公司vs小公司
        player_configs = {
            'player_a': {
                'type': 'dqn', 'learning_enabled': True,
                'initial_strategy': 28, 'strategy_bias': -3,
                'learning_rate': 0.001, 'initial_epsilon': 0.9,
                'memory_size': 6000, 'predictor_capacity': 48,
                'market_advantage': 0.8, 'risk_preference': -0.2
            },
            'player_b': {
                'type': 'dqn', 'learning_enabled': True,
                'initial_strategy': 32, 'strategy_bias': 3,
                'learning_rate': 0.001, 'initial_epsilon': 0.7,
                'memory_size': 12000, 'predictor_capacity': 64,
                'market_advantage': 1.2, 'risk_preference': 0.2
            }
        }
    
    config = ExperimentConfig(
        experiment_name=f"非对称博弈实验_{asymmetry_type}",
        experiment_type="asymmetric",
        total_rounds=total_rounds,
        num_runs=num_runs,
        player_configs=player_configs,
        ai_config={
            'dqn_params': {
                'learning_rate': 0.001,
                'epsilon': 0.8,
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
    
    return AsymmetricGameExperiment(config) 