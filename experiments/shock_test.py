"""
环境冲击测试实验
测试博弈系统在外部冲击下的稳定性和适应性
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging
import random
import os

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


class ShockTestExperiment:
    """
    环境冲击测试实验
    在标准博弈过程中引入外部市场冲击
    """
    
    def __init__(self, config: ExperimentConfig):
        """
        初始化冲击测试实验
        
        Args:
            config: 实验配置
        """
        self.config = config
        self.game_config = GameConfig()
        
        # 设置实验名称
        self.experiment_name = config.experiment_name or "market_shock_test"
        
        # 创建游戏框架
        self.game_framework = GameFramework(self.game_config)
        
        # 创建市场环境
        self.market_env = MarketEnvironment(self.game_config)
        
        # 初始化AI智能体
        self._initialize_agents()
        
        # 初始化数据存储
        self._initialize_data_storage()
        
        # 配置冲击事件
        self._configure_shocks()
        
        logger.info(f"冲击测试实验初始化完成: {self.experiment_name}")
        
        # 实验状态
        self.current_round = 0
        self.shock_applied = False
        self.active_shocks = []
        
        # 玩家标识
        self.player_a = "player_a"
        self.player_b = "player_b"
        
        logger.info(f"已配置 {len(self.market_shocks)} 个市场冲击事件")
        
        self.experiment_logger = ExperimentLogger(self.experiment_name)
        self.data_collector = DataCollector()
    
    def _initialize_agents(self):
        """初始化AI智能体"""
        # 创建DQN智能体配置
        agent_a_config = {
            'agent_name': "Player_A",
            'state_size': self.game_config.DQN_STATE_SIZE,
            'action_size': self.game_config.DQN_ACTION_SIZE,
            'learning_rate': 0.01,
            'gamma': 0.95,  # 折扣因子
            'epsilon_start': 0.1,  # 探索率
            'epsilon_decay': 0.995
        }
        
        agent_b_config = {
            'agent_name': "Player_B",
            'state_size': self.game_config.DQN_STATE_SIZE,
            'action_size': self.game_config.DQN_ACTION_SIZE,
            'learning_rate': 0.01,
            'gamma': 0.95,  # 折扣因子
            'epsilon_start': 0.1,  # 探索率
            'epsilon_decay': 0.995
        }
        
        # 创建策略预测器配置
        predictor_a_config = {
            'agent_name': "Predictor_A",
            'input_features': 10,
            'hidden_size': 32,
            'sequence_length': 5,
            'learning_rate': 0.001
        }
        
        predictor_b_config = {
            'agent_name': "Predictor_B",
            'input_features': 10,
            'hidden_size': 32,
            'sequence_length': 5,
            'learning_rate': 0.001
        }
        
        # 创建DQN智能体
        self.agent_a = DQNAgent(agent_a_config)
        self.agent_b = DQNAgent(agent_b_config)
        
        # 创建策略预测器
        self.predictor_a = StrategyPredictor(predictor_a_config)
        self.predictor_b = StrategyPredictor(predictor_b_config)
        
        logger.info("AI智能体初始化完成")
    
    def apply_shock(self, round_num: int) -> Dict[str, any]:
        """
        应用市场冲击
        
        Args:
            round_num: 当前轮次
        
        Returns:
            包含冲击信息的字典
        """
        active_shock_info = {}
        
        # 检查新的冲击
        for shock in self.market_shocks:
            if shock['round'] == round_num:
                shock_copy = shock.copy()
                shock_copy['end_round'] = round_num + shock['duration']
                shock_copy['start_round'] = round_num
                self.active_shocks.append(shock_copy)
                
                # 记录冲击事件
                self.shock_events.append({
                    'round': round_num,
                    'type': shock['type'],
                    'intensity': shock.get('intensity', 1.0),
                    'duration': shock['duration']
                })
                
                logger.info(f"轮次 {round_num}: 应用市场冲击 - 类型: {shock['type']}")
                
                if 'intensity' in shock:
                    logger.info(f"冲击强度: {shock['intensity']}")
                if 'duration' in shock:
                    logger.info(f"冲击持续时间: {shock['duration']} 轮")
        
        # 更新活跃冲击状态
        new_active_shocks = []
        for shock in self.active_shocks:
            if round_num <= shock['end_round']:
                new_active_shocks.append(shock)
                active_shock_info[shock['type']] = {
                    'intensity': shock.get('intensity', 1.0),
                    'remaining': shock['end_round'] - round_num,
                    'progress': (round_num - shock['start_round']) / shock['duration']
                }
                
                # 对市场环境应用冲击效果
                if shock['type'] == 'demand_surge':
                    # 需求激增，乘以强度因子
                    self.market_env.demand_multiplier = shock.get('intensity', 1.0)
                    
                elif shock['type'] == 'demand_drop':
                    # 需求下降，强度为衰减因子
                    self.market_env.demand_multiplier = shock.get('intensity', 1.0)
                    
                elif shock['type'] == 'supply_shortage':
                    # 供给减少，强度为衰减因子
                    self.market_env.supply_multiplier = shock.get('intensity', 1.0)
                    
                elif shock['type'] == 'price_regulation':
                    # 价格上限规制
                    self.market_env.max_price = shock.get('max_price', 50)
                    
                elif shock['type'] == 'competition_increase':
                    # 竞争加剧，对供需均有影响
                    competition_factor = shock.get('intensity', 1.0)
                    self.market_env.supply_multiplier *= competition_factor
                    self.market_env.demand_multiplier /= (competition_factor * 0.8)  # 需求被分流但不会完全按比例减少
                    
                elif shock['type'] == 'market_disruption':
                    # 市场混乱，同时影响供给和需求
                    self.market_env.demand_multiplier = shock.get('demand_effect', 0.7)
                    self.market_env.supply_multiplier = shock.get('supply_effect', 1.2)
                    # 价格波动增加
                    self.market_env.price_volatility = 1.5
                    
                elif shock['type'] == 'technology_shift':
                    # 技术革新，降低成本提高需求
                    self.market_env.demand_multiplier = shock.get('demand_boost', 1.3)
                    self.market_env.cost_multiplier = shock.get('cost_reduction', 0.8)
                    
                elif shock['type'] == 'market_crash':
                    # 市场崩溃，全面负面影响
                    self.market_env.demand_multiplier = shock.get('demand_effect', 0.4)
                    self.market_env.supply_multiplier = shock.get('supply_effect', 0.5)
                    self.market_env.price_volatility = 2.0
                    self.market_env.max_price = self.market_env.max_price * shock.get('price_effect', 0.7)
                
                # 记录冲击影响
                self.shock_impacts.append({
                    'round': round_num,
                    'type': shock['type'],
                    'demand_multiplier': self.market_env.demand_multiplier,
                    'supply_multiplier': self.market_env.supply_multiplier,
                    'max_price': self.market_env.max_price,
                    'price_volatility': getattr(self.market_env, 'price_volatility', 1.0),
                    'cost_multiplier': getattr(self.market_env, 'cost_multiplier', 1.0)
                })
        
        self.active_shocks = new_active_shocks
        
        # 如果没有活跃冲击，恢复正常状态
        if not self.active_shocks:
            self.market_env.demand_multiplier = 1.0
            self.market_env.supply_multiplier = 1.0
            self.market_env.max_price = 50
            if hasattr(self.market_env, 'price_volatility'):
                self.market_env.price_volatility = 1.0
            if hasattr(self.market_env, 'cost_multiplier'):
                self.market_env.cost_multiplier = 1.0
        
        return active_shock_info
        
    def run_experiment(self) -> ExperimentResult:
        """
        运行冲击测试实验
        
        Returns:
            实验结果
        """
        # 创建实验结果对象
        result = ExperimentResult(
            experiment_config=self.config,
            start_time=datetime.now()
        )
        
        total_rounds = self.config.total_rounds
        num_repetitions = self.config.repetitions if hasattr(self.config, 'repetitions') else 1  # 默认只运行1次
        
        logger.info(f"开始冲击测试实验: {total_rounds} 轮 x {num_repetitions} 次重复")
        
        # 运行多次实验
        for repetition in range(num_repetitions):
            logger.info(f"开始第 {repetition+1}/{num_repetitions} 轮冲击测试")
            
            # 重置实验状态
            self.current_round = 0
            self.active_shocks = []
            self.market_env.reset()
            self.agent_a.reset()
            self.agent_b.reset()
            
            # 记录每轮结果
            round_results = []
            
            for round_num in range(1, total_rounds + 1):
                self.current_round = round_num
                
                # 应用市场冲击
                active_shock_info = self.apply_shock(round_num)
                
                # 检查阶段转换
                if round_num == 50:
                    logger.info(f"进入学习阶段: 轮次 {round_num}")
                elif round_num == 200:
                    logger.info(f"进入均衡阶段: 轮次 {round_num}")
                
                # 获取当前市场状态
                market_state = self.market_env.get_state()
                
                # 创建用于DQN的状态向量
                state_vector = np.array([
                    market_state['demand'], 
                    market_state['supply'],
                    market_state['avg_price'],
                    market_state['time_period'],
                    market_state['competition'],
                    market_state['order_rate'],
                    market_state['avg_revenue_a'],
                    market_state['avg_revenue_b'],
                    market_state['strategy_variance_a'],
                    market_state['strategy_variance_b'],
                    market_state['avg_distance']
                ], dtype=np.float32)
                
                # 选择策略 (定价)
                action_result_a = self.agent_a.select_action(state_vector)
                action_result_b = self.agent_b.select_action(state_vector)
                
                # 从ActionResult中提取策略值
                strategy_a = self.agent_a.action_to_strategy(action_result_a.action)
                strategy_b = self.agent_b.action_to_strategy(action_result_b.action)
                
                # 预测对手策略
                # 在前20轮使用默认值，之后才尝试预测
                predicted_a = 30.0
                predicted_b = 30.0
                
                if len(self.strategies_a) > 20 and len(self.strategies_b) > 20:
                    try:
                        # 有足够历史数据时进行预测
                        prediction_result_b = self.predictor_a.predict_strategy("player_b")
                        prediction_result_a = self.predictor_b.predict_strategy("player_a")
                        
                        if hasattr(prediction_result_b, 'predicted_strategy'):
                            predicted_b = prediction_result_b.predicted_strategy
                        
                        if hasattr(prediction_result_a, 'predicted_strategy'):
                            predicted_a = prediction_result_a.predicted_strategy
                    except Exception as e:
                        logger.warning(f"预测对手策略失败: {e}")
                
                # 执行策略，获得奖励
                player_a_state = {'strategy': strategy_a, 'predicted_opponent': predicted_b}
                player_b_state = {'strategy': strategy_b, 'predicted_opponent': predicted_a}
                
                # 使用play_round方法执行博弈
                strategies = {"player_a": strategy_a, "player_b": strategy_b}
                round_result = self.game_framework.play_round(round_num, strategies)
                
                # 获取结果
                result_a = {
                    'revenue': round_result['payoffs']["player_a"],
                    'orders': round_result['market_data'].get('total_orders', 0) // 2  # 简单平分订单
                }
                
                result_b = {
                    'revenue': round_result['payoffs']["player_b"],
                    'orders': round_result['market_data'].get('total_orders', 0) // 2  # 简单平分订单
                }
                
                # 更新模型
                next_market_state = self.market_env.update(result_a, result_b)
                
                # 将下一个状态转换为向量
                next_state_vector = np.array([
                    next_market_state['demand'], 
                    next_market_state['supply'],
                    next_market_state['avg_price'],
                    next_market_state['time_period'],
                    next_market_state['competition'],
                    next_market_state['order_rate'],
                    next_market_state['avg_revenue_a'],
                    next_market_state['avg_revenue_b'],
                    next_market_state['strategy_variance_a'],
                    next_market_state['strategy_variance_b'],
                    next_market_state['avg_distance']
                ], dtype=np.float32)
                
                # 记录策略和收益历史
                self.strategies_a.append(strategy_a)
                self.strategies_b.append(strategy_b)
                self.market_states.append(market_state)
                self.revenues_a.append(result_a['revenue'])
                self.revenues_b.append(result_b['revenue'])
                
                # 处理智能体学习
                self.agent_a.store_experience(state_vector, action_result_a.action, result_a['revenue'], next_state_vector, False)
                self.agent_b.store_experience(state_vector, action_result_b.action, result_b['revenue'], next_state_vector, False)
                
                # 训练模型
                self.agent_a.train()
                self.agent_b.train()
                
                # 更新预测器的历史数据
                self.predictor_a.update_history(strategy_a, market_state, result_a['revenue'])
                self.predictor_b.update_history(strategy_b, market_state, result_b['revenue'])
                
                # 训练预测器（有足够数据时）
                if len(self.strategies_a) > 20 and len(self.strategies_b) > 20:
                    # 使用最近的数据训练
                    try:
                        self.predictor_a.train_on_batch(self.strategies_a[-20:], self.market_states[-20:], 
                                                      self.revenues_a[-20:], self.strategies_b[-20:])
                        self.predictor_b.train_on_batch(self.strategies_b[-20:], self.market_states[-20:], 
                                                      self.revenues_b[-20:], self.strategies_a[-20:])
                    except Exception as e:
                        logger.warning(f"训练预测器失败: {e}")
                
                # 记录本轮结果
                result_obj = RoundResult(
                    round_number=round_num,
                    player_a_strategy=strategy_a,
                    player_b_strategy=strategy_b,
                    player_a_revenue=result_a['revenue'],
                    player_b_revenue=result_b['revenue'],
                    market_state=market_state
                )
                
                # 添加到结果中
                result.round_results.append(result_obj)
                
                # 更新进度
                if round_num % 50 == 0:
                    active_shock_type = ", ".join([f"{s['type']}" for s in self.active_shocks]) if self.active_shocks else "无"
                    logger.info(f"轮次 {round_num}: 策略A={strategy_a:.2f}, 策略B={strategy_b:.2f}, 活跃冲击={active_shock_type}")
            
            logger.info(f"完成第 {repetition+1} 轮实验，共 {total_rounds} 个轮次")
            
            # 将本次重复实验的结果保存到总结果中
            round_results = result.round_results
            
            # 重置结果对象，准备下一次重复实验
            if repetition < num_repetitions - 1:
                result = ExperimentResult(
                    experiment_config=self.config,
                    start_time=datetime.now()
                )
        
        # 完成实验
        result.end_time = datetime.now()
        result.total_rounds = total_rounds
        result.players = [self.player_a, self.player_b]
        
        # 计算最终策略和总收益
        final_strategies = {
            self.player_a: np.mean([r.player_a_strategy for r in result.round_results]),
            self.player_b: np.mean([r.player_b_strategy for r in result.round_results])
        }
        
        total_revenues = {
            self.player_a: sum(r.player_a_revenue for r in result.round_results),
            self.player_b: sum(r.player_b_revenue for r in result.round_results)
        }
        
        result.final_strategies = final_strategies
        result.total_revenues = total_revenues
        
        # 记录实验结果
        self.experiment_logger.log_experiment_end(result)
        
        logger.info(f"实验结束: {self.experiment_name}")
        logger.info(f"最终策略: {final_strategies}")
        logger.info(f"总收益: {total_revenues}")
        
        # 分析冲击效果
        shock_analysis = self.analyze_shock_effects(result)
        result.shock_analysis = shock_analysis
        
        # 可视化冲击效果
        figures_dir = os.path.join("results", "figures")
        os.makedirs(figures_dir, exist_ok=True)
        
        visualization_path = os.path.join(
            figures_dir, 
            f"shock_analysis_{self.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        )
        
        self.visualize_shock_effects(result, save_path=visualization_path)
        result.shock_visualization_path = visualization_path
        
        return result 

    def _initialize_data_storage(self):
        """初始化数据记录"""
        # 创建数据收集器
        self.data_collector = DataCollector()
        
        # 实验日志记录
        experiment_name = self.experiment_name or "market_shock_test"
        self.logger = ExperimentLogger(experiment_name)
        
        # 用于记录策略、状态和收益历史的列表
        self.strategies_a = []
        self.strategies_b = []
        self.market_states = []
        self.revenues_a = []
        self.revenues_b = []
        
        # 冲击事件记录
        self.shock_events = []
        self.shock_impacts = []

    def _configure_shocks(self):
        """配置冲击事件"""
        # 配置不同类型的市场冲击
        if hasattr(self.config, 'market_config') and 'market_shocks' in self.config.market_config:
            self.market_shocks = self.config.market_config['market_shocks']
        else:
            # 默认冲击设置
            self.market_shocks = [
                {'round': 50, 'type': 'demand_surge', 'intensity': 1.5, 'duration': 20},
                {'round': 100, 'type': 'supply_shortage', 'intensity': 0.7, 'duration': 30},
                {'round': 150, 'type': 'price_regulation', 'max_price': 35, 'duration': 50}
            ]
            
            # 添加更多类型的冲击
            if random.random() < 0.3:  # 30%的概率添加额外冲击
                extra_shocks = [
                    {'round': 180, 'type': 'competition_increase', 'intensity': 1.5, 'duration': 25},
                    {'round': 350, 'type': 'demand_drop', 'intensity': 0.6, 'duration': 35}
                ]
                self.market_shocks.extend(extra_shocks)
                
            # 可能添加更复杂的冲击组合
            if random.random() < 0.2:  # 20%的概率添加复合冲击
                compound_shocks = [
                    {'round': 230, 'type': 'market_disruption', 'demand_effect': 0.7, 'supply_effect': 1.2, 'duration': 40},
                    {'round': 300, 'type': 'technology_shift', 'cost_reduction': 0.8, 'demand_boost': 1.3, 'duration': 60}
                ]
                self.market_shocks.extend(compound_shocks)
                
            # 有小概率添加极端冲击事件
            if random.random() < 0.1:  # 10%的概率添加极端冲击
                extreme_shock = {
                    'round': 400, 
                    'type': 'market_crash', 
                    'demand_effect': 0.4,
                    'supply_effect': 0.5,
                    'price_effect': 0.7,
                    'duration': 30
                }
                self.market_shocks.append(extreme_shock)
                
        logger.info(f"已配置 {len(self.market_shocks)} 个市场冲击事件")
        
        # 初始化冲击跟踪
        self.active_shocks = []

    def analyze_shock_effects(self, result: ExperimentResult) -> Dict:
        """
        分析冲击效果对市场和玩家策略的影响
        
        Args:
            result: 实验结果对象
        
        Returns:
            冲击效果分析结果
        """
        analysis = {
            'shock_events': self.shock_events,
            'shock_impacts': [],
            'strategy_changes': [],
            'revenue_changes': [],
            'shock_type_analysis': {}  # 按冲击类型分类的分析
        }
        
        if not self.shock_events or not result.round_results:
            return analysis
        
        # 按冲击类型分组
        shock_types = set(shock['type'] for shock in self.shock_events)
        for shock_type in shock_types:
            analysis['shock_type_analysis'][shock_type] = {
                'count': 0,
                'avg_strategy_change_a': 0,
                'avg_strategy_change_b': 0,
                'avg_revenue_change_a': 0,
                'avg_revenue_change_b': 0,
                'recovery_time': 0  # 恢复时间（轮次）
            }
        
        # 分析每个冲击事件
        for shock in self.shock_events:
            shock_round = shock['round']
            shock_end = shock_round + shock['duration']
            shock_type = shock['type']
            
            # 获取冲击前后的数据
            pre_shock_rounds = range(max(0, shock_round - 20), shock_round)
            during_shock_rounds = range(shock_round, min(shock_end, len(result.round_results)))
            post_shock_rounds = range(shock_end, min(shock_end + 20, len(result.round_results)))
            recovery_rounds = range(shock_end, min(shock_end + 50, len(result.round_results)))
            
            # 收集策略和收益数据
            pre_strategies_a = [result.round_results[r].player_a_strategy for r in pre_shock_rounds if r < len(result.round_results)]
            pre_strategies_b = [result.round_results[r].player_b_strategy for r in pre_shock_rounds if r < len(result.round_results)]
            during_strategies_a = [result.round_results[r].player_a_strategy for r in during_shock_rounds if r < len(result.round_results)]
            during_strategies_b = [result.round_results[r].player_b_strategy for r in during_shock_rounds if r < len(result.round_results)]
            post_strategies_a = [result.round_results[r].player_a_strategy for r in post_shock_rounds if r < len(result.round_results)]
            post_strategies_b = [result.round_results[r].player_b_strategy for r in post_shock_rounds if r < len(result.round_results)]
            
            pre_revenues_a = [result.round_results[r].player_a_revenue for r in pre_shock_rounds if r < len(result.round_results)]
            pre_revenues_b = [result.round_results[r].player_b_revenue for r in pre_shock_rounds if r < len(result.round_results)]
            during_revenues_a = [result.round_results[r].player_a_revenue for r in during_shock_rounds if r < len(result.round_results)]
            during_revenues_b = [result.round_results[r].player_b_revenue for r in during_shock_rounds if r < len(result.round_results)]
            post_revenues_a = [result.round_results[r].player_a_revenue for r in post_shock_rounds if r < len(result.round_results)]
            post_revenues_b = [result.round_results[r].player_b_revenue for r in post_shock_rounds if r < len(result.round_results)]
            
            # 计算均值
            avg_pre_strategy_a = np.mean(pre_strategies_a) if pre_strategies_a else 0
            avg_pre_strategy_b = np.mean(pre_strategies_b) if pre_strategies_b else 0
            avg_during_strategy_a = np.mean(during_strategies_a) if during_strategies_a else 0
            avg_during_strategy_b = np.mean(during_strategies_b) if during_strategies_b else 0
            avg_post_strategy_a = np.mean(post_strategies_a) if post_strategies_a else 0
            avg_post_strategy_b = np.mean(post_strategies_b) if post_strategies_b else 0
            
            avg_pre_revenue_a = np.mean(pre_revenues_a) if pre_revenues_a else 0
            avg_pre_revenue_b = np.mean(pre_revenues_b) if pre_revenues_b else 0
            avg_during_revenue_a = np.mean(during_revenues_a) if during_revenues_a else 0
            avg_during_revenue_b = np.mean(during_revenues_b) if during_revenues_b else 0
            avg_post_revenue_a = np.mean(post_revenues_a) if post_revenues_a else 0
            avg_post_revenue_b = np.mean(post_revenues_b) if post_revenues_b else 0
            
            # 计算策略和收益变化率
            strategy_change_a = (avg_during_strategy_a - avg_pre_strategy_a) / max(1, avg_pre_strategy_a)
            strategy_change_b = (avg_during_strategy_b - avg_pre_strategy_b) / max(1, avg_pre_strategy_b)
            revenue_change_a = (avg_during_revenue_a - avg_pre_revenue_a) / max(1, avg_pre_revenue_a)
            revenue_change_b = (avg_during_revenue_b - avg_pre_revenue_b) / max(1, avg_pre_revenue_b)
            
            # 计算恢复时间
            recovery_time = 0
            if recovery_rounds and pre_revenues_a and pre_revenues_b:
                avg_pre_total = (avg_pre_revenue_a + avg_pre_revenue_b) / 2
                for i, r in enumerate(recovery_rounds):
                    if r < len(result.round_results):
                        recovery_revenue_a = result.round_results[r].player_a_revenue
                        recovery_revenue_b = result.round_results[r].player_b_revenue
                        recovery_avg = (recovery_revenue_a + recovery_revenue_b) / 2
                        if recovery_avg >= avg_pre_total * 0.95:  # 恢复到冲击前95%的水平
                            recovery_time = i
                            break
            
            # 记录分析结果
            impact = {
                'shock_round': shock_round,
                'shock_type': shock_type,
                'duration': shock['duration'],
                'pre_shock': {
                    'avg_strategy_a': avg_pre_strategy_a,
                    'avg_strategy_b': avg_pre_strategy_b,
                    'avg_revenue_a': avg_pre_revenue_a,
                    'avg_revenue_b': avg_pre_revenue_b
                },
                'during_shock': {
                    'avg_strategy_a': avg_during_strategy_a,
                    'avg_strategy_b': avg_during_strategy_b,
                    'avg_revenue_a': avg_during_revenue_a,
                    'avg_revenue_b': avg_during_revenue_b
                },
                'post_shock': {
                    'avg_strategy_a': avg_post_strategy_a,
                    'avg_strategy_b': avg_post_strategy_b,
                    'avg_revenue_a': avg_post_revenue_a,
                    'avg_revenue_b': avg_post_revenue_b
                },
                'changes': {
                    'strategy_change_a': strategy_change_a,
                    'strategy_change_b': strategy_change_b,
                    'revenue_change_a': revenue_change_a,
                    'revenue_change_b': revenue_change_b,
                    'recovery_time': recovery_time
                }
            }
            
            analysis['shock_impacts'].append(impact)
            
            # 记录策略和收益变化
            analysis['strategy_changes'].append({
                'shock_type': shock_type,
                'round': shock_round,
                'player_a': strategy_change_a,
                'player_b': strategy_change_b
            })
            
            analysis['revenue_changes'].append({
                'shock_type': shock_type,
                'round': shock_round,
                'player_a': revenue_change_a,
                'player_b': revenue_change_b,
                'recovery_time': recovery_time
            })
            
            # 更新冲击类型分析
            analysis['shock_type_analysis'][shock_type]['count'] += 1
            analysis['shock_type_analysis'][shock_type]['avg_strategy_change_a'] += strategy_change_a
            analysis['shock_type_analysis'][shock_type]['avg_strategy_change_b'] += strategy_change_b
            analysis['shock_type_analysis'][shock_type]['avg_revenue_change_a'] += revenue_change_a
            analysis['shock_type_analysis'][shock_type]['avg_revenue_change_b'] += revenue_change_b
            analysis['shock_type_analysis'][shock_type]['recovery_time'] += recovery_time
        
        # 计算每种冲击类型的平均值
        for shock_type, data in analysis['shock_type_analysis'].items():
            if data['count'] > 0:
                data['avg_strategy_change_a'] /= data['count']
                data['avg_strategy_change_b'] /= data['count']
                data['avg_revenue_change_a'] /= data['count']
                data['avg_revenue_change_b'] /= data['count']
                data['recovery_time'] /= data['count']
        
        return analysis
    
    def visualize_shock_effects(self, result: ExperimentResult, save_path: str = None):
        """
        可视化冲击效果
        
        Args:
            result: 实验结果对象
            save_path: 图表保存路径
        """
        if not result.round_results or not self.shock_events:
            logger.warning("没有足够的数据来可视化冲击效果")
            return
        
        import matplotlib.pyplot as plt
        import matplotlib.font_manager as fm
        from matplotlib.gridspec import GridSpec
        
        # 设置中文字体
        font_path = r"C:\Windows\Fonts\STZHONGS.TTF"
        if os.path.exists(font_path):
            fontprop = fm.FontProperties(fname=font_path)
            plt.rcParams['font.family'] = fontprop.get_name()
            plt.rcParams['axes.unicode_minus'] = False
        
        # 创建多个子图
        fig = plt.figure(figsize=(18, 24))
        gs = GridSpec(6, 2, figure=fig)  # 增加到6行
        
        # 提取数据
        rounds = [r.round_number for r in result.round_results]
        strategies_a = [r.player_a_strategy for r in result.round_results]
        strategies_b = [r.player_b_strategy for r in result.round_results]
        revenues_a = [r.player_a_revenue for r in result.round_results]
        revenues_b = [r.player_b_revenue for r in result.round_results]
        
        # 计算收益差值的绝对值
        revenue_abs_diff = [abs(a - b) for a, b in zip(revenues_a, revenues_b)]
        
        # 将轮次转换为天
        # 每24小时为1天，收益聚合为每天总收益
        days = [r // 24 + 1 for r in rounds]  # 轮次转换为天数（从1开始）
        unique_days = sorted(list(set(days)))
        
        # 计算每天的总收益
        daily_revenues_a = {}
        daily_revenues_b = {}
        daily_abs_diff = {}
        daily_strategies_a = {}  # 计算每天的平均策略
        daily_strategies_b = {}
        daily_count = {}  # 记录每天的数据点数量
        
        for day in unique_days:
            daily_revenues_a[day] = 0
            daily_revenues_b[day] = 0
            daily_abs_diff[day] = 0
            daily_strategies_a[day] = 0
            daily_strategies_b[day] = 0
            daily_count[day] = 0
        
        for i, day in enumerate(days):
            daily_revenues_a[day] += revenues_a[i]
            daily_revenues_b[day] += revenues_b[i]
            daily_abs_diff[day] += revenue_abs_diff[i]
            daily_strategies_a[day] += strategies_a[i]
            daily_strategies_b[day] += strategies_b[i]
            daily_count[day] += 1
        
        # 计算每天平均策略
        for day in unique_days:
            if daily_count[day] > 0:
                daily_strategies_a[day] /= daily_count[day]
                daily_strategies_b[day] /= daily_count[day]
        
        # 转换为列表形式
        days_list = list(unique_days)
        daily_rev_a_list = [daily_revenues_a[day] for day in days_list]
        daily_rev_b_list = [daily_revenues_b[day] for day in days_list]
        daily_abs_diff_list = [daily_abs_diff[day] for day in days_list]
        daily_strat_a_list = [daily_strategies_a[day] for day in days_list]
        daily_strat_b_list = [daily_strategies_b[day] for day in days_list]
        
        # 分析冲击效果
        shock_analysis = self.analyze_shock_effects(result)
        
        # 绘制策略图
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(days_list, daily_strat_a_list, 'b-', label='玩家A策略')
        ax1.plot(days_list, daily_strat_b_list, 'r-', label='玩家B策略')
        ax1.set_title('策略随时间变化（日均）', fontproperties=fontprop, fontsize=14)
        ax1.set_xlabel('天数', fontproperties=fontprop)
        ax1.set_ylabel('策略值（价格）', fontproperties=fontprop)
        ax1.legend(prop=fontprop)
        ax1.grid(True)
        
        # 标记冲击事件（转换为天）
        for shock in self.shock_events:
            shock_day = shock['round'] // 24 + 1
            shock_end_day = (shock['round'] + shock['duration']) // 24 + 1
            shock_type = shock['type']
            
            # 不同类型的冲击用不同颜色标记
            color = {
                'demand_surge': 'green',
                'supply_shortage': 'red',
                'price_regulation': 'purple',
                'competition_increase': 'orange',
                'demand_drop': 'brown',
                'market_disruption': 'magenta',
                'technology_shift': 'cyan',
                'market_crash': 'black'
            }.get(shock_type, 'gray')
            
            # 在策略图中标记冲击区间
            ax1.axvspan(shock_day, shock_end_day, alpha=0.2, color=color)
            ax1.axvline(x=shock_day, color=color, linestyle='--')
            ax1.text(shock_day, min(daily_strat_a_list + daily_strat_b_list) - 2, 
                      f"{shock_type}", rotation=90, 
                      verticalalignment='bottom', fontproperties=fontprop)
        
        # 计算每天收益差值的绝对值（先计算每天总收益，再取差值绝对值）
        daily_abs_diff_list = [abs(a - b) for a, b in zip(daily_rev_a_list, daily_rev_b_list)]
        
        # 创建上下两个子图，显示全部30天
        ax_top = fig.add_subplot(gs[1, :])
        ax_bottom = fig.add_subplot(gs[2, :], sharex=ax_top)
        
        # 定义新的配色方案 (将RGB值转换为matplotlib可识别的格式)
        color_a = (250/255, 170/255, 137/255)  # 橙红色
        color_b = (255/255, 220/255, 126/255)  # 黄色
        color_diff = (126/255, 208/255, 248/255)  # 蓝色
        
        # 上部分：收益曲线
        ax_top.plot(days_list, daily_rev_a_list, label='玩家A收益', alpha=0.8, color=color_a, linewidth=1.5)
        ax_top.plot(days_list, daily_rev_b_list, label='玩家B收益', alpha=0.8, color=color_b, linewidth=1.5)
        ax_top.set_title(f'市场冲击测试日收益对比 (30天)', fontproperties=fontprop, fontsize=14)
        ax_top.set_ylabel('日收益（元/天）', fontproperties=fontprop)
        ax_top.legend(prop=fontprop)
        ax_top.grid(True, alpha=0.3)
        
        # 设置x轴刻度为整数天
        ax_top.set_xticks(list(range(1, max(days_list)+1, 2)))  # 每隔2天显示一个刻度
        
        # 标记冲击区间
        for shock in self.shock_events:
            shock_day = shock['round'] // 24 + 1
            shock_end_day = (shock['round'] + shock['duration']) // 24 + 1
            shock_type = shock['type']
            
            color = {
                'demand_surge': 'green',
                'supply_shortage': 'red',
                'price_regulation': 'purple',
                'competition_increase': 'orange',
                'demand_drop': 'brown',
                'market_disruption': 'magenta',
                'technology_shift': 'cyan',
                'market_crash': 'black'
            }.get(shock_type, 'gray')
            
            ax_top.axvspan(shock_day, shock_end_day, alpha=0.2, color=color)
            ax_top.axvline(x=shock_day, color=color, linestyle='--')
            # 添加标签（只在上图显示）
            ax_top.text(shock_day, min(daily_rev_a_list + daily_rev_b_list) - 2, 
                      f"{shock_type}", rotation=90, 
                      verticalalignment='bottom', fontproperties=fontprop, fontsize=8)
        
        # 下部分：收益差值绝对值
        ax_bottom.plot(days_list, daily_abs_diff_list, label='收益差值绝对值', color=color_diff, alpha=0.8, linewidth=1.5)
        ax_bottom.fill_between(days_list, daily_abs_diff_list, alpha=0.3, color=color_diff)
        ax_bottom.set_xlabel('天数', fontproperties=fontprop)
        ax_bottom.set_ylabel('收益差值绝对值（元/天）', fontproperties=fontprop)
        ax_bottom.grid(True, alpha=0.3)
        ax_bottom.legend(prop=fontprop)
        
        # 标记冲击区间（下部图）
        for shock in self.shock_events:
            shock_day = shock['round'] // 24 + 1
            shock_end_day = (shock['round'] + shock['duration']) // 24 + 1
            color = {
                'demand_surge': 'green',
                'supply_shortage': 'red',
                'price_regulation': 'purple',
                'competition_increase': 'orange',
                'demand_drop': 'brown',
                'market_disruption': 'magenta',
                'technology_shift': 'cyan',
                'market_crash': 'black'
            }.get(shock['type'], 'gray')
            
            ax_bottom.axvspan(shock_day, shock_end_day, alpha=0.2, color=color)
            ax_bottom.axvline(x=shock_day, color=color, linestyle='--')
        
        # 绘制冲击影响分析
        if shock_analysis['strategy_changes'] and shock_analysis['revenue_changes']:
            # 变化率对比图
            ax3 = fig.add_subplot(gs[4, 0])
            shock_types = [sc['shock_type'] for sc in shock_analysis['strategy_changes']]
            strategy_changes_a = [sc['player_a'] * 100 for sc in shock_analysis['strategy_changes']]  # 转为百分比
            strategy_changes_b = [sc['player_b'] * 100 for sc in shock_analysis['strategy_changes']]
            revenue_changes_a = [rc['player_a'] * 100 for rc in shock_analysis['revenue_changes']]
            revenue_changes_b = [rc['player_b'] * 100 for rc in shock_analysis['revenue_changes']]
            
            x = np.arange(len(shock_types))
            width = 0.2
            
            # 策略和收益变化率比较
            ax3.bar(x - 1.5*width, strategy_changes_a, width, label='玩家A策略变化率', color='skyblue')
            ax3.bar(x - 0.5*width, strategy_changes_b, width, label='玩家B策略变化率', color='lightcoral')
            ax3.bar(x + 0.5*width, revenue_changes_a, width, label='玩家A收益变化率', color='blue')
            ax3.bar(x + 1.5*width, revenue_changes_b, width, label='玩家B收益变化率', color='red')
            
            ax3.set_title('冲击影响分析（百分比变化）', fontproperties=fontprop, fontsize=14)
            ax3.set_xticks(x)
            ax3.set_xticklabels(shock_types, fontproperties=fontprop, rotation=45)
            ax3.set_ylabel('变化率（%）', fontproperties=fontprop)
            ax3.legend(prop=fontprop)
            ax3.grid(True, axis='y')
            
            # 添加水平线表示无变化基准线
            ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            
            # 恢复时间分析
            ax4 = fig.add_subplot(gs[4, 1])
            recovery_times = [rc.get('recovery_time', 0) for rc in shock_analysis['revenue_changes']]
            
            ax4.bar(x, recovery_times, width*2, label='恢复时间', color='purple')
            ax4.set_title('冲击后恢复时间分析', fontproperties=fontprop, fontsize=14)
            ax4.set_xticks(x)
            ax4.set_xticklabels(shock_types, fontproperties=fontprop, rotation=45)
            ax4.set_ylabel('恢复轮次', fontproperties=fontprop)
            ax4.grid(True, axis='y')
            
            # 冲击类型分析对比
            if 'shock_type_analysis' in shock_analysis and shock_analysis['shock_type_analysis']:
                ax5 = fig.add_subplot(gs[5, :])
                
                shock_types_analysis = list(shock_analysis['shock_type_analysis'].keys())
                avg_strategy_changes_a = [data['avg_strategy_change_a'] * 100 for _, data in shock_analysis['shock_type_analysis'].items()]
                avg_strategy_changes_b = [data['avg_strategy_change_b'] * 100 for _, data in shock_analysis['shock_type_analysis'].items()]
                avg_revenue_changes_a = [data['avg_revenue_change_a'] * 100 for _, data in shock_analysis['shock_type_analysis'].items()]
                avg_revenue_changes_b = [data['avg_revenue_change_b'] * 100 for _, data in shock_analysis['shock_type_analysis'].items()]
                recovery_times = [data['recovery_time'] for _, data in shock_analysis['shock_type_analysis'].items()]
                
                x_types = np.arange(len(shock_types_analysis))
                
                # 创建分组柱状图
                ax5.bar(x_types - 0.3, avg_strategy_changes_a, width=0.15, color='skyblue', label='A策略变化率(%)')
                ax5.bar(x_types - 0.15, avg_strategy_changes_b, width=0.15, color='lightcoral', label='B策略变化率(%)')
                ax5.bar(x_types, avg_revenue_changes_a, width=0.15, color='blue', label='A收益变化率(%)')
                ax5.bar(x_types + 0.15, avg_revenue_changes_b, width=0.15, color='red', label='B收益变化率(%)')
                
                # 绘制次坐标轴的恢复时间
                ax5_twin = ax5.twinx()
                ax5_twin.plot(x_types, recovery_times, 'o-', color='purple', label='恢复时间(轮次)')
                ax5_twin.set_ylabel('恢复轮次', fontproperties=fontprop)
                
                ax5.set_title('各类冲击平均影响对比', fontproperties=fontprop, fontsize=14)
                ax5.set_xticks(x_types)
                ax5.set_xticklabels(shock_types_analysis, fontproperties=fontprop, rotation=45)
                ax5.set_ylabel('变化率（%）', fontproperties=fontprop)
                ax5.legend(prop=fontprop, loc='upper left')
                ax5_twin.legend(prop=fontprop, loc='upper right')
                ax5.grid(True)
        
        plt.tight_layout()
        
        # 保存图表
        if save_path:
            plt.savefig(save_path)
            logger.info(f"冲击效果可视化已保存至: {save_path}")
        else:
            plt.savefig(f"results/figures/shock_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            logger.info("冲击效果可视化已保存")
        
        plt.close() 