"""
博弈论理论框架
定义不完全信息动态博弈和纳什均衡的核心概念
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Set, Callable, Optional
from dataclasses import dataclass, field
from enum import Enum
import logging
import itertools
import hashlib

logger = logging.getLogger(__name__)

class PlayerType(Enum):
    """玩家类型枚举"""
    AGGRESSIVE = "aggressive"     # 激进型
    CONSERVATIVE = "conservative" # 保守型
    BALANCED = "balanced"         # 平衡型


@dataclass
class InformationSet:
    """信息集类"""
    player_id: str                            # 所属玩家
    observable_history: List[Dict[str, Any]]  # 可观察历史
    observable_market: Dict[str, Any]         # 可观察市场状态
    unique_id: str = None                     # 唯一标识符
    
    def __post_init__(self):
        if self.unique_id is None:
            # 生成唯一标识符
            history_str = str(self.observable_history)
            market_str = str(self.observable_market)
            combined = f"{self.player_id}:{history_str}:{market_str}"
            self.unique_id = hashlib.md5(combined.encode()).hexdigest()
    
    def __hash__(self):
        return hash(self.unique_id)
    
    def __eq__(self, other):
        if not isinstance(other, InformationSet):
            return False
        return self.unique_id == other.unique_id


@dataclass
class Action:
    """行动类"""
    player_id: str        # 执行行动的玩家
    action_type: str      # 行动类型
    action_value: float   # 行动值
    
    def __hash__(self):
        return hash((self.player_id, self.action_type, self.action_value))
    
    def __eq__(self, other):
        if not isinstance(other, Action):
            return False
        return (self.player_id == other.player_id and 
                self.action_type == other.action_type and 
                self.action_value == other.action_value)


@dataclass
class History:
    """历史类"""
    actions: List[Action] = field(default_factory=list)  # 行动序列
    market_states: List[Dict[str, Any]] = field(default_factory=list)  # 市场状态序列
    observation_noise_scale: float = 0.1  # 观察噪声标准差（占实际值比例）
    price_observability: float = 0.8  # 观察到对手真实价格的概率
    
    def add_action(self, action: Action, market_state: Dict[str, Any]):
        """添加行动到历史"""
        self.actions.append(action)
        self.market_states.append(market_state.copy())
    
    def get_player_actions(self, player_id: str) -> List[Action]:
        """获取特定玩家的行动"""
        return [a for a in self.actions if a.player_id == player_id]
    
    def get_observable_history(self, player_id: str) -> List[Dict[str, Any]]:
        """获取特定玩家可观察的历史"""
        observable = []
        for i, action in enumerate(self.actions):
            # 玩家可以观察自己的所有行动和其他玩家的部分行动
            if action.player_id == player_id:
                observable.append({
                    'round': i,
                    'player': action.player_id,
                    'action_type': action.action_type,
                    'action_value': action.action_value,
                    'market_state': self._filter_observable_market(player_id, self.market_states[i])
                })
            else:
                # 对其他玩家的行动，只能观察部分信息，根据可观察性概率决定是否有噪声
                if np.random.random() < self.price_observability:
                    # 可以观察到真实价格，但带有噪声
                    observable.append({
                        'round': i,
                        'player': action.player_id,
                        'action_type': action.action_type,
                        'action_value_estimate': self._add_observation_noise(action.action_value),
                        'market_state': self._filter_observable_market(player_id, self.market_states[i])
                    })
                else:
                    # 无法观察到价格，只知道采取了行动
                    observable.append({
                        'round': i,
                        'player': action.player_id,
                        'action_type': action.action_type,
                        'action_observed': False,
                        'market_state': self._filter_observable_market(player_id, self.market_states[i])
                    })
        return observable
    
    def _filter_observable_market(self, player_id: str, market_state: Dict[str, Any]) -> Dict[str, Any]:
        """过滤玩家可观察的市场状态"""
        observable = {}
        for key, value in market_state.items():
            # 只保留玩家可观察的市场因素
            if key in ['demand_level', 'time_period', 'location_factor']:
                observable[key] = value
        return observable
    
    def _add_observation_noise(self, value: float) -> float:
        """添加观察噪声"""
        # 添加指定程度的观察噪声
        noise = np.random.normal(0, value * self.observation_noise_scale)
        return value + noise


class BeliefSystem:
    """信念系统类"""
    
    def __init__(self, player_ids: List[str], player_types: List[str],
                belief_update_rate: float = 0.2,
                prior_strength: float = 2.0,
                likelihood_smoothing: float = 0.1):
        """
        初始化信念系统
        
        Args:
            player_ids: 玩家ID列表
            player_types: 玩家类型列表
            belief_update_rate: 信念更新速率 (0-1)
            prior_strength: 先验信念强度
            likelihood_smoothing: 似然平滑因子
        """
        self.player_ids = player_ids
        self.player_types = player_types
        self.beliefs = {}
        self.belief_update_rate = belief_update_rate
        self.prior_strength = prior_strength
        self.likelihood_smoothing = likelihood_smoothing
        self.initialize_beliefs()
    
    def initialize_beliefs(self):
        """初始化信念为均匀分布"""
        for player in self.player_ids:
            self.beliefs[player] = {}
    
    def get_belief(self, player_id: str, info_set: InformationSet) -> Dict[str, Dict[str, float]]:
        """获取特定信息集上的信念"""
        if info_set.unique_id not in self.beliefs[player_id]:
            # 如果该信息集上没有信念，初始化为均匀分布
            self.beliefs[player_id][info_set.unique_id] = {}
            for opponent in [p for p in self.player_ids if p != player_id]:
                self.beliefs[player_id][info_set.unique_id][opponent] = {
                    type_name: 1.0/len(self.player_types) for type_name in self.player_types
                }
        
        return self.beliefs[player_id][info_set.unique_id]
    
    def update_belief(self, player_id: str, info_set: InformationSet, 
                      opponent_id: str, observed_action: Action, 
                      type_likelihoods: Dict[str, float]):
        """更新信念"""
        # 获取当前信念
        current_belief = self.get_belief(player_id, info_set)
        
        # 应用贝叶斯更新
        prior = current_belief.get(opponent_id, {})
        if not prior:
            prior = {type_name: 1.0/len(self.player_types) for type_name in self.player_types}
            current_belief[opponent_id] = prior
        
        # 添加平滑因子，确保似然不为零
        smoothed_likelihoods = {
            type_name: max(self.likelihood_smoothing, likelihood)
            for type_name, likelihood in type_likelihoods.items()
        }
        
        # 贝叶斯更新
        normalization_constant = 0.0
        posterior = {}
        
        for type_name in self.player_types:
            prior_prob = prior.get(type_name, 1.0/len(self.player_types))
            likelihood = smoothed_likelihoods.get(type_name, self.likelihood_smoothing)
            
            # 计算后验概率 (未归一化)
            posterior[type_name] = prior_prob * likelihood
            normalization_constant += posterior[type_name]
        
        # 归一化
        if normalization_constant > 0:
            for type_name in posterior:
                posterior[type_name] /= normalization_constant
        
        # 应用学习率
        for type_name in posterior:
            old_belief = prior.get(type_name, 1.0/len(self.player_types))
            posterior[type_name] = (
                old_belief * (1 - self.belief_update_rate) + 
                posterior[type_name] * self.belief_update_rate
            )
        
        # 更新信念
        current_belief[opponent_id] = posterior


class DynamicBayesianGame:
    """不完全信息动态博弈类"""
    
    def __init__(self, player_ids: List[str], player_types: List[str], 
                 action_space: Dict[str, List[float]] = None,
                 payoff_function: Callable = None,
                 observation_noise_scale: float = 0.1,
                 price_observability: float = 0.8):
        """
        初始化不完全信息动态博弈
        
        Args:
            player_ids: 玩家ID列表
            player_types: 玩家类型列表
            action_space: 玩家行动空间
            payoff_function: 收益函数
            observation_noise_scale: 观察噪声比例
            price_observability: 价格可观察概率
        """
        self.player_ids = player_ids
        self.player_types = player_types
        self.action_space = action_space or {}
        self.payoff_function = payoff_function or self._default_payoff_function
        
        # 初始化信念系统
        self.belief_system = BeliefSystem(player_ids, player_types)
        
        # 初始化历史
        self.history = History(
            observation_noise_scale=observation_noise_scale,
            price_observability=price_observability
        )
        
        # 策略注册
        self.strategies = {}
        
        logger.info(f"不完全信息动态博弈初始化完成: {len(player_ids)}个玩家, {len(player_types)}种类型")
    
    def get_information_set(self, player_id: str, market_state: Dict[str, Any]) -> InformationSet:
        """获取玩家当前的信息集"""
        observable_history = self.history.get_observable_history(player_id)
        observable_market = self.history._filter_observable_market(player_id, market_state)
        
        return InformationSet(
            player_id=player_id,
            observable_history=observable_history,
            observable_market=observable_market
        )
    
    def add_round(self, actions: Dict[str, Action], market_state: Dict[str, Any]):
        """添加一轮博弈结果"""
        for action in actions.values():
            self.history.add_action(action, market_state)
    
    def register_strategy(self, player_id: str, strategy_function: Callable):
        """注册玩家策略函数"""
        self.strategies[player_id] = strategy_function
    
    def _default_payoff_function(self, player_id: str, action: Action,
                                opponent_actions: Dict[str, Action], 
                                market_state: Dict[str, Any]) -> float:
        """默认收益函数"""
        # 简单的价格竞争收益函数
        price = action.action_value
        competitor_prices = [a.action_value for a in opponent_actions.values()]
        
        if not competitor_prices:
            return price * 10  # 垄断情况
        
        avg_competitor_price = np.mean(competitor_prices)
        
        # 价格越低相对于竞争对手，订单越多，但价格降低影响收益
        if price < avg_competitor_price:
            order_volume = 15 * (avg_competitor_price / price)
        elif price > avg_competitor_price:
            order_volume = 10 * (avg_competitor_price / price)
            else:
            order_volume = 12
            
        return price * order_volume
        

class NashEquilibriumAnalyzer:
    """纳什均衡分析器"""
    
    def __init__(self, game: DynamicBayesianGame, tolerance: float = 0.1):
        """
        初始化纳什均衡分析器
        
        Args:
            game: 博弈实例
            tolerance: 均衡检验容忍度
        """
        self.game = game
        self.tolerance = tolerance
        logger.info(f"纳什均衡分析器初始化完成，容忍度: {tolerance}")
    
    def check_nash_equilibrium(self, strategies: Dict[str, float], 
                              payoffs: Dict[str, float]) -> Tuple[bool, Dict[str, Any]]:
        """
        检查当前策略组合是否构成纳什均衡
        
        Args:
            strategies: 当前策略组合
            payoffs: 当前收益组合
        
        Returns:
            是否为纳什均衡及详细分析
        """
        is_equilibrium = True
        analysis = {
            'players_at_equilibrium': {},
            'max_beneficial_deviation': 0.0,
            'equilibrium_type': 'unknown'
        }
        
        for player_id in strategies.keys():
            current_strategy = strategies[player_id]
            current_payoff = payoffs[player_id]
            
            # 检查是否存在有利偏离策略
            max_deviation_payoff = current_payoff
            best_deviation_strategy = current_strategy
            
            # 在行动空间中测试所有可能的偏离策略
            test_strategies = self.game.action_space.get(player_id, 
                                                        [i for i in range(10, 51)])
            
            for test_strategy in test_strategies:
                if abs(test_strategy - current_strategy) < self.tolerance:
                    continue
                    
                # 估计偏离策略的收益
                estimated_payoff = self._estimate_deviation_payoff(
                    player_id, test_strategy, strategies, payoffs
                )
                
                if estimated_payoff > max_deviation_payoff + self.tolerance:
                    max_deviation_payoff = estimated_payoff
                    best_deviation_strategy = test_strategy
                    is_equilibrium = False
            
            # 记录分析结果
            analysis['players_at_equilibrium'][player_id] = {
                'at_equilibrium': abs(best_deviation_strategy - current_strategy) < self.tolerance,
                'best_response': best_deviation_strategy,
                'deviation_gain': max_deviation_payoff - current_payoff
            }
            
            analysis['max_beneficial_deviation'] = max(
                analysis['max_beneficial_deviation'],
                max_deviation_payoff - current_payoff
            )
        
        # 确定均衡类型
        if is_equilibrium:
            analysis['equilibrium_type'] = 'nash_equilibrium'
        elif analysis['max_beneficial_deviation'] < self.tolerance * 2:
            analysis['equilibrium_type'] = 'approximate_equilibrium'
        else:
            analysis['equilibrium_type'] = 'not_equilibrium'
        
        return is_equilibrium, analysis
    
    def _estimate_deviation_payoff(self, player_id: str, deviation_strategy: float,
                                  current_strategies: Dict[str, float],
                                  current_payoffs: Dict[str, float]) -> float:
        """估计偏离策略的收益"""
        # 计算相对价格
        other_strategies = [s for p, s in current_strategies.items() if p != player_id]
        avg_other_strategy = np.mean(other_strategies) if other_strategies else 25.0
        relative_price = deviation_strategy / avg_other_strategy
        
        # 收益模型
        if relative_price > 1.1:  # 高价策略
            order_volume_factor = 0.7 / relative_price
            price_premium = relative_price
        elif relative_price < 0.9:  # 低价策略
            order_volume_factor = 1.3 * (2 - relative_price)
            price_premium = relative_price
        else:  # 中等策略
            order_volume_factor = 1.0
            price_premium = 1.0
        
        # 计算预期收益
        base_payoff = current_payoffs.get(player_id, 20.0)
        estimated_payoff = base_payoff * order_volume_factor * price_premium
        
        # 添加市场不确定性
        noise = np.random.normal(0, estimated_payoff * 0.1)
        return estimated_payoff + noise


# ... existing code ... 