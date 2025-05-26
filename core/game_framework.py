"""
博弈框架核心模块
定义博弈的基本要素和执行逻辑
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import logging

from config.game_config import GameConfig


class GamePhase(Enum):
    """博弈阶段枚举"""
    EXPLORATION = "exploration"    # 探索期
    LEARNING = "learning"         # 学习期  
    EQUILIBRIUM = "equilibrium"   # 均衡期


@dataclass
class GameState:
    """博弈状态类，记录当前博弈的所有关键信息"""
    
    round_number: int = 0
    phase: GamePhase = GamePhase.EXPLORATION
    
    # 当前策略状态
    current_strategies: Dict[str, float] = field(default_factory=dict)
    
    # 市场状态
    market_state: Dict[str, Any] = field(default_factory=dict)
    
    # 历史信息
    strategy_history: List[Dict[str, float]] = field(default_factory=list)
    payoff_history: List[Dict[str, float]] = field(default_factory=list)
    market_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # 博弈统计信息
    total_orders: int = 0
    total_revenue: float = 0.0
    
    def update_round(self, round_num: int, config: GameConfig):
        """更新轮次和阶段"""
        self.round_number = round_num
        
        if round_num <= config.EXPLORATION_ROUNDS:
            self.phase = GamePhase.EXPLORATION
        elif round_num <= config.LEARNING_ROUNDS:
            self.phase = GamePhase.LEARNING  
        else:
            self.phase = GamePhase.EQUILIBRIUM
    
    def record_round_data(self, strategies: Dict[str, float], 
                         payoffs: Dict[str, float], 
                         market_data: Dict[str, Any]):
        """记录单轮博弈数据"""
        self.current_strategies = strategies.copy()
        self.strategy_history.append(strategies.copy())
        self.payoff_history.append(payoffs.copy())
        self.market_history.append(market_data.copy())
        
        # 更新统计信息
        self.total_orders += market_data.get('total_orders', 0)
        self.total_revenue += sum(payoffs.values())
    
    def get_recent_strategies(self, window: int = 10) -> Dict[str, List[float]]:
        """获取最近N轮的策略历史"""
        recent_history = self.strategy_history[-window:]
        result = {}
        
        if recent_history:
            for player in recent_history[0].keys():
                result[player] = [round_data[player] for round_data in recent_history]
        
        return result
    
    def get_recent_payoffs(self, window: int = 10) -> Dict[str, List[float]]:
        """获取最近N轮的收益历史"""
        recent_history = self.payoff_history[-window:]
        result = {}
        
        if recent_history:
            for player in recent_history[0].keys():
                result[player] = [round_data[player] for round_data in recent_history]
        
        return result
    
    def calculate_strategy_stability(self, player: str, window: int = 20) -> float:
        """计算策略稳定性（方差的倒数）"""
        recent_strategies = self.get_recent_strategies(window)
        
        if player not in recent_strategies or len(recent_strategies[player]) < 2:
            return 0.0
        
        strategy_variance = np.var(recent_strategies[player])
        return 1.0 / (1.0 + strategy_variance)  # 稳定性指标
    
    def is_converged(self, window: int = 30, tolerance: float = 0.1) -> bool:
        """检查策略是否已收敛"""
        if len(self.strategy_history) < window:
            return False
        
        recent_strategies = self.get_recent_strategies(window)
        
        for player_strategies in recent_strategies.values():
            if len(player_strategies) < window:
                return False
            
            # 计算最近策略的标准差
            std = np.std(player_strategies)
            if std > tolerance:
                return False
        
        return True


class GameFramework:
    """博弈框架主类，负责协调整个博弈过程"""
    
    def __init__(self, config: GameConfig):
        """
        初始化博弈框架
        
        Args:
            config: 博弈配置对象
        """
        self.config = config
        self.game_state = GameState()
        self.logger = self._setup_logger()
        
        # 初始化随机种子
        np.random.seed(config.RANDOM_SEED)
        
        # 存储玩家信息
        self.players = {}
        
        self.logger.info(f"博弈框架初始化完成，参与者数量: {config.NUM_PLAYERS}")
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger('GameFramework')
        logger.setLevel(getattr(logging, self.config.LOG_LEVEL))
        
        if not logger.handlers:
            # 控制台处理器
            if self.config.LOG_TO_CONSOLE:
                console_handler = logging.StreamHandler()
                console_formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
                console_handler.setFormatter(console_formatter)
                logger.addHandler(console_handler)
            
            # 文件处理器
            if self.config.LOG_TO_FILE:
                file_handler = logging.FileHandler('results/logs/game_framework.log')
                file_formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
                file_handler.setFormatter(file_formatter)
                logger.addHandler(file_handler)
        
        return logger
    
    def validate_strategies(self, strategies: Dict[str, float]) -> bool:
        """验证策略的有效性"""
        for player, strategy in strategies.items():
            if not (self.config.MIN_PRICE_THRESHOLD <= strategy <= self.config.MAX_PRICE_THRESHOLD):
                self.logger.warning(f"玩家 {player} 的策略 {strategy} 超出有效范围")
                return False
        return True
    
    def calculate_strategy_change_cost(self, old_strategy: float, new_strategy: float) -> float:
        """计算策略变化成本"""
        change = abs(new_strategy - old_strategy)
        return self.config.STRATEGY_STABILITY_PENALTY * change
    
    def get_market_information(self, player: str, round_num: int) -> Dict[str, Any]:
        """获取特定玩家在当前轮次可观察到的市场信息"""
        base_info = {
            'round_number': round_num,
            'phase': self.game_state.phase.value,
            'time_period': self.config.get_time_period(round_num % 24),
        }
        
        # 添加历史信息（有限的对手观察）
        if len(self.game_state.strategy_history) > 0:
            # 玩家只能观察到对手的部分信息（通过行为推测）
            recent_strategies = self.game_state.get_recent_strategies(5)
            base_info['opponent_behavior_estimate'] = self._estimate_opponent_behavior(
                player, recent_strategies
            )
        
        # 添加自身历史收益
        recent_payoffs = self.game_state.get_recent_payoffs(10)
        if player in recent_payoffs:
            base_info['my_recent_payoffs'] = recent_payoffs[player]
        
        return base_info
    
    def _estimate_opponent_behavior(self, player: str, recent_strategies: Dict[str, List[float]]) -> Dict[str, Any]:
        """估计对手行为模式（不完全信息博弈的体现）"""
        opponents = [p for p in recent_strategies.keys() if p != player]
        behavior_estimate = {}
        
        for opponent in opponents:
            if opponent in recent_strategies and recent_strategies[opponent]:
                strategies = recent_strategies[opponent]
                behavior_estimate[opponent] = {
                    'avg_strategy': np.mean(strategies),
                    'strategy_trend': strategies[-1] - strategies[0] if len(strategies) > 1 else 0,
                    'volatility': np.std(strategies) if len(strategies) > 1 else 0,
                    'recent_strategy_estimate': strategies[-1] + np.random.normal(0, 1)  # 添加观察噪声
                }
        
        return behavior_estimate
    
    def check_nash_equilibrium(self, strategies: Dict[str, float], 
                              payoffs: Dict[str, float], 
                              tolerance: float = None) -> Tuple[bool, Dict[str, Any]]:
        """检查当前策略组合是否构成纳什均衡"""
        if tolerance is None:
            tolerance = self.config.EQUILIBRIUM_TOLERANCE
        
        is_equilibrium = True
        analysis = {
            'players_at_equilibrium': {},
            'max_beneficial_deviation': 0.0,
            'equilibrium_type': 'unknown'
        }
        
        for player in strategies.keys():
            current_strategy = strategies[player]
            current_payoff = payoffs[player]
            
            # 测试偏离策略的收益
            max_deviation_payoff = current_payoff
            best_deviation_strategy = current_strategy
            
            # 在策略空间中搜索更好的策略
            test_strategies = np.arange(
                self.config.MIN_PRICE_THRESHOLD,
                self.config.MAX_PRICE_THRESHOLD + self.config.PRICE_STEP,
                self.config.PRICE_STEP
            )
            
            for test_strategy in test_strategies:
                if abs(test_strategy - current_strategy) < tolerance:
                    continue
                
                # 创建测试策略组合
                test_strategy_profile = strategies.copy()
                test_strategy_profile[player] = test_strategy
                
                # 这里需要重新计算收益（简化版本）
                estimated_payoff = self._estimate_deviation_payoff(
                    player, test_strategy, strategies, payoffs
                )
                
                if estimated_payoff > max_deviation_payoff + tolerance:
                    max_deviation_payoff = estimated_payoff
                    best_deviation_strategy = test_strategy
                    is_equilibrium = False
            
            analysis['players_at_equilibrium'][player] = {
                'at_equilibrium': abs(best_deviation_strategy - current_strategy) < tolerance,
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
        elif analysis['max_beneficial_deviation'] < tolerance * 2:
            analysis['equilibrium_type'] = 'approximate_equilibrium'
        else:
            analysis['equilibrium_type'] = 'not_equilibrium'
        
        return is_equilibrium, analysis
    
    def _estimate_deviation_payoff(self, player: str, deviation_strategy: float,
                                  current_strategies: Dict[str, float],
                                  current_payoffs: Dict[str, float]) -> float:
        """
        估计偏离策略的收益（简化版本）
        在实际应用中，这里应该调用完整的市场仿真
        """
        # 简化的收益估计：基于相对定价策略
        other_strategies = [s for p, s in current_strategies.items() if p != player]
        avg_other_strategy = np.mean(other_strategies) if other_strategies else 25.0
        
        # 如果定价更高，可能获得更高单价但订单量减少
        # 如果定价更低，可能获得更多订单但单价降低
        relative_price = deviation_strategy / avg_other_strategy
        
        # 简化的收益模型
        if relative_price > 1.1:  # 高价策略
            order_volume_factor = 0.7 / relative_price
            price_premium = relative_price
        elif relative_price < 0.9:  # 低价策略
            order_volume_factor = 1.3 * (2 - relative_price)
            price_premium = relative_price
        else:  # 中等策略
            order_volume_factor = 1.0
            price_premium = 1.0
        
        base_payoff = current_payoffs.get(player, 20.0)
        estimated_payoff = base_payoff * order_volume_factor * price_premium
        
        # 添加一些随机性来模拟市场不确定性
        noise = np.random.normal(0, estimated_payoff * 0.1)
        return estimated_payoff + noise
    
    def get_game_statistics(self) -> Dict[str, Any]:
        """获取当前博弈的统计信息"""
        stats = {
            'basic_info': {
                'current_round': self.game_state.round_number,
                'current_phase': self.game_state.phase.value,
                'total_rounds_played': len(self.game_state.strategy_history),
                'convergence_status': self.game_state.is_converged()
            },
            'strategy_analysis': {},
            'payoff_analysis': {},
            'stability_metrics': {}
        }
        
        if self.game_state.strategy_history:
            # 策略分析
            recent_strategies = self.game_state.get_recent_strategies(30)
            for player, strategies in recent_strategies.items():
                if strategies:
                    stats['strategy_analysis'][player] = {
                        'current_strategy': strategies[-1],
                        'mean_strategy': np.mean(strategies),
                        'strategy_variance': np.var(strategies),
                        'strategy_range': (min(strategies), max(strategies))
                    }
            
            # 收益分析  
            recent_payoffs = self.game_state.get_recent_payoffs(30)
            for player, payoffs in recent_payoffs.items():
                if payoffs:
                    stats['payoff_analysis'][player] = {
                        'current_payoff': payoffs[-1],
                        'mean_payoff': np.mean(payoffs),
                        'payoff_variance': np.var(payoffs),
                        'total_payoff': sum(payoffs)
                    }
            
            # 稳定性指标
            for player in self.config.PLAYER_NAMES:
                stats['stability_metrics'][player] = {
                    'strategy_stability': self.game_state.calculate_strategy_stability(player),
                    'recent_strategy_changes': self._calculate_recent_changes(player)
                }
        
        return stats
    
    def _calculate_recent_changes(self, player: str, window: int = 10) -> List[float]:
        """计算最近的策略变化"""
        recent_strategies = self.game_state.get_recent_strategies(window + 1)
        
        if player not in recent_strategies or len(recent_strategies[player]) < 2:
            return []
        
        strategies = recent_strategies[player]
        changes = [strategies[i] - strategies[i-1] for i in range(1, len(strategies))]
        return changes
    
    def save_game_state(self, filepath: str):
        """保存当前博弈状态"""
        import pickle
        
        save_data = {
            'config': self.config.to_dict(),
            'game_state': self.game_state,
            'round_number': self.game_state.round_number
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        self.logger.info(f"博弈状态已保存至: {filepath}")
    
    def load_game_state(self, filepath: str):
        """加载博弈状态"""
        import pickle
        
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        self.game_state = save_data['game_state']
        
        self.logger.info(f"博弈状态已加载自: {filepath}")
    
    def reset_game(self):
        """重置博弈状态"""
        self.game_state = GameState()
        self.logger.info("博弈状态已重置")
    
    def register_player(self, player, player_type='ai'):
        """
        注册新玩家到博弈框架
        
        Args:
            player: 玩家对象
            player_type: 玩家类型 ('ai', 'human', 'rule_based')
            
        Returns:
            玩家ID
        """
        player_id = f"player_{chr(65 + len(self.players))}"
        self.players[player_id] = {
            'player': player,
            'type': player_type
        }
        
        self.logger.info(f"注册玩家 {player_id}，类型: {player_type}")
        return player_id

    def play_round(self, round_num: int, strategies: Dict[str, float]) -> Dict[str, Any]:
        """
        执行一轮博弈
        
        Args:
            round_num: 当前轮次
            strategies: 各玩家策略字典 {player_id: strategy_value}
            
        Returns:
            轮次结果
        """
        # 验证策略有效性
        if not self.validate_strategies(strategies):
            self.logger.warning(f"轮次 {round_num} 策略无效")
            return {'error': 'invalid_strategies'}
        
        # 更新博弈状态
        self.game_state.update_round(round_num, self.config)
        
        # 计算收益 (简化示例)
        payoffs = {}
        market_data = {'total_orders': 10, 'total_revenue': 0}
        
        for player_id, strategy in strategies.items():
            # 简单收益函数，实际项目中可能更复杂
            base_payoff = 100 - abs(strategy - 30) * 2  # 以30为最优值的简单曲线
            competitive_effect = 0
            
            # 考虑竞争效应
            for other_id, other_strategy in strategies.items():
                if other_id != player_id:
                    # 价格低于对手获得竞争优势
                    if strategy < other_strategy:
                        competitive_effect += (other_strategy - strategy) * 0.5
                    else:
                        competitive_effect -= (strategy - other_strategy) * 0.2
            
            # 计算最终收益
            final_payoff = base_payoff + competitive_effect
            payoffs[player_id] = max(0, final_payoff)
            market_data['total_revenue'] += final_payoff
        
        # 记录轮次数据
        self.game_state.record_round_data(
            strategies=strategies,
            payoffs=payoffs,
            market_data=market_data
        )
        
        # 生成结果
        result = {
            'round_number': round_num,
            'strategies': strategies,
            'payoffs': payoffs,
            'market_data': market_data,
            'phase': self.game_state.phase.value
        }
        
        self.logger.debug(f"轮次 {round_num} 完成: {strategies} -> {payoffs}")
        return result


if __name__ == "__main__":
    # 测试博弈框架
    config = GameConfig()
    framework = GameFramework(config)
    
    # 测试策略验证
    test_strategies = {'司机A': 25.0, '司机B': 30.0}
    print(f"策略验证结果: {framework.validate_strategies(test_strategies)}")
    
    # 测试市场信息获取
    market_info = framework.get_market_information('司机A', 1)
    print(f"市场信息: {market_info}")
    
    print("博弈框架测试完成 ✓")