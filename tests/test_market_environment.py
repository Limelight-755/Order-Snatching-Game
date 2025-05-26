"""
市场环境测试
"""

import unittest
import numpy as np
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

from core.market_environment import MarketEnvironment
from config.game_config import GameConfig


class TestMarketEnvironment(unittest.TestCase):
    """市场环境测试类"""
    
    def setUp(self):
        """测试前置设置"""
        self.config = GameConfig()
        self.market = MarketEnvironment(self.config)
    
    def test_initialization(self):
        """测试初始化"""
        self.assertIsNotNone(self.market.config)
        self.assertEqual(self.market.current_round, 0)
        self.assertIsNotNone(self.market.market_state)
    
    def test_market_state_update(self):
        """测试市场状态更新"""
        # 模拟玩家动作
        player_actions = {'司机A': 25, '司机B': 30}
        
        initial_round = self.market.current_round
        state = self.market.update_market_state(player_actions)
        
        # 验证轮数增加
        self.assertEqual(self.market.current_round, initial_round + 1)
        
        # 验证状态结构
        self.assertIn('round', state)
        self.assertIn('players', state)
        self.assertIn('market_conditions', state)
        
        # 验证价格在合理范围内
        for player, price in player_actions.items():
            self.assertIn(player, state['players'])
    
    def test_demand_calculation(self):
        """测试需求计算"""
        prices = [25, 30]
        demands = self.market.calculate_demand(prices)
        
        self.assertEqual(len(demands), len(prices))
        
        # 验证需求为非负数
        for demand in demands:
            self.assertGreaterEqual(demand, 0)
        
        # 验证低价格通常对应更高需求（在基本情况下）
        if len(set(prices)) > 1:  # 如果价格不同
            min_price_idx = prices.index(min(prices))
            max_price_idx = prices.index(max(prices))
            # 在基本市场条件下，低价格应有更高需求
            self.assertGreaterEqual(demands[min_price_idx], demands[max_price_idx] * 0.8)
    
    def test_reward_calculation(self):
        """测试奖励计算"""
        player_actions = {'司机A': 25, '司机B': 30}
        
        # 更新市场状态
        self.market.update_market_state(player_actions)
        
        # 计算奖励
        rewards = self.market.calculate_rewards(player_actions)
        
        self.assertIn('司机A', rewards)
        self.assertIn('司机B', rewards)
        
        # 验证奖励为数值
        for player, reward in rewards.items():
            self.assertIsInstance(reward, (int, float))
    
    def test_market_conditions(self):
        """测试市场条件"""
        conditions = self.market.get_market_conditions()
        
        # 验证市场条件包含必要信息
        expected_keys = ['base_demand', 'competition_factor', 'time_factor', 'weather_factor']
        for key in expected_keys:
            self.assertIn(key, conditions)
        
        # 验证因子在合理范围内
        self.assertGreater(conditions['base_demand'], 0)
        self.assertGreaterEqual(conditions['competition_factor'], 0)
    
    def test_phase_detection(self):
        """测试阶段检测"""
        # 测试探索阶段
        self.market.current_round = 25
        self.assertTrue(self.market.is_exploration_phase())
        self.assertFalse(self.market.is_learning_phase())
        self.assertFalse(self.market.is_equilibrium_phase())
        
        # 测试学习阶段
        self.market.current_round = 100
        self.assertFalse(self.market.is_exploration_phase())
        self.assertTrue(self.market.is_learning_phase())
        self.assertFalse(self.market.is_equilibrium_phase())
        
        # 测试均衡阶段
        self.market.current_round = 300
        self.assertFalse(self.market.is_exploration_phase())
        self.assertFalse(self.market.is_learning_phase())
        self.assertTrue(self.market.is_equilibrium_phase())
    
    def test_reset_functionality(self):
        """测试重置功能"""
        # 先运行几轮
        for _ in range(5):
            player_actions = {'司机A': 25, '司机B': 30}
            self.market.update_market_state(player_actions)
        
        initial_round = self.market.current_round
        self.assertGreater(initial_round, 0)
        
        # 重置市场
        self.market.reset()
        
        # 验证重置效果
        self.assertEqual(self.market.current_round, 0)
    
    def test_state_consistency(self):
        """测试状态一致性"""
        # 连续运行多轮，检查状态一致性
        player_actions = {'司机A': 25, '司机B': 30}
        
        states = []
        for _ in range(3):
            state = self.market.update_market_state(player_actions)
            states.append(state)
        
        # 验证轮数递增
        for i in range(1, len(states)):
            self.assertEqual(states[i]['round'], states[i-1]['round'] + 1)
        
        # 验证状态结构一致
        for state in states:
            self.assertIn('round', state)
            self.assertIn('players', state)
            self.assertIn('market_conditions', state)


class TestMarketEnvironmentIntegration(unittest.TestCase):
    """市场环境集成测试"""
    
    def test_multi_player_interaction(self):
        """测试多玩家交互"""
        config = GameConfig()
        market = MarketEnvironment(config)
        
        # 模拟多轮博弈
        results = []
        for round_num in range(10):
            # 模拟不同的价格策略
            if round_num < 5:
                actions = {'司机A': 20, '司机B': 35}  # 一个低价，一个高价
            else:
                actions = {'司机A': 30, '司机B': 25}  # 策略调整
            
            state = market.update_market_state(actions)
            rewards = market.calculate_rewards(actions)
            
            results.append({
                'round': round_num + 1,
                'actions': actions,
                'rewards': rewards,
                'state': state
            })
        
        # 验证结果完整性
        self.assertEqual(len(results), 10)
        
        # 验证奖励变化（应该反映策略效果）
        early_rewards = [r['rewards'] for r in results[:5]]
        late_rewards = [r['rewards'] for r in results[5:]]
        
        # 至少应该有一些奖励变化
        self.assertTrue(len(set(str(r) for r in early_rewards + late_rewards)) > 1)
    
    def test_extreme_price_scenarios(self):
        """测试极端价格场景"""
        config = GameConfig()
        market = MarketEnvironment(config)
        
        # 测试最低价格
        min_price = config.price_range[0]
        low_actions = {'司机A': min_price, '司机B': min_price}
        
        low_state = market.update_market_state(low_actions)
        low_rewards = market.calculate_rewards(low_actions)
        
        # 测试最高价格
        market.reset()
        max_price = config.price_range[1]
        high_actions = {'司机A': max_price, '司机B': max_price}
        
        high_state = market.update_market_state(high_actions)
        high_rewards = market.calculate_rewards(high_actions)
        
        # 验证两种情况都能正常处理
        self.assertIsNotNone(low_rewards)
        self.assertIsNotNone(high_rewards)
        
        # 验证状态有效
        for state in [low_state, high_state]:
            self.assertIn('round', state)
            self.assertIn('players', state)


if __name__ == '__main__':
    unittest.main()