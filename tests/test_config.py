"""
配置模块测试
"""

import unittest
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

from config.game_config import GameConfig


class TestGameConfig(unittest.TestCase):
    """游戏配置测试类"""
    
    def setUp(self):
        """测试前置设置"""
        self.config = GameConfig()
    
    def test_default_config(self):
        """测试默认配置"""
        self.assertEqual(self.config.total_rounds, 500)
        self.assertEqual(self.config.price_range, (10, 50))
        self.assertEqual(self.config.exploration_rounds, 50)
        self.assertEqual(self.config.learning_rounds, 150)
        self.assertEqual(self.config.equilibrium_rounds, 300)
    
    def test_custom_config(self):
        """测试自定义配置"""
        custom_config = GameConfig(
            total_rounds=300,
            price_range=(15, 45),
            exploration_rounds=30
        )
        
        self.assertEqual(custom_config.total_rounds, 300)
        self.assertEqual(custom_config.price_range, (15, 45))
        self.assertEqual(custom_config.exploration_rounds, 30)
    
    def test_config_validation(self):
        """测试配置验证"""
        # 测试价格范围验证
        with self.assertRaises(ValueError):
            GameConfig(price_range=(50, 10))  # 错误的价格范围
        
        # 测试轮数验证
        with self.assertRaises(ValueError):
            GameConfig(total_rounds=0)  # 无效轮数
    
    def test_phase_calculation(self):
        """测试阶段计算"""
        config = GameConfig(total_rounds=500)
        
        self.assertTrue(config.is_exploration_phase(25))
        self.assertTrue(config.is_learning_phase(100))
        self.assertTrue(config.is_equilibrium_phase(300))
        
        self.assertFalse(config.is_exploration_phase(100))
        self.assertFalse(config.is_learning_phase(300))
        self.assertFalse(config.is_equilibrium_phase(25))


if __name__ == '__main__':
    unittest.main() 