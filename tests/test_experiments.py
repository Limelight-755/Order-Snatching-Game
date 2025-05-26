"""
实验模块测试
"""

import unittest
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

from config.game_config import GameConfig
from experiments.experiment_utils import ExperimentConfig, ExperimentRunner
from experiments.symmetric_game import SymmetricGameExperiment
from experiments.asymmetric_game import AsymmetricGameExperiment


class TestExperimentConfig(unittest.TestCase):
    """实验配置测试类"""
    
    def test_experiment_config_creation(self):
        """测试实验配置创建"""
        config = ExperimentConfig(
            name="test_experiment",
            description="测试实验",
            num_rounds=100,
            players=['玩家A', '玩家B'],
            player_configs={
                '玩家A': {'type': 'ai', 'learning_rate': 0.01},
                '玩家B': {'type': 'ai', 'learning_rate': 0.01}
            }
        )
        
        self.assertEqual(config.name, "test_experiment")
        self.assertEqual(config.num_rounds, 100)
        self.assertEqual(len(config.players), 2)
        self.assertIn('玩家A', config.player_configs)
    
    def test_config_validation(self):
        """测试配置验证"""
        with self.assertRaises(ValueError):
            ExperimentConfig(
                name="",  # 空名称
                description="测试",
                num_rounds=100,
                players=['玩家A'],
                player_configs={}
            )


class TestSymmetricGameExperiment(unittest.TestCase):
    """对称博弈实验测试类"""
    
    def setUp(self):
        """测试前置设置"""
        self.game_config = GameConfig()
        self.game_config.total_rounds = 20  # 减少测试轮数
        
        self.exp_config = ExperimentConfig(
            name="test_symmetric",
            description="对称博弈测试",
            num_rounds=20,
            players=['司机A', '司机B'],
            player_configs={
                '司机A': {'type': 'ai', 'learning_rate': 0.01, 'exploration_rate': 0.1},
                '司机B': {'type': 'ai', 'learning_rate': 0.01, 'exploration_rate': 0.1}
            }
        )
    
    def test_experiment_initialization(self):
        """测试实验初始化"""
        experiment = SymmetricGameExperiment(self.game_config, self.exp_config)
        
        self.assertIsNotNone(experiment.game_framework)
        self.assertEqual(experiment.config.total_rounds, 20)
        self.assertEqual(len(experiment.exp_config.players), 2)
    
    def test_experiment_setup(self):
        """测试实验设置"""
        experiment = SymmetricGameExperiment(self.game_config, self.exp_config)
        experiment.setup()
        
        # 验证AI智能体已创建
        self.assertIsNotNone(experiment.ai_agents)
        self.assertEqual(len(experiment.ai_agents), 2)
    
    def test_quick_run(self):
        """测试快速运行实验"""
        # 进一步减少轮数以加快测试
        self.game_config.total_rounds = 5
        self.exp_config.num_rounds = 5
        
        experiment = SymmetricGameExperiment(self.game_config, self.exp_config)
        results = experiment.run()
        
        # 验证结果结构
        self.assertIn('total_rounds', results)
        self.assertIn('players', results)
        self.assertIn('round_results', results)
        self.assertEqual(results['total_rounds'], 5)
        self.assertEqual(len(results['round_results']), 5)


class TestAsymmetricGameExperiment(unittest.TestCase):
    """非对称博弈实验测试类"""
    
    def setUp(self):
        """测试前置设置"""
        self.game_config = GameConfig()
        self.game_config.total_rounds = 20
        
        self.exp_config = ExperimentConfig(
            name="test_asymmetric",
            description="非对称博弈测试",
            num_rounds=20,
            players=['经验司机', '新手司机'],
            player_configs={
                '经验司机': {
                    'type': 'ai',
                    'learning_rate': 0.015,
                    'exploration_rate': 0.08,
                    'experience_bonus': 1.2
                },
                '新手司机': {
                    'type': 'ai',
                    'learning_rate': 0.008,
                    'exploration_rate': 0.15,
                    'experience_bonus': 1.0
                }
            }
        )
    
    def test_asymmetric_setup(self):
        """测试非对称实验设置"""
        experiment = AsymmetricGameExperiment(self.game_config, self.exp_config)
        experiment.setup()
        
        # 验证不同配置的AI智能体
        self.assertIsNotNone(experiment.ai_agents)
        self.assertEqual(len(experiment.ai_agents), 2)
        
        # 验证配置差异
        experienced_config = self.exp_config.player_configs['经验司机']
        novice_config = self.exp_config.player_configs['新手司机']
        
        self.assertGreater(experienced_config['learning_rate'], 
                          novice_config['learning_rate'])
        self.assertLess(experienced_config['exploration_rate'],
                       novice_config['exploration_rate'])


class TestExperimentRunner(unittest.TestCase):
    """实验运行器测试类"""
    
    def test_runner_initialization(self):
        """测试运行器初始化"""
        config = GameConfig()
        runner = ExperimentRunner(config)
        
        self.assertEqual(runner.config, config)
        self.assertIsNotNone(runner.logger)
    
    def test_experiment_registration(self):
        """测试实验注册"""
        config = GameConfig()
        runner = ExperimentRunner(config)
        
        exp_config = ExperimentConfig(
            name="test_exp",
            description="测试",
            num_rounds=10,
            players=['A', 'B'],
            player_configs={'A': {}, 'B': {}}
        )
        
        runner.register_experiment("test", exp_config)
        self.assertIn("test", runner.experiments)


if __name__ == '__main__':
    unittest.main() 