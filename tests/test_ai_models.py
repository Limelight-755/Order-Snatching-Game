"""
AI模型测试
"""

import unittest
import numpy as np
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

from ai_models.dqn_agent import DQNAgent
from ai_models.lstm_predictor import LSTMPredictor


class TestDQNAgent(unittest.TestCase):
    """DQN智能体测试类"""
    
    def setUp(self):
        """测试前置设置"""
        self.config = {
            'state_size': 10,
            'action_size': 41,  # 10-50的价格策略
            'learning_rate': 0.001,
            'hidden_units': [64, 32],
            'epsilon': 0.1,
            'epsilon_decay': 0.995,
            'epsilon_min': 0.01
        }
        self.agent = DQNAgent(self.config)
    
    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.agent.state_size, 10)
        self.assertEqual(self.agent.action_size, 41)
        self.assertIsNotNone(self.agent.q_network)
        self.assertIsNotNone(self.agent.target_network)
    
    def test_action_selection(self):
        """测试动作选择"""
        state = np.random.rand(10)
        action = self.agent.choose_action(state)
        
        self.assertIsInstance(action, int)
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, 41)
    
    def test_memory_operations(self):
        """测试记忆操作"""
        state = np.random.rand(10)
        action = 5
        reward = 10.0
        next_state = np.random.rand(10)
        done = False
        
        # 存储经验
        self.agent.remember(state, action, reward, next_state, done)
        self.assertEqual(len(self.agent.memory), 1)
        
        # 清空记忆
        self.agent.memory.clear()
        self.assertEqual(len(self.agent.memory), 0)
    
    def test_epsilon_decay(self):
        """测试epsilon衰减"""
        initial_epsilon = self.agent.epsilon
        self.agent.decay_epsilon()
        self.assertLessEqual(self.agent.epsilon, initial_epsilon)
        self.assertGreaterEqual(self.agent.epsilon, self.agent.epsilon_min)


class TestLSTMPredictor(unittest.TestCase):
    """LSTM预测器测试类"""
    
    def setUp(self):
        """测试前置设置"""
        self.config = {
            'input_size': 5,
            'hidden_size': 32,
            'output_size': 1,
            'sequence_length': 10,
            'learning_rate': 0.001
        }
        self.predictor = LSTMPredictor(self.config)
    
    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.predictor.input_size, 5)
        self.assertEqual(self.predictor.hidden_size, 32)
        self.assertEqual(self.predictor.output_size, 1)
        self.assertEqual(self.predictor.sequence_length, 10)
    
    def test_prediction(self):
        """测试预测功能"""
        # 创建测试序列
        sequence = np.random.rand(10, 5)
        prediction = self.predictor.predict(sequence)
        
        self.assertIsNotNone(prediction)
        self.assertIsInstance(prediction, (int, float, np.ndarray))
    
    def test_data_preprocessing(self):
        """测试数据预处理"""
        # 测试数据归一化
        data = np.random.rand(100, 5) * 100  # 随机数据
        normalized_data = self.predictor.normalize_data(data)
        
        self.assertEqual(normalized_data.shape, data.shape)
        self.assertLessEqual(normalized_data.max(), 1.0)
        self.assertGreaterEqual(normalized_data.min(), 0.0)
    
    def test_sequence_creation(self):
        """测试序列创建"""
        data = np.random.rand(50, 5)
        sequences, targets = self.predictor.create_sequences(data, target_col=0)
        
        expected_length = len(data) - self.predictor.sequence_length
        self.assertEqual(len(sequences), expected_length)
        self.assertEqual(len(targets), expected_length)
        self.assertEqual(sequences[0].shape, (self.predictor.sequence_length, 5))


class TestModelIntegration(unittest.TestCase):
    """模型集成测试"""
    
    def test_dqn_lstm_interaction(self):
        """测试DQN和LSTM的交互"""
        # 创建DQN智能体
        dqn_config = {
            'state_size': 10,
            'action_size': 41,
            'learning_rate': 0.001
        }
        dqn_agent = DQNAgent(dqn_config)
        
        # 创建LSTM预测器
        lstm_config = {
            'input_size': 5,
            'hidden_size': 32,
            'output_size': 1,
            'sequence_length': 10,
            'learning_rate': 0.001
        }
        lstm_predictor = LSTMPredictor(lstm_config)
        
        # 模拟一轮交互
        state = np.random.rand(10)
        action = dqn_agent.choose_action(state)
        
        # 使用LSTM预测对手策略
        opponent_history = np.random.rand(10, 5)
        predicted_strategy = lstm_predictor.predict(opponent_history)
        
        # 验证输出
        self.assertIsInstance(action, int)
        self.assertIsNotNone(predicted_strategy)


if __name__ == '__main__':
    unittest.main() 