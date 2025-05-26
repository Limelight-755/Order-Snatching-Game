"""
DQN决策智能体
基于深度强化学习的博弈策略优化模型
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque, namedtuple
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# 经验元组定义
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


@dataclass
class ActionResult:
    """动作结果数据类"""
    action: int
    action_value: float
    confidence: float
    exploration_type: str  # 'exploit', 'explore', 'random'


class QNetwork(nn.Module):
    """
    Q网络
    用于估计状态-动作价值函数
    """
    
    def __init__(self, state_size: int, action_size: int, hidden_sizes: List[int] = None,
                 dropout: float = 0.1):
        """
        初始化Q网络
        
        Args:
            state_size: 状态空间维度
            action_size: 动作空间维度  
            hidden_sizes: 隐藏层尺寸列表
            dropout: Dropout比例
        """
        super(QNetwork, self).__init__()
        
        if hidden_sizes is None:
            hidden_sizes = [128, 64, 32]
        
        self.state_size = state_size
        self.action_size = action_size
        
        # 构建网络层
        layers = []
        input_size = state_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(dropout)
            ])
            input_size = hidden_size
        
        # 移除最后的dropout
        layers = layers[:-1]
        
        # 输出层
        layers.append(nn.Linear(input_size, action_size))
        
        self.network = nn.Sequential(*layers)
        
        # 价值流和优势流（Dueling DQN）
        self.value_stream = nn.Sequential(
            nn.Linear(input_size, input_size // 2),
            nn.ReLU(),
            nn.Linear(input_size // 2, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(input_size, input_size // 2),
            nn.ReLU(),
            nn.Linear(input_size // 2, action_size)
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            state: 输入状态 [batch_size, state_size]
            
        Returns:
            Q值 [batch_size, action_size]
        """
        # 基础特征提取
        features = self.network[:-1](state)  # 去掉最后的输出层
        
        # Dueling DQN架构
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values


class ReplayBuffer:
    """
    经验回放缓冲区
    存储和采样经验用于训练
    """
    
    def __init__(self, capacity: int, batch_size: int = 32):
        """
        初始化回放缓冲区
        
        Args:
            capacity: 缓冲区容量
            batch_size: 批次大小
        """
        self.buffer = deque(maxlen=capacity)
        self.batch_size = batch_size
        self.capacity = capacity
        
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool):
        """添加经验到缓冲区"""
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size: int = None) -> List[Experience]:
        """采样一批经验"""
        if batch_size is None:
            batch_size = self.batch_size
        return random.sample(self.buffer, min(len(self.buffer), batch_size))
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def is_ready(self, min_size: int = None) -> bool:
        """检查是否有足够经验进行训练"""
        if min_size is None:
            min_size = self.batch_size
        return len(self.buffer) >= min_size


class DQNAgent:
    """
    DQN智能体
    使用深度Q学习进行决策优化
    """
    
    def __init__(self, config: Dict):
        """
        初始化DQN智能体
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 网络参数
        self.state_size = config.get('state_size', 15)  # 状态特征维度
        self.action_size = config.get('action_size', 41)  # 动作空间：10-50元价格阈值
        self.hidden_sizes = config.get('hidden_sizes', [128, 64, 32])
        
        # 学习参数
        self.learning_rate = config.get('learning_rate', 0.0005)
        self.gamma = config.get('gamma', 0.99)  # 折扣因子
        self.epsilon = config.get('epsilon_start', 1.0)  # 探索率
        self.epsilon_min = config.get('epsilon_min', 0.01)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.target_update_freq = config.get('target_update_freq', 100)  # 目标网络更新频率
        
        # 经验回放
        self.buffer_size = config.get('buffer_size', 10000)
        self.batch_size = config.get('batch_size', 32)
        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)
        
        # 创建网络
        self.q_network = QNetwork(
            self.state_size, self.action_size, self.hidden_sizes
        ).to(self.device)
        
        # 目标网络
        self.target_network = QNetwork(
            self.state_size, self.action_size, self.hidden_sizes
        ).to(self.device)
        
        # 复制权重到目标网络
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # 优化器
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        
        # 训练计数器
        self.training_step = 0
        self.update_step = 0
        
        # 性能跟踪
        self.losses = []
        self.episode_rewards = []
        self.exploration_history = []
        
        logger.info(f"DQN智能体初始化完成，设备: {self.device}")
        logger.info(f"状态空间: {self.state_size}, 动作空间: {self.action_size}")
    
    def preprocess_state(self, market_state: Dict, personal_state: Dict, 
                        opponent_info: Dict = None) -> np.ndarray:
        """
        预处理状态信息为网络输入
        
        Args:
            market_state: 市场状态
            personal_state: 个人状态
            opponent_info: 对手信息（可选）
            
        Returns:
            处理后的状态向量
        """
        features = []
        
        # 市场特征
        features.extend([
            market_state.get('order_count', 0) / 100.0,  # 归一化订单数量
            market_state.get('avg_price', 30) / 50.0,    # 归一化平均价格
            market_state.get('competition_level', 0.5),   # 竞争水平
            market_state.get('time_factor', 0.5),         # 时间因子
            market_state.get('location_factor', 0.5),     # 位置因子
        ])
        
        # 个人状态特征
        features.extend([
            personal_state.get('current_strategy', 30) / 50.0,  # 当前策略
            personal_state.get('recent_revenue', 0) / 1000.0,   # 最近收益
            personal_state.get('acceptance_rate', 0.5),         # 接单率
            personal_state.get('waiting_time', 0) / 60.0,       # 等待时间
            personal_state.get('round_number', 0) / 500.0,      # 轮次进度
        ])
        
        # 对手信息（如果可用）
        if opponent_info:
            features.extend([
                opponent_info.get('predicted_strategy', 30) / 50.0,  # 预测对手策略
                opponent_info.get('prediction_confidence', 0.5),     # 预测置信度
                opponent_info.get('opponent_trend', 0),               # 对手趋势（-1,0,1）
            ])
        else:
            features.extend([0.6, 0.5, 0])  # 默认值
        
        # 历史特征（简化的移动平均）
        history_features = personal_state.get('history_features', [0, 0])
        features.extend(history_features)
        
        # 确保特征维度正确
        features = features[:self.state_size]
        while len(features) < self.state_size:
            features.append(0.0)
            
        return np.array(features, dtype=np.float32)
    
    def action_to_strategy(self, action: int) -> float:
        """将动作索引转换为策略值"""
        return 10.0 + action  # 动作0-40对应策略10-50
    
    def strategy_to_action(self, strategy: float) -> int:
        """将策略值转换为动作索引"""
        return int(np.clip(strategy - 10, 0, 40))
    
    def select_action(self, state: np.ndarray, training: bool = True) -> ActionResult:
        """
        选择动作
        
        Args:
            state: 当前状态
            training: 是否在训练模式
            
        Returns:
            动作选择结果
        """
        # epsilon-greedy策略
        exploration_type = "exploit"
        
        if training and random.random() < self.epsilon:
            # 探索：随机选择动作
            action = random.randint(0, self.action_size - 1)
            action_value = 0.0
            confidence = 0.0
            exploration_type = "explore"
        else:
            # 利用：选择最优动作
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            self.q_network.eval()
            
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                action = q_values.argmax().item()
                action_value = q_values.max().item()
                
                # 计算决策置信度（基于Q值分布）
                q_probs = F.softmax(q_values, dim=1)
                confidence = q_probs.max().item()
        
        strategy = self.action_to_strategy(action)
        
        return ActionResult(
            action=action,
            action_value=action_value,
            confidence=confidence,
            exploration_type=exploration_type
        )
    
    def store_experience(self, state: np.ndarray, action: int, reward: float,
                        next_state: np.ndarray, done: bool):
        """存储经验到回放缓冲区"""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def calculate_reward(self, revenue: float, prev_revenue: float, 
                        market_state: Dict, personal_state: Dict) -> float:
        """
        计算奖励函数
        
        Args:
            revenue: 当前收益
            prev_revenue: 上轮收益
            market_state: 市场状态
            personal_state: 个人状态
            
        Returns:
            计算的奖励值
        """
        # 基础收益奖励
        revenue_reward = (revenue - prev_revenue) / 100.0  # 归一化收益变化
        
        # 接单率奖励
        acceptance_rate = personal_state.get('acceptance_rate', 0.5)
        acceptance_reward = (acceptance_rate - 0.5) * 0.5  # 鼓励适中的接单率
        
        # 竞争效应奖励
        competition_level = market_state.get('competition_level', 0.5)
        if competition_level > 0.8:  # 高竞争环境下的额外奖励
            competition_reward = revenue_reward * 0.2
        else:
            competition_reward = 0
        
        # 策略稳定性惩罚（防止过度频繁调整）
        strategy_change = abs(personal_state.get('strategy_change', 0))
        stability_penalty = -strategy_change / 50.0 * 0.1
        
        # 总奖励
        total_reward = revenue_reward + acceptance_reward + competition_reward + stability_penalty
        
        return np.clip(total_reward, -1.0, 1.0)  # 限制奖励范围
    
    def train(self) -> Optional[float]:
        """
        训练DQN网络
        
        Returns:
            训练损失（如果进行了训练）
        """
        if not self.replay_buffer.is_ready():
            return None
        
        # 采样经验
        experiences = self.replay_buffer.sample()
        
        # 准备训练数据
        states = torch.FloatTensor([e.state for e in experiences]).to(self.device)
        actions = torch.LongTensor([e.action for e in experiences]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in experiences]).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in experiences]).to(self.device)
        dones = torch.BoolTensor([e.done for e in experiences]).to(self.device)
        
        # 当前Q值
        self.q_network.train()
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # 目标Q值（Double DQN）
        with torch.no_grad():
            # 使用主网络选择动作
            next_actions = self.q_network(next_states).argmax(1, keepdim=True)
            # 使用目标网络评估Q值
            next_q_values = self.target_network(next_states).gather(1, next_actions)
            target_q_values = rewards.unsqueeze(1) + (self.gamma * next_q_values * (~dones).unsqueeze(1))
        
        # 计算损失
        loss = F.mse_loss(current_q_values, target_q_values)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # 更新计数器
        self.training_step += 1
        
        # 更新目标网络
        if self.training_step % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            self.update_step += 1
            logger.debug(f"目标网络已更新，更新步数: {self.update_step}")
        
        # 衰减探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # 记录损失
        loss_value = loss.item()
        self.losses.append(loss_value)
        
        return loss_value
    
    def update_exploration_rate(self, round_number: int, total_rounds: int):
        """
        根据博弈轮次更新探索率
        
        Args:
            round_number: 当前轮次
            total_rounds: 总轮次
        """
        # 分阶段调整探索率
        progress = round_number / total_rounds
        
        if progress < 0.1:  # 前10%轮次：高探索
            target_epsilon = 0.8
        elif progress < 0.4:  # 10%-40%轮次：中等探索
            target_epsilon = 0.3
        else:  # 40%后：低探索，主要利用
            target_epsilon = 0.05
        
        # 平滑过渡
        self.epsilon = max(target_epsilon, self.epsilon_min)
    
    def get_training_stats(self) -> Dict:
        """获取训练统计信息"""
        return {
            'training_steps': self.training_step,
            'target_updates': self.update_step,
            'epsilon': self.epsilon,
            'buffer_size': len(self.replay_buffer),
            'avg_loss': np.mean(self.losses[-100:]) if self.losses else 0,
            'avg_episode_reward': np.mean(self.episode_rewards[-10:]) if self.episode_rewards else 0
        }
    
    def save_model(self, filepath: str):
        """保存模型"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_step': self.training_step,
            'epsilon': self.epsilon,
            'config': self.config,
            'losses': self.losses,
            'episode_rewards': self.episode_rewards
        }, filepath)
        logger.info(f"DQN模型已保存到: {filepath}")
    
    def load_model(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_step = checkpoint.get('training_step', 0)
        self.epsilon = checkpoint.get('epsilon', self.epsilon_min)
        self.losses = checkpoint.get('losses', [])
        self.episode_rewards = checkpoint.get('episode_rewards', [])
        logger.info(f"DQN模型已从 {filepath} 加载")
    
    def reset_for_new_episode(self):
        """
        重置智能体状态，准备下一轮学习
        """
        # 重置探索率
        self.epsilon = self.config.get('epsilon_start', 1.0)
        
        # 清空经验缓冲区
        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)
        
        # 重置训练统计
        self.training_steps = 0
        self.total_reward = 0
        self.episode_rewards = []
        
        # 重置历史状态
        if hasattr(self, 'last_state'):
            self.last_state = None
            
        if hasattr(self, 'last_action'):
            self.last_action = None
            
        logger.debug(f"智能体已重置，设置新探索率为 {self.epsilon}")
        
    def reset(self):
        """
        重置智能体状态，兼容实验框架
        """
        self.reset_for_new_episode()
    
    def choose_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        选择动作（别名方法，兼容实验框架）
        
        Args:
            state: 状态向量
            training: 是否在训练中
            
        Returns:
            所选择的动作索引
        """
        action_result = self.select_action(state, training)
        return action_result.action 