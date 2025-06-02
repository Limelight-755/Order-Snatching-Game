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
            config: 智能体配置字典，包含网络参数、学习参数等
        """
        # 基本配置
        self.config = config
        self.player_id = config.get('player_id', 'unknown')
        
        # 网络参数
        self.state_size = config.get('state_size', 15)
        self.action_size = config.get('action_size', 41)
        self.hidden_size = config.get('hidden_size', 128)
        
        # 学习参数
        self.learning_rate = config.get('learning_rate', 0.001)
        self.epsilon = config.get('epsilon', 1.0)
        self.epsilon_min = config.get('epsilon_min', 0.02)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.gamma = config.get('gamma', 0.95)
        
        # 经验回放参数
        self.memory_size = config.get('memory_size', 10000)
        self.batch_size = config.get('batch_size', 32)
        
        # 训练参数
        self.target_update_frequency = config.get('target_update_frequency', 10)
        self.min_replay_size = config.get('min_replay_size', 100)
        
        # 轮次计数
        self._round_count = 0
        
        # 设备设置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化神经网络
        self._initialize_networks()
        
        # 初始化经验回放缓冲区
        self.memory = []
        
        # 训练统计
        self.training_step = 0
        self.total_loss = 0.0
        self.training_history = []
        
        logger.info(f"DQN智能体 {self.player_id} 初始化完成")
    
    def _initialize_networks(self):
        # 创建网络
        self.q_network = QNetwork(
            self.state_size, self.action_size, [self.hidden_size]
        ).to(self.device)
        
        # 目标网络
        self.target_network = QNetwork(
            self.state_size, self.action_size, [self.hidden_size]
        ).to(self.device)
        
        # 复制权重到目标网络
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # 优化器
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
    
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
    
    def select_action(self, state: np.ndarray, training: bool = True,
                     opponent_strategy: float = None) -> ActionResult:
        """
        选择动作
        
        Args:
            state: 当前状态
            training: 是否在训练模式
            opponent_strategy: 对手当前策略（用于收敛学习）
            
        Returns:
            动作选择结果
        """
        # epsilon-greedy策略
        exploration_type = "exploit"
        
        if training and random.random() < self.epsilon:
            # 在均衡期，如果有对手策略信息，倾向于选择接近的策略
            if (hasattr(self, '_round_count') and self._round_count > 360 and 
                opponent_strategy is not None):
                # 均衡期的探索偏向对手策略附近
                opponent_action = self.strategy_to_action(opponent_strategy)
                # 在对手策略附近±5的范围内随机选择
                min_action = max(0, opponent_action - 5)
                max_action = min(self.action_size - 1, opponent_action + 5)
                action = random.randint(min_action, max_action)
                exploration_type = "convergence_explore"
            else:
                # 常规探索：随机选择动作
                action = random.randint(0, self.action_size - 1)
                exploration_type = "explore"
            
            action_value = 0.0
            confidence = 0.0
        else:
            # 利用：选择最优动作
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            self.q_network.eval()
            
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                
                # 在均衡期，如果策略差距较大，倾向于选择更接近对手的策略
                if (hasattr(self, '_round_count') and self._round_count > 360 and 
                    opponent_strategy is not None):
                    
                    current_best_action = q_values.argmax().item()
                    current_best_strategy = self.action_to_strategy(current_best_action)
                    strategy_diff = abs(current_best_strategy - opponent_strategy)
                    
                    # 如果策略差距超过8元，考虑向对手策略靠拢
                    if strategy_diff > 8.0:
                        opponent_action = self.strategy_to_action(opponent_strategy)
                        # 在对手策略附近寻找较高Q值的动作
                        nearby_actions = range(max(0, opponent_action - 3), 
                                             min(self.action_size, opponent_action + 4))
                        nearby_q_values = q_values[0][list(nearby_actions)]
                        
                        if len(nearby_q_values) > 0:
                            # 选择对手策略附近Q值最高的动作
                            best_nearby_idx = nearby_q_values.argmax().item()
                            action = list(nearby_actions)[best_nearby_idx]
                            action_value = nearby_q_values[best_nearby_idx].item()
                            exploration_type = "convergence_exploit"
                        else:
                            action = current_best_action
                            action_value = q_values.max().item()
                    else:
                        action = current_best_action
                        action_value = q_values.max().item()
                else:
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
        self.memory.append((state, action, reward, next_state, done))
    
    def calculate_reward(self, revenue: float, prev_revenue: float, 
                        market_state: Dict, personal_state: Dict,
                        opponent_strategy: float = None, opponent_revenue: float = None) -> float:
        """
        计算奖励函数（支持低价策略版）
        
        Args:
            revenue: 当前收益
            prev_revenue: 上轮收益
            market_state: 市场状态
            personal_state: 个人状态
            opponent_strategy: 对手策略（用于收敛奖励）
            opponent_revenue: 对手收益（用于收益均衡奖励）
            
        Returns:
            计算的奖励值
        """
        # 基础收益奖励（收益越高，奖励越大）
        base_reward = revenue / 40.0  # 收益除以40作为基础奖励
        
        # 收益增长奖励（收益增长时给予奖励）
        growth_reward = max(0, (revenue - prev_revenue) / 25.0)  # 增长部分的奖励
        
        # 合理定价策略奖励（支持低价策略）
        current_strategy = personal_state.get('current_strategy', 30)
        pricing_reward = 0
        
        if 20.0 <= current_strategy <= 30.0:  # 合理低价区间，给予大奖励
            # 越接近25元，奖励越高
            optimal_price = 25.0
            distance_from_optimal = abs(current_strategy - optimal_price)
            pricing_reward = (1.0 - distance_from_optimal / 5.0) * 0.4  # 最高0.4奖励
        elif 30.0 < current_strategy <= 35.0:  # 中等价格，给予小奖励
            distance_from_30 = current_strategy - 30.0
            pricing_reward = (1.0 - distance_from_30 / 5.0) * 0.2  # 最高0.2奖励
        elif current_strategy > 35.0:  # 高价策略，给予惩罚
            excess_price = current_strategy - 35.0
            pricing_reward = -min(0.5, excess_price / 10.0)  # 最高-0.5惩罚
        elif current_strategy < 20.0:  # 过低价格也给轻微惩罚（避免恶性竞争）
            pricing_reward = -0.1
        
        # 策略收敛奖励（鼓励收敛到合理低价）
        convergence_reward = 0
        if opponent_strategy is not None:
            strategy_diff = abs(current_strategy - opponent_strategy)
            avg_strategy = (current_strategy + opponent_strategy) / 2
            
            # 如果双方都在合理价格区间且策略接近，给予奖励
            if 20.0 <= avg_strategy <= 30.0 and strategy_diff < 5.0:
                convergence_reward = (5.0 - strategy_diff) / 5.0 * 0.3  # 最高0.3奖励
            elif strategy_diff < 3.0:  # 策略非常接近时也给小奖励
                convergence_reward = (3.0 - strategy_diff) / 3.0 * 0.15
        
        # 收益均衡奖励（双赢奖励，但优先考虑合理定价）
        balance_reward = 0
        if opponent_revenue is not None and opponent_revenue > 0:
            # 双方收益都较高时给予奖励
            min_revenue = min(revenue, opponent_revenue)
            if min_revenue > 20.0:  # 双方收益都超过20元
                balance_reward = min_revenue / 150.0  # 调整奖励强度
            
            # 收益接近时给予额外奖励
            revenue_diff = abs(revenue - opponent_revenue)
            max_revenue = max(revenue, opponent_revenue)
            if max_revenue > 0:
                revenue_similarity = 1.0 - (revenue_diff / max_revenue)
                if revenue_similarity > 0.8:  # 收益非常接近
                    balance_reward += 0.15
        
        # 市场适应性奖励（根据订单接受情况）
        acceptance_rate = personal_state.get('acceptance_rate', 0.5)
        if acceptance_rate > 0.7:  # 高接单率说明定价合理
            market_reward = (acceptance_rate - 0.7) / 0.3 * 0.2  # 最高0.2奖励
        elif acceptance_rate < 0.3:  # 低接单率可能说明定价过高
            market_reward = -0.2
        else:
            market_reward = 0
        
        # 总奖励
        total_reward = (base_reward + growth_reward + pricing_reward + 
                       convergence_reward + balance_reward + market_reward)
        
        # 确保最低奖励，避免过度惩罚影响学习
        total_reward = max(0.05, total_reward)
        
        return np.clip(total_reward, 0.05, 2.5)  # 限制在0.05-2.5范围内
    
    def train(self) -> Optional[float]:
        """
        训练DQN网络
        
        Returns:
            训练损失（如果进行了训练）
        """
        if len(self.memory) < self.min_replay_size:
            return None
        
        # 采样经验
        experiences = random.sample(self.memory, self.batch_size)
        
        # 准备训练数据
        states = torch.FloatTensor([e[0] for e in experiences]).to(self.device)
        actions = torch.LongTensor([e[1] for e in experiences]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in experiences]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in experiences]).to(self.device)
        dones = torch.BoolTensor([e[4] for e in experiences]).to(self.device)
        
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
        if self.training_step % self.target_update_frequency == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # 衰减探索率
        self._decay_epsilon()
        
        # 记录损失
        loss_value = loss.item()
        self.total_loss += loss_value
        self.training_history.append(loss_value)
        
        return loss_value
    
    def _decay_epsilon(self):
        """衰减探索率"""
        # 在360轮前基本完成衰减，确保均衡期稳定
        if self.epsilon > self.epsilon_min:
            # 调整衰减速度，让探索率在360轮时达到最小值
            decay_factor = 0.9985  # 更慢的衰减，但在360轮前达到最小值
            self.epsilon = max(self.epsilon_min, self.epsilon * decay_factor)
            
        # 在均衡期（360轮后）进一步降低探索率到0.001
        if hasattr(self, '_round_count') and self._round_count > 360:
            target_epsilon = 0.001  # 均衡期目标探索率
            self.epsilon = max(target_epsilon, self.epsilon * 0.99)  # 快速降到0.001
    
    def get_training_stats(self) -> Dict:
        """获取训练统计信息"""
        return {
            'training_steps': self.training_step,
            'total_loss': self.total_loss,
            'epsilon': self.epsilon,
            'memory_size': len(self.memory),
            'avg_loss': np.mean(self.training_history[-100:]) if self.training_history else 0,
            'training_history': self.training_history
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
            'total_loss': self.total_loss,
            'training_history': self.training_history
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
        self.total_loss = checkpoint.get('total_loss', 0.0)
        self.training_history = checkpoint.get('training_history', [])
        logger.info(f"DQN模型已从 {filepath} 加载")
    
    def reset_for_new_episode(self):
        """
        重置智能体状态，准备下一轮学习
        """
        # 重置探索率
        self.epsilon = self.config.get('epsilon_start', 1.0)
        
        # 清空经验缓冲区
        self.memory = []
        
        # 重置训练统计
        self.training_steps = 0
        self.total_reward = 0
        self.training_history = []
        
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
    
    def choose_action(self, state: np.ndarray, training: bool = True,
                     opponent_strategy: float = None) -> int:
        """
        选择动作（别名方法，兼容实验框架）
        
        Args:
            state: 状态向量
            training: 是否在训练中
            opponent_strategy: 对手策略（可选）
            
        Returns:
            所选择的动作索引
        """
        action_result = self.select_action(state, training, opponent_strategy)
        return action_result.action
    
    def update_round(self, round_number: int):
        """
        更新当前轮次数
        
        Args:
            round_number: 当前轮次
        """
        self._round_count = round_number
        
        # 在均衡期（360轮后）将探索率降到0.001以确保策略稳定
        if round_number > 360:  # 均衡期
            target_epsilon = 0.001  # 均衡期目标探索率
            if self.epsilon > target_epsilon:
                self.epsilon = max(target_epsilon, self.epsilon * 0.98)  # 快速降到0.001 