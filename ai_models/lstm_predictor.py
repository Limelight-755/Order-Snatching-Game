"""
LSTM策略预测模型
用于预测对手的策略选择和市场状态演化
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """预测结果数据类"""
    predicted_strategy: float
    confidence: float
    market_trend: str
    prediction_horizon: int


class LSTMPredictor(nn.Module):
    """
    LSTM预测网络
    用于预测时间序列数据
    """
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, 
                 output_size: int, dropout: float = 0.1):
        super(LSTMPredictor, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # 注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=dropout
        )
        
        # 输出层
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
        
        # 预测置信度输出
        self.confidence_layer = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入序列 [batch_size, seq_len, input_size]
            
        Returns:
            预测值和置信度
        """
        batch_size = x.size(0)
        
        # LSTM处理
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # 注意力机制
        # 转换维度适配注意力层 [seq_len, batch_size, hidden_size]
        lstm_out_transposed = lstm_out.transpose(0, 1)
        attended_out, attention_weights = self.attention(
            lstm_out_transposed, lstm_out_transposed, lstm_out_transposed
        )
        
        # 使用最后一个时间步的输出
        final_output = attended_out[-1]  # [batch_size, hidden_size]
        
        # 预测输出
        prediction = self.fc_layers(final_output)
        confidence = self.confidence_layer(final_output)
        
        return prediction, confidence
    
    def init_hidden(self, batch_size: int, device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor]:
        """初始化隐藏状态"""
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        cell = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        return hidden, cell


class StrategyPredictor:
    """
    策略预测器
    专门用于博弈中的策略预测任务
    """
    
    def __init__(self, config: Dict):
        """
        初始化策略预测器
        
        Args:
            config: 配置字典，包含模型参数
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 模型参数
        self.sequence_length = config.get('sequence_length', 20)
        self.input_features = config.get('input_features', 10)  # 确保使用10个输入特征
        self.hidden_size = config.get('hidden_size', 64)
        self.num_layers = config.get('num_layers', 2)
        self.learning_rate = config.get('learning_rate', 0.001)
        
        # 创建模型
        self.model = LSTMPredictor(
            input_size=self.input_features,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            output_size=1,  # 预测单个策略值
            dropout=0.1
        ).to(self.device)
        
        # 优化器和损失函数
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        
        # 历史数据存储
        self.strategy_history = []
        self.market_history = []
        self.revenue_history = []
        
        logger.info(f"策略预测器初始化完成，设备: {self.device}")
    
    def preprocess_data(self, strategies: List[float], market_states: List[Dict], 
                       revenues: List[float]) -> torch.Tensor:
        """
        预处理输入数据
        
        Args:
            strategies: 策略历史
            market_states: 市场状态历史
            revenues: 收益历史
            
        Returns:
            处理后的特征张量，确保有10个输入特征
        """
        features = []
        
        for i in range(len(strategies)):
            feature_vector = [
                strategies[i],  # 当前策略
                revenues[i] if i < len(revenues) else 0,  # 当前收益
            ]
            
            # 添加市场状态特征
            if i < len(market_states):
                market_state = market_states[i]
                feature_vector.extend([
                    market_state.get('demand', 0),
                    market_state.get('supply', 0),
                    market_state.get('avg_price', 0),
                    market_state.get('competition', 0),
                    market_state.get('order_rate', 0),
                    market_state.get('time_period', 0),
                ])
            else:
                feature_vector.extend([0] * 6)
            
            # 添加策略变化率（如果可计算）
            if i > 0:
                strategy_change = strategies[i] - strategies[i-1]
                revenue_change = revenues[i] - revenues[i-1] if i < len(revenues) else 0
            else:
                strategy_change = 0
                revenue_change = 0
                
            feature_vector.extend([strategy_change, revenue_change])
            
            # 确保特征向量长度为10
            assert len(feature_vector) == 10, f"特征向量长度应为10，但得到了{len(feature_vector)}"
            
            features.append(feature_vector)
        
        return torch.tensor(features, dtype=torch.float32)
    
    def update_history(self, strategy: float, market_state: Dict, revenue: float):
        """更新历史数据"""
        self.strategy_history.append(strategy)
        self.market_history.append(market_state)
        self.revenue_history.append(revenue)
        
        # 保持历史长度在合理范围内
        max_history = self.sequence_length * 5
        if len(self.strategy_history) > max_history:
            self.strategy_history = self.strategy_history[-max_history:]
            self.market_history = self.market_history[-max_history:]
            self.revenue_history = self.revenue_history[-max_history:]
    
    def predict_strategy(self, opponent_id: str, prediction_horizon: int = 1) -> PredictionResult:
        """
        预测对手的下一步策略
        
        Args:
            opponent_id: 对手ID
            prediction_horizon: 预测时间窗口
            
        Returns:
            预测结果
        """
        if len(self.strategy_history) < self.sequence_length:
            # 历史数据不足，返回平均策略
            avg_strategy = np.mean(self.strategy_history) if self.strategy_history else 30.0
            return PredictionResult(
                predicted_strategy=avg_strategy,
                confidence=0.3,
                market_trend="insufficient_data",
                prediction_horizon=prediction_horizon
            )
        
        # 准备输入数据
        recent_data = self.preprocess_data(
            self.strategy_history[-self.sequence_length:],
            self.market_history[-self.sequence_length:],
            self.revenue_history[-self.sequence_length:]
        )
        
        # 调整数据形状为 [1, seq_len, features]
        input_data = recent_data.unsqueeze(0).to(self.device)
        
        # 模型预测
        self.model.eval()
        with torch.no_grad():
            prediction, confidence = self.model(input_data)
            
        predicted_strategy = prediction.cpu().item()
        confidence_score = confidence.cpu().item()
        
        # 分析市场趋势
        market_trend = self._analyze_market_trend()
        
        # 确保策略在有效范围内
        predicted_strategy = np.clip(predicted_strategy, 10.0, 50.0)
        
        return PredictionResult(
            predicted_strategy=predicted_strategy,
            confidence=confidence_score,
            market_trend=market_trend,
            prediction_horizon=prediction_horizon
        )
    
    def _analyze_market_trend(self) -> str:
        """分析市场趋势"""
        if len(self.revenue_history) < 3:
            return "stable"
        
        recent_revenues = self.revenue_history[-3:]
        if recent_revenues[-1] > recent_revenues[-2] > recent_revenues[-3]:
            return "rising"
        elif recent_revenues[-1] < recent_revenues[-2] < recent_revenues[-3]:
            return "falling"
        else:
            return "stable"
    
    def train_on_batch(self, strategies: List[float], market_states: List[Dict], 
                      revenues: List[float], targets: List[float]):
        """
        在一批数据上训练模型
        
        Args:
            strategies: 策略序列
            market_states: 市场状态序列
            revenues: 收益序列
            targets: 目标策略值
        """
        if len(strategies) < self.sequence_length + 1:
            return
            
        self.model.train()
        
        # 创建训练样本
        sequences = []
        labels = []
        
        for i in range(len(strategies) - self.sequence_length):
            # 输入序列
            seq_strategies = strategies[i:i+self.sequence_length]
            seq_markets = market_states[i:i+self.sequence_length]
            seq_revenues = revenues[i:i+self.sequence_length]
            
            # 预处理序列
            seq_features = self.preprocess_data(seq_strategies, seq_markets, seq_revenues)
            sequences.append(seq_features)
            
            # 目标值（下一个策略）
            if i+self.sequence_length < len(targets):
                labels.append(targets[i+self.sequence_length])
        
        if not sequences:
            return
            
        # 转换为张量
        X = torch.stack(sequences).to(self.device)
        y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1).to(self.device)
        
        # 前向传播
        predictions, confidence = self.model(X)
        
        # 计算损失
        prediction_loss = self.criterion(predictions, y)
        
        # 置信度正则化（鼓励高置信度的准确预测）
        accuracy = torch.abs(predictions - y)
        confidence_loss = torch.mean(confidence * accuracy)
        
        total_loss = prediction_loss + 0.1 * confidence_loss
        
        # 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        logger.debug(f"训练损失: {total_loss.item():.4f}, 预测损失: {prediction_loss.item():.4f}")
    
    def save_model(self, filepath: str):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, filepath)
        logger.info(f"模型已保存到: {filepath}")
    
    def load_model(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"模型已从 {filepath} 加载")
    
    def get_prediction_confidence(self) -> float:
        """获取最近预测的平均置信度"""
        if not hasattr(self, '_recent_confidences'):
            return 0.5
        return np.mean(self._recent_confidences[-10:])  # 最近10次预测的平均置信度 