"""
博弈配置模块
定义所有博弈相关的参数和设置
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any


@dataclass
class GameConfig:
    """博弈配置类，包含所有可配置的参数"""
    
    # ============ 基本博弈参数 ============
    # 博弈轮次设置
    MAX_ROUNDS: int = 500
    EXPLORATION_ROUNDS: int = 50  # 探索期轮次
    LEARNING_ROUNDS: int = 150    # 学习期轮次 (50-200)
    EQUILIBRIUM_ROUNDS: int = 300 # 均衡期轮次 (200-500)
    
    # 参与者设置
    NUM_PLAYERS: int = 2
    PLAYER_NAMES: List[str] = None
    
    def __post_init__(self):
        if self.PLAYER_NAMES is None:
            self.PLAYER_NAMES = [f"司机{chr(65+i)}" for i in range(self.NUM_PLAYERS)]
    
    # ============ 策略空间参数 ============
    # 定价策略空间
    MIN_PRICE_THRESHOLD: float = 10.0  # 最低定价阈值
    MAX_PRICE_THRESHOLD: float = 50.0  # 最高定价阈值
    PRICE_STEP: float = 0.5            # 价格调整步长
    MAX_PRICE_CHANGE: float = 5.0      # 单轮最大价格变化
    
    # ============ 市场环境参数 ============
    # 订单生成参数
    BASE_ORDER_RATE: float = 0.5       # 基础订单到达率 (订单/分钟)
    BASE_PRICE_MEAN: float = 25.0      # 基础价格均值
    BASE_PRICE_STD: float = 8.0        # 基础价格标准差
    DEMAND_VOLATILITY: float = 0.3     # 需求波动系数
    
    # 位置分布
    LOCATION_DISTRIBUTION: Dict[str, float] = None
    
    def __post_init__(self):
        if self.LOCATION_DISTRIBUTION is None:
            self.LOCATION_DISTRIBUTION = {
                'hotspot': 0.3,  # 热点区域
                'normal': 0.5,   # 普通区域
                'remote': 0.2    # 偏远区域
            }
    
    # 时间段影响系数
    TIME_EFFECTS: Dict[str, Dict[str, float]] = None
    
    def __post_init__(self):
        if self.TIME_EFFECTS is None:
            self.TIME_EFFECTS = {
                'peak': {      # 高峰期 (7-9, 17-19)
                    'order_multiplier': 1.2,
                    'price_multiplier': 1.3
                },
                'normal': {    # 平峰期 (10-16)
                    'order_multiplier': 0.8,
                    'price_multiplier': 0.9
                },
                'low': {       # 低峰期 (20-6)
                    'order_multiplier': 0.4,
                    'price_multiplier': 0.7
                }
            }
    
    # 地理位置影响
    LOCATION_EFFECTS: Dict[str, Dict[str, float]] = None
    
    def __post_init__(self):
        if self.LOCATION_EFFECTS is None:
            self.LOCATION_EFFECTS = {
                'hotspot': {    # 热点区域
                    'order_density_boost': 1.5,
                    'price_premium': 1.2
                },
                'normal': {     # 普通区域
                    'order_density_boost': 1.0,
                    'price_premium': 1.0
                },
                'remote': {     # 偏远区域
                    'order_density_boost': 0.4,
                    'price_premium': 1.3
                }
            }
    
    # ============ AI模型参数 ============
    # LSTM预测模型参数
    LSTM_HIDDEN_SIZE: int = 64
    LSTM_NUM_LAYERS: int = 2
    LSTM_SEQUENCE_LENGTH: int = 10    # 历史序列长度
    LSTM_LEARNING_RATE: float = 0.001
    LSTM_BATCH_SIZE: int = 32
    
    # DQN参数
    DQN_STATE_SIZE: int = 11          # 状态向量维度
    DQN_ACTION_SIZE: int = 21         # 动作空间大小 (价格调整范围)
    DQN_HIDDEN_SIZE: int = 128
    DQN_LEARNING_RATE: float = 0.0005
    DQN_MEMORY_SIZE: int = 10000
    DQN_BATCH_SIZE: int = 64
    DQN_TARGET_UPDATE: int = 100      # 目标网络更新频率
    
    # 探索策略参数
    EPSILON_START: float = 0.8        # 初始探索率
    EPSILON_END: float = 0.05         # 最终探索率
    EPSILON_DECAY_ROUNDS: int = 200   # 探索率衰减轮次
    
    # ============ 收益计算参数 ============
    # 基础收益计算
    BASE_INCOME_RATE: float = 0.8     # 基础收入比例
    WAITING_TIME_PENALTY: float = 0.1 # 等待时间惩罚系数
    COMPETITION_BONUS: float = 0.1    # 竞争优势奖励系数
    STRATEGY_STABILITY_PENALTY: float = 0.05  # 策略变化惩罚
    
    # 收益计算权重
    REVENUE_WEIGHTS: Dict[str, float] = None
    
    def __post_init__(self):
        if self.REVENUE_WEIGHTS is None:
            self.REVENUE_WEIGHTS = {
                'base_revenue': 1.0,
                'competition_advantage': 0.1,
                'strategy_stability': -0.05,
                'waiting_time_penalty': -0.1
            }
    
    # ============ 博弈分析参数 ============
    # 均衡检测参数
    EQUILIBRIUM_TOLERANCE: float = 0.1      # 均衡容忍度
    CONVERGENCE_WINDOW: int = 50            # 收敛检测窗口
    STABILITY_THRESHOLD: float = 0.05       # 稳定性阈值
    
    # 分析指标参数
    ANALYSIS_WINDOW: int = 30               # 分析窗口大小
    
    # ============ 实验场景参数 ============
    # 对称博弈参数
    SYMMETRIC_GAME: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.SYMMETRIC_GAME is None:
            self.SYMMETRIC_GAME = {
                'equal_learning_rates': True,
                'equal_risk_preferences': True,
                'initial_strategy_range': (20.0, 30.0)
            }
    
    # 非对称博弈参数
    ASYMMETRIC_GAME: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.ASYMMETRIC_GAME is None:
            self.ASYMMETRIC_GAME = {
                'player_types': ['aggressive', 'conservative'],
                'learning_rate_ratios': [1.5, 0.8],
                'risk_preferences': [0.8, 1.2],
                'initial_strategies': [35.0, 20.0]
            }
    
    # 环境冲击参数
    SHOCK_TEST: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.SHOCK_TEST is None:
            self.SHOCK_TEST = {
                'shock_rounds': [150, 300],  # 冲击发生轮次
                'shock_types': ['demand_surge', 'policy_change'],
                'shock_magnitude': [1.5, 0.7],  # 冲击强度
                'shock_duration': [20, 30]      # 冲击持续时间
            }
    
    # ============ 日志和输出参数 ============
    # 日志设置
    LOG_LEVEL: str = "INFO"
    LOG_TO_FILE: bool = True
    LOG_TO_CONSOLE: bool = True
    
    # 可视化设置
    PLOT_EVERY_N_ROUNDS: int = 50     # 绘图频率
    SAVE_PLOTS: bool = True
    PLOT_DPI: int = 300
    
    # 模型保存设置
    SAVE_MODELS: bool = True
    MODEL_SAVE_INTERVAL: int = 100    # 模型保存间隔
    
    # ============ 系统性能参数 ============
    # 多线程设置
    USE_MULTIPROCESSING: bool = False
    NUM_WORKERS: int = 2
    
    # 随机种子
    RANDOM_SEED: int = 42
    
    def get_epsilon_for_round(self, round_num: int) -> float:
        """根据轮次计算当前探索率"""
        if round_num <= self.EPSILON_DECAY_ROUNDS:
            progress = round_num / self.EPSILON_DECAY_ROUNDS
            return self.EPSILON_START + (self.EPSILON_END - self.EPSILON_START) * progress
        else:
            return self.EPSILON_END
    
    def get_time_period(self, hour: int) -> str:
        """根据小时确定时间段类型"""
        if hour in [7, 8, 17, 18]:
            return 'peak'
        elif hour in range(10, 16):
            return 'normal'
        else:
            return 'low'
    
    def validate_config(self) -> bool:
        """验证配置参数的有效性"""
        try:
            # 验证基本参数
            assert self.MAX_ROUNDS > 0, "博弈轮次必须大于0"
            assert self.NUM_PLAYERS >= 2, "参与者数量必须至少为2"
            
            # 验证策略空间
            assert self.MIN_PRICE_THRESHOLD < self.MAX_PRICE_THRESHOLD, "价格阈值范围无效"
            assert self.PRICE_STEP > 0, "价格步长必须大于0"
            
            # 验证市场参数
            assert self.BASE_ORDER_RATE > 0, "订单到达率必须大于0"
            assert self.BASE_PRICE_STD > 0, "价格标准差必须大于0"
            
            # 验证AI参数
            assert self.LSTM_HIDDEN_SIZE > 0, "LSTM隐藏层大小必须大于0"
            assert self.DQN_STATE_SIZE > 0, "DQN状态空间大小必须大于0"
            assert 0 <= self.EPSILON_START <= 1, "探索率必须在[0,1]范围内"
            assert 0 <= self.EPSILON_END <= 1, "探索率必须在[0,1]范围内"
            
            return True
            
        except AssertionError as e:
            print(f"配置验证失败: {e}")
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """将配置转换为字典格式"""
        return {
            key: getattr(self, key) 
            for key in dir(self) 
            if not key.startswith('_') and not callable(getattr(self, key))
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'GameConfig':
        """从字典创建配置对象"""
        config = cls()
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config


# 创建默认配置实例
default_config = GameConfig()

# 验证默认配置
if not default_config.validate_config():
    raise ValueError("默认配置验证失败")

print("博弈配置模块加载完成")