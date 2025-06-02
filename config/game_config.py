"""
博弈配置模块
定义所有博弈相关的参数和设置
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any


@dataclass
class GameConfig:
    """博弈配置类，包含所有可配置的参数"""
    
    # ============ 基本博弈参数 ============
    # 博弈轮次设置
    MAX_ROUNDS: int = 720
    EXPLORATION_ROUNDS: int = 120  # 探索期轮次 (0-120)
    LEARNING_ROUNDS: int = 240     # 学习期轮次 (121-360)
    EQUILIBRIUM_ROUNDS: int = 360  # 均衡期轮次 (361-720)
    
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
    EPSILON_END: float = 0.02         # 最终探索率（降低以减少均衡期变动）
    EPSILON_DECAY_ROUNDS: int = 360   # 探索率衰减轮次（在均衡期开始前基本完成衰减）
    
    # ============ 收益计算参数 ============
    # 基础收益计算
    BASE_INCOME_RATE: float = 0.9     # 基础收入比例
    WAITING_TIME_COST: float = 0.05   # 等待时间成本系数（重命名并优化）
    COMPETITION_BONUS: float = 0.1    # 竞争优势奖励系数
    STRATEGY_STABILITY_PENALTY: float = 0.02  # 策略变化惩罚
    OPERATION_COST_PER_ORDER: float = 0.5  # 每单运营成本
    
    # 新增：收益收敛参数（强化版）
    REVENUE_CONVERGENCE_ENABLED: bool = True  # 是否启用收益收敛机制
    SIMILARITY_BONUS_RATE: float = 0.3        # 策略相似度奖励比例（提高到最高30%）
    COOPERATION_BONUS_RATE: float = 0.25      # 合作奖励比例（提高到最高25%）
    HIGH_STRATEGY_BONUS_RATE: float = 0.15    # 高策略奖励比例（提高到最高15%）
    OPTIMAL_STRATEGY_RANGE: tuple = (30.0, 45.0)  # 最优策略范围（扩大范围）
    REVENUE_BALANCE_ADJUSTMENT: float = 0.2   # 收益均衡调整比例（提高到最高20%）
    CONVERGENCE_INCENTIVE_MULTIPLIER: float = 2.0  # 收敛激励乘数（新增）
    
    # 收益计算权重（更新版）
    REVENUE_WEIGHTS: Dict[str, float] = None
    
    def __post_init__(self):
        if self.REVENUE_WEIGHTS is None:
            self.REVENUE_WEIGHTS = {
                'base_revenue': 1.0,
                'competition_advantage': 0.1,
                'strategy_stability': -0.05,
                'waiting_time_penalty': -0.1,
                'similarity_bonus': 0.2,      # 新增
                'cooperation_bonus': 0.15,    # 新增  
                'high_strategy_bonus': 0.1,   # 新增
                'revenue_balance': 0.1        # 新增
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
    
    # ============ 贝叶斯博弈参数 ============
    # 玩家类型参数
    PLAYER_TYPE_NAMES: List[str] = field(default_factory=lambda: ['aggressive', 'conservative', 'balanced'])
    
    # 不完全信息参数
    OBSERVATION_NOISE_SCALE: float = 0.1  # 观察噪声标准差（占实际值比例）
    PRICE_OBSERVABILITY: float = 0.8     # 观察到对手真实价格的概率
    
    # 信念参数
    BELIEF_UPDATE_RATE: float = 0.2      # 信念更新速率
    PRIOR_STRENGTH: float = 2.0          # 先验信念强度
    LIKELIHOOD_SMOOTHING: float = 0.1    # 似然平滑因子
    
    # 信息集参数
    MAX_HISTORY_LENGTH: int = 10         # 可观察历史最大长度
    HISTORY_DISCOUNT_FACTOR: float = 0.9 # 历史信息折扣因子
    
    # 均衡精炼参数(Equilibrium Refinement)
    INTUITIVE_CRITERION_ENABLED: bool = True  # 是否启用直觉标准检验
    DIVINITY_CRITERION_ENABLED: bool = False  # 是否启用神性标准检验
    TREMBLING_HAND_EPSILON: float = 0.02  # 扰动手完美性检验扰动参数
    D1_CRITERION_ENABLED: bool = False  # 是否启用D1标准检验
    FORWARD_INDUCTION_ENABLED: bool = True  # 是否启用前向归纳
    
    # 信息结构参数
    INFORMATION_STRUCTURE: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.INFORMATION_STRUCTURE is None:
            self.INFORMATION_STRUCTURE = {
                'symmetric_information': False,  # 信息是否对称
                'public_history': True,  # 历史是否公开
                'private_history_length': 5,  # 私有历史长度
                'strategic_information_disclosure': False,  # 是否允许策略性信息披露
                'information_acquisition_cost': 0.0,  # 信息获取成本
                'noisy_signal_variance': 0.2  # 噪声信号方差
            }
    
    # 博弈类型参数
    GAME_TYPE: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.GAME_TYPE is None:
            self.GAME_TYPE = {
                'signaling_game': True,  # 是否为信号博弈
                'repeated_game': True,  # 是否为重复博弈
                'finite_horizon': True,  # 是否为有限视野
                'time_varying_types': False,  # 类型是否随时间变化
                'multidimensional_types': False,  # 是否为多维类型
                'two_sided_private_information': True  # 是否为双边私有信息
            }
    
    # 市场状态可观察性
    MARKET_OBSERVABILITY: Dict[str, bool] = None
    
    def __post_init__(self):
        if self.MARKET_OBSERVABILITY is None:
            self.MARKET_OBSERVABILITY = {
                'demand_level': True,         # 需求水平可观察
                'time_period': True,          # 时段可观察
                'competitor_prices': False,   # 竞争对手价格不可直接观察
                'orders': False,              # 他人订单不可直接观察
                'location_factors': True      # 位置因素可观察
            }
    
    # ============ 贝叶斯智能体参数 ============
    # Bayesian DQN参数
    BAYESIAN_MEMORY_SIZE: int = 10000    # 贝叶斯DQN记忆容量
    BAYESIAN_BATCH_SIZE: int = 64        # 贝叶斯DQN批处理大小
    BAYESIAN_HIDDEN_SIZE: int = 128      # 贝叶斯DQN隐藏层大小
    BAYESIAN_LEARNING_RATE: float = 0.001  # 贝叶斯DQN学习率
    
    # 贝叶斯探索参数
    BAYESIAN_EPSILON: float = 0.8        # 贝叶斯探索初始概率
    BAYESIAN_EPSILON_MIN: float = 0.05   # 贝叶斯探索最小概率
    BAYESIAN_EPSILON_DECAY: float = 0.995  # 贝叶斯探索衰减率
    
    # 状态空间参数
    STATE_SIZE: int = 5                  # 状态向量维度
    ACTION_SIZE: int = 21                # 动作空间大小
    MEMORY_SIZE: int = 10000             # 经验回放缓冲区大小
    BATCH_SIZE: int = 64                 # 批量大小
    GAMMA: float = 0.95                  # 折扣因子
    EPSILON: float = 1.0                 # 初始探索率
    EPSILON_MIN: float = 0.01            # 最小探索率
    EPSILON_DECAY: float = 0.995         # 探索率衰减
    LEARNING_RATE: float = 0.001         # 学习率
    
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
            
            # 验证贝叶斯博弈参数
            assert 0 <= self.OBSERVATION_NOISE_SCALE <= 1, "观察噪声比例必须在[0,1]范围内"
            assert 0 <= self.PRICE_OBSERVABILITY <= 1, "价格可观察性必须在[0,1]范围内"
            assert self.MAX_HISTORY_LENGTH > 0, "历史长度必须大于0"
            assert 0 <= self.HISTORY_DISCOUNT_FACTOR <= 1, "历史折扣因子必须在[0,1]范围内"
            
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

# 扩展博弈论分析参数
DOMINANCE_THRESHOLD = 0.2  # 策略支配阈值

# 新增博弈类型参数
GAME_TYPE = "dynamic"  # 博弈类型: standard, dynamic
EQUILIBRIUM_CONCEPT = "nash"  # 均衡概念: nash

# 博弈论理论分析参数
PARETO_ANALYSIS_ENABLED = True  # 启用帕累托最优性分析
NASH_ANALYSIS_ENABLED = True  # 启用纳什均衡分析

# 信息结构参数
INFORMATION_STRUCTURE = "imperfect"  # 信息结构: perfect, imperfect
PRICE_OBSERVABILITY = 0.8  # 价格观察准确度
NOISY_OBSERVATION_ENABLED = True  # 启用噪声观察