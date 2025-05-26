"""
实验工具函数和数据结构
包含实验配置、结果记录和分析工具
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
from enum import Enum  # 添加枚举类型导入

logger = logging.getLogger(__name__)


# 辅助函数：处理numpy类型的转换，为JSON序列化做准备
def convert_numpy_types(obj):
    """将NumPy类型转换为Python原生类型，以便JSON序列化"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


@dataclass
class ExperimentConfig:
    """实验配置数据类"""
    # 基础实验参数
    experiment_name: str
    experiment_type: str  # 'symmetric', 'asymmetric', 'shock_test'
    total_rounds: int = 500
    num_runs: int = 10  # 重复实验次数
    
    # 玩家配置
    player_configs: Dict[str, Dict] = field(default_factory=dict)
    
    # 市场环境配置
    market_config: Dict = field(default_factory=dict)
    
    # AI模型配置
    ai_config: Dict = field(default_factory=dict)
    
    # 分析配置
    analysis_config: Dict = field(default_factory=dict)
    
    # 输出配置
    save_detailed_logs: bool = True
    save_frequency: int = 50
    plot_real_time: bool = False
    
    def __post_init__(self):
        """初始化后处理"""
        if not self.player_configs:
            self.player_configs = {
                'player_a': {'type': 'dqn', 'learning_enabled': True},
                'player_b': {'type': 'dqn', 'learning_enabled': True}
            }
        
        if not self.market_config:
            self.market_config = {
                'base_order_rate': 50,
                'price_sensitivity': 0.3,
                'competition_factor': 0.2
            }
        
        if not self.analysis_config:
            self.analysis_config = {
                'nash_threshold': 0.05,
                'convergence_window': 50,
                'stability_threshold': 0.1
            }


@dataclass
class RoundResult:
    """单轮结果数据类"""
    round_number: int
    player_a_strategy: float
    player_b_strategy: float
    player_a_revenue: float
    player_b_revenue: float
    market_state: Dict
    nash_distance: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        # 处理market_state中的特殊类型（如枚举）
        market_state_copy = {}
        for key, value in self.market_state.items():
            # 处理枚举类型
            if isinstance(value, Enum):
                market_state_copy[key] = value.value
            else:
                market_state_copy[key] = value

        # 构建结果字典并转换所有NumPy类型
        result = {
            'round_number': self.round_number,
            'player_a_strategy': self.player_a_strategy,
            'player_b_strategy': self.player_b_strategy,
            'player_a_revenue': self.player_a_revenue,
            'player_b_revenue': self.player_b_revenue,
            'market_state': market_state_copy,
            'nash_distance': self.nash_distance
        }
        
        # 使用全局convert_numpy_types函数处理NumPy类型
        return convert_numpy_types(result)


@dataclass
class ExperimentResult:
    """实验结果数据类"""
    experiment_config: ExperimentConfig
    start_time: datetime
    end_time: Optional[datetime] = None
    total_duration: Optional[float] = None
    
    # 轮次结果
    round_results: List[RoundResult] = field(default_factory=list)
    
    # 汇总统计
    final_strategies: Dict[str, float] = field(default_factory=dict)
    total_revenues: Dict[str, float] = field(default_factory=dict)
    convergence_round: Optional[int] = None
    nash_equilibrium_found: bool = False
    stability_achieved: bool = False
    
    # 阶段统计
    phase_statistics: Dict[str, Dict] = field(default_factory=dict)
    
    def add_round_result(self, result: RoundResult):
        """添加轮次结果"""
        self.round_results.append(result)
    
    def calculate_summary_statistics(self):
        """计算汇总统计"""
        if not self.round_results:
            return
        
        # 计算最终策略
        final_window = self.round_results[-20:]  # 最后20轮
        self.final_strategies = {
            'player_a': np.mean([r.player_a_strategy for r in final_window]),
            'player_b': np.mean([r.player_b_strategy for r in final_window])
        }
        
        # 计算总收益
        self.total_revenues = {
            'player_a': sum(r.player_a_revenue for r in self.round_results),
            'player_b': sum(r.player_b_revenue for r in self.round_results)
        }
        
        # 检测收敛
        self._detect_convergence()
        
        # 分析阶段统计
        self._analyze_phases()
    
    def _detect_convergence(self):
        """检测策略收敛"""
        window_size = 50
        convergence_threshold = 0.05
        
        for i in range(window_size, len(self.round_results)):
            window = self.round_results[i-window_size:i]
            
            # 计算策略变化
            a_strategies = [r.player_a_strategy for r in window]
            b_strategies = [r.player_b_strategy for r in window]
            
            a_std = np.std(a_strategies)
            b_std = np.std(b_strategies)
            
            if a_std < convergence_threshold and b_std < convergence_threshold:
                self.convergence_round = self.round_results[i].round_number
                break
    
    def _analyze_phases(self):
        """分析不同阶段的统计"""
        total_rounds = len(self.round_results)
        
        # 定义阶段
        phases = {
            'exploration': (0, int(0.1 * total_rounds)),
            'learning': (int(0.1 * total_rounds), int(0.4 * total_rounds)),
            'equilibrium': (int(0.4 * total_rounds), total_rounds)
        }
        
        for phase_name, (start, end) in phases.items():
            if end > len(self.round_results):
                continue
                
            phase_data = self.round_results[start:end]
            
            self.phase_statistics[phase_name] = {
                'avg_strategy_a': np.mean([r.player_a_strategy for r in phase_data]),
                'avg_strategy_b': np.mean([r.player_b_strategy for r in phase_data]),
                'strategy_stability_a': 1.0 / (1.0 + np.std([r.player_a_strategy for r in phase_data])),
                'strategy_stability_b': 1.0 / (1.0 + np.std([r.player_b_strategy for r in phase_data])),
                'avg_revenue_a': np.mean([r.player_a_revenue for r in phase_data]),
                'avg_revenue_b': np.mean([r.player_b_revenue for r in phase_data]),
                'round_range': (start, end)
            }
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        result_dict = {
            'experiment_config': {
                'experiment_name': self.experiment_config.experiment_name,
                'experiment_type': self.experiment_config.experiment_type,
                'total_rounds': self.experiment_config.total_rounds,
                'num_runs': self.experiment_config.num_runs
            },
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'total_duration': self.total_duration,
            'final_strategies': convert_numpy_types(self.final_strategies),
            'total_revenues': convert_numpy_types(self.total_revenues),
            'convergence_round': self.convergence_round,
            'nash_equilibrium_found': bool(self.nash_equilibrium_found) if hasattr(self.nash_equilibrium_found, 'dtype') else self.nash_equilibrium_found,
            'stability_achieved': bool(self.stability_achieved) if hasattr(self.stability_achieved, 'dtype') else self.stability_achieved,
            'phase_statistics': convert_numpy_types(self.phase_statistics),
            'round_results_count': len(self.round_results),
            'players': ['player_a', 'player_b'],
            'experiment_type': self.experiment_config.experiment_type,
            'total_rounds': self.experiment_config.total_rounds
        }
        
        # 添加动态属性（如asymmetric_analysis）
        if hasattr(self, 'asymmetric_analysis'):
            result_dict['asymmetric_analysis'] = convert_numpy_types(self.asymmetric_analysis)
        
        # 添加轮次结果数据
        round_results_data = []
        for r in self.round_results:
            round_results_data.append(r.to_dict())
        result_dict['round_results_data'] = round_results_data
        
        return result_dict
    
    def save_to_json(self, filepath: str):
        """保存结果到JSON文件"""
        try:
            # 先将数据转换为字典
            data_dict = self.to_dict()
            
            # 使用自定义JSON编码器处理NumPy类型
            class NumpyEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, (np.integer, np.int64, np.int32)):
                        return int(obj)
                    elif isinstance(obj, (np.floating, np.float64, np.float32)):
                        return float(obj)
                    elif isinstance(obj, (np.ndarray,)):
                        return obj.tolist()
                    elif isinstance(obj, (np.bool_)):
                        return bool(obj)
                    return super(NumpyEncoder, self).default(obj)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data_dict, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
            logger.info(f"实验结果已保存到: {filepath}")
        except Exception as e:
            logger.error(f"保存结果失败: {e}")
            import traceback
            traceback.print_exc()


class ExperimentLogger:
    """实验日志记录器"""
    
    def __init__(self, experiment_name: str, log_dir: str = "results/logs"):
        """
        初始化日志记录器
        
        Args:
            experiment_name: 实验名称
            log_dir: 日志目录
        """
        self.experiment_name = experiment_name
        self.log_dir = log_dir
        self.logger = logging.getLogger(f"experiment.{experiment_name}")
        
        # 创建日志目录
        import os
        os.makedirs(log_dir, exist_ok=True)
        
        # 设置日志文件
        log_file = os.path.join(log_dir, f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
        # 配置日志格式
        handler = logging.FileHandler(log_file, encoding='utf-8')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def log_experiment_start(self, config: ExperimentConfig):
        """记录实验开始"""
        self.logger.info(f"实验开始: {config.experiment_name}")
        self.logger.info(f"实验类型: {config.experiment_type}")
        self.logger.info(f"总轮次: {config.total_rounds}")
        self.logger.info(f"重复次数: {config.num_runs}")
    
    def log_round_result(self, result: RoundResult):
        """记录轮次结果"""
        self.logger.debug(f"轮次 {result.round_number}: "
                         f"策略A={result.player_a_strategy:.2f}, "
                         f"策略B={result.player_b_strategy:.2f}, "
                         f"收益A={result.player_a_revenue:.2f}, "
                         f"收益B={result.player_b_revenue:.2f}")
    
    def log_phase_transition(self, phase: str, round_number: int):
        """记录阶段转换"""
        self.logger.info(f"进入{phase}阶段: 轮次 {round_number}")
    
    def log_convergence(self, round_number: int):
        """记录收敛检测"""
        self.logger.info(f"策略收敛检测: 轮次 {round_number}")
    
    def log_nash_equilibrium(self, round_number: int, strategies: Dict[str, float]):
        """记录纳什均衡"""
        self.logger.info(f"纳什均衡检测: 轮次 {round_number}, 策略 {strategies}")
    
    def log_experiment_end(self, result: ExperimentResult):
        """记录实验结束"""
        self.logger.info(f"实验结束: {result.experiment_config.experiment_name}")
        self.logger.info(f"最终策略: {result.final_strategies}")
        self.logger.info(f"总收益: {result.total_revenues}")
        self.logger.info(f"收敛轮次: {result.convergence_round}")


class DataCollector:
    """数据收集器"""
    
    def __init__(self):
        """初始化数据收集器"""
        self.collected_data = {
            'strategies': {'player_a': [], 'player_b': []},
            'revenues': {'player_a': [], 'player_b': []},
            'market_states': [],
            'decisions': {'player_a': [], 'player_b': []},
            'predictions': {'player_a': [], 'player_b': []}
        }
    
    def collect_round_data(self, round_result: RoundResult, 
                          additional_data: Dict = None):
        """收集轮次数据"""
        # 基础数据
        self.collected_data['strategies']['player_a'].append(round_result.player_a_strategy)
        self.collected_data['strategies']['player_b'].append(round_result.player_b_strategy)
        self.collected_data['revenues']['player_a'].append(round_result.player_a_revenue)
        self.collected_data['revenues']['player_b'].append(round_result.player_b_revenue)
        self.collected_data['market_states'].append(round_result.market_state)
        
        # 额外数据
        if additional_data:
            for key, value in additional_data.items():
                if key not in self.collected_data:
                    self.collected_data[key] = []
                self.collected_data[key].append(value)
    
    def get_dataframe(self) -> pd.DataFrame:
        """获取pandas DataFrame格式的数据"""
        # 基础数据
        data = {
            'round': range(len(self.collected_data['strategies']['player_a'])),
            'strategy_a': self.collected_data['strategies']['player_a'],
            'strategy_b': self.collected_data['strategies']['player_b'],
            'revenue_a': self.collected_data['revenues']['player_a'],
            'revenue_b': self.collected_data['revenues']['player_b']
        }
        
        # 市场状态数据
        if self.collected_data['market_states']:
            for key in self.collected_data['market_states'][0].keys():
                data[f'market_{key}'] = [ms.get(key, 0) for ms in self.collected_data['market_states']]
        
        return pd.DataFrame(data)
    
    def export_to_csv(self, filepath: str):
        """导出数据到CSV文件"""
        df = self.get_dataframe()
        df.to_csv(filepath, index=False, encoding='utf-8')
        logger.info(f"数据已导出到: {filepath}")
    
    def get_summary_statistics(self) -> Dict:
        """获取汇总统计"""
        df = self.get_dataframe()
        
        return {
            'strategy_stats': {
                'player_a': {
                    'mean': df['strategy_a'].mean(),
                    'std': df['strategy_a'].std(),
                    'min': df['strategy_a'].min(),
                    'max': df['strategy_a'].max()
                },
                'player_b': {
                    'mean': df['strategy_b'].mean(),
                    'std': df['strategy_b'].std(),
                    'min': df['strategy_b'].min(),
                    'max': df['strategy_b'].max()
                }
            },
            'revenue_stats': {
                'player_a': {
                    'total': df['revenue_a'].sum(),
                    'mean': df['revenue_a'].mean(),
                    'std': df['revenue_a'].std()
                },
                'player_b': {
                    'total': df['revenue_b'].sum(),
                    'mean': df['revenue_b'].mean(),
                    'std': df['revenue_b'].std()
                }
            },
            'correlation': {
                'strategy_correlation': df['strategy_a'].corr(df['strategy_b']),
                'revenue_correlation': df['revenue_a'].corr(df['revenue_b'])
            }
        }


def create_experiment_config(experiment_type: str, **kwargs) -> ExperimentConfig:
    """
    创建实验配置的便捷函数
    
    Args:
        experiment_type: 实验类型
        **kwargs: 其他配置参数
        
    Returns:
        实验配置对象
    """
    # 预定义配置模板
    templates = {
        'symmetric': {
            'experiment_name': '对称博弈实验',
            'player_configs': {
                'player_a': {'type': 'dqn', 'learning_enabled': True, 'initial_strategy': 30},
                'player_b': {'type': 'dqn', 'learning_enabled': True, 'initial_strategy': 30}
            }
        },
        'asymmetric': {
            'experiment_name': '非对称博弈实验',
            'player_configs': {
                'player_a': {'type': 'dqn', 'learning_enabled': True, 'initial_strategy': 25},
                'player_b': {'type': 'dqn', 'learning_enabled': True, 'initial_strategy': 35}
            }
        },
        'shock_test': {
            'experiment_name': '冲击测试实验',
            'market_config': {
                'shock_enabled': True,
                'shock_rounds': [200, 300],
                'shock_magnitude': 0.5
            }
        }
    }
    
    # 基础配置
    base_config = templates.get(experiment_type, {})
    base_config['experiment_type'] = experiment_type
    
    # 合并用户提供的配置
    base_config.update(kwargs)
    
    return ExperimentConfig(**base_config)


def compare_experiment_results(results: List[ExperimentResult]) -> Dict:
    """
    比较多个实验结果
    
    Args:
        results: 实验结果列表
        
    Returns:
        比较分析结果
    """
    comparison = {
        'experiments': [],
        'performance_ranking': [],
        'strategy_comparison': {},
        'convergence_comparison': {},
        'stability_comparison': {}
    }
    
    for result in results:
        exp_name = result.experiment_config.experiment_name
        comparison['experiments'].append(exp_name)
        
        # 性能比较
        total_revenue = sum(result.total_revenues.values())
        comparison['performance_ranking'].append((exp_name, total_revenue))
        
        # 策略比较
        comparison['strategy_comparison'][exp_name] = result.final_strategies
        
        # 收敛比较
        comparison['convergence_comparison'][exp_name] = result.convergence_round
        
        # 稳定性比较
        if result.phase_statistics.get('equilibrium'):
            eq_stats = result.phase_statistics['equilibrium']
            stability = (eq_stats['strategy_stability_a'] + eq_stats['strategy_stability_b']) / 2
            comparison['stability_comparison'][exp_name] = stability
    
    # 排序
    comparison['performance_ranking'].sort(key=lambda x: x[1], reverse=True)
    
    return comparison 