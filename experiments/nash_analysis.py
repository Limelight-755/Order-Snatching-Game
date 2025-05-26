"""
纳什均衡分析实验
分析各种博弈设置下的纳什均衡形成
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import logging

from analysis.nash_analyzer import NashEquilibriumAnalyzer

logger = logging.getLogger(__name__)


class NashEquilibriumAnalyzer:
    """
    封装对纳什均衡分析的实验功能
    提供实验性的均衡分析方法
    """
    
    def __init__(self, convergence_threshold: float = 0.05, stability_window: int = 20):
        """
        初始化纳什均衡分析器
        
        Args:
            convergence_threshold: 收敛阈值
            stability_window: 稳定性检测窗口大小
        """
        self.analyzer = NashEquilibriumAnalyzer(
            convergence_threshold=convergence_threshold,
            stability_window=stability_window
        )
        
        logger.info(f"纳什均衡分析实验初始化: 阈值={convergence_threshold}, 窗口={stability_window}")
    
    def analyze_experiment(self, experiment_results):
        """分析实验结果中的纳什均衡"""
        # 基本实现示例
        return self.analyzer.analyze_nash_equilibrium(experiment_results) 