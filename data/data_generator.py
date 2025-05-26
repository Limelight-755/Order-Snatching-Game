"""
数据生成器
生成博弈分析所需的模拟数据
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
import random

logger = logging.getLogger(__name__)


class DataGenerator:
    """
    数据生成器
    生成博弈实验所需的各种模拟数据
    """
    
    def __init__(self, seed: int = 42):
        """
        初始化数据生成器
        
        Args:
            seed: 随机种子
        """
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
        self.logger = logging.getLogger(__name__)
    
    def generate_demand_data(self, time_periods: int, 
                           base_demand: float = 100.0,
                           seasonal_amplitude: float = 20.0,
                           trend_slope: float = 0.1,
                           noise_std: float = 10.0) -> np.ndarray:
        """
        生成需求数据
        
        Args:
            time_periods: 时间段数量
            base_demand: 基础需求量
            seasonal_amplitude: 季节性波动幅度
            trend_slope: 趋势斜率
            noise_std: 噪声标准差
            
        Returns:
            np.ndarray: 需求数据序列
        """
        try:
            # 生成时间序列
            t = np.arange(time_periods)
            
            # 趋势项
            trend = trend_slope * t
            
            # 季节性项（简化为正弦波）
            seasonal = seasonal_amplitude * np.sin(2 * np.pi * t / 24)  # 24小时周期
            
            # 随机噪声
            noise = np.random.normal(0, noise_std, time_periods)
            
            # 组合需求
            demand = base_demand + trend + seasonal + noise
            
            # 确保需求非负
            demand = np.maximum(demand, 0)
            
            return demand
            
        except Exception as e:
            self.logger.error(f"生成需求数据时出错: {e}")
            raise
    
    def generate_price_data(self, time_periods: int,
                          base_price: float = 30.0,
                          volatility: float = 0.1,
                          price_range: Tuple[float, float] = (10.0, 50.0)) -> np.ndarray:
        """
        生成价格数据
        
        Args:
            time_periods: 时间段数量
            base_price: 基础价格
            volatility: 价格波动率
            price_range: 价格范围
            
        Returns:
            np.ndarray: 价格数据序列
        """
        try:
            # 生成随机游走价格
            returns = np.random.normal(0, volatility, time_periods)
            log_prices = np.cumsum(returns) + np.log(base_price)
            prices = np.exp(log_prices)
            
            # 约束价格范围
            prices = np.clip(prices, price_range[0], price_range[1])
            
            return prices
            
        except Exception as e:
            self.logger.error(f"生成价格数据时出错: {e}")
            raise
    
    def generate_competitor_data(self, time_periods: int,
                               num_competitors: int = 2,
                               strategy_types: List[str] = None) -> Dict[str, np.ndarray]:
        """
        生成竞争对手数据
        
        Args:
            time_periods: 时间段数量
            num_competitors: 竞争对手数量
            strategy_types: 策略类型列表
            
        Returns:
            Dict: 竞争对手数据
        """
        try:
            if strategy_types is None:
                strategy_types = ['aggressive', 'conservative', 'adaptive']
            
            competitors_data = {}
            
            for i in range(num_competitors):
                competitor_id = f"competitor_{i+1}"
                strategy = random.choice(strategy_types)
                
                # 根据策略类型生成不同的行为模式
                if strategy == 'aggressive':
                    # 激进策略：高波动，低价格
                    prices = self.generate_price_data(
                        time_periods, base_price=25.0, volatility=0.15
                    )
                elif strategy == 'conservative':
                    # 保守策略：低波动，中等价格
                    prices = self.generate_price_data(
                        time_periods, base_price=35.0, volatility=0.05
                    )
                else:  # adaptive
                    # 适应性策略：中等波动，动态价格
                    prices = self.generate_price_data(
                        time_periods, base_price=30.0, volatility=0.10
                    )
                
                competitors_data[competitor_id] = {
                    'strategy': strategy,
                    'prices': prices,
                    'market_share': np.random.uniform(0.1, 0.3, time_periods)
                }
            
            return competitors_data
            
        except Exception as e:
            self.logger.error(f"生成竞争对手数据时出错: {e}")
            raise
    
    def generate_external_factors(self, time_periods: int) -> Dict[str, np.ndarray]:
        """
        生成外部因素数据
        
        Args:
            time_periods: 时间段数量
            
        Returns:
            Dict: 外部因素数据
        """
        try:
            external_factors = {}
            
            # 天气因素 (0-1，0表示恶劣天气，1表示良好天气)
            weather = np.random.beta(2, 2, time_periods)
            
            # 交通状况 (0-1，0表示拥堵，1表示畅通)
            traffic = np.random.beta(1.5, 1.5, time_periods)
            
            # 经济指标 (标准化)
            economic_index = np.random.normal(0, 1, time_periods)
            
            # 节假日指标 (二元变量)
            holidays = np.random.binomial(1, 0.1, time_periods)  # 10%概率是节假日
            
            # 事件冲击 (偶发事件)
            events = np.random.binomial(1, 0.05, time_periods)  # 5%概率有特殊事件
            
            external_factors = {
                'weather': weather,
                'traffic': traffic,
                'economic_index': economic_index,
                'holidays': holidays,
                'events': events
            }
            
            return external_factors
            
        except Exception as e:
            self.logger.error(f"生成外部因素数据时出错: {e}")
            raise
    
    def generate_geographic_data(self, num_locations: int = 10) -> Dict[str, Any]:
        """
        生成地理位置数据
        
        Args:
            num_locations: 位置数量
            
        Returns:
            Dict: 地理数据
        """
        try:
            locations = {}
            
            for i in range(num_locations):
                location_id = f"location_{i+1}"
                
                # 随机生成坐标 (简化为2D平面)
                x = np.random.uniform(-10, 10)
                y = np.random.uniform(-10, 10)
                
                # 位置特征
                population_density = np.random.uniform(100, 1000)  # 人口密度
                commercial_level = np.random.uniform(0, 1)  # 商业活跃度
                transport_hub = np.random.binomial(1, 0.3)  # 是否是交通枢纽
                
                locations[location_id] = {
                    'coordinates': (x, y),
                    'population_density': population_density,
                    'commercial_level': commercial_level,
                    'transport_hub': transport_hub
                }
            
            return locations
            
        except Exception as e:
            self.logger.error(f"生成地理数据时出错: {e}")
            raise
    
    def generate_player_profiles(self, num_players: int = 2) -> Dict[str, Dict[str, Any]]:
        """
        生成玩家档案数据
        
        Args:
            num_players: 玩家数量
            
        Returns:
            Dict: 玩家档案
        """
        try:
            profiles = {}
            
            personality_types = ['aggressive', 'conservative', 'balanced', 'adaptive']
            experience_levels = ['novice', 'intermediate', 'expert']
            
            for i in range(num_players):
                player_id = f"player_{i+1}"
                
                profile = {
                    'player_id': player_id,
                    'personality': random.choice(personality_types),
                    'experience': random.choice(experience_levels),
                    'risk_tolerance': np.random.uniform(0, 1),
                    'learning_rate': np.random.uniform(0.01, 0.1),
                    'exploration_rate': np.random.uniform(0.1, 0.5),
                    'initial_capital': np.random.uniform(1000, 5000),
                    'reputation_score': np.random.uniform(0.5, 1.0)
                }
                
                profiles[player_id] = profile
            
            return profiles
            
        except Exception as e:
            self.logger.error(f"生成玩家档案时出错: {e}")
            raise
    
    def generate_historical_data(self, days: int = 30, 
                               hours_per_day: int = 24) -> pd.DataFrame:
        """
        生成历史数据
        
        Args:
            days: 天数
            hours_per_day: 每天小时数
            
        Returns:
            pd.DataFrame: 历史数据
        """
        try:
            total_periods = days * hours_per_day
            
            # 生成时间戳
            start_date = datetime.now() - timedelta(days=days)
            timestamps = [start_date + timedelta(hours=i) for i in range(total_periods)]
            
            # 生成各类数据
            demand = self.generate_demand_data(total_periods)
            prices = self.generate_price_data(total_periods)
            external = self.generate_external_factors(total_periods)
            
            # 构建DataFrame
            data = {
                'timestamp': timestamps,
                'demand': demand,
                'average_price': prices,
                'weather': external['weather'],
                'traffic': external['traffic'],
                'economic_index': external['economic_index'],
                'is_holiday': external['holidays'],
                'special_event': external['events']
            }
            
            df = pd.DataFrame(data)
            
            # 添加时间特征
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['is_weekend'] = df['day_of_week'].isin([5, 6])
            
            return df
            
        except Exception as e:
            self.logger.error(f"生成历史数据时出错: {e}")
            raise
    
    def generate_training_data(self, num_samples: int = 1000) -> Dict[str, np.ndarray]:
        """
        生成训练数据
        
        Args:
            num_samples: 样本数量
            
        Returns:
            Dict: 训练数据
        """
        try:
            # 特征数据
            features = {
                'demand': np.random.normal(100, 20, num_samples),
                'competitor_price': np.random.uniform(10, 50, num_samples),
                'weather': np.random.beta(2, 2, num_samples),
                'traffic': np.random.beta(1.5, 1.5, num_samples),
                'hour_of_day': np.random.randint(0, 24, num_samples),
                'day_of_week': np.random.randint(0, 7, num_samples)
            }
            
            # 标签数据（收益）
            # 简化的收益函数：基于需求、价格和外部因素
            base_revenue = features['demand'] * 0.5
            price_effect = np.where(features['competitor_price'] > 30, 1.2, 0.8)
            weather_effect = features['weather'] * 0.3 + 0.7
            traffic_effect = features['traffic'] * 0.2 + 0.8
            
            revenue = base_revenue * price_effect * weather_effect * traffic_effect
            revenue += np.random.normal(0, 10, num_samples)  # 添加噪声
            
            # 构建训练集
            X = np.column_stack([
                features['demand'],
                features['competitor_price'],
                features['weather'],
                features['traffic'],
                features['hour_of_day'],
                features['day_of_week']
            ])
            
            y = revenue
            
            return {
                'X': X,
                'y': y,
                'feature_names': ['demand', 'competitor_price', 'weather', 
                                'traffic', 'hour_of_day', 'day_of_week']
            }
            
        except Exception as e:
            self.logger.error(f"生成训练数据时出错: {e}")
            raise
    
    def save_data(self, data: Any, filepath: str) -> None:
        """
        保存数据到文件
        
        Args:
            data: 要保存的数据
            filepath: 文件路径
        """
        try:
            if filepath.endswith('.csv') and isinstance(data, pd.DataFrame):
                data.to_csv(filepath, index=False)
            elif filepath.endswith('.npy') and isinstance(data, np.ndarray):
                np.save(filepath, data)
            elif filepath.endswith('.json'):
                import json
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2, default=str)
            else:
                # 使用pickle作为默认保存方式
                import pickle
                with open(filepath, 'wb') as f:
                    pickle.dump(data, f)
            
            self.logger.info(f"数据已保存到: {filepath}")
            
        except Exception as e:
            self.logger.error(f"保存数据时出错: {e}")
            raise 