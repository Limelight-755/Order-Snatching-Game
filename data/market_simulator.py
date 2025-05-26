"""
市场模拟器
提供真实的出租车市场环境模拟
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
import random
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MarketState:
    """市场状态"""
    timestamp: datetime
    total_demand: float
    total_supply: float
    average_price: float
    weather_factor: float
    traffic_factor: float
    competition_intensity: float
    market_efficiency: float
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'timestamp': self.timestamp,
            'total_demand': self.total_demand,
            'total_supply': self.total_supply,
            'average_price': self.average_price,
            'weather_factor': self.weather_factor,
            'traffic_factor': self.traffic_factor,
            'competition_intensity': self.competition_intensity,
            'market_efficiency': self.market_efficiency
        }


@dataclass
class OrderInfo:
    """订单信息"""
    order_id: str
    pickup_location: Tuple[float, float]
    destination: Tuple[float, float]
    distance: float
    base_price: float
    surge_multiplier: float
    passenger_willingness_to_pay: float
    time_constraint: float  # 时间敏感度
    
    def calculate_max_price(self) -> float:
        """计算乘客愿意支付的最高价格"""
        return self.passenger_willingness_to_pay * self.surge_multiplier


@dataclass
class DriverInfo:
    """司机信息"""
    driver_id: str
    location: Tuple[float, float]
    available: bool
    rating: float
    experience_level: int
    efficiency_score: float
    
    def calculate_acceptance_probability(self, order: OrderInfo, offered_price: float) -> float:
        """计算接受订单的概率"""
        # 基于价格、距离、司机特征计算接受概率
        price_factor = min(1.0, offered_price / (order.base_price * 0.8))
        distance_factor = max(0.1, 1.0 - order.distance / 20.0)  # 距离越远接受概率越低
        efficiency_factor = self.efficiency_score
        
        probability = (price_factor * 0.4 + distance_factor * 0.3 + efficiency_factor * 0.3)
        return min(1.0, max(0.0, probability))


class MarketSimulator:
    """
    市场模拟器
    模拟出租车市场的真实运行环境
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化市场模拟器
        
        Args:
            config: 配置参数
        """
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
        
        # 初始化市场状态
        self.current_time = datetime.now()
        self.market_history = []
        self.active_orders = []
        self.available_drivers = []
        
        # 地理和环境设置
        self.city_bounds = self.config.get('city_bounds', (-10, -10, 10, 10))  # (min_x, min_y, max_x, max_y)
        self.num_zones = self.config.get('num_zones', 16)
        self.zones = self._initialize_zones()
        
        # 初始化司机和订单
        self._initialize_drivers()
        
        self.logger.info("市场模拟器初始化完成")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'num_drivers': 100,
            'base_demand_rate': 50,  # 每小时基础订单数
            'demand_variance': 0.3,
            'weather_impact': 0.2,
            'traffic_impact': 0.15,
            'surge_threshold': 1.5,  # 供需比例超过此值时启动动态定价
            'max_surge_multiplier': 3.0,
            'city_bounds': (-10, -10, 10, 10),
            'num_zones': 16,
            'base_price_per_km': 2.5,
            'minimum_fare': 8.0
        }
    
    def _initialize_zones(self) -> List[Dict[str, Any]]:
        """初始化城市区域"""
        zones = []
        min_x, min_y, max_x, max_y = self.city_bounds
        
        # 创建网格化区域
        rows = int(np.sqrt(self.num_zones))
        cols = int(np.ceil(self.num_zones / rows))
        
        x_step = (max_x - min_x) / cols
        y_step = (max_y - min_y) / rows
        
        for i in range(rows):
            for j in range(cols):
                if len(zones) >= self.num_zones:
                    break
                
                zone_id = f"zone_{i}_{j}"
                center_x = min_x + (j + 0.5) * x_step
                center_y = min_y + (i + 0.5) * y_step
                
                # 随机分配区域特征
                zone = {
                    'zone_id': zone_id,
                    'center': (center_x, center_y),
                    'bounds': (min_x + j * x_step, min_y + i * y_step,
                             min_x + (j + 1) * x_step, min_y + (i + 1) * y_step),
                    'population_density': np.random.uniform(0.5, 2.0),
                    'commercial_level': np.random.uniform(0.3, 1.0),
                    'transport_hub': np.random.choice([True, False], p=[0.2, 0.8])
                }
                zones.append(zone)
        
        return zones
    
    def _initialize_drivers(self) -> None:
        """初始化司机"""
        num_drivers = self.config.get('num_drivers', 100)
        min_x, min_y, max_x, max_y = self.city_bounds
        
        for i in range(num_drivers):
            driver = DriverInfo(
                driver_id=f"driver_{i}",
                location=(
                    np.random.uniform(min_x, max_x),
                    np.random.uniform(min_y, max_y)
                ),
                available=True,
                rating=np.random.uniform(4.0, 5.0),
                experience_level=np.random.randint(1, 6),
                efficiency_score=np.random.uniform(0.6, 1.0)
            )
            self.available_drivers.append(driver)
    
    def generate_orders(self, time_period: int = 1) -> List[OrderInfo]:
        """
        生成订单
        
        Args:
            time_period: 时间段（小时）
            
        Returns:
            List[OrderInfo]: 生成的订单列表
        """
        try:
            # 计算基础需求
            base_demand = self.config.get('base_demand_rate', 50) * time_period
            
            # 时间因素
            hour = self.current_time.hour
            day_of_week = self.current_time.weekday()
            
            # 时间调整因子
            if 7 <= hour <= 9 or 17 <= hour <= 19:  # 早晚高峰
                time_factor = 1.8
            elif 22 <= hour or hour <= 6:  # 深夜
                time_factor = 0.3
            else:  # 其他时间
                time_factor = 1.0
            
            # 周末调整
            if day_of_week >= 5:  # 周末
                if 10 <= hour <= 14 or 19 <= hour <= 23:
                    time_factor *= 1.3
                else:
                    time_factor *= 0.8
            
            # 天气和交通影响
            weather_factor = self._get_weather_factor()
            traffic_factor = self._get_traffic_factor()
            
            # 最终需求
            adjusted_demand = base_demand * time_factor * weather_factor * traffic_factor
            demand_variance = self.config.get('demand_variance', 0.3)
            num_orders = max(0, int(np.random.normal(adjusted_demand, adjusted_demand * demand_variance)))
            
            # 生成订单
            orders = []
            for i in range(num_orders):
                order = self._create_random_order(f"order_{int(self.current_time.timestamp())}_{i}")
                orders.append(order)
            
            return orders
            
        except Exception as e:
            self.logger.error(f"生成订单时出错: {e}")
            return []
    
    def _create_random_order(self, order_id: str) -> OrderInfo:
        """创建随机订单"""
        min_x, min_y, max_x, max_y = self.city_bounds
        
        # 随机起点和终点
        pickup = (
            np.random.uniform(min_x, max_x),
            np.random.uniform(min_y, max_y)
        )
        destination = (
            np.random.uniform(min_x, max_x),
            np.random.uniform(min_y, max_y)
        )
        
        # 计算距离
        distance = np.sqrt((destination[0] - pickup[0])**2 + (destination[1] - pickup[1])**2)
        
        # 基础价格
        base_price_per_km = self.config.get('base_price_per_km', 2.5)
        minimum_fare = self.config.get('minimum_fare', 8.0)
        base_price = max(minimum_fare, distance * base_price_per_km)
        
        # 计算动态定价倍数
        surge_multiplier = self._calculate_surge_multiplier()
        
        # 乘客支付意愿
        willingness_base = base_price * np.random.uniform(0.8, 1.5)
        time_sensitivity = np.random.uniform(0.5, 1.0)  # 时间敏感度影响支付意愿
        
        return OrderInfo(
            order_id=order_id,
            pickup_location=pickup,
            destination=destination,
            distance=distance,
            base_price=base_price,
            surge_multiplier=surge_multiplier,
            passenger_willingness_to_pay=willingness_base,
            time_constraint=time_sensitivity
        )
    
    def _calculate_surge_multiplier(self) -> float:
        """计算动态定价倍数"""
        # 计算供需比
        available_drivers = len([d for d in self.available_drivers if d.available])
        current_demand = len(self.active_orders)
        
        if available_drivers == 0:
            supply_demand_ratio = 0
        else:
            supply_demand_ratio = available_drivers / max(current_demand, 1)
        
        surge_threshold = self.config.get('surge_threshold', 1.5)
        max_surge = self.config.get('max_surge_multiplier', 3.0)
        
        if supply_demand_ratio < 1.0 / surge_threshold:
            # 需求远大于供给
            surge = min(max_surge, 1.0 + (1.0 / surge_threshold - supply_demand_ratio) * 2)
        else:
            surge = 1.0
        
        return surge
    
    def _get_weather_factor(self) -> float:
        """获取天气影响因子"""
        # 简化的天气模型
        base_weather = 1.0
        
        # 随机天气事件
        weather_event = np.random.choice(
            ['sunny', 'cloudy', 'rainy', 'stormy'],
            p=[0.6, 0.25, 0.12, 0.03]
        )
        
        weather_impact = {
            'sunny': 1.0,
            'cloudy': 1.05,
            'rainy': 1.3,
            'stormy': 1.8
        }
        
        return weather_impact.get(weather_event, 1.0)
    
    def _get_traffic_factor(self) -> float:
        """获取交通影响因子"""
        hour = self.current_time.hour
        
        # 高峰期交通拥堵增加需求
        if 7 <= hour <= 9 or 17 <= hour <= 19:
            return np.random.uniform(1.2, 1.5)
        elif 22 <= hour or hour <= 6:
            return np.random.uniform(0.8, 1.0)
        else:
            return np.random.uniform(0.9, 1.1)
    
    def simulate_market_matching(self, player_strategies: Dict[str, float], 
                               new_orders: List[OrderInfo]) -> Dict[str, Any]:
        """
        模拟市场匹配过程
        
        Args:
            player_strategies: 玩家策略 {player_id: pricing_strategy}
            new_orders: 新增订单
            
        Returns:
            Dict: 匹配结果
        """
        try:
            # 更新活跃订单
            self.active_orders.extend(new_orders)
            
            # 为每个玩家分配订单
            player_results = {}
            total_orders_matched = 0
            total_revenue_generated = 0
            
            for player_id, strategy in player_strategies.items():
                # 计算该玩家的订单匹配
                matched_orders, revenue, market_share = self._match_orders_for_player(
                    player_id, strategy, self.active_orders.copy()
                )
                
                player_results[player_id] = {
                    'matched_orders': matched_orders,
                    'revenue': revenue,
                    'market_share': market_share,
                    'num_orders': len(matched_orders),
                    'avg_price': revenue / max(len(matched_orders), 1),
                    'strategy': strategy
                }
                
                total_orders_matched += len(matched_orders)
                total_revenue_generated += revenue
            
            # 更新市场状态
            market_state = self._update_market_state(player_results, new_orders)
            
            # 清理已匹配的订单
            self._cleanup_matched_orders(player_results)
            
            return {
                'market_state': market_state,
                'player_results': player_results,
                'total_orders': len(new_orders),
                'total_matched': total_orders_matched,
                'matching_efficiency': total_orders_matched / max(len(new_orders), 1),
                'total_revenue': total_revenue_generated
            }
            
        except Exception as e:
            self.logger.error(f"市场匹配模拟时出错: {e}")
            raise
    
    def _match_orders_for_player(self, player_id: str, strategy: float, 
                               available_orders: List[OrderInfo]) -> Tuple[List[OrderInfo], float, float]:
        """为特定玩家匹配订单"""
        matched_orders = []
        total_revenue = 0.0
        
        # 策略转换为实际定价
        pricing_multiplier = strategy / 30.0  # 假设策略范围是10-50，转换为0.33-1.67倍数
        
        for order in available_orders:
            # 计算该玩家的报价
            player_price = order.base_price * order.surge_multiplier * pricing_multiplier
            
            # 检查乘客是否接受
            max_price = order.calculate_max_price()
            if player_price <= max_price:
                # 乘客接受，检查司机是否接受
                available_driver = self._find_available_driver(order.pickup_location)
                if available_driver:
                    acceptance_prob = available_driver.calculate_acceptance_probability(order, player_price)
                    if np.random.random() < acceptance_prob:
                        matched_orders.append(order)
                        total_revenue += player_price
                        available_driver.available = False  # 司机变为忙碌状态
        
        # 计算市场份额（简化）
        total_market_size = len(available_orders)
        market_share = len(matched_orders) / max(total_market_size, 1)
        
        return matched_orders, total_revenue, market_share
    
    def _find_available_driver(self, pickup_location: Tuple[float, float]) -> Optional[DriverInfo]:
        """找到可用的司机"""
        available = [d for d in self.available_drivers if d.available]
        if not available:
            return None
        
        # 找到距离最近的司机
        min_distance = float('inf')
        closest_driver = None
        
        for driver in available:
            distance = np.sqrt(
                (driver.location[0] - pickup_location[0])**2 + 
                (driver.location[1] - pickup_location[1])**2
            )
            if distance < min_distance:
                min_distance = distance
                closest_driver = driver
        
        return closest_driver
    
    def _update_market_state(self, player_results: Dict[str, Any], 
                           new_orders: List[OrderInfo]) -> MarketState:
        """更新市场状态"""
        # 计算市场指标
        total_demand = len(new_orders)
        total_supply = len([d for d in self.available_drivers if d.available])
        
        # 计算平均价格
        all_revenues = []
        all_orders = []
        for result in player_results.values():
            all_revenues.append(result['revenue'])
            all_orders.extend(result['matched_orders'])
        
        avg_price = sum(all_revenues) / max(len(all_orders), 1)
        
        # 外部因素
        weather_factor = self._get_weather_factor()
        traffic_factor = self._get_traffic_factor()
        
        # 竞争强度
        strategies = [result['strategy'] for result in player_results.values()]
        competition_intensity = np.std(strategies) / max(np.mean(strategies), 1) if strategies else 0
        
        # 市场效率
        total_matched = sum(len(result['matched_orders']) for result in player_results.values())
        market_efficiency = total_matched / max(total_demand, 1)
        
        market_state = MarketState(
            timestamp=self.current_time,
            total_demand=total_demand,
            total_supply=total_supply,
            average_price=avg_price,
            weather_factor=weather_factor,
            traffic_factor=traffic_factor,
            competition_intensity=competition_intensity,
            market_efficiency=market_efficiency
        )
        
        # 记录历史
        self.market_history.append(market_state)
        
        return market_state
    
    def _cleanup_matched_orders(self, player_results: Dict[str, Any]) -> None:
        """清理已匹配的订单"""
        matched_order_ids = set()
        for result in player_results.values():
            for order in result['matched_orders']:
                matched_order_ids.add(order.order_id)
        
        # 移除已匹配的订单
        self.active_orders = [
            order for order in self.active_orders 
            if order.order_id not in matched_order_ids
        ]
        
        # 重置司机状态（简化：假设每轮结束后司机变为可用）
        for driver in self.available_drivers:
            driver.available = True
    
    def advance_time(self, hours: int = 1) -> None:
        """推进时间"""
        self.current_time += timedelta(hours=hours)
    
    def get_market_summary(self) -> Dict[str, Any]:
        """获取市场摘要"""
        if not self.market_history:
            return {
                'total_periods': 0,
                'avg_demand': 0,
                'avg_supply': 0,
                'avg_price': 0,
                'avg_efficiency': 0
            }
        
        demands = [state.total_demand for state in self.market_history]
        supplies = [state.total_supply for state in self.market_history]
        prices = [state.average_price for state in self.market_history]
        efficiencies = [state.market_efficiency for state in self.market_history]
        
        return {
            'total_periods': len(self.market_history),
            'avg_demand': np.mean(demands),
            'avg_supply': np.mean(supplies),
            'avg_price': np.mean(prices),
            'avg_efficiency': np.mean(efficiencies),
            'demand_volatility': np.std(demands),
            'price_volatility': np.std(prices),
            'max_efficiency': np.max(efficiencies),
            'min_efficiency': np.min(efficiencies)
        }
    
    def reset(self) -> None:
        """重置市场状态"""
        self.current_time = datetime.now()
        self.market_history = []
        self.active_orders = []
        self._initialize_drivers()
        self.logger.info("市场模拟器已重置")
    
    def export_market_data(self) -> pd.DataFrame:
        """导出市场数据"""
        try:
            if not self.market_history:
                return pd.DataFrame()
            
            data = []
            for state in self.market_history:
                data.append(state.to_dict())
            
            df = pd.DataFrame(data)
            return df
            
        except Exception as e:
            self.logger.error(f"导出市场数据时出错: {e}")
            return pd.DataFrame() 