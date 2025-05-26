"""
市场环境模拟
模拟订单生成、司机匹配和市场竞争机制
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import logging
import math
import random
from datetime import datetime, timedelta

from config.game_config import GameConfig

logger = logging.getLogger(__name__)


class LocationType(Enum):
    """地理位置类型枚举"""
    HOTSPOT = "hotspot"    # 热点区域
    NORMAL = "normal"      # 普通区域  
    REMOTE = "remote"      # 偏远区域


class TimePeriod(Enum):
    """时间段类型枚举"""
    PEAK = "peak"       # 高峰期
    NORMAL = "normal"   # 平峰期
    LOW = "low"        # 低峰期


@dataclass
class Order:
    """订单数据类"""
    order_id: str
    price: float
    location_type: LocationType
    time_period: TimePeriod
    timestamp: float
    waiting_time_limit: float = 5.0  # 乘客等待时间限制（分钟）
    is_accepted: bool = False
    accepted_by: Optional[str] = None
    
    def __post_init__(self):
        if self.order_id is None:
            self.order_id = f"order_{random.randint(10000, 99999)}"


@dataclass
class MarketState:
    """市场状态数据类"""
    current_time: float  # 当前时间（小时）
    time_period: TimePeriod
    order_rate: float    # 当前订单到达率
    price_mean: float    # 当前价格均值
    price_std: float     # 当前价格标准差
    total_orders: int = 0
    accepted_orders: int = 0
    total_revenue: float = 0.0
    
    # 司机竞争状态
    driver_strategies: Dict[str, float] = field(default_factory=dict)
    market_supply_level: float = 1.0  # 市场供给水平
    competition_intensity: float = 0.0  # 竞争激烈程度


class MarketEnvironment:
    """
    市场环境类
    模拟订单生成和市场竞争机制
    """
    
    def __init__(self, config: GameConfig):
        """
        初始化市场环境
        
        Args:
            config: 博弈配置对象
        """
        self.config = config
        
        # 确保配置中有LOCATION_DISTRIBUTION
        if self.config.LOCATION_DISTRIBUTION is None:
            self.config.LOCATION_DISTRIBUTION = {
                'hotspot': 0.3,  # 热点区域
                'normal': 0.5,   # 普通区域
                'remote': 0.2    # 偏远区域
            }
        
        # 确保配置中有TIME_EFFECTS
        if self.config.TIME_EFFECTS is None:
            self.config.TIME_EFFECTS = {
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
        
        # 确保配置中有LOCATION_EFFECTS
        if self.config.LOCATION_EFFECTS is None:
            self.config.LOCATION_EFFECTS = {
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
        
        # 市场状态
        self.current_time = 0  # 模拟时间
        self.current_orders = []
        self.order_history = []
        self.market_state = {
            'demand_level': 1.0,
            'competition_level': 0.5,
            'avg_price': config.BASE_PRICE_MEAN,
            'order_count': 0,
            'accepted_orders': 0,
            'total_revenue': 0.0,
            'time_period': TimePeriod.NORMAL,
            'location_factor': 1.0
        }
        
        # 市场冲击相关属性
        self.demand_multiplier = 1.0  # 需求乘数
        self.supply_multiplier = 1.0  # 供给乘数
        self.max_price = self.config.MAX_PRICE_THRESHOLD  # 价格上限
        self.price_volatility = 1.0  # 价格波动性
        self.cost_multiplier = 1.0  # 成本乘数
        
        # 滑动窗口分析数据
        self.recent_orders = []
        self.recent_revenues = {'player_a': [], 'player_b': []}
        self.recent_strategies = {'player_a': [], 'player_b': []}
        self.window_size = 10  # 滑动窗口大小
        
        logger.info("市场环境初始化完成")
    
    def reset(self):
        """重置市场环境"""
        self.current_time = 0
        self.current_orders = []
        self.order_history = []
        self.market_state = {
            'demand_level': 1.0,
            'competition_level': 0.5,
            'avg_price': self.config.BASE_PRICE_MEAN,
            'order_count': 0,
            'accepted_orders': 0,
            'total_revenue': 0.0,
            'time_period': TimePeriod.NORMAL,
            'location_factor': 1.0
        }
        
        # 重置冲击相关属性
        self.demand_multiplier = 1.0
        self.supply_multiplier = 1.0
        self.max_price = self.config.MAX_PRICE_THRESHOLD
        self.price_volatility = 1.0
        self.cost_multiplier = 1.0
        
        # 重置滑动窗口数据
        self.recent_orders = []
        self.recent_revenues = {'player_a': [], 'player_b': []}
        self.recent_strategies = {'player_a': [], 'player_b': []}
        
        logger.debug("市场环境已重置")
    
    def get_current_state(self) -> Dict[str, Any]:
        """获取当前市场状态"""
        return self.market_state.copy()
    
    def get_state(self) -> Dict[str, float]:
        """
        获取当前市场状态，用于智能体决策
        
        Returns:
            包含市场状态指标的词典
        """
        # 计算滑动窗口统计
        avg_order_count = len(self.recent_orders) / max(1, self.window_size)
        
        avg_revenue_a = np.mean(self.recent_revenues['player_a']) if self.recent_revenues['player_a'] else 0
        avg_revenue_b = np.mean(self.recent_revenues['player_b']) if self.recent_revenues['player_b'] else 0
        
        strategy_variance_a = np.var(self.recent_strategies['player_a']) if len(self.recent_strategies['player_a']) > 1 else 0
        strategy_variance_b = np.var(self.recent_strategies['player_b']) if len(self.recent_strategies['player_b']) > 1 else 0
        
        # 需求和供给级别
        demand_level = self.market_state['demand_level'] * self.demand_multiplier
        supply_level = self.market_state['location_factor'] * self.supply_multiplier
        
        time_period_value = 0
        if self.market_state['time_period'] == TimePeriod.PEAK:
            time_period_value = 1.0
        elif self.market_state['time_period'] == TimePeriod.NORMAL:
            time_period_value = 0.5
        else:  # LOW
            time_period_value = 0.2
        
        return {
            'demand': demand_level,
            'supply': supply_level,
            'avg_price': min(self.market_state['avg_price'], self.max_price),
            'time_period': time_period_value,
            'competition': self.market_state['competition_level'],
            'order_rate': avg_order_count,
            'avg_revenue_a': avg_revenue_a,
            'avg_revenue_b': avg_revenue_b,
            'strategy_variance_a': strategy_variance_a,
            'strategy_variance_b': strategy_variance_b,
            'avg_distance': 5.0,  # 默认平均距离
            'total_orders': self.market_state['order_count']
        }
    
    def update(self, result_a: Dict, result_b: Dict) -> Dict[str, float]:
        """
        更新市场状态
        
        Args:
            result_a: 玩家A的行动结果
            result_b: 玩家B的行动结果
        
        Returns:
            更新后的市场状态
        """
        # 更新市场时间
        self.update_market_time()
        
        # 更新冲击效果
        self.update_shock_effects()
        
        # 更新滑动窗口数据
        if 'revenue' in result_a:
            self.recent_revenues['player_a'].append(result_a['revenue'])
            if len(self.recent_revenues['player_a']) > self.window_size:
                self.recent_revenues['player_a'].pop(0)
        
        if 'revenue' in result_b:
            self.recent_revenues['player_b'].append(result_b['revenue'])
            if len(self.recent_revenues['player_b']) > self.window_size:
                self.recent_revenues['player_b'].pop(0)
        
        if 'strategy' in result_a:
            self.recent_strategies['player_a'].append(result_a['strategy'])
            if len(self.recent_strategies['player_a']) > self.window_size:
                self.recent_strategies['player_a'].pop(0)
        
        if 'strategy' in result_b:
            self.recent_strategies['player_b'].append(result_b['strategy'])
            if len(self.recent_strategies['player_b']) > self.window_size:
                self.recent_strategies['player_b'].pop(0)
        
        # 更新订单计数
        if 'orders' in result_a and 'orders' in result_b:
            total_orders = result_a['orders'] + result_b['orders']
            self.recent_orders.append(total_orders)
            if len(self.recent_orders) > self.window_size:
                self.recent_orders.pop(0)
            
            self.market_state['order_count'] += total_orders
            self.market_state['accepted_orders'] += total_orders
        
        # 更新总收益
        if 'revenue' in result_a and 'revenue' in result_b:
            self.market_state['total_revenue'] += (result_a['revenue'] + result_b['revenue'])
        
        return self.get_state()
    
    def get_time_period(self, hour: float) -> TimePeriod:
        """根据小时确定时间段类型"""
        hour_int = int(hour) % 24
        
        if hour_int in [7, 8, 17, 18]:
            return TimePeriod.PEAK
        elif hour_int in range(10, 16):
            return TimePeriod.NORMAL
        else:
            return TimePeriod.LOW
    
    def update_market_time(self, time_increment: float = 1.0):
        """
        更新市场时间并调整相应的市场参数
        
        Args:
            time_increment: 时间增量（小时）
        """
        self.current_time += time_increment
        self.market_state['time_period'] = self.get_time_period(self.current_time)
        
        # 根据时间段调整市场参数
        time_effects = self.config.TIME_EFFECTS[self.market_state['time_period'].value]
        self.market_state['demand_level'] = (
            self.config.BASE_ORDER_RATE * time_effects['order_multiplier']
        )
        self.market_state['avg_price'] = (
            self.config.BASE_PRICE_MEAN * time_effects['price_multiplier']
        )
        
        # 添加随机波动
        volatility = self.config.DEMAND_VOLATILITY
        self.market_state['demand_level'] *= (1 + np.random.normal(0, volatility))
        self.market_state['avg_price'] *= (1 + np.random.normal(0, volatility * 0.5))
        
        # 确保参数在合理范围内
        self.market_state['demand_level'] = max(0.1, self.market_state['demand_level'])
        self.market_state['avg_price'] = max(10.0, self.market_state['avg_price'])
        
        # 应用冲击乘数
        self.market_state['demand_level'] *= self.demand_multiplier
        self.market_state['avg_price'] = min(self.market_state['avg_price'], self.max_price)
    
    def apply_competition_effects(self, driver_strategies: Dict[str, float]):
        """
        应用司机策略对市场的竞争效应
        
        Args:
            driver_strategies: 司机策略字典 {司机名: 定价阈值}
        """
        self.market_state['driver_strategies'] = driver_strategies.copy()
        
        # 计算平均定价水平
        avg_pricing = np.mean(list(driver_strategies.values()))
        
        # 计算市场供给调整（定价越高，供给越少）
        supply_adjustment = np.exp(-0.1 * (avg_pricing - self.config.BASE_PRICE_MEAN))
        self.market_state['location_factor'] = max(0.3, min(2.0, supply_adjustment))
        
        # 应用供给乘数
        self.market_state['location_factor'] *= self.supply_multiplier
        
        # 计算竞争激烈程度（策略差异越大，竞争越激烈）
        strategy_variance = np.var(list(driver_strategies.values()))
        self.market_state['competition_level'] = min(1.0, strategy_variance / 100.0)
        
        # 根据竞争情况调整订单分布
        competition_factor = 1.0 + self.market_state['competition_level'] * 0.2
        self.market_state['demand_level'] *= competition_factor
        
        logger.debug(f"竞争效应 - 平均定价: {avg_pricing:.2f}, "
                     f"供给调整: {supply_adjustment:.2f}, "
                     f"竞争强度: {self.market_state['competition_level']:.2f}")
    
    def generate_orders(self, duration_minutes: float = 60.0) -> List[Order]:
        """
        在指定时间段内生成订单
        
        Args:
            duration_minutes: 生成订单的时间段长度（分钟）
        
        Returns:
            生成的订单列表
        """
        orders = []
        
        # 计算期望订单数量
        expected_orders = self.market_state['demand_level'] * (duration_minutes / 60.0)
        
        # 使用泊松分布生成实际订单数量
        actual_order_count = np.random.poisson(expected_orders)
        
        for i in range(actual_order_count):
            order = self._generate_single_order(i, duration_minutes)
            orders.append(order)
        
        # 更新统计信息
        self.market_state['order_count'] += len(orders)
        self.order_history.extend(orders)
        
        logger.debug(f"生成订单 {len(orders)} 个，期望 {expected_orders:.1f} 个")
        
        return orders
    
    def _generate_single_order(self, order_index: int, duration_minutes: float) -> Order:
        """生成单个订单"""
        
        # 随机选择地理位置类型
        location_type = self._sample_location_type()
        
        # 生成订单价格，应用价格波动性
        price_std = self.config.BASE_PRICE_STD * self.price_volatility
        base_price = np.random.normal(
            self.market_state['avg_price'],
            price_std
        )
        
        # 应用地理位置影响
        location_effects = self.config.LOCATION_EFFECTS[location_type.value]
        price = base_price * location_effects['price_premium']
        
        # 应用市场供给影响
        price *= self.market_state['location_factor']
        
        # 确保价格在合理范围内并受到max_price限制
        price = max(self.config.MIN_PRICE_THRESHOLD, 
                   min(self.max_price, price))
        
        # 生成订单时间戳（在duration_minutes内随机分布）
        timestamp = self.current_time + (
            np.random.uniform(0, duration_minutes) / 60.0
        )
        
        # 生成等待时间限制（根据地理位置调整）
        if location_type == LocationType.HOTSPOT:
            waiting_time = np.random.uniform(3.0, 8.0)
        elif location_type == LocationType.REMOTE:
            waiting_time = np.random.uniform(8.0, 15.0)
        else:
            waiting_time = np.random.uniform(5.0, 10.0)
        
        return Order(
            order_id=f"order_{int(timestamp*1000)}_{order_index}",
            price=round(price, 2),
            location_type=location_type,
            time_period=self.market_state['time_period'],
            timestamp=timestamp,
            waiting_time_limit=waiting_time
        )
    
    def _sample_location_type(self) -> LocationType:
        """根据概率分布采样地理位置类型"""
        rand = np.random.random()
        cumulative_prob = 0.0
        
        # 转换字典的键为字符串，以防它们本来就是LocationType
        location_dict = {
            loc if isinstance(loc, str) else loc.value: prob
            for loc, prob in self.config.LOCATION_DISTRIBUTION.items()
        }
        
        for location_str, prob in location_dict.items():
            cumulative_prob += prob
            if rand <= cumulative_prob:
                # 将字符串转换为枚举类型
                if location_str == 'hotspot':
                    return LocationType.HOTSPOT
                elif location_str == 'remote':
                    return LocationType.REMOTE
                elif location_str == 'normal':
                    return LocationType.NORMAL
        
        return LocationType.NORMAL  # 默认返回普通区域
    
    def process_driver_decisions(self, orders: List[Order], 
                                driver_strategies: Dict[str, float]) -> Dict[str, List[Order]]:
        """
        处理司机的接单决策
        
        Args:
            orders: 可用订单列表
            driver_strategies: 司机策略 {司机名: 定价阈值}
        
        Returns:
            司机接受的订单 {司机名: 订单列表}
        """
        driver_orders = {driver: [] for driver in driver_strategies.keys()}
        
        # 为订单分配优先级（模拟订单出现的时间顺序）
        orders_with_priority = sorted(orders, key=lambda x: x.timestamp)
        
        for order in orders_with_priority:
            # 找到愿意接受此订单的司机
            willing_drivers = []
            
            for driver, threshold in driver_strategies.items():
                if order.price >= threshold:
                    # 添加一些随机性来模拟其他因素（距离、疲劳等）
                    acceptance_probability = self._calculate_acceptance_probability(
                        order, threshold, driver
                    )
                    
                    if np.random.random() < acceptance_probability:
                        willing_drivers.append(driver)
            
            # 如果有司机愿意接单，随机选择一个（模拟平台分配或司机抢单）
            if willing_drivers:
                selected_driver = np.random.choice(willing_drivers)
                order.is_accepted = True
                order.accepted_by = selected_driver
                driver_orders[selected_driver].append(order)
                
                # 更新市场统计
                self.market_state['accepted_orders'] += 1
                self.market_state['total_revenue'] += order.price
        
        logger.debug(f"订单分配完成 - 总订单: {len(orders)}, "
                     f"已接受: {self.market_state['accepted_orders']}")
        
        return driver_orders
    
    def _calculate_acceptance_probability(self, order: Order, threshold: float, driver: str) -> float:
        """
        计算司机接受订单的概率
        
        考虑因素：
        - 价格相对于阈值的比例
        - 地理位置类型
        - 市场竞争程度
        - 随机因素
        """
        base_probability = 0.8  # 基础接受概率
        
        # 价格因素：价格越高于阈值，接受概率越大
        price_ratio = order.price / threshold
        price_factor = min(1.2, price_ratio ** 0.5)
        
        # 地理位置因素
        location_factor = {
            LocationType.HOTSPOT: 1.1,  # 热点区域更愿意接单
            LocationType.NORMAL: 1.0,
            LocationType.REMOTE: 0.9    # 偏远区域稍微降低意愿
        }[order.location_type]
        
        # 竞争因素：竞争越激烈，越不愿意错过订单
        competition_factor = 1.0 + self.market_state['competition_level'] * 0.2
        
        # 时间因素：高峰期更愿意接单
        time_factor = {
            TimePeriod.PEAK: 1.1,
            TimePeriod.NORMAL: 1.0,
            TimePeriod.LOW: 0.95
        }[order.time_period]
        
        # 随机因素（模拟司机的个人状态变化）
        random_factor = np.random.uniform(0.9, 1.1)
        
        final_probability = (base_probability * price_factor * location_factor * 
                           competition_factor * time_factor * random_factor)
        
        return min(1.0, final_probability)
    
    def calculate_driver_revenues(self, driver_orders: Dict[str, List[Order]], 
                                 strategies: Dict[str, float]) -> Dict[str, float]:
        """
        计算司机的收益
        
        Args:
            driver_orders: 司机接受的订单
            strategies: 司机策略
        
        Returns:
            司机收益字典
        """
        revenues = {}
        
        for driver, orders in driver_orders.items():
            if not orders:
                revenues[driver] = 0.0
                continue
            
            # 基础收入：订单价格总和
            gross_revenue = sum(order.price for order in orders)
            
            # 等待时间成本
            avg_waiting_time = self._estimate_waiting_time(driver, orders, strategies)
            waiting_cost = avg_waiting_time * self.config.WAITING_TIME_PENALTY
            
            # 运营成本（受cost_multiplier影响）
            operation_cost = len(orders) * self.config.OPERATION_COST_PER_ORDER * self.cost_multiplier
            
            # 策略稳定性奖励（减少频繁变化的成本）
            stability_bonus = self._calculate_stability_bonus(driver)
            
            # 竞争优势奖励
            competition_bonus = self._calculate_competition_bonus(driver, revenues)
            
            # 最终收益
            net_revenue = (gross_revenue - waiting_cost - operation_cost + 
                          stability_bonus + competition_bonus)
            
            revenues[driver] = max(0.0, net_revenue)  # 确保收益非负
        
        logger.debug(f"司机收益计算完成: {revenues}")
        return revenues
    
    def _estimate_waiting_time(self, driver: str, orders: List[Order], 
                              strategies: Dict[str, float]) -> float:
        """
        估计司机的平均等待时间
        
        基于策略阈值和市场状况估算
        """
        if not orders:
            # 如果没有接到订单，等待时间较长
            threshold = strategies.get(driver, self.config.BASE_PRICE_MEAN)
            base_waiting = 30.0  # 基础等待时间（分钟）
            
            # 阈值越高，等待时间越长
            threshold_factor = threshold / self.config.BASE_PRICE_MEAN
            adjusted_waiting = base_waiting * threshold_factor ** 0.8
            
            return min(120.0, adjusted_waiting)  # 最大等待时间2小时
        
        # 有订单时，根据订单间隔估算等待时间
        order_count = len(orders)
        time_span = 60.0  # 假设1小时内的订单
        
        if order_count > 0:
            avg_interval = time_span / order_count
            return max(5.0, avg_interval * 0.8)  # 最小等待5分钟
        
        return 15.0  # 默认等待时间
    
    def _calculate_stability_bonus(self, driver: str) -> float:
        """计算策略稳定性奖励"""
        # 这里需要访问历史策略数据，简化处理
        # 在实际实现中应该从game_state获取历史数据
        return 0.0  # 暂时返回0，后续可以结合历史数据计算
    
    def _calculate_competition_bonus(self, driver: str, current_revenues: Dict[str, float]) -> float:
        """计算竞争优势奖励"""
        if len(current_revenues) < 2:
            return 0.0
        
        # 如果当前司机收益最高，给予奖励
        driver_revenue = current_revenues.get(driver, 0.0)
        max_revenue = max(current_revenues.values())
        
        if driver_revenue == max_revenue and driver_revenue > 0:
            return driver_revenue * self.config.COMPETITION_BONUS
        
        return 0.0
    
    def apply_market_shock(self, shock_type: str, magnitude: float, duration: int):
        """
        应用市场冲击
        
        Args:
            shock_type: 冲击类型 ('demand_surge', 'policy_change', 'competition_entry')
            magnitude: 冲击强度
            duration: 冲击持续时间（轮次）
        """
        logger.info(f"应用市场冲击: {shock_type}, 强度: {magnitude}, 持续: {duration}轮")
        
        if shock_type == 'demand_surge':
            # 需求激增
            self.market_state['demand_level'] *= magnitude
            self.market_state['avg_price'] *= (1 + (magnitude - 1) * 0.5)
            
        elif shock_type == 'policy_change':
            # 政策变化（如限制定价）
            if magnitude < 1.0:  # 限制性政策
                max_allowed_price = self.config.MAX_PRICE_THRESHOLD * magnitude
                self.market_state['avg_price'] = min(self.market_state['avg_price'], 
                                                  max_allowed_price)
            
        elif shock_type == 'competition_entry':
            # 新竞争者进入
            self.market_state['demand_level'] *= (1 / magnitude)  # 订单被分流
            self.market_state['competition_level'] += 0.3
        
        # 记录冲击信息（用于后续恢复）
        if not hasattr(self, 'active_shocks'):
            self.active_shocks = []
        
        self.active_shocks.append({
            'type': shock_type,
            'magnitude': magnitude,
            'remaining_duration': duration,
            'original_state': {
                'demand_level': self.market_state['demand_level'] / magnitude,
                'avg_price': self.market_state['avg_price'],
                'competition_level': self.market_state['competition_level']
            }
        })
    
    def update_shock_effects(self):
        """更新冲击效果（每轮调用）"""
        if not hasattr(self, 'active_shocks'):
            return
        
        expired_shocks = []
        
        for i, shock in enumerate(self.active_shocks):
            shock['remaining_duration'] -= 1
            
            if shock['remaining_duration'] <= 0:
                expired_shocks.append(i)
                # 恢复原始状态（简化处理）
                logger.info(f"市场冲击 {shock['type']} 效果结束")
        
        # 移除过期的冲击
        for i in reversed(expired_shocks):
            self.active_shocks.pop(i)
    
    def get_market_summary(self) -> Dict[str, Any]:
        """获取市场状态摘要"""
        acceptance_rate = (
            self.market_state['accepted_orders'] / max(1, self.market_state['order_count'])
        )
        
        avg_order_price = 0.0
        if self.order_history:
            avg_order_price = np.mean([order.price for order in self.order_history[-100:]])
        
        return {
            'current_time': self.current_time,
            'time_period': self.market_state['time_period'].value,
            'demand_level': self.market_state['demand_level'],
            'avg_price': self.market_state['avg_price'],
            'price_std': self.config.BASE_PRICE_STD,
            'order_count': self.market_state['order_count'],
            'acceptance_rate': acceptance_rate,
            'avg_order_price': avg_order_price,
            'location_factor': self.market_state['location_factor'],
            'competition_level': self.market_state['competition_level'],
            'total_revenue': self.market_state['total_revenue']
        }
    
    def save_market_data(self, filepath: str):
        """保存市场数据"""
        market_data = {
            'current_state': self.market_state,
            'order_history': self.order_history,
            'config_summary': self.config.to_dict()
        }
        
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(market_data, f)
        
        logger.info(f"市场数据已保存至: {filepath}")


if __name__ == "__main__":
    # 测试市场环境
    from config.game_config import GameConfig
    
    print("测试市场环境模拟器...")
    config = GameConfig()
    market = MarketEnvironment(config)
    
    # 测试订单生成
    print("\n=== 测试订单生成 ===")
    orders = market.generate_orders(60.0)  # 生成1小时的订单
    print(f"生成订单数量: {len(orders)}")
    if orders:
        print(f"第一个订单: ID={orders[0].order_id}, 价格={orders[0].price:.2f}, "
              f"位置={orders[0].location_type.value}")
    
    # 测试司机决策处理
    print("\n=== 测试司机决策处理 ===")
    driver_strategies = {'司机A': 25.0, '司机B': 30.0}
    market.apply_competition_effects(driver_strategies)
    driver_orders = market.process_driver_decisions(orders, driver_strategies)
    
    for driver, accepted_orders in driver_orders.items():
        print(f"{driver} 接受订单数: {len(accepted_orders)}")
    
    # 测试收益计算
    print("\n=== 测试收益计算 ===")
    revenues = market.calculate_driver_revenues(driver_orders, driver_strategies)
    for driver, revenue in revenues.items():
        print(f"{driver} 收益: {revenue:.2f}")
    
    # 测试市场摘要
    print("\n=== 市场状态摘要 ===")
    summary = market.get_market_summary()
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    print("\n市场环境模拟器测试完成 ✓")