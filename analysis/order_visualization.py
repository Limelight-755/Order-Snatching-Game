"""
订单可视化工具
提供网约车订单的地理热力图和时段分布图表功能
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import logging
import pandas as pd
import os
import matplotlib.font_manager as fm
from matplotlib import rcParams
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
import random
from collections import defaultdict
from core.market_environment import Order, LocationType, TimePeriod

# 配置日志
logger = logging.getLogger(__name__)

# 设置中文字体路径
chinese_font_path = r"C:\Windows\Fonts\STZHONGS.TTF"

class OrderVisualization:
    """
    订单可视化工具类
    提供网约车订单的地理热力图和时段分布图表功能
    """
    
    def __init__(self):
        """初始化订单可视化工具"""
        self.setup_chinese_font()
        
        # 地理区域映射坐标 (x, y, 宽度, 高度)
        # 重新设计区域布局，使所有区域自然接壤
        self.area_coords = {
            # 热点区域在地图中心
            LocationType.HOTSPOT: (0.25, 0.25, 0.5, 0.5),  
            
            # 普通区域环绕四周，与热点区域接壤
            LocationType.NORMAL: [
                (0.25, 0.0, 0.5, 0.25),   # 上方区块
                (0.75, 0.25, 0.25, 0.5),  # 右方区块
                (0.25, 0.75, 0.5, 0.25),  # 下方区块
                (0.0, 0.25, 0.25, 0.5)    # 左方区块
            ],  
            
            # 偏远区域在四角，与普通区域接壤
            LocationType.REMOTE: [
                (0.0, 0.0, 0.25, 0.25),   # 左上
                (0.75, 0.0, 0.25, 0.25),  # 右上
                (0.0, 0.75, 0.25, 0.25),  # 左下
                (0.75, 0.75, 0.25, 0.25)  # 右下
            ]  
        }
        
        # 时间段颜色映射
        self.time_period_colors = {
            TimePeriod.PEAK: '#FF7F7F',  # 高峰期 - 红色
            TimePeriod.NORMAL: '#7FBF7F',  # 平峰期 - 绿色
            TimePeriod.LOW: '#7F7FFF'   # 低峰期 - 蓝色
        }
        
        # 地理位置名称映射（用于图表显示）
        self.location_names = {
            LocationType.HOTSPOT: "热点区域",
            LocationType.NORMAL: "普通区域",
            LocationType.REMOTE: "偏远区域"
        }
        
        # 时间段名称映射（用于图表显示）
        self.time_period_names = {
            TimePeriod.PEAK: "高峰期",
            TimePeriod.NORMAL: "平峰期",
            TimePeriod.LOW: "低峰期"
        }
        
        # 时间段对应的小时（根据项目设置）
        self.time_period_hours = {
            TimePeriod.PEAK: [7, 8, 9, 17, 18, 19],   # 高峰期小时
            TimePeriod.NORMAL: [10, 11, 12, 13, 14, 15, 16],  # 平峰期小时
            TimePeriod.LOW: [0, 1, 2, 3, 4, 5, 6, 20, 21, 22, 23]  # 低峰期小时
        }
        
        # 区域背景色 - 调整为更鲜明且接近真实地图的颜色
        self.area_bg_colors = {
            LocationType.HOTSPOT: '#FFCCCC',  # 热点区域背景 - 浅红色
            LocationType.NORMAL: '#CCFFCC',   # 普通区域背景 - 浅绿色
            LocationType.REMOTE: '#CCCCFF'    # 偏远区域背景 - 浅蓝色
        }
        
        logger.info("订单可视化工具初始化完成")
    
    def setup_chinese_font(self):
        """设置中文字体"""
        if os.path.exists(chinese_font_path):
            self.font_prop = fm.FontProperties(fname=chinese_font_path)
            plt.rcParams['font.family'] = self.font_prop.get_name()
            plt.rcParams['font.sans-serif'] = [self.font_prop.get_name()] + plt.rcParams.get('font.sans-serif', [])
            plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
            logger.info(f"已成功配置中文字体: {chinese_font_path}")
        else:
            # 回退到常见中文字体名称
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi', 'FangSong', 'Arial Unicode MS']
            plt.rcParams['axes.unicode_minus'] = False
            self.font_prop = None
            logger.warning(f"找不到指定的字体文件: {chinese_font_path}，使用备用字体")
    
    def generate_daily_order_heatmap(self, orders: List[Order], save_path: Optional[str] = None, 
                                    grid_size: int = 50, price_range: Tuple[float, float] = None):
        """
        生成一天的订单生成地理热力图，价格越高颜色越深
        
        Args:
            orders: 一天内的订单列表
            save_path: 图表保存路径
            grid_size: 网格大小（决定热力图精细度）
            price_range: 价格范围，如果不提供则根据数据自动计算
        """
        if not orders:
            logger.warning("没有订单数据，无法生成热力图")
            return
            
        # 按小时排序订单
        sorted_orders = sorted(orders, key=lambda o: o.timestamp % 24)
        
        # 创建网格和价格数据
        grid_x = np.linspace(0, 1, grid_size)
        grid_y = np.linspace(0, 1, grid_size)
        grid_data = np.zeros((grid_size, grid_size))
        count_data = np.zeros((grid_size, grid_size))
        
        # 为每个订单生成地理坐标并填充价格数据
        for order in sorted_orders:
            # 根据地理位置类型分配坐标
            x, y = self._generate_order_coordinates(order.location_type)
            
            # 转换到网格索引
            xi = int(x * (grid_size-1))
            yi = int(y * (grid_size-1))
            
            # 累加价格和计数
            grid_data[yi, xi] += order.price
            count_data[yi, xi] += 1
        
        # 计算平均价格
        with np.errstate(divide='ignore', invalid='ignore'):
            avg_price = np.divide(grid_data, count_data)
            avg_price = np.nan_to_num(avg_price)  # 替换 NaN 和 inf
        
        # 如果没有提供价格范围，则根据数据计算
        if price_range is None:
            min_price = max(1, np.min(avg_price[avg_price > 0]))
            max_price = np.max(avg_price)
            price_range = (min_price, max_price)
        
        # 创建图形，提高DPI设置增加清晰度
        fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
        
        # 绘制地理区域底图
        self._draw_area_boundaries(ax)
        
        # 创建热力图
        im = ax.imshow(avg_price, cmap='YlOrRd', origin='lower', 
                      extent=[0, 1, 0, 1], alpha=0.7,
                      vmin=price_range[0], vmax=price_range[1])
        
        # 添加彩色条
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label('平均订单价格（元）', fontproperties=self.font_prop)
        
        # 添加标题和标签
        hour_range = f"{int(sorted_orders[0].timestamp % 24):02d}:00 - {int(sorted_orders[-1].timestamp % 24):02d}:59"
        ax.set_title(f"一天订单地理分布热力图 ({hour_range})", fontproperties=self.font_prop)
        ax.set_xlabel('经度', fontproperties=self.font_prop)
        ax.set_ylabel('纬度', fontproperties=self.font_prop)
        
        # 隐藏坐标轴刻度
        ax.set_xticks([])
        ax.set_yticks([])
        
        # 添加图例
        legend_elements = [
            Rectangle((0, 0), 1, 1, color='#FF8080', alpha=0.7, label=self.location_names[LocationType.HOTSPOT]),
            Rectangle((0, 0), 1, 1, color='#80CC80', alpha=0.7, label=self.location_names[LocationType.NORMAL]),
            Rectangle((0, 0), 1, 1, color='#8080FF', alpha=0.7, label=self.location_names[LocationType.REMOTE])
        ]
        ax.legend(handles=legend_elements, loc='upper left', prop=self.font_prop)
        
        # 添加统计信息
        order_counts = {}
        for loc_type in [LocationType.HOTSPOT, LocationType.NORMAL, LocationType.REMOTE]:
            order_counts[self.location_names[loc_type]] = sum(1 for o in orders if o.location_type == loc_type)
        
        info_text = "\n".join([
            f"总订单数: {len(orders)}",
            f"热点区域: {order_counts[self.location_names[LocationType.HOTSPOT]]}",
            f"普通区域: {order_counts[self.location_names[LocationType.NORMAL]]}",
            f"偏远区域: {order_counts[self.location_names[LocationType.REMOTE]]}"
        ])
        
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontproperties=self.font_prop,
               va='top', ha='left', bbox=dict(facecolor='white', alpha=0.7))
        
        # 保存图表，使用高DPI设置
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"地理热力图已保存至: {save_path}")
        
        return fig, ax
    
    def generate_time_location_bar_chart(self, orders: List[Order], save_path: Optional[str] = None):
        """
        生成一天中不同时段不同地理位置的订单量柱状图
        
        Args:
            orders: 一天内的订单列表
            save_path: 图表保存路径
        """
        if not orders:
            logger.warning("没有订单数据，无法生成柱状图")
            return
            
        # 按时段和地理位置统计订单量
        order_counts = {}
        for period in [TimePeriod.PEAK, TimePeriod.NORMAL, TimePeriod.LOW]:
            period_name = self.time_period_names[period]
            order_counts[period_name] = {}
            
            for loc_type in [LocationType.HOTSPOT, LocationType.NORMAL, LocationType.REMOTE]:
                loc_name = self.location_names[loc_type]
                order_counts[period_name][loc_name] = sum(
                    1 for o in orders if o.time_period == period and o.location_type == loc_type
                )
        
        # 准备数据
        periods = list(order_counts.keys())
        locations = [self.location_names[loc] for loc in [LocationType.HOTSPOT, LocationType.NORMAL, LocationType.REMOTE]]
        
        # 创建图表，提高DPI设置增加清晰度
        fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
        
        # 设置条形宽度和位置
        width = 0.25
        x = np.arange(len(periods))
        
        # 绘制分组条形图
        for i, location in enumerate(locations):
            counts = [order_counts[period][location] for period in periods]
            offset = width * (i - 1)
            rects = ax.bar(x + offset, counts, width, label=location, 
                         color=self.time_period_colors[list(self.time_period_names.keys())[i]])
            
            # 添加数据标签
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{int(height)}',
                          xy=(rect.get_x() + rect.get_width() / 2, height),
                          xytext=(0, 3),  # 3点垂直偏移
                          textcoords="offset points",
                          ha='center', va='bottom',
                          fontproperties=self.font_prop)
        
        # 添加标题和标签
        ax.set_title('一天中不同时段各地理位置订单量分布', fontproperties=self.font_prop)
        ax.set_xlabel('时段', fontproperties=self.font_prop)
        ax.set_ylabel('订单数量', fontproperties=self.font_prop)
        ax.set_xticks(x)
        ax.set_xticklabels(periods, fontproperties=self.font_prop)
        ax.legend(prop=self.font_prop)
        
        # 添加网格
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 保存图表，使用高DPI设置
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"时段-地点柱状图已保存至: {save_path}")
        
        return fig, ax
    
    def generate_24h_order_distribution(self, orders: List[Order], save_path: Optional[str] = None):
        """
        生成24小时订单分布图，按地理位置类型分组
        
        Args:
            orders: 一天内的订单列表
            save_path: 图表保存路径
        """
        if not orders:
            logger.warning("没有订单数据，无法生成24小时分布图")
            return
        
        # 创建24小时桶
        hours = list(range(24))
        order_counts = {
            self.location_names[LocationType.HOTSPOT]: [0] * 24,
            self.location_names[LocationType.NORMAL]: [0] * 24,
            self.location_names[LocationType.REMOTE]: [0] * 24
        }
        
        # 统计每小时各地理位置的订单量
        for order in orders:
            hour = int(order.timestamp % 24)
            location_name = self.location_names[order.location_type]
            order_counts[location_name][hour] += 1
        
        # 创建图表，提高DPI设置增加清晰度
        fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
        
        # 获取位置名称列表
        locations = list(order_counts.keys())
        
        # 创建堆叠面积图
        for i, location in enumerate(locations):
            color = plt.cm.tab10(i)
            ax.fill_between(hours, order_counts[location], alpha=0.7, color=color, label=location)
        
        # 添加标题和标签
        ax.set_title('24小时订单分布 (按地理位置分组)', fontproperties=self.font_prop)
        ax.set_xlabel('小时', fontproperties=self.font_prop)
        ax.set_ylabel('订单数量', fontproperties=self.font_prop)
        
        # 设置x轴范围和刻度
        ax.set_xlim(0, 23)
        ax.set_xticks(range(0, 24, 2))
        ax.set_xticklabels([f'{h:02d}:00' for h in range(0, 24, 2)])
        
        # 添加网格和图例
        ax.grid(alpha=0.3)
        ax.legend(prop=self.font_prop)
        
        # 标记高峰期、平峰期和低峰期
        peak_hours = [7, 8, 17, 18]
        normal_hours = list(range(10, 16))
        
        # 添加时段背景色
        for hour in range(24):
            if hour in peak_hours:
                ax.axvspan(hour - 0.5, hour + 0.5, alpha=0.2, color='red')
            elif hour in normal_hours:
                ax.axvspan(hour - 0.5, hour + 0.5, alpha=0.2, color='green')
            else:
                ax.axvspan(hour - 0.5, hour + 0.5, alpha=0.2, color='blue')
        
        # 添加时段标注
        ax.text(7.5, ax.get_ylim()[1] * 0.95, '早高峰', fontproperties=self.font_prop, 
               ha='center', color='darkred', bbox=dict(facecolor='white', alpha=0.7))
        ax.text(13, ax.get_ylim()[1] * 0.95, '平峰期', fontproperties=self.font_prop, 
               ha='center', color='darkgreen', bbox=dict(facecolor='white', alpha=0.7))
        ax.text(17.5, ax.get_ylim()[1] * 0.95, '晚高峰', fontproperties=self.font_prop, 
               ha='center', color='darkred', bbox=dict(facecolor='white', alpha=0.7))
        ax.text(22, ax.get_ylim()[1] * 0.95, '低峰期', fontproperties=self.font_prop, 
               ha='center', color='darkblue', bbox=dict(facecolor='white', alpha=0.7))
        
        # 保存图表，使用高DPI设置
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"24小时订单分布图已保存至: {save_path}")
        
        return fig, ax
    
    def _generate_order_coordinates(self, location_type: LocationType) -> Tuple[float, float]:
        """
        根据地理位置类型生成一个随机坐标
        
        Args:
            location_type: 地理位置类型
            
        Returns:
            (x, y) 坐标元组，范围0-1
        """
        coords = self.area_coords.get(location_type)
        if not coords:
            # 默认随机坐标
            return random.random(), random.random()
        
        if isinstance(coords, list):
            # 随机选择一个区域
            area = random.choice(coords)
            x0, y0, w, h = area
            return x0 + random.random() * w, y0 + random.random() * h
        else:
            # 单个区域
            x0, y0, w, h = coords
            return x0 + random.random() * w, y0 + random.random() * h
    
    def _draw_area_boundaries(self, ax):
        """
        在图表上绘制地理区域边界
        
        Args:
            ax: matplotlib坐标轴对象
        """
        # 绘制热点区域
        hotspot = self.area_coords[LocationType.HOTSPOT]
        rect = Rectangle((hotspot[0], hotspot[1]), hotspot[2], hotspot[3], 
                        fill=False, edgecolor='red', linestyle='-', linewidth=2)
        ax.add_patch(rect)
        ax.text(hotspot[0] + hotspot[2]/2, hotspot[1] + hotspot[3]/2, 
               self.location_names[LocationType.HOTSPOT], 
               fontproperties=self.font_prop,
               color='darkred', ha='center', va='center',
               bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
        
        # 绘制普通区域
        for i, area in enumerate(self.area_coords[LocationType.NORMAL]):
            rect = Rectangle((area[0], area[1]), area[2], area[3], 
                          fill=False, edgecolor='green', linestyle='-', linewidth=1.5)
            ax.add_patch(rect)
        
        # 绘制偏远区域
        for i, area in enumerate(self.area_coords[LocationType.REMOTE]):
            rect = Rectangle((area[0], area[1]), area[2], area[3], 
                          fill=False, edgecolor='blue', linestyle='-', linewidth=1.5)
            ax.add_patch(rect)
    
    def generate_day_simulation_orders(self, count: int = 1000) -> List[Order]:
        """
        生成一天的模拟订单数据用于可视化测试
        
        Args:
            count: 要生成的订单总数
            
        Returns:
            订单列表
        """
        from datetime import datetime
        import uuid
        
        orders = []
        
        # 时间段和区域分布
        time_period_probs = {
            TimePeriod.PEAK: 0.5,    # 高峰期50%的订单
            TimePeriod.NORMAL: 0.3,  # 平峰期30%的订单
            TimePeriod.LOW: 0.2      # 低峰期20%的订单
        }
        
        location_probs = {
            LocationType.HOTSPOT: 0.4,  # 热点区域40%的订单
            LocationType.NORMAL: 0.4,   # 普通区域40%的订单
            LocationType.REMOTE: 0.2    # 偏远区域20%的订单
        }
        
        # 价格范围
        price_ranges = {
            LocationType.HOTSPOT: (25, 40),
            LocationType.NORMAL: (20, 35),
            LocationType.REMOTE: (30, 50)
        }
        
        # 生成每个时间段对应的小时
        time_hours = {
            TimePeriod.PEAK: [7, 8, 17, 18],
            TimePeriod.NORMAL: list(range(10, 16)),
            TimePeriod.LOW: list(range(0, 7)) + list(range(19, 24))
        }
        
        # 生成订单
        for _ in range(count):
            # 随机选择时间段
            time_period = random.choices(
                list(time_period_probs.keys()), 
                weights=list(time_period_probs.values())
            )[0]
            
            # 根据时间段随机选择小时
            hour = random.choice(time_hours[time_period])
            minute = random.randint(0, 59)
            timestamp = hour + minute / 60.0
            
            # 随机选择地理位置
            location = random.choices(
                list(location_probs.keys()),
                weights=list(location_probs.values())
            )[0]
            
            # 根据地理位置生成价格
            price_min, price_max = price_ranges[location]
            price = round(random.uniform(price_min, price_max), 2)
            
            # 创建订单
            order = Order(
                order_id=f"sim_{uuid.uuid4().hex[:8]}",
                price=price,
                location_type=location,
                time_period=time_period,
                timestamp=timestamp,
                waiting_time_limit=random.uniform(3, 15)
            )
            
            orders.append(order)
        
        return orders

    def generate_time_periods_scatter_plot(self, orders: List[Order], save_path: Optional[str] = None):
        """
        生成分时段订单散点分布图，为高峰期/平峰期/低峰期分别绘制子图，每个时段只取3小时数据
        
        Args:
            orders: 一天内的订单列表
            save_path: 图表保存路径
        """
        if not orders:
            logger.warning("没有订单数据，无法生成散点图")
            return
            
        # 创建一个3行1列的图形，提高DPI设置以增加清晰度
        fig, axes = plt.subplots(3, 1, figsize=(10, 15), dpi=300)
        
        # 按照时间段筛选订单
        period_orders = {}
        for period in [TimePeriod.PEAK, TimePeriod.NORMAL, TimePeriod.LOW]:
            period_orders[period] = [o for o in orders if o.time_period == period]
        
        # 获取所有订单价格的范围，用于统一的颜色映射
        all_prices = [order.price for order in orders]
        min_price = min(all_prices) if all_prices else 15
        max_price = max(all_prices) if all_prices else 50
        
        # 创建一个Normalize对象用于价格颜色映射
        norm = plt.Normalize(min_price, max_price)
        
        # 生成每个时间段的散点图
        for idx, period in enumerate([TimePeriod.PEAK, TimePeriod.NORMAL, TimePeriod.LOW]):
            ax = axes[idx]
            selected_orders = period_orders[period]
            
            # 选择该时段的3个小时（如果数据足够）
            hours_to_use = self.time_period_hours[period][:3]  # 取前3个小时
            if len(hours_to_use) < 3 and self.time_period_hours[period]:
                # 如果不足3小时，就用所有可用小时
                hours_to_use = self.time_period_hours[period]
            
            hour_str = ", ".join([f"{h:02d}:00" for h in hours_to_use])
            
            # 过滤出这3个小时的订单
            filtered_orders = [
                o for o in selected_orders 
                if int(o.timestamp % 24) in hours_to_use
            ]
            
            # 如果过滤后数据太少，就使用该时段的所有数据
            if len(filtered_orders) < 50:
                filtered_orders = selected_orders
                hour_str = "所有" + self.time_period_names[period] + "小时"
            
            # 绘制区域背景
            self._draw_area_background(ax)
            
            # 准备散点数据
            x_coords = []
            y_coords = []
            colors = []
            sizes = []
            prices = []
            
            for order in filtered_orders:
                # 生成坐标
                x, y = self._generate_order_coordinates(order.location_type)
                x_coords.append(x)
                y_coords.append(y)
                prices.append(order.price)
                
                # 点大小也可以根据价格设置
                sizes.append(20 + (order.price - min_price) / (max_price - min_price) * 60)
            
            # 绘制散点图
            scatter = ax.scatter(x_coords, y_coords, c=prices, s=sizes, alpha=0.8, 
                             cmap='YlOrRd', norm=norm, edgecolor='black', linewidth=0.5)
            
            # 绘制区域边界
            self._draw_area_boundaries(ax)
            
            # 添加标题和标签
            ax.set_title(f"{self.time_period_names[period]}订单分布 ({hour_str})", fontproperties=self.font_prop, fontsize=14)
            ax.set_xlabel('经度', fontproperties=self.font_prop)
            ax.set_ylabel('纬度', fontproperties=self.font_prop)
            
            # 隐藏坐标轴刻度
            ax.set_xticks([])
            ax.set_yticks([])
            
            # 添加订单统计信息
            order_counts = {}
            for loc_type in [LocationType.HOTSPOT, LocationType.NORMAL, LocationType.REMOTE]:
                order_counts[self.location_names[loc_type]] = sum(1 for o in filtered_orders if o.location_type == loc_type)
            
            info_text = "\n".join([
                f"总订单数: {len(filtered_orders)}",
                f"热点区域: {order_counts[self.location_names[LocationType.HOTSPOT]]}",
                f"普通区域: {order_counts[self.location_names[LocationType.NORMAL]]}",
                f"偏远区域: {order_counts[self.location_names[LocationType.REMOTE]]}"
            ])
            
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontproperties=self.font_prop,
                   va='top', ha='left', bbox=dict(facecolor='white', alpha=0.7))
            
            # 添加区域类型图例
            area_patches = [
                Rectangle((0, 0), 1, 1, facecolor=self.area_bg_colors[LocationType.HOTSPOT], 
                         edgecolor='red', alpha=0.4, label=self.location_names[LocationType.HOTSPOT]),
                Rectangle((0, 0), 1, 1, facecolor=self.area_bg_colors[LocationType.NORMAL], 
                         edgecolor='green', alpha=0.4, label=self.location_names[LocationType.NORMAL]),
                Rectangle((0, 0), 1, 1, facecolor=self.area_bg_colors[LocationType.REMOTE], 
                         edgecolor='blue', alpha=0.4, label=self.location_names[LocationType.REMOTE])
            ]
            
            # 添加价格说明到图例
            price_dot = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF7700', 
                                 markersize=10, label='价格（颜色越深价格越高）')
            
            # 合并所有图例元素
            legend_elements = area_patches + [price_dot]
            ax.legend(handles=legend_elements, loc='upper right', 
                    prop=self.font_prop, framealpha=0.7, title="区域类型")
        
        # 添加颜色条，表示价格
        cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = plt.colorbar(scatter, cax=cax)
        cbar.set_label('订单价格 (元)', fontproperties=self.font_prop)
        
        plt.tight_layout(rect=[0, 0, 0.9, 1])
        
        # 保存图表，使用高DPI提高清晰度
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"时段订单散点分布图已保存至: {save_path}")
        
        return fig, axes
    
    def _draw_area_background(self, ax):
        """
        绘制区域背景
        
        Args:
            ax: matplotlib坐标轴对象
        """
        # 绘制热点区域背景
        hotspot = self.area_coords[LocationType.HOTSPOT]
        rect = Rectangle((hotspot[0], hotspot[1]), hotspot[2], hotspot[3], 
                        fill=True, facecolor=self.area_bg_colors[LocationType.HOTSPOT], 
                        edgecolor='red', alpha=0.4, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(hotspot[0] + hotspot[2]/2, hotspot[1] + hotspot[3]/2, 
               self.location_names[LocationType.HOTSPOT], 
               fontproperties=self.font_prop,
               color='darkred', ha='center', va='center',
               bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
        
        # 绘制普通区域背景
        for i, area in enumerate(self.area_coords[LocationType.NORMAL]):
            rect = Rectangle((area[0], area[1]), area[2], area[3], 
                          fill=True, facecolor=self.area_bg_colors[LocationType.NORMAL], 
                          edgecolor='green', alpha=0.4, linewidth=1)
            ax.add_patch(rect)
            # 只为其中一个普通区域添加文本标签
            if i == 0:
                ax.text(area[0] + area[2]/2, area[1] + area[3]/2, 
                      self.location_names[LocationType.NORMAL], 
                      fontproperties=self.font_prop,
                      color='darkgreen', ha='center', va='center',
                      bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
        
        # 绘制偏远区域背景
        for i, area in enumerate(self.area_coords[LocationType.REMOTE]):
            rect = Rectangle((area[0], area[1]), area[2], area[3], 
                          fill=True, facecolor=self.area_bg_colors[LocationType.REMOTE], 
                          edgecolor='blue', alpha=0.4, linewidth=1)
            ax.add_patch(rect)
            # 只为其中一个偏远区域添加文本标签
            if i == 0:
                ax.text(area[0] + area[2]/2, area[1] + area[3]/2, 
                      self.location_names[LocationType.REMOTE], 
                      fontproperties=self.font_prop,
                      color='darkblue', ha='center', va='center',
                      bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))


def demo():
    """
    生成示例图表展示功能
    """
    import matplotlib.pyplot as plt
    from pathlib import Path
    import os
    
    # 确保结果目录存在
    os.makedirs("results/plots", exist_ok=True)
    
    # 创建可视化工具实例
    viz = OrderVisualization()
    
    # 生成模拟数据
    sample_orders = viz.generate_day_simulation_orders(count=2000)
    
    # 生成地理热力图
    viz.generate_daily_order_heatmap(
        sample_orders, 
        save_path="results/plots/order_heatmap.png"
    )
    
    # 生成时段-地点柱状图
    viz.generate_time_location_bar_chart(
        sample_orders,
        save_path="results/plots/time_location_bar_chart.png"
    )
    
    # 生成24小时订单分布图
    viz.generate_24h_order_distribution(
        sample_orders,
        save_path="results/plots/24h_order_distribution.png"
    )
    
    # 生成时段订单散点分布图
    viz.generate_time_periods_scatter_plot(
        sample_orders,
        save_path="results/plots/time_periods_scatter_plot.png"
    )
    
    print("示例图表已生成到results/plots目录")

if __name__ == "__main__":
    demo() 