#!/usr/bin/env python3
"""
订单可视化工具运行脚本
生成订单地理热力图和时段分布图表
"""

import sys
import os
import logging
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

# 导入项目模块
from analysis.order_visualization import OrderVisualization
from core.market_environment import Order, LocationType, TimePeriod
import matplotlib.pyplot as plt

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def main():
    """主函数"""
    # 确保结果目录存在
    os.makedirs("results/plots", exist_ok=True)
    
    # 创建可视化工具实例
    viz = OrderVisualization()
    
    logger.info("生成模拟订单数据...")
    # 生成模拟数据（2000个订单，覆盖一天24小时）
    sample_orders = viz.generate_day_simulation_orders(count=2000)
    
    logger.info("生成分时段订单散点图...")
    # 生成分时段订单散点图（代替原来的热力图）
    viz.generate_time_periods_scatter_plot(
        sample_orders,
        save_path="results/plots/time_periods_scatter_plot.png"
    )
    
    logger.info("生成时段-地点柱状图...")
    # 生成时段-地点柱状图
    viz.generate_time_location_bar_chart(
        sample_orders,
        save_path="results/plots/time_location_bar_chart.png"
    )
    
    logger.info("生成24小时订单分布图...")
    # 生成24小时订单分布图
    viz.generate_24h_order_distribution(
        sample_orders,
        save_path="results/plots/24h_order_distribution.png"
    )
    
    logger.info("可视化完成！图表已保存到 results/plots/ 目录")
    logger.info("- 分时段订单散点图: results/plots/time_periods_scatter_plot.png")
    logger.info("- 时段-地点柱状图: results/plots/time_location_bar_chart.png")
    logger.info("- 24小时订单分布图: results/plots/24h_order_distribution.png")


if __name__ == "__main__":
    main() 