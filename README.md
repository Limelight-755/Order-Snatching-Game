# 网约车司机动态博弈定价模型

## 项目概述

本项目构建了一个基于博弈论的网约车司机动态定价策略模型，重点展现司机间的策略互动、学习适应和均衡演化过程。项目采用完全模拟的数据环境，通过LSTM预测技术和DQN决策技术支撑多轮动态博弈的实现。

## 核心特色

- **博弈论框架**：完整的多人博弈系统，包含策略空间、收益函数、信息结构
- **AI技术融合**：LSTM用于策略预测，DQN用于策略优化
- **动态演化**：博弈演化过程，展现从探索到均衡的完整过程
- **多场景测试**：对称博弈、非对称博弈等多种场景
- **纳什均衡求解**：创新的动态学习与静态检验相结合的纳什均衡求解方法

## 项目架构与功能模块

```
博弈论大作业/
├── README.md                          # 项目说明文档
├── requirements.txt                   # 依赖包列表
├── main.py                            # 主程序入口
├── config/                            # 配置模块
│   ├── __init__.py
│   └── game_config.py                 # 博弈参数配置
├── core/                              # 核心逻辑模块
│   ├── __init__.py
│   ├── game_framework.py              # 博弈框架核心
│   ├── game_theory_framework.py       # 博弈论理论框架
│   └── market_environment.py          # 市场环境模拟
├── ai_models/                         # AI智能体模块
│   ├── __init__.py
│   ├── dqn_agent.py                   # DQN决策智能体
│   ├── lstm_predictor.py              # LSTM策略预测模型
│   └── model_utils.py                 # 模型工具函数
├── data/                              # 数据模块
│   ├── __init__.py
│   ├── data_generator.py              # 数据模拟生成器
│   └── market_simulator.py            # 市场数据模拟
├── analysis/                          # 分析模块
│   ├── __init__.py
│   ├── nash_analyzer.py               # Nash均衡分析
│   ├── convergence_analyzer.py        # 收敛性分析
│   └── visualization_utils.py         # 可视化工具
├── experiments/                       # 实验模块
│   ├── __init__.py
│   ├── symmetric_game.py              # 对称博弈实验
│   ├── asymmetric_game.py             # 非对称博弈实验
│   └── experiment_analysis.py         # 实验结果分析工具
└── results/                           # 实验结果目录
```

## 环境配置

```bash
# 创建虚拟环境（推荐）
conda create -n game_theory python=3.8
conda activate game_theory

# 或使用virtualenv
python -m venv game_theory_env
# Windows
game_theory_env\Scripts\activate
# Linux/Mac
source game_theory_env/bin/activate

# 安装依赖
pip install -r requirements.txt

# 确认项目配置完毕
python validate_model.py
```

## 运行步骤

**1. 运行特定博弈实验**

```bash
# 运行对称博弈实验（两个司机具有相同条件）
python main.py symmetric

# 运行非对称博弈实验（两个司机具有不同条件）
python main.py asymmetric

# 指定实验轮数
python main.py symmetric -r 100
```

**2. 查看实验结果**

实验结束后，结果将保存在`results/`目录下，包括：
- 策略演化图表
- 收益分析报告
- Nash均衡检测结果
- 收敛性分析结果

**3. 数据分析**

使用分析模块查看详细结果：
```bash
# 分析最近一次实验结果
python analysis/analyze_results.py

# 比较多次实验结果
python analysis/compare_experiments.py
```
