# 网约车司机动态博弈定价模型

## 项目概述

本项目构建了一个基于博弈论的网约车司机动态定价策略模型，重点展现司机间的策略互动、学习适应和均衡演化过程。项目采用完全模拟的数据环境，通过LSTM预测技术和DQN决策技术支撑多轮动态博弈的实现。

## 核心特色

- **博弈论框架**：完整的多人博弈系统，包含策略空间、收益函数、信息结构
- **AI技术融合**：LSTM用于策略预测，DQN用于策略优化
- **动态演化**：500轮博弈演化过程，展现从探索到均衡的完整过程
- **多场景测试**：对称博弈、非对称博弈、环境冲击等多种场景

## 项目架构与功能模块

```
博弈论大作业/
├── README.md                          # 项目说明文档
├── dynamic_game_model_report.md       # 详细技术报告
├── requirements.txt                   # 依赖包列表
├── main.py                            # 主程序入口
├── demo.py                            # 功能演示脚本
├── validate_model.py                  # 模型验证脚本
├── simple_test.py                     # 简单集成测试脚本
├── test_market.py                     # 市场测试脚本
├── config/                            # 配置模块
│   ├── __init__.py
│   └── game_config.py                 # 博弈参数配置
├── core/                              # 核心逻辑模块
│   ├── __init__.py
│   ├── game_framework.py              # 博弈框架核心
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
│   ├── performance_evaluator.py       # 性能评估
│   ├── statistical_analyzer.py        # 统计分析
│   └── visualization_utils.py         # 可视化工具
├── experiments/                       # 实验模块
│   ├── __init__.py
│   ├── experiment_utils.py            # 实验工具
│   ├── symmetric_game.py              # 对称博弈实验
│   ├── asymmetric_game.py             # 非对称博弈实验
│   ├── shock_test.py                  # 冲击测试
│   └── nash_analysis.py               # Nash分析工具
├── tests/                             # 测试模块
│   ├── __init__.py
│   ├── test_config.py                 # 配置模块测试
│   ├── test_market_environment.py     # 市场环境测试
│   ├── test_ai_models.py              # AI模型测试
│   ├── test_experiments.py            # 实验模块测试
│   └── run_all_tests.py               # 测试套件运行器
└── results/                           # 实验结果目录（不含内容）
```

### 核心模块功能

#### 配置模块 (config/)
- **game_config.py**：设置博弈参数、策略范围、市场环境参数等全局配置

#### 核心逻辑模块 (core/)
- **game_framework.py**：实现博弈论核心规则、Nash均衡检测、策略评估
- **market_environment.py**：模拟网约车市场环境，包括订单生成、价格匹配、收益计算

#### AI智能体模块 (ai_models/)
- **lstm_predictor.py**：使用LSTM网络预测对手策略和市场变化
- **dqn_agent.py**：使用深度强化学习优化定价策略
- **model_utils.py**：提供神经网络工具函数和数据预处理

#### 数据模块 (data/)
- **data_generator.py**：生成模拟订单、用户需求和地理位置数据
- **market_simulator.py**：模拟市场供需变化、价格波动和竞争环境

#### 分析模块 (analysis/)
- **nash_analyzer.py**：分析策略是否达到Nash均衡及均衡稳定性
- **convergence_analyzer.py**：评估策略收敛性和学习效率
- **performance_evaluator.py**：评估智能体性能和整体博弈质量
- **statistical_analyzer.py**：进行统计分析和假设检验
- **visualization_utils.py**：生成各类可视化图表及动画

#### 实验模块 (experiments/)
- **experiment_utils.py**：提供实验框架和数据收集工具
- **symmetric_game.py**：实现对称博弈实验（相同条件下的竞争）
- **asymmetric_game.py**：实现非对称博弈实验（不同条件下的竞争）

## 🚀 快速开始

### 环境配置

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
```

### 模型验证

运行模型验证脚本以确保环境配置正确：

```bash
python validate_model.py
```

这个脚本会检查：
- ✅ 所有模块是否正确导入
- ✅ 核心功能是否正常工作
- ✅ AI模型是否能正确初始化
- ✅ 数据生成和处理功能
- ✅ 实验框架完整性

> **如果验证失败，请检查**：依赖包是否全部安装成功、Python版本是否符合要求、项目目录结构是否完整

### 运行实验

**1. 功能演示**
```bash
python demo.py
```

**2. 运行特定博弈实验**

```bash
# 运行对称博弈实验（两个司机具有相同条件）
python main.py symmetric

# 运行非对称博弈实验（两个司机具有不同条件）
python main.py asymmetric

# 运行市场冲击实验（测试环境变化下的适应性）
python main.py shock

# 运行所有实验
python main.py all

# 指定实验轮数
python main.py symmetric -r 100
```

## 📊 主要功能

### 1. 市场环境模拟

市场环境模拟了网约车平台上的供需关系、订单分配和收益计算：

- **订单生成**：根据时间、地点、需求等因素动态生成订单
- **价格匹配**：根据司机定价策略匹配订单
- **收益计算**：计算司机在不同策略下的收益
- **市场动态**：模拟高峰低谷、天气变化等外部因素

### 2. 博弈框架

核心的博弈论框架实现了：

- **策略空间**：司机可选择的价格策略范围（10-50元）
- **收益矩阵**：不同策略组合下的收益计算
- **Nash均衡**：检测系统是否达到均衡状态
- **信息结构**：完全信息与不完全信息博弈设置

### 3. AI决策系统

AI智能体通过学习优化决策策略：

- **DQN决策**：深度Q网络进行策略优化
- **LSTM预测**：预测对手策略和市场变化
- **经验回放**：存储并学习历史经验
- **探索-利用平衡**：在策略搜索和利用之间取得平衡

### 4. 结果分析

系统提供丰富的分析工具：

- **策略演化分析**：策略如何随时间变化
- **Nash均衡分析**：检测是否达到稳定均衡
- **收敛性分析**：策略是否及如何收敛
- **统计分析**：对实验结果进行统计验证
- **可视化工具**：生成直观的图表和动画

## 📈 结果解读

实验运行后会生成多种图表，帮助理解博弈过程：

- **策略演化图**：展示玩家策略如何随时间变化
- **收益对比图**：比较不同玩家的收益表现
- **纳什均衡图**：显示策略与均衡点的关系
- **市场分析图**：展示市场供需和价格动态

## 🛠️ 定制与扩展

### 调整实验参数

编辑 `config/game_config.py` 来调整实验参数：

```python
# 基本参数
total_rounds = 500          # 总博弈轮数
price_range = (10, 50)      # 价格策略范围
exploration_rounds = 50     # 探索阶段轮数

# AI参数
learning_rate = 0.001       # 学习率
epsilon = 0.1              # 探索率
batch_size = 32            # 批次大小
```
