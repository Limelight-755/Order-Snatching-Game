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
│   ├── nash_analysis.py               # Nash分析工具
│   └── experiment_analysis.py         # 实验结果分析工具
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
- **shock_test.py**：实现市场冲击测试（环境变化下的策略适应）
- **nash_analysis.py**：提供纳什均衡分析工具
- **experiment_analysis.py**：综合分析实验结果和比较不同实验

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

# 博弈论实验框架

## 对称博弈实验

对称博弈实验模拟了具有相同条件的网约车司机之间的策略竞争。在这种博弈中，参与者拥有相同的学习能力、市场地位和初始条件。

### 实验特点

1. **参与者特点**:
   - 相同的学习率(0.01)
   - 相同的探索率衰减曲线
   - 相同的网络结构和初始化方式
   - 相同的市场位置和接单能力

2. **均衡预期**:
   - 两个智能体往往会收敛到相似的策略
   - Nash均衡通常出现在策略空间的中间区域
   - 收益水平相近，竞争较为平衡

3. **运行参数**:
   - 默认500轮博弈
   - 学习率: 0.01
   - 初始探索率: 0.9 → 0.5 → 0.1
   - 价格范围: 10-50元

### 运行实验

```bash
python main.py symmetric
```

### 实验结果

实验结果展示了策略收敛过程、Nash均衡形成和双方收益变化，通常在300-400轮后能观察到稳定的均衡策略组合。

## 非对称博弈实验

非对称博弈实验模拟了不同条件的网约车司机之间的策略竞争。参与者之间存在能力差异、信息不对称或市场地位不同等情况。

### 实验特点

1. **参与者差异**:
   - 不同的学习率(0.015 vs 0.008)
   - 不同的探索率(高探索vs低探索)
   - 不同的经验加成(1.2 vs 1.0)
   - 不同的效率得分(0.9 vs 0.7)

2. **均衡特性**:
   - 策略收敛路径明显不同
   - 可能形成领导者-跟随者模式
   - Nash均衡可能偏向具有优势的参与者
   - 收益差距通常更明显

3. **运行参数**:
   - 默认500轮博弈
   - 经验司机: 学习率0.015, 探索率0.08
   - 新手司机: 学习率0.008, 探索率0.15
   - 价格范围: 10-50元

### 运行实验

```bash
python main.py asymmetric
```

### 实验结果

实验结果通常显示优势方占据更有利的市场地位，最终达到的Nash均衡反映了参与者之间的能力差异。

## 市场冲击实验

市场冲击实验是一个模拟市场环境中突发事件对策略演化的影响的实验框架。实验通过模拟不同类型的市场冲击，分析智能体如何调整其策略来应对变化。

### 冲击类型

实验支持多种类型的市场冲击，包括：

1. **需求激增 (demand_surge)**: 市场需求突然增加，导致订单量增加
2. **供应短缺 (supply_shortage)**: 市场供应减少，可用资源受限
3. **价格管制 (price_regulation)**: 政府实施价格上限，限制最高定价
4. **竞争加剧 (competition_increase)**: 市场竞争者增加，争夺有限市场
5. **需求下降 (demand_drop)**: 市场需求突然下降，订单量减少
6. **市场扰动 (market_disruption)**: 同时影响供需两侧的复合型冲击
7. **技术变革 (technology_shift)**: 技术进步导致成本降低，同时刺激需求
8. **市场崩溃 (market_crash)**: 极端情况下的全面负面冲击

### 冲击参数

每种冲击可以通过以下参数进行配置：

- **round**: 冲击发生的轮次
- **type**: 冲击类型
- **intensity**: 冲击强度（影响倍数）
- **duration**: 冲击持续时间（轮次）
- 其他特定参数（如price_regulation的max_price）

### 冲击效果分析

实验结束后，会生成详细的冲击效果分析，包括：

1. 各种冲击对策略的影响
2. 收益变化率分析
3. 恢复时间分析
4. 不同类型冲击的比较分析

### 运行实验

```bash
python main.py shock
```

### 实验结果

实验结果保存在`results/figures`目录下，包含详细的冲击效果可视化分析图表。
