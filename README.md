# 稳定币价格预测系统 (Stablecoin Price Prediction)

基于深度学习的稳定币价格走势预测系统，利用分钟级订单簿深度数据预测未来价格变化方向。

## 🎯 项目目标

通过分析 USDCUSDT 等稳定币的订单簿深度数据，使用 LSTM 和 Transformer 模型预测未来时间周期内的价格走势：

| 预测类别 | 条件 | 含义 |
|----------|------|------|
| 下降 (0) | `Δprice < -0.0001` | 价格将下跌 |
| 不变 (1) | `-0.0001 ≤ Δprice ≤ 0.0001` | 价格将保持稳定 |
| 上升 (2) | `Δprice > 0.0001` | 价格将上涨 |

## 📁 项目结构

```
stablecoin_price_prediction/
├── config/
│   └── config.yaml                 # 配置文件
├── data/
│   ├── orderbooks/                 # 原始订单簿数据
│   └── processed/                  # 预处理后的数据
├── experiments/                    # 实验记录目录
│   ├── experiment_log.md           # 实验日志
│   └── hyperparameter_tuning.md    # 调参记录
├── src/
│   ├── data/                       # 数据处理模块
│   ├── features/                   # 特征工程模块
│   ├── models/                     # 模型定义与训练
│   ├── utils/                      # 工具函数
│   └── main.py                     # 主程序入口
├── outputs/
│   ├── models/                     # 保存的模型
│   └── logs/                       # 训练日志
├── notebooks/                      # Jupyter 笔记本
├── PROGRESS.md                     # 项目进展记录
├── README.md                       # 项目说明（本文件）
└── requirements.txt                # 依赖包
```

## 📊 数据说明

### 数据来源
- 路径: `data/orderbooks/`
- 格式: CSV 文件，每日一个文件
- 命名: `{SYMBOL}_depth_{DATE}.csv`

### 数据字段
| 字段 | 说明 |
|------|------|
| `ts_ms` | 毫秒时间戳 |
| `ts_utc8` | UTC+8 时间字符串 |
| `symbol` | 交易对名称 |
| `lastUpdateId` | 订单簿更新ID |
| `bid1_px` ~ `bid5_px` | 5档买盘价格 |
| `bid1_qty` ~ `bid5_qty` | 5档买盘数量 |
| `ask1_px` ~ `ask5_px` | 5档卖盘价格 |
| `ask1_qty` ~ `ask5_qty` | 5档卖盘数量 |

---

## 📋 项目记录规范

为了有序推进项目并追踪实验结果，项目采用以下记录文件：

### 1. PROGRESS.md - 项目进展记录

**位置**: 项目根目录

**用途**: 追踪整体开发进度、里程碑和每日工作日志

**更新规则**:
- 每完成一个阶段时更新里程碑状态
- 每日开发结束时记录当日工作内容
- 遇到重大决策或变更时记录原因

**格式示例**:
```markdown
## 里程碑

| 阶段 | 目标 | 状态 | 完成日期 |
|------|------|------|----------|
| Phase 1 | 项目结构搭建 | ✅ 已完成 | 2025-12-07 |

## 开发日志

### 2025-12-07
- 完成项目初始化
- 实现数据加载模块
```

---

### 2. experiments/experiment_log.md - 实验日志

**位置**: `experiments/` 目录

**用途**: 记录每次模型训练的详细信息、结果和分析

**更新规则**:
- 每次训练实验完成后添加新记录
- 实验按顺序编号（#001, #002, ...）
- 必须记录完整的配置参数和评估指标
- 记录观察分析和改进方向

**必须包含的内容**:
- 实验编号和日期
- 完整配置（模型类型、超参数、数据划分）
- 评估结果（Accuracy、F1、各类别指标）
- 混淆矩阵
- 观察分析
- 下一步改进方向

---

### 3. experiments/hyperparameter_tuning.md - 调参记录

**位置**: `experiments/` 目录

**用途**: 追踪超参数搜索过程，快速对比不同配置的效果

**更新规则**:
- 每次调参实验后添加一行记录
- 按模型类型分表记录（LSTM、Transformer）
- 找到更优配置时更新"最佳配置"部分

**记录字段**:
- 实验ID
- 关键超参数
- 验证集指标
- 训练耗时
- 备注

---

## 🚀 快速开始

### 环境准备

```bash
# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 训练模型

```bash
# 训练 LSTM 模型
python src/main.py train --model lstm --config config/config.yaml

# 训练 Transformer 模型
python src/main.py train --model transformer --config config/config.yaml
```

### 评估模型

```bash
python src/main.py evaluate --model-path outputs/models/best_model.pt
```

---

## ⚙️ 配置参数

主要配置参数（位于 `config/config.yaml`）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `lookback_window` | 60 | 回看窗口（分钟） |
| `prediction_horizon` | 5 | 预测周期（分钟） |
| `price_threshold` | 0.0001 | 价格变化阈值 |
| `batch_size` | 64 | 批次大小 |
| `learning_rate` | 0.001 | 学习率 |
| `epochs` | 100 | 最大训练轮数 |

---

## 📈 评估指标

| 指标 | 基线目标 | 良好目标 |
|------|----------|----------|
| 总体准确率 | > 40% | > 50% |
| Macro F1 | > 0.35 | > 0.45 |
| 上升类 Recall | > 30% | > 40% |
| 下降类 Recall | > 30% | > 40% |

> **注意**: 由于稳定币价格波动极小，这是一个具有挑战性的预测任务。随机基线约为 33%（三分类）。

---

## 🔧 技术栈

- **深度学习框架**: PyTorch
- **数据处理**: Pandas, NumPy
- **机器学习**: Scikit-learn
- **可视化**: Matplotlib, Seaborn
- **日志记录**: TensorBoard

---

## 📝 开发规范

### 代码提交
- 功能开发完成后及时提交
- 提交信息使用中文，格式：`<type>: <description>`
- 类型包括：feat（新功能）、fix（修复）、docs（文档）、refactor（重构）

### 实验管理
- 每次重要实验必须记录到 `experiment_log.md`
- 调参实验记录到 `hyperparameter_tuning.md`
- 发现有效改进时更新 `PROGRESS.md`

### 模型保存
- 最佳模型保存为 `outputs/models/best_model.pt`
- 包含完整的模型权重和配置信息
