# 项目进展记录

## 里程碑

| 阶段 | 目标 | 状态 | 完成日期 |
|------|------|------|----------|
| Phase 1 | 项目结构搭建 | ✅ 已完成 | 2025-12-07 |
| Phase 2 | 数据处理模块 | ⏳ 验证中 | - |
| Phase 3 | 模型实现 | 💬 初步实现，待讨论 | - |
| Phase 4 | 训练与评估 | ⏸️ 未开始 | - |
| Phase 5 | 优化与调参 | ⏸️ 未开始 | - |

---

## 开发日志

### 2025-12-07

#### 项目初始化
- 分析了 USDCUSDT 订单簿数据结构
- 创建了完整的实现计划
- 建立了项目目录结构

#### 数据处理模块 (验证中)
- `loader.py`: 数据加载和验证
- `preprocessor.py`: 中间价格计算、标签生成、归一化
- `dataset.py`: PyTorch Dataset 和 DataLoader

#### 特征工程
- `feature_engineering.py`: 实现了 70+ 个特征
  - 基础价格特征 (mid_price, spread)
  - 订单簿特征 (深度、累计量)
  - 不平衡特征 (各档位不平衡度)
  - 技术指标 (MA, EMA, RSI, Bollinger Bands)
  - 时序变化特征 (收益率、波动率)

#### 模型实现 (初步完成，待讨论)
- `lstm_model.py`: LSTMClassifier, SimpleLSTMClassifier, GRUClassifier
- `transformer_model.py`: TransformerClassifier, TransformerWithCLS
- `trainer.py`: 训练器框架

> **注意**: 模型架构需要进一步讨论确定后再最终定稿

---

## 当前工作

- [ ] 验证数据处理流程
- [ ] 生成特征数据
- [ ] 检查标签分布
- [ ] 讨论模型架构

---
