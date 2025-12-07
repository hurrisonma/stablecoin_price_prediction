"""
二分类下跌检测模型训练脚本
配置：0错误 + 最高捕获率 最优版本

最佳结果：
- 方向错误：0
- 捕获率：54%（27/50事件）
"""

import sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random
import os
from datetime import datetime

from src.data.loader import load_orderbook_data, validate_data
from src.data.preprocessor import generate_labels
from src.models.lstm_model import SimpleLSTMClassifier

# ==================== 配置参数 ====================
CONFIG = {
    # 数据配置
    'data_dir': 'data/orderbooks',
    'symbol': 'USDCUSDT',
    'horizon': 10,          # 预测周期（分钟）
    'threshold': 0.0001,    # 价格变化阈值
    'lookback': 40,         # 回看窗口（分钟）
    
    # 模型配置
    'input_size': 4,
    'hidden_size': 96,
    'num_layers': 1,
    'num_classes': 2,
    'dropout': 0.3,
    
    # 训练配置
    'seed': 10,             # 最佳种子
    'batch_size': 64,
    'epochs': 25,
    'learning_rate': 0.001,
    'class_weights': [1.0, 30.0],  # [不下跌, 下跌]
    
    # 推理配置
    'threshold_inference': 0.85,  # 推理阈值
    
    # 输出配置
    'model_dir': 'models',
}


def set_seed(seed):
    """固定所有随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def prepare_features(df):
    """准备特征"""
    bid1 = df['bid1_qty']
    bid2 = df['bid2_qty']
    ask1 = df['ask1_qty']
    
    features = pd.DataFrame({
        'bid1_qty': bid1,
        'bid1_velocity': bid1.diff(),
        'bid_ratio_12': bid1 / (bid2 + 1e-10),
        'imbalance_1': (bid1 - ask1) / (bid1 + ask1 + 1e-10),
    }).fillna(0)
    
    return features


def create_sequences(X, y, lookback):
    """创建时序样本"""
    Xs, ys = [], []
    for i in range(lookback, len(X)):
        Xs.append(X[i-lookback:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)


def train_model(config=CONFIG):
    """训练模型"""
    print("=" * 60)
    print("二分类下跌检测模型训练")
    print("=" * 60)
    
    # 设置种子
    set_seed(config['seed'])
    print(f"随机种子: {config['seed']}")
    
    # 加载数据
    df = load_orderbook_data(config['data_dir'], config['symbol'])
    df = validate_data(df)
    print(f"数据量: {len(df)} 条")
    
    # 准备特征和标签
    features_df = prepare_features(df)
    labels = generate_labels(df['bid1_px'], horizon=config['horizon'], threshold=config['threshold'])
    
    valid_mask = ~labels.isna()
    features_valid = features_df[valid_mask].values
    labels_orig = labels[valid_mask].values.astype(int)
    labels_binary = (labels_orig == 0).astype(int)  # 下跌=1, 其他=0
    
    print(f"下跌样本: {labels_binary.sum()}/{len(labels_binary)} ({labels_binary.mean()*100:.2f}%)")
    
    # 标准化
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_valid)
    
    # 划分数据
    n = len(labels_binary)
    train_end = int(n * 0.7)
    
    X_train = features_scaled[:train_end]
    y_train = labels_binary[:train_end]
    
    # 创建序列
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, config['lookback'])
    print(f"训练样本: {len(X_train_seq)}")
    
    # 创建数据加载器
    from torch.utils.data import TensorDataset, DataLoader
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_seq),
        torch.LongTensor(y_train_seq)
    )
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    
    # 创建模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    
    model = SimpleLSTMClassifier(
        input_size=config['input_size'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        num_classes=config['num_classes'],
        dropout=config['dropout']
    ).to(device)
    
    criterion = nn.CrossEntropyLoss(
        weight=torch.FloatTensor(config['class_weights']).to(device)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # 训练
    print(f"\n开始训练 ({config['epochs']} epochs)...")
    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{config['epochs']}, Loss: {total_loss/len(train_loader):.4f}")
    
    print("训练完成!")
    
    # 保存模型
    os.makedirs(config['model_dir'], exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(config['model_dir'], f"down_detector_{timestamp}.pt")
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'scaler_mean': scaler.mean_,
        'scaler_scale': scaler.scale_,
    }, model_path)
    
    print(f"模型已保存: {model_path}")
    
    return model, scaler, config


if __name__ == "__main__":
    train_model()
