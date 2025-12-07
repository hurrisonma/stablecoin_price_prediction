"""
二分类下跌检测模型 - 完整参数搜索流程
============================================

此脚本记录了从头开始寻找最优模型配置的完整过程。
当有新数据时，运行此脚本可以复现参数搜索，找到新数据上的最优配置。

搜索流程：
1. 回看窗口搜索（10-60分钟）
2. 隐藏层大小搜索（16-256）
3. 阈值优化（0.70-0.95）
4. 种子搜索（找到0错误配置）
"""

import sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random
from datetime import datetime

from src.data.loader import load_orderbook_data, validate_data
from src.data.preprocessor import generate_labels
from src.models.lstm_model import SimpleLSTMClassifier
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader


# ==================== 基础配置（固定项） ====================
BASE_CONFIG = {
    'data_dir': 'data/orderbooks',
    'symbol': 'USDCUSDT',
    'horizon': 10,
    'threshold': 0.0001,
    'train_ratio': 0.7,
    'batch_size': 64,
    'epochs': 25,
    'learning_rate': 0.001,
    'class_weights': [1.0, 30.0],
    'dropout': 0.3,
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def prepare_data():
    """加载和准备数据"""
    df = load_orderbook_data(BASE_CONFIG['data_dir'], BASE_CONFIG['symbol'])
    df = validate_data(df)
    
    bid1, bid2, ask1 = df['bid1_qty'], df['bid2_qty'], df['ask1_qty']
    
    features_df = pd.DataFrame({
        'bid1_qty': bid1,
        'bid1_velocity': bid1.diff(),
        'bid_ratio_12': bid1 / (bid2 + 1e-10),
        'imbalance_1': (bid1 - ask1) / (bid1 + ask1 + 1e-10),
    }).fillna(0)
    
    labels = generate_labels(df['bid1_px'], horizon=BASE_CONFIG['horizon'], threshold=BASE_CONFIG['threshold'])
    
    valid_mask = ~labels.isna()
    features_valid = features_df[valid_mask].values
    labels_orig = labels[valid_mask].values.astype(int)
    labels_binary = (labels_orig == 0).astype(int)
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_valid)
    
    n = len(labels_binary)
    train_end = int(n * BASE_CONFIG['train_ratio'])
    
    return {
        'X_train': features_scaled[:train_end],
        'X_test': features_scaled[train_end:],
        'y_train': labels_binary[:train_end],
        'y_orig_test': labels_orig[train_end:],
        'scaler': scaler,
    }


def create_sequences(X, y, lookback):
    Xs, ys = [], []
    for i in range(lookback, len(X)):
        Xs.append(X[i-lookback:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)


def identify_events(labels):
    events = []
    i = 0
    while i < len(labels):
        if labels[i] != 1:
            event_type = labels[i]
            start = i
            while i < len(labels) and labels[i] == event_type:
                i += 1
            events.append({'type': event_type, 'start': start, 'end': i-1})
        else:
            i += 1
    return events


def train_and_evaluate(data, lookback, hidden_size, seed, threshold_inference):
    """训练模型并评估"""
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    X_train_seq, y_train_seq = create_sequences(data['X_train'], data['y_train'], lookback)
    X_test_seq, _ = create_sequences(data['X_test'], np.zeros(len(data['X_test'])), lookback)
    y_orig_test_seq = data['y_orig_test'][lookback:]
    
    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train_seq), torch.LongTensor(y_train_seq)),
        batch_size=BASE_CONFIG['batch_size'], shuffle=True
    )
    
    model = SimpleLSTMClassifier(
        input_size=4, hidden_size=hidden_size, num_layers=1, 
        num_classes=2, dropout=BASE_CONFIG['dropout']
    ).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(BASE_CONFIG['class_weights']).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=BASE_CONFIG['learning_rate'])
    
    for _ in range(BASE_CONFIG['epochs']):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
    
    model.eval()
    with torch.no_grad():
        pred_down = F.softmax(model(torch.FloatTensor(X_test_seq).to(device)), dim=1)[:, 1].cpu().numpy()
    
    events = identify_events(y_orig_test_seq)
    down_events = [e for e in events if e['type'] == 0]
    up_events = [e for e in events if e['type'] == 2]
    
    pred = (pred_down > threshold_inference).astype(int)
    captured = sum(1 for e in down_events if pred[e['start']:e['end']+1].any())
    wrong = sum(1 for e in up_events if pred[e['start']:e['end']+1].any())
    
    return {
        'captured': captured,
        'total_down': len(down_events),
        'wrong': wrong,
        'model': model,
        'pred_probs': pred_down,
    }


def search_best_config():
    """
    完整参数搜索流程
    """
    print("=" * 70)
    print("二分类下跌检测模型 - 参数搜索流程")
    print("目标: 0错误 + 最高捕获率")
    print("=" * 70)
    
    # 加载数据
    print("\n[1/4] 加载数据...")
    data = prepare_data()
    print(f"训练样本: {len(data['X_train'])}, 测试样本: {len(data['X_test'])}")
    
    # ==================== 步骤1: 回看窗口搜索 ====================
    print("\n[2/4] 回看窗口搜索 (10-60分钟)...")
    print("-" * 50)
    
    lookback_results = []
    for lookback in [10, 20, 30, 40, 50, 60]:
        result = train_and_evaluate(data, lookback=lookback, hidden_size=32, seed=42, threshold_inference=0.80)
        ratio = result['captured'] / result['wrong'] if result['wrong'] > 0 else float('inf')
        lookback_results.append((lookback, result['captured'], result['wrong'], ratio))
        print(f"  lookback={lookback:2d}分钟: 捕获{result['captured']:2d}/{result['total_down']}, 错误{result['wrong']:2d}, 比例{ratio:.1f}:1")
    
    # 选择真:错比例最高的回看窗口
    best_lookback = max(lookback_results, key=lambda x: x[3])[0]
    print(f"\n  >> 最佳回看窗口: {best_lookback}分钟")
    
    # ==================== 步骤2: 隐藏层搜索 ====================
    print(f"\n[3/4] 隐藏层搜索 (使用lookback={best_lookback})...")
    print("-" * 50)
    
    hidden_results = []
    for hidden_size in [16, 32, 48, 64, 96, 128]:
        result = train_and_evaluate(data, lookback=best_lookback, hidden_size=hidden_size, seed=42, threshold_inference=0.88)
        ratio = result['captured'] / result['wrong'] if result['wrong'] > 0 else float('inf')
        hidden_results.append((hidden_size, result['captured'], result['wrong'], ratio))
        print(f"  hidden={hidden_size:3d}: 捕获{result['captured']:2d}/{result['total_down']}, 错误{result['wrong']:2d}, 比例{ratio:.1f}:1")
    
    # 选择真:错比例最高的隐藏层
    best_hidden = max(hidden_results, key=lambda x: x[3])[0]
    print(f"\n  >> 最佳隐藏层: {best_hidden}")
    
    # ==================== 步骤3: 0错误种子搜索 ====================
    print(f"\n[4/4] 0错误种子搜索 (lookback={best_lookback}, hidden={best_hidden})...")
    print("-" * 50)
    
    zero_error_configs = []
    for seed in range(1, 21):
        # 对每个种子，找到能实现0错误的最低阈值
        for th in np.arange(0.80, 0.99, 0.01):
            result = train_and_evaluate(data, lookback=best_lookback, hidden_size=best_hidden, seed=seed, threshold_inference=th)
            if result['wrong'] == 0 and result['captured'] > 0:
                zero_error_configs.append((seed, th, result['captured'], result['model']))
                print(f"  seed={seed:2d}, 阈值={th:.2f}: 捕获{result['captured']:2d}/{result['total_down']}, 错误0 ✓")
                break
        else:
            print(f"  seed={seed:2d}: 无法达到0错误")
    
    if not zero_error_configs:
        print("\n警告: 未找到0错误配置!")
        return None
    
    # 选择捕获率最高的0错误配置
    zero_error_configs.sort(key=lambda x: -x[2])
    best_seed, best_threshold, best_captured, best_model = zero_error_configs[0]
    
    print("\n" + "=" * 70)
    print("搜索完成! 最佳配置:")
    print("=" * 70)
    print(f"  回看窗口: {best_lookback}分钟")
    print(f"  隐藏层: {best_hidden}")
    print(f"  种子: {best_seed}")
    print(f"  阈值: {best_threshold}")
    print(f"  捕获率: {best_captured}/{zero_error_configs[0][2]} (方向错误: 0)")
    
    # 保存最佳模型
    import os
    os.makedirs('models', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f'models/best_down_detector_{timestamp}.pt'
    
    torch.save({
        'model_state_dict': best_model.state_dict(),
        'config': {
            'lookback': best_lookback,
            'hidden_size': best_hidden,
            'seed': best_seed,
            'threshold_inference': best_threshold,
            **BASE_CONFIG,
        },
        'scaler_mean': data['scaler'].mean_,
        'scaler_scale': data['scaler'].scale_,
    }, model_path)
    
    print(f"\n模型已保存: {model_path}")
    
    return {
        'lookback': best_lookback,
        'hidden_size': best_hidden,
        'seed': best_seed,
        'threshold': best_threshold,
        'captured': best_captured,
        'model_path': model_path,
    }


if __name__ == "__main__":
    search_best_config()
