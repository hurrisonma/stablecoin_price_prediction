"""
数据预处理模块
负责计算中间价格、生成标签和特征归一化
"""

from typing import Tuple, Dict, Optional
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def calculate_mid_price(df: pd.DataFrame) -> pd.Series:
    """
    计算中间价格 (mid-price)
    
    Args:
        df: 包含 bid1_px 和 ask1_px 的 DataFrame
    
    Returns:
        中间价格序列
    """
    mid_price = (df['bid1_px'] + df['ask1_px']) / 2
    return mid_price


def generate_labels(mid_prices: pd.Series, 
                   horizon: int, 
                   threshold: float = 0.0001) -> pd.Series:
    """
    生成价格变化标签 (三分类)
    
    Args:
        mid_prices: 中间价格序列
        horizon: 预测周期 (分钟数)
        threshold: 价格变化阈值
    
    Returns:
        标签序列:
            0 - 下降 (price_change < -threshold)
            1 - 不变 (-threshold <= price_change <= threshold)
            2 - 上升 (price_change > threshold)
    """
    # 计算未来价格
    future_price = mid_prices.shift(-horizon)
    
    # 计算价格变化
    price_change = future_price - mid_prices
    
    # 生成标签
    labels = pd.Series(1, index=mid_prices.index)  # 默认为不变
    labels[price_change < -threshold] = 0  # 下降
    labels[price_change > threshold] = 2   # 上升
    
    # 最后 horizon 个样本没有未来价格，标记为 NaN
    labels.iloc[-horizon:] = np.nan
    
    return labels


def get_label_distribution(labels: pd.Series) -> Dict[str, float]:
    """
    获取标签分布
    
    Args:
        labels: 标签序列
    
    Returns:
        各类别占比字典
    """
    # 移除 NaN
    valid_labels = labels.dropna()
    
    total = len(valid_labels)
    distribution = {
        '下降 (0)': (valid_labels == 0).sum() / total * 100,
        '不变 (1)': (valid_labels == 1).sum() / total * 100,
        '上升 (2)': (valid_labels == 2).sum() / total * 100
    }
    
    return distribution


def normalize_features(features: pd.DataFrame, 
                       method: str = 'zscore',
                       scaler: Optional[object] = None) -> Tuple[pd.DataFrame, object]:
    """
    特征归一化
    
    Args:
        features: 特征 DataFrame
        method: 归一化方法 ('zscore' 或 'minmax')
        scaler: 预训练的 scaler (用于测试集)
    
    Returns:
        归一化后的特征 DataFrame 和 scaler 对象
    """
    if scaler is None:
        if method == 'zscore':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"不支持的归一化方法: {method}")
        
        # 拟合并转换
        normalized_values = scaler.fit_transform(features)
    else:
        # 仅转换 (使用已拟合的 scaler)
        normalized_values = scaler.transform(features)
    
    # 转换回 DataFrame
    normalized_df = pd.DataFrame(
        normalized_values,
        columns=features.columns,
        index=features.index
    )
    
    return normalized_df, scaler


def prepare_data(df: pd.DataFrame, 
                 features_df: pd.DataFrame,
                 horizon: int = 5,
                 threshold: float = 0.0001,
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.15) -> Dict:
    """
    准备训练、验证和测试数据
    
    Args:
        df: 原始数据 DataFrame
        features_df: 特征 DataFrame
        horizon: 预测周期
        threshold: 价格变化阈值
        train_ratio: 训练集比例
        val_ratio: 验证集比例
    
    Returns:
        包含划分后数据的字典
    """
    # 计算中间价格和标签
    mid_prices = calculate_mid_price(df)
    labels = generate_labels(mid_prices, horizon, threshold)
    
    # 移除标签为 NaN 的样本
    valid_mask = ~labels.isna()
    features_valid = features_df[valid_mask].copy()
    labels_valid = labels[valid_mask].copy()
    
    # 按时间顺序划分数据
    n_samples = len(labels_valid)
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))
    
    # 划分数据
    train_features = features_valid.iloc[:train_end]
    train_labels = labels_valid.iloc[:train_end]
    
    val_features = features_valid.iloc[train_end:val_end]
    val_labels = labels_valid.iloc[train_end:val_end]
    
    test_features = features_valid.iloc[val_end:]
    test_labels = labels_valid.iloc[val_end:]
    
    # 归一化 (仅在训练集上拟合)
    train_normalized, scaler = normalize_features(train_features)
    val_normalized, _ = normalize_features(val_features, scaler=scaler)
    test_normalized, _ = normalize_features(test_features, scaler=scaler)
    
    # 打印数据集信息
    print(f"\n数据集划分:")
    print(f"  训练集: {len(train_labels)} 条")
    print(f"  验证集: {len(val_labels)} 条")
    print(f"  测试集: {len(test_labels)} 条")
    
    # 打印标签分布
    print(f"\n标签分布:")
    for name, subset in [('训练集', train_labels), ('验证集', val_labels), ('测试集', test_labels)]:
        dist = get_label_distribution(subset)
        print(f"  {name}: 下降 {dist['下降 (0)']:.1f}% | 不变 {dist['不变 (1)']:.1f}% | 上升 {dist['上升 (2)']:.1f}%")
    
    return {
        'train': {
            'features': train_normalized.values,
            'labels': train_labels.values.astype(int)
        },
        'val': {
            'features': val_normalized.values,
            'labels': val_labels.values.astype(int)
        },
        'test': {
            'features': test_normalized.values,
            'labels': test_labels.values.astype(int)
        },
        'scaler': scaler,
        'feature_names': list(features_df.columns)
    }


if __name__ == "__main__":
    # 测试代码
    import os
    from loader import load_orderbook_data, validate_data
    
    # 获取项目根目录
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(project_root, "data", "orderbooks")
    
    # 加载和验证数据
    df = load_orderbook_data(data_dir, "USDCUSDT")
    df = validate_data(df)
    
    # 计算中间价格
    mid_prices = calculate_mid_price(df)
    print(f"\n中间价格统计:")
    print(f"  均值: {mid_prices.mean():.6f}")
    print(f"  标准差: {mid_prices.std():.6f}")
    
    # 生成标签
    labels = generate_labels(mid_prices, horizon=5, threshold=0.0001)
    
    # 打印标签分布
    dist = get_label_distribution(labels)
    print(f"\n标签分布 (horizon=5, threshold=0.0001):")
    for label, pct in dist.items():
        print(f"  {label}: {pct:.2f}%")
