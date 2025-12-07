"""
特征工程模块 V3 - 精简版
专为稳定币订单簿数据设计的特征

核心设计原则：
1. 聚焦一档加速度（最重要）
2. 一档速度和不平衡作为辅助
3. 一档二档比值变化捕捉聪明钱
4. 累计加速度捕捉趋势
"""

import pandas as pd
import numpy as np
from typing import List, Optional


def create_features(df: pd.DataFrame, 
                    rolling_windows: List[int] = [5, 15, 30],
                    use_slim: bool = True) -> pd.DataFrame:
    """
    创建订单簿动态特征
    
    Args:
        df: 原始订单簿数据 DataFrame
        rolling_windows: 滚动窗口大小列表（分钟）
        use_slim: 是否使用精简特征集 (约20个 vs 57个)
    
    Returns:
        特征 DataFrame
    """
    if use_slim:
        features = create_slim_features(df, rolling_windows)
    else:
        features = create_full_features(df, rolling_windows)
    
    # 处理无穷大和NaN
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.ffill().bfill().fillna(0)
    
    print(f"创建了 {len(features.columns)} 个特征")
    print(f"特征列表: {list(features.columns)}")
    
    return features


def create_slim_features(df: pd.DataFrame, 
                         rolling_windows: List[int] = [5, 15, 30]) -> pd.DataFrame:
    """
    创建精简特征集 (~20个特征)
    """
    features = pd.DataFrame(index=df.index)
    
    bid1 = df['bid1_qty']
    ask1 = df['ask1_qty']
    bid2 = df['bid2_qty']
    ask2 = df['ask2_qty']
    
    # ========== 1. 一档加速度 (3个) ⭐⭐⭐ 最重要 ==========
    bid1_velocity = bid1.diff()
    ask1_velocity = ask1.diff()
    
    features['bid1_accel'] = bid1_velocity.diff()
    features['ask1_accel'] = ask1_velocity.diff()
    features['net_accel_1'] = features['bid1_accel'] - features['ask1_accel']
    
    # ========== 2. 一档速度 (3个) ⭐⭐ ==========
    features['bid1_velocity'] = bid1_velocity
    features['ask1_velocity'] = ask1_velocity
    features['net_velocity_1'] = bid1_velocity - ask1_velocity
    
    # ========== 3. 不平衡及其变化 (3个) ⭐⭐ ==========
    imbalance_1 = (bid1 - ask1) / (bid1 + ask1 + 1e-10)
    features['imbalance_1'] = imbalance_1
    features['imbalance_1_velocity'] = imbalance_1.diff()
    features['imbalance_1_accel'] = features['imbalance_1_velocity'].diff()
    
    # ========== 4. 一档二档比值变化 (3个) ⭐ 聪明钱 ==========
    bid_ratio_12 = bid1 / (bid2 + 1e-10)
    ask_ratio_12 = ask1 / (ask2 + 1e-10)
    
    features['bid_ratio_12_velocity'] = bid_ratio_12.diff()
    features['ask_ratio_12_velocity'] = ask_ratio_12.diff()
    features['ratio_diff_velocity'] = (bid_ratio_12 - ask_ratio_12).diff()
    
    # ========== 5. 累计加速度 (9个) ==========
    for window in rolling_windows:
        # 累计净加速度
        features[f'net_accel_1_sum_{window}'] = features['net_accel_1'].rolling(window=window).sum()
        features[f'net_accel_1_mean_{window}'] = features['net_accel_1'].rolling(window=window).mean()
        # 加速度波动
        features[f'accel_std_{window}'] = features['net_accel_1'].rolling(window=window).std()
    
    return features


def create_full_features(df: pd.DataFrame, 
                         rolling_windows: List[int] = [5, 15, 30]) -> pd.DataFrame:
    """
    创建完整特征集 (57个特征)
    保留原有的所有特征，用于对比实验
    """
    features = pd.DataFrame(index=df.index)
    
    # 1. 一档深度特征
    tier1_features = compute_tier1_features(df)
    features = pd.concat([features, tier1_features], axis=1)
    
    # 2. 二档深度特征
    tier2_features = compute_tier2_features(df)
    features = pd.concat([features, tier2_features], axis=1)
    
    # 3. 不平衡特征
    imbalance_features = compute_imbalance_features(df)
    features = pd.concat([features, imbalance_features], axis=1)
    
    # 4. 一档二档比值特征
    ratio_features = compute_ratio_features(df)
    features = pd.concat([features, ratio_features], axis=1)
    
    # 5. 累计特征
    cumulative_features = compute_cumulative_features(features, rolling_windows)
    features = pd.concat([features, cumulative_features], axis=1)
    
    # 6. 价差特征
    spread_features = compute_spread_features(df)
    features = pd.concat([features, spread_features], axis=1)
    
    return features


def compute_tier1_features(df: pd.DataFrame) -> pd.DataFrame:
    """计算一档深度特征"""
    features = pd.DataFrame(index=df.index)
    
    bid1_qty = df['bid1_qty']
    ask1_qty = df['ask1_qty']
    
    features['bid1_velocity'] = bid1_qty.diff()
    features['ask1_velocity'] = ask1_qty.diff()
    features['net_velocity_1'] = features['bid1_velocity'] - features['ask1_velocity']
    features['bid1_accel'] = features['bid1_velocity'].diff()
    features['ask1_accel'] = features['ask1_velocity'].diff()
    features['net_accel_1'] = features['bid1_accel'] - features['ask1_accel']
    features['bid1_velocity_pct'] = bid1_qty.pct_change().clip(-1, 1)
    features['ask1_velocity_pct'] = ask1_qty.pct_change().clip(-1, 1)
    
    return features


def compute_tier2_features(df: pd.DataFrame) -> pd.DataFrame:
    """计算二档深度特征"""
    features = pd.DataFrame(index=df.index)
    
    bid2_qty = df['bid2_qty']
    ask2_qty = df['ask2_qty']
    
    features['bid2_velocity'] = bid2_qty.diff()
    features['ask2_velocity'] = ask2_qty.diff()
    features['bid2_accel'] = features['bid2_velocity'].diff()
    features['ask2_accel'] = features['ask2_velocity'].diff()
    features['net_velocity_2'] = features['bid2_velocity'] - features['ask2_velocity']
    features['net_accel_2'] = features['bid2_accel'] - features['ask2_accel']
    
    return features


def compute_imbalance_features(df: pd.DataFrame) -> pd.DataFrame:
    """计算不平衡特征"""
    features = pd.DataFrame(index=df.index)
    
    bid1 = df['bid1_qty']
    ask1 = df['ask1_qty']
    features['imbalance_1'] = (bid1 - ask1) / (bid1 + ask1 + 1e-10)
    
    bid2 = df['bid2_qty']
    ask2 = df['ask2_qty']
    features['imbalance_2'] = (bid2 - ask2) / (bid2 + ask2 + 1e-10)
    
    total_bid = bid1 + bid2
    total_ask = ask1 + ask2
    features['imbalance_total'] = (total_bid - total_ask) / (total_bid + total_ask + 1e-10)
    
    features['imbalance_1_velocity'] = features['imbalance_1'].diff()
    features['imbalance_2_velocity'] = features['imbalance_2'].diff()
    features['imbalance_1_accel'] = features['imbalance_1_velocity'].diff()
    features['imbalance_2_accel'] = features['imbalance_2_velocity'].diff()
    
    return features


def compute_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """计算一档二档比值特征"""
    features = pd.DataFrame(index=df.index)
    
    bid1 = df['bid1_qty']
    bid2 = df['bid2_qty']
    ask1 = df['ask1_qty']
    ask2 = df['ask2_qty']
    
    features['bid_ratio_12'] = bid1 / (bid2 + 1e-10)
    features['ask_ratio_12'] = ask1 / (ask2 + 1e-10)
    features['bid_ratio_12_velocity'] = features['bid_ratio_12'].diff()
    features['ask_ratio_12_velocity'] = features['ask_ratio_12'].diff()
    features['ratio_diff'] = features['bid_ratio_12'] - features['ask_ratio_12']
    features['ratio_diff_velocity'] = features['ratio_diff'].diff()
    
    return features


def compute_cumulative_features(features: pd.DataFrame, 
                                 windows: List[int] = [5, 15, 30]) -> pd.DataFrame:
    """计算累计/滚动特征"""
    cumulative = pd.DataFrame(index=features.index)
    
    key_features = ['net_velocity_1', 'net_accel_1', 'imbalance_1', 'imbalance_1_velocity']
    
    for col in key_features:
        if col not in features.columns:
            continue
        for window in windows:
            cumulative[f'{col}_sum_{window}'] = features[col].rolling(window=window).sum()
            cumulative[f'{col}_mean_{window}'] = features[col].rolling(window=window).mean()
    
    if 'net_accel_1' in features.columns:
        for window in windows:
            cumulative[f'accel_std_{window}'] = features['net_accel_1'].rolling(window=window).std()
    
    return cumulative


def compute_spread_features(df: pd.DataFrame) -> pd.DataFrame:
    """计算价差特征"""
    features = pd.DataFrame(index=df.index)
    
    features['spread'] = df['ask1_px'] - df['bid1_px']
    features['spread_velocity'] = features['spread'].diff()
    mid_price = (df['ask1_px'] + df['bid1_px']) / 2
    features['spread_ratio'] = features['spread'] / (mid_price + 1e-10)
    
    return features


def get_feature_importance_ranking() -> dict:
    """返回特征重要性排序"""
    return {
        '⭐⭐⭐ 最重要 (一档加速度)': [
            'bid1_accel', 'ask1_accel', 'net_accel_1'
        ],
        '⭐⭐ 重要 (一档速度+不平衡)': [
            'bid1_velocity', 'ask1_velocity', 'net_velocity_1',
            'imbalance_1', 'imbalance_1_velocity', 'imbalance_1_accel'
        ],
        '⭐ 辅助 (聪明钱指标)': [
            'bid_ratio_12_velocity', 'ask_ratio_12_velocity', 'ratio_diff_velocity'
        ],
        '参考 (累计)': [
            'net_accel_1_sum_5/15/30', 'net_accel_1_mean_5/15/30', 'accel_std_5/15/30'
        ]
    }


# 兼容旧接口
def compute_orderbook_features(df: pd.DataFrame) -> pd.DataFrame:
    return create_features(df, use_slim=False)


if __name__ == "__main__":
    import os
    import sys
    
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, project_root)
    
    from src.data.loader import load_orderbook_data, validate_data
    
    data_dir = os.path.join(project_root, "data", "orderbooks")
    df = load_orderbook_data(data_dir, "USDCUSDT")
    df = validate_data(df)
    
    # 精简特征
    print("="*60)
    print("精简特征集:")
    features_slim = create_features(df, use_slim=True)
    print(f"特征数量: {len(features_slim.columns)}")
    
    # 完整特征
    print("\n" + "="*60)
    print("完整特征集:")
    features_full = create_features(df, use_slim=False)
    print(f"特征数量: {len(features_full.columns)}")
