"""
特征工程模块 V2
专为稳定币订单簿数据设计的特征

核心设计原则：
1. 聚焦订单簿动态变化，而非价格技术指标
2. 加速度（二阶导数）是最重要的特征
3. 一档数据最重要，二档作为辅助信号（聪明钱）
"""

import pandas as pd
import numpy as np
from typing import List, Optional


def create_features(df: pd.DataFrame, 
                    rolling_windows: List[int] = [5, 15, 30]) -> pd.DataFrame:
    """
    创建订单簿动态特征
    
    Args:
        df: 原始订单簿数据 DataFrame
        rolling_windows: 滚动窗口大小列表（分钟）
    
    Returns:
        特征 DataFrame
    """
    features = pd.DataFrame(index=df.index)
    
    # 1. 一档深度特征（最重要）
    tier1_features = compute_tier1_features(df)
    features = pd.concat([features, tier1_features], axis=1)
    
    # 2. 二档深度特征（辅助信号）
    tier2_features = compute_tier2_features(df)
    features = pd.concat([features, tier2_features], axis=1)
    
    # 3. 不平衡特征
    imbalance_features = compute_imbalance_features(df)
    features = pd.concat([features, imbalance_features], axis=1)
    
    # 4. 一档二档比值特征（聪明钱指标）
    ratio_features = compute_ratio_features(df)
    features = pd.concat([features, ratio_features], axis=1)
    
    # 5. 累计特征（滚动窗口）
    cumulative_features = compute_cumulative_features(features, rolling_windows)
    features = pd.concat([features, cumulative_features], axis=1)
    
    # 6. 价差特征
    spread_features = compute_spread_features(df)
    features = pd.concat([features, spread_features], axis=1)
    
    # 处理无穷大和NaN
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.ffill().bfill().fillna(0)
    
    print(f"创建了 {len(features.columns)} 个特征")
    print(f"特征列表: {list(features.columns)}")
    
    return features


def compute_tier1_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算一档深度特征（最重要）
    
    包含：
    - 深度变化速度（一阶导数）
    - 深度变化加速度（二阶导数）⭐ 最重要
    """
    features = pd.DataFrame(index=df.index)
    
    # 原始深度
    bid1_qty = df['bid1_qty']
    ask1_qty = df['ask1_qty']
    
    # === 速度（一阶导数）===
    features['bid1_velocity'] = bid1_qty.diff()
    features['ask1_velocity'] = ask1_qty.diff()
    
    # 净速度（买方 - 卖方）
    features['net_velocity_1'] = features['bid1_velocity'] - features['ask1_velocity']
    
    # === 加速度（二阶导数）⭐ 最重要 ===
    features['bid1_accel'] = features['bid1_velocity'].diff()
    features['ask1_accel'] = features['ask1_velocity'].diff()
    
    # 净加速度
    features['net_accel_1'] = features['bid1_accel'] - features['ask1_accel']
    
    # === 相对变化率 ===
    # 避免除零，使用平滑处理
    features['bid1_velocity_pct'] = bid1_qty.pct_change().clip(-1, 1)
    features['ask1_velocity_pct'] = ask1_qty.pct_change().clip(-1, 1)
    
    return features


def compute_tier2_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算二档深度特征（辅助信号，聪明钱埋伏）
    """
    features = pd.DataFrame(index=df.index)
    
    bid2_qty = df['bid2_qty']
    ask2_qty = df['ask2_qty']
    
    # 速度
    features['bid2_velocity'] = bid2_qty.diff()
    features['ask2_velocity'] = ask2_qty.diff()
    
    # 加速度（重要性低于一档）
    features['bid2_accel'] = features['bid2_velocity'].diff()
    features['ask2_accel'] = features['ask2_velocity'].diff()
    
    # 净变化
    features['net_velocity_2'] = features['bid2_velocity'] - features['ask2_velocity']
    features['net_accel_2'] = features['bid2_accel'] - features['ask2_accel']
    
    return features


def compute_imbalance_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算不平衡特征
    
    不平衡度 = (买 - 卖) / (买 + 卖)
    范围: [-1, 1]
    """
    features = pd.DataFrame(index=df.index)
    
    # 一档不平衡
    bid1 = df['bid1_qty']
    ask1 = df['ask1_qty']
    features['imbalance_1'] = (bid1 - ask1) / (bid1 + ask1 + 1e-10)
    
    # 二档不平衡
    bid2 = df['bid2_qty']
    ask2 = df['ask2_qty']
    features['imbalance_2'] = (bid2 - ask2) / (bid2 + ask2 + 1e-10)
    
    # 综合不平衡（一档+二档）
    total_bid = bid1 + bid2
    total_ask = ask1 + ask2
    features['imbalance_total'] = (total_bid - total_ask) / (total_bid + total_ask + 1e-10)
    
    # === 不平衡的速度 ===
    features['imbalance_1_velocity'] = features['imbalance_1'].diff()
    features['imbalance_2_velocity'] = features['imbalance_2'].diff()
    
    # === 不平衡的加速度 ⭐ ===
    features['imbalance_1_accel'] = features['imbalance_1_velocity'].diff()
    features['imbalance_2_accel'] = features['imbalance_2_velocity'].diff()
    
    return features


def compute_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算一档二档比值特征（聪明钱指标）
    
    思路：如果二档相对一档变强，说明聪明钱在二档埋伏
    """
    features = pd.DataFrame(index=df.index)
    
    bid1 = df['bid1_qty']
    bid2 = df['bid2_qty']
    ask1 = df['ask1_qty']
    ask2 = df['ask2_qty']
    
    # 一档/二档比值
    features['bid_ratio_12'] = bid1 / (bid2 + 1e-10)
    features['ask_ratio_12'] = ask1 / (ask2 + 1e-10)
    
    # 比值的变化（速度）
    features['bid_ratio_12_velocity'] = features['bid_ratio_12'].diff()
    features['ask_ratio_12_velocity'] = features['ask_ratio_12'].diff()
    
    # 买卖比值对比
    # 正值表示买方在一档更集中，负值表示卖方在一档更集中
    features['ratio_diff'] = features['bid_ratio_12'] - features['ask_ratio_12']
    features['ratio_diff_velocity'] = features['ratio_diff'].diff()
    
    return features


def compute_cumulative_features(features: pd.DataFrame, 
                                 windows: List[int] = [5, 15, 30]) -> pd.DataFrame:
    """
    计算累计/滚动特征
    
    累计过去N分钟的变化，捕捉趋势
    """
    cumulative = pd.DataFrame(index=features.index)
    
    # 关键特征的滚动累计
    key_features = [
        'net_velocity_1', 'net_accel_1',
        'imbalance_1', 'imbalance_1_velocity'
    ]
    
    for col in key_features:
        if col not in features.columns:
            continue
        for window in windows:
            # 滚动求和
            cumulative[f'{col}_sum_{window}'] = features[col].rolling(window=window).sum()
            # 滚动均值
            cumulative[f'{col}_mean_{window}'] = features[col].rolling(window=window).mean()
    
    # 加速度的滚动标准差（波动程度）
    if 'net_accel_1' in features.columns:
        for window in windows:
            cumulative[f'accel_std_{window}'] = features['net_accel_1'].rolling(window=window).std()
    
    return cumulative


def compute_spread_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算价差特征
    """
    features = pd.DataFrame(index=df.index)
    
    # 价差
    features['spread'] = df['ask1_px'] - df['bid1_px']
    
    # 价差变化
    features['spread_velocity'] = features['spread'].diff()
    
    # 相对价差（相对于mid-price）
    mid_price = (df['ask1_px'] + df['bid1_px']) / 2
    features['spread_ratio'] = features['spread'] / (mid_price + 1e-10)
    
    return features


def get_feature_importance_ranking() -> dict:
    """
    返回特征重要性排序（基于讨论确定）
    
    用于特征选择和模型解释
    """
    return {
        '⭐⭐⭐ 最重要': [
            'bid1_accel', 'ask1_accel', 'net_accel_1',
            'imbalance_1_accel'
        ],
        '⭐⭐ 重要': [
            'bid1_velocity', 'ask1_velocity', 'net_velocity_1',
            'imbalance_1', 'imbalance_1_velocity'
        ],
        '⭐ 辅助': [
            'bid2_accel', 'ask2_accel', 'net_accel_2',
            'bid_ratio_12_velocity', 'ask_ratio_12_velocity',
            'ratio_diff_velocity'
        ],
        '参考': [
            'imbalance_2', 'spread', 'spread_velocity',
            '累计指标'
        ]
    }


# 保留旧接口兼容性
def compute_orderbook_features(df: pd.DataFrame) -> pd.DataFrame:
    """兼容旧接口"""
    return create_features(df)


def compute_technical_indicators(prices: pd.Series, 
                                  windows: List[int] = [5, 10, 20]) -> pd.DataFrame:
    """
    技术指标（对稳定币意义不大，仅保留接口）
    
    注意：稳定币价格几乎恒定，这些指标信号很弱
    """
    features = pd.DataFrame(index=prices.index)
    
    # 只保留最基础的
    for window in windows:
        features[f'price_std_{window}'] = prices.rolling(window=window).std()
    
    return features


if __name__ == "__main__":
    # 测试代码
    import os
    import sys
    
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, project_root)
    
    from src.data.loader import load_orderbook_data, validate_data
    
    # 加载数据
    data_dir = os.path.join(project_root, "data", "orderbooks")
    df = load_orderbook_data(data_dir, "USDCUSDT")
    df = validate_data(df)
    
    # 创建特征
    features = create_features(df)
    
    print(f"\n特征形状: {features.shape}")
    print(f"\n特征统计:")
    print(features.describe())
    
    # 打印重要性排序
    print("\n特征重要性排序:")
    for level, cols in get_feature_importance_ranking().items():
        print(f"  {level}: {cols}")
