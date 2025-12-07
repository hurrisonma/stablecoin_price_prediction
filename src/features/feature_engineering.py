"""
特征工程模块
从订单簿数据中提取各类特征
"""

import pandas as pd
import numpy as np
from typing import List, Optional


def create_features(df: pd.DataFrame, 
                    include_technical: bool = True,
                    rolling_windows: List[int] = [5, 10, 20]) -> pd.DataFrame:
    """
    创建所有特征
    
    Args:
        df: 原始订单簿数据 DataFrame
        include_technical: 是否包含技术指标
        rolling_windows: 滚动窗口大小列表
    
    Returns:
        特征 DataFrame
    """
    features = pd.DataFrame(index=df.index)
    
    # 1. 基础价格特征
    price_features = compute_price_features(df)
    features = pd.concat([features, price_features], axis=1)
    
    # 2. 订单簿特征
    orderbook_features = compute_orderbook_features(df)
    features = pd.concat([features, orderbook_features], axis=1)
    
    # 3. 深度不平衡特征
    imbalance_features = compute_imbalance_features(df)
    features = pd.concat([features, imbalance_features], axis=1)
    
    # 4. 技术指标
    if include_technical:
        mid_price = (df['bid1_px'] + df['ask1_px']) / 2
        tech_features = compute_technical_indicators(mid_price, rolling_windows)
        features = pd.concat([features, tech_features], axis=1)
    
    # 5. 时序变化特征
    change_features = compute_change_features(df, rolling_windows)
    features = pd.concat([features, change_features], axis=1)
    
    # 处理无穷大值
    features = features.replace([np.inf, -np.inf], np.nan)
    
    # 填充 NaN (使用前向填充)
    features = features.fillna(method='ffill')
    features = features.fillna(method='bfill')
    features = features.fillna(0)  # 如果还有 NaN，用0填充
    
    print(f"创建了 {len(features.columns)} 个特征")
    print(f"特征列表: {list(features.columns)}")
    
    return features


def compute_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算基础价格特征
    
    Args:
        df: 订单簿数据
    
    Returns:
        价格特征 DataFrame
    """
    features = pd.DataFrame(index=df.index)
    
    # 中间价格
    features['mid_price'] = (df['bid1_px'] + df['ask1_px']) / 2
    
    # 买卖价差
    features['spread'] = df['ask1_px'] - df['bid1_px']
    
    # 价差比率
    features['spread_ratio'] = features['spread'] / features['mid_price']
    
    # 各档位价格与中间价格的距离
    for i in range(1, 6):
        features[f'bid{i}_dist'] = features['mid_price'] - df[f'bid{i}_px']
        features[f'ask{i}_dist'] = df[f'ask{i}_px'] - features['mid_price']
    
    return features


def compute_orderbook_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算订单簿特征
    
    Args:
        df: 订单簿数据
    
    Returns:
        订单簿特征 DataFrame
    """
    features = pd.DataFrame(index=df.index)
    
    # 总买卖量
    bid_qty_cols = [f'bid{i}_qty' for i in range(1, 6)]
    ask_qty_cols = [f'ask{i}_qty' for i in range(1, 6)]
    
    features['total_bid_qty'] = df[bid_qty_cols].sum(axis=1)
    features['total_ask_qty'] = df[ask_qty_cols].sum(axis=1)
    features['total_qty'] = features['total_bid_qty'] + features['total_ask_qty']
    
    # 买卖量比率
    features['bid_ask_qty_ratio'] = (features['total_bid_qty'] / 
                                      (features['total_ask_qty'] + 1e-10))
    
    # 加权深度 (价格 × 数量)
    bid_depth = 0
    ask_depth = 0
    for i in range(1, 6):
        bid_depth += df[f'bid{i}_px'] * df[f'bid{i}_qty']
        ask_depth += df[f'ask{i}_px'] * df[f'ask{i}_qty']
    
    features['bid_depth'] = bid_depth
    features['ask_depth'] = ask_depth
    features['depth_ratio'] = bid_depth / (ask_depth + 1e-10)
    
    # 深度压力差
    features['depth_pressure'] = bid_depth - ask_depth
    
    # 各档位数量占比
    for i in range(1, 6):
        features[f'bid{i}_qty_ratio'] = df[f'bid{i}_qty'] / (features['total_bid_qty'] + 1e-10)
        features[f'ask{i}_qty_ratio'] = df[f'ask{i}_qty'] / (features['total_ask_qty'] + 1e-10)
    
    return features


def compute_imbalance_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算深度不平衡特征
    
    Args:
        df: 订单簿数据
    
    Returns:
        不平衡特征 DataFrame
    """
    features = pd.DataFrame(index=df.index)
    
    # 各档位不平衡
    for i in range(1, 6):
        bid_qty = df[f'bid{i}_qty']
        ask_qty = df[f'ask{i}_qty']
        
        # 简单不平衡: (bid - ask) / (bid + ask)
        features[f'imbalance_{i}'] = ((bid_qty - ask_qty) / 
                                       (bid_qty + ask_qty + 1e-10))
    
    # 加权不平衡 (权重随档位递减)
    weights = np.array([1.0, 0.8, 0.6, 0.4, 0.2])
    weighted_bid = 0
    weighted_ask = 0
    
    for i, w in enumerate(weights, 1):
        weighted_bid += w * df[f'bid{i}_qty']
        weighted_ask += w * df[f'ask{i}_qty']
    
    features['weighted_imbalance'] = ((weighted_bid - weighted_ask) / 
                                       (weighted_bid + weighted_ask + 1e-10))
    
    # 累计不平衡
    cumulative_imbalance = 0
    for i in range(1, 6):
        bid_qty = df[f'bid{i}_qty']
        ask_qty = df[f'ask{i}_qty']
        cumulative_imbalance += (bid_qty - ask_qty)
        features[f'cumulative_imbalance_{i}'] = cumulative_imbalance
    
    return features


def compute_technical_indicators(prices: pd.Series, 
                                  windows: List[int] = [5, 10, 20]) -> pd.DataFrame:
    """
    计算技术指标
    
    Args:
        prices: 价格序列
        windows: 滚动窗口大小列表
    
    Returns:
        技术指标 DataFrame
    """
    features = pd.DataFrame(index=prices.index)
    
    for window in windows:
        # 移动平均
        features[f'ma_{window}'] = prices.rolling(window=window).mean()
        
        # 价格与MA的距离
        features[f'price_ma_{window}_dist'] = prices - features[f'ma_{window}']
        
        # 指数移动平均
        features[f'ema_{window}'] = prices.ewm(span=window, adjust=False).mean()
        
        # 波动率 (滚动标准差)
        features[f'volatility_{window}'] = prices.rolling(window=window).std()
        
        # 最高价/最低价
        features[f'rolling_max_{window}'] = prices.rolling(window=window).max()
        features[f'rolling_min_{window}'] = prices.rolling(window=window).min()
        
        # 价格位置 (相对于最高最低价)
        features[f'price_position_{window}'] = (
            (prices - features[f'rolling_min_{window}']) / 
            (features[f'rolling_max_{window}'] - features[f'rolling_min_{window}'] + 1e-10)
        )
    
    # RSI (相对强弱指标)
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    features['rsi'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    bb_window = 20
    bb_ma = prices.rolling(window=bb_window).mean()
    bb_std = prices.rolling(window=bb_window).std()
    
    features['bb_upper'] = bb_ma + 2 * bb_std
    features['bb_lower'] = bb_ma - 2 * bb_std
    features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / (bb_ma + 1e-10)
    features['bb_position'] = (prices - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'] + 1e-10)
    
    return features


def compute_change_features(df: pd.DataFrame, 
                            windows: List[int] = [1, 5, 10]) -> pd.DataFrame:
    """
    计算时序变化特征
    
    Args:
        df: 订单簿数据
        windows: 变化窗口列表
    
    Returns:
        变化特征 DataFrame
    """
    features = pd.DataFrame(index=df.index)
    
    # 中间价格
    mid_price = (df['bid1_px'] + df['ask1_px']) / 2
    
    # 价格收益率
    for window in windows:
        features[f'return_{window}'] = mid_price.pct_change(periods=window)
        features[f'log_return_{window}'] = np.log(mid_price / mid_price.shift(window))
    
    # 总数量
    bid_qty_cols = [f'bid{i}_qty' for i in range(1, 6)]
    ask_qty_cols = [f'ask{i}_qty' for i in range(1, 6)]
    total_qty = df[bid_qty_cols + ask_qty_cols].sum(axis=1)
    
    # 数量变化
    for window in windows:
        features[f'qty_change_{window}'] = total_qty.pct_change(periods=window)
    
    # 价差变化
    spread = df['ask1_px'] - df['bid1_px']
    for window in windows:
        features[f'spread_change_{window}'] = spread.diff(periods=window)
    
    # 不平衡变化
    imbalance = (df['bid1_qty'] - df['ask1_qty']) / (df['bid1_qty'] + df['ask1_qty'] + 1e-10)
    for window in windows:
        features[f'imbalance_change_{window}'] = imbalance.diff(periods=window)
    
    return features


def select_features(features: pd.DataFrame, 
                    method: str = 'variance',
                    threshold: float = 0.01) -> List[str]:
    """
    特征选择
    
    Args:
        features: 特征 DataFrame
        method: 选择方法 ('variance' 或 'correlation')
        threshold: 阈值
    
    Returns:
        选中的特征名列表
    """
    selected = []
    
    if method == 'variance':
        # 基于方差的选择
        variances = features.var()
        selected = variances[variances > threshold].index.tolist()
    
    elif method == 'correlation':
        # 移除高度相关的特征
        corr_matrix = features.corr().abs()
        upper_tri = np.triu(np.ones(corr_matrix.shape), k=1)
        upper_tri_df = pd.DataFrame(upper_tri, 
                                    columns=corr_matrix.columns,
                                    index=corr_matrix.index)
        
        # 找出高度相关的特征对
        to_drop = []
        for col in corr_matrix.columns:
            if col not in to_drop:
                # 找出与当前特征高度相关的其他特征
                high_corr = corr_matrix.index[
                    (corr_matrix[col] > threshold) & 
                    (upper_tri_df[col] == 1)
                ].tolist()
                to_drop.extend(high_corr)
        
        selected = [col for col in features.columns if col not in to_drop]
    
    print(f"特征选择 ({method}): {len(features.columns)} -> {len(selected)} 个特征")
    
    return selected


if __name__ == "__main__":
    # 测试代码
    import os
    import sys
    
    # 添加项目根目录到路径
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
    
    # 检查缺失值
    missing = features.isnull().sum()
    if missing.any():
        print(f"\n缺失值:")
        print(missing[missing > 0])
