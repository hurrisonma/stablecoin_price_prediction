"""
数据加载模块
负责加载和验证订单簿数据
"""

import os
import glob
from typing import Optional
import pandas as pd
import numpy as np
from tqdm import tqdm


def load_orderbook_data(data_dir: str, symbol: str, 
                        start_date: Optional[str] = None,
                        end_date: Optional[str] = None) -> pd.DataFrame:
    """
    加载并合并指定品种的所有订单簿数据
    
    Args:
        data_dir: 数据目录路径
        symbol: 交易对名称 (如 "USDCUSDT")
        start_date: 起始日期 (格式: "YYYY-MM-DD")，可选
        end_date: 结束日期 (格式: "YYYY-MM-DD")，可选
    
    Returns:
        合并后的 DataFrame
    """
    # 查找所有匹配的文件
    pattern = os.path.join(data_dir, f"{symbol}_depth_*.csv")
    files = sorted(glob.glob(pattern))
    
    if not files:
        raise FileNotFoundError(f"未找到符合条件的数据文件: {pattern}")
    
    print(f"找到 {len(files)} 个 {symbol} 数据文件")
    
    # 过滤日期范围
    if start_date or end_date:
        filtered_files = []
        for f in files:
            # 从文件名提取日期
            filename = os.path.basename(f)
            date_str = filename.split("_depth_")[1].replace(".csv", "")
            
            if start_date and date_str < start_date:
                continue
            if end_date and date_str > end_date:
                continue
            filtered_files.append(f)
        files = filtered_files
        print(f"日期过滤后剩余 {len(files)} 个文件")
    
    # 加载并合并所有文件
    dfs = []
    for file_path in tqdm(files, desc="加载数据文件"):
        try:
            df = pd.read_csv(file_path)
            dfs.append(df)
        except Exception as e:
            print(f"警告: 无法加载文件 {file_path}: {e}")
            continue
    
    if not dfs:
        raise ValueError("没有成功加载任何数据文件")
    
    # 合并所有数据
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # 按时间戳排序
    combined_df = combined_df.sort_values('ts_ms').reset_index(drop=True)
    
    print(f"数据加载完成: {len(combined_df)} 条记录")
    print(f"时间范围: {combined_df['ts_utc8'].iloc[0]} ~ {combined_df['ts_utc8'].iloc[-1]}")
    
    return combined_df


def validate_data(df: pd.DataFrame, 
                  remove_duplicates: bool = True,
                  fill_missing: bool = True) -> pd.DataFrame:
    """
    数据验证和清洗
    
    Args:
        df: 输入的 DataFrame
        remove_duplicates: 是否移除重复记录
        fill_missing: 是否填充缺失值
    
    Returns:
        清洗后的 DataFrame
    """
    original_len = len(df)
    
    # 1. 检查必要字段
    required_columns = [
        'ts_ms', 'symbol',
        'bid1_px', 'bid1_qty', 'bid2_px', 'bid2_qty', 'bid3_px', 'bid3_qty',
        'bid4_px', 'bid4_qty', 'bid5_px', 'bid5_qty',
        'ask1_px', 'ask1_qty', 'ask2_px', 'ask2_qty', 'ask3_px', 'ask3_qty',
        'ask4_px', 'ask4_qty', 'ask5_px', 'ask5_qty'
    ]
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"缺少必要字段: {missing_columns}")
    
    # 2. 移除重复记录
    if remove_duplicates:
        df = df.drop_duplicates(subset=['ts_ms'], keep='first')
        duplicates_removed = original_len - len(df)
        if duplicates_removed > 0:
            print(f"移除 {duplicates_removed} 条重复记录")
    
    # 3. 检查缺失值
    missing_counts = df[required_columns].isnull().sum()
    if missing_counts.any():
        print("检测到缺失值:")
        print(missing_counts[missing_counts > 0])
        
        if fill_missing:
            # 使用前向填充
            df = df.fillna(method='ffill')
            # 如果开头有缺失，使用后向填充
            df = df.fillna(method='bfill')
            print("已使用前向/后向填充处理缺失值")
    
    # 4. 检查价格异常值
    price_columns = [col for col in df.columns if '_px' in col]
    for col in price_columns:
        # 检查是否有非正数价格
        invalid_prices = (df[col] <= 0).sum()
        if invalid_prices > 0:
            print(f"警告: {col} 有 {invalid_prices} 个非正数价格")
            # 用前一个有效值填充
            df[col] = df[col].replace(0, np.nan).fillna(method='ffill')
    
    # 5. 检查数量异常值
    qty_columns = [col for col in df.columns if '_qty' in col]
    for col in qty_columns:
        # 检查是否有负数数量
        invalid_qty = (df[col] < 0).sum()
        if invalid_qty > 0:
            print(f"警告: {col} 有 {invalid_qty} 个负数数量")
            df[col] = df[col].clip(lower=0)
    
    # 6. 检查时间连续性
    df['time_diff'] = df['ts_ms'].diff()
    expected_interval = 60000  # 1分钟 = 60000毫秒
    
    # 允许一定误差
    irregular_intervals = ((df['time_diff'] < expected_interval - 1000) | 
                          (df['time_diff'] > expected_interval + 1000))
    irregular_count = irregular_intervals.sum()
    
    if irregular_count > 0:
        print(f"注意: 发现 {irregular_count} 个不规则时间间隔")
    
    # 删除辅助列
    df = df.drop(columns=['time_diff'])
    
    print(f"数据验证完成: {len(df)} 条有效记录 (原始 {original_len} 条)")
    
    return df.reset_index(drop=True)


def get_data_statistics(df: pd.DataFrame) -> dict:
    """
    获取数据统计信息
    
    Args:
        df: 数据 DataFrame
    
    Returns:
        统计信息字典
    """
    stats = {
        'total_records': len(df),
        'time_range': {
            'start': df['ts_utc8'].iloc[0] if 'ts_utc8' in df.columns else None,
            'end': df['ts_utc8'].iloc[-1] if 'ts_utc8' in df.columns else None
        },
        'bid1_price': {
            'mean': df['bid1_px'].mean(),
            'std': df['bid1_px'].std(),
            'min': df['bid1_px'].min(),
            'max': df['bid1_px'].max()
        },
        'ask1_price': {
            'mean': df['ask1_px'].mean(),
            'std': df['ask1_px'].std(),
            'min': df['ask1_px'].min(),
            'max': df['ask1_px'].max()
        },
        'spread': {
            'mean': (df['ask1_px'] - df['bid1_px']).mean(),
            'std': (df['ask1_px'] - df['bid1_px']).std()
        }
    }
    
    return stats


if __name__ == "__main__":
    # 测试代码
    import os
    
    # 获取项目根目录
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(project_root, "data", "orderbooks")
    
    # 加载数据
    df = load_orderbook_data(data_dir, "USDCUSDT")
    
    # 验证数据
    df = validate_data(df)
    
    # 打印统计信息
    stats = get_data_statistics(df)
    print("\n数据统计:")
    print(f"总记录数: {stats['total_records']}")
    print(f"时间范围: {stats['time_range']['start']} ~ {stats['time_range']['end']}")
    print(f"Bid1 价格均值: {stats['bid1_price']['mean']:.6f}")
    print(f"Ask1 价格均值: {stats['ask1_price']['mean']:.6f}")
    print(f"Spread 均值: {stats['spread']['mean']:.6f}")
