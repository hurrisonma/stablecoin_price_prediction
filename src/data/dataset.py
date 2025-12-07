"""
PyTorch Dataset 模块
用于创建时序滑动窗口数据集
"""

from typing import Tuple, Dict, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class OrderbookDataset(Dataset):
    """
    订单簿时序数据集
    
    将特征数据转换为滑动窗口格式，用于 LSTM/Transformer 等时序模型
    """
    
    def __init__(self, 
                 features: np.ndarray, 
                 labels: np.ndarray, 
                 lookback: int):
        """
        初始化数据集
        
        Args:
            features: 特征数组，形状为 (n_samples, n_features)
            labels: 标签数组，形状为 (n_samples,)
            lookback: 回看窗口大小
        """
        self.features = features
        self.labels = labels
        self.lookback = lookback
        
        # 计算有效样本数量
        self.n_samples = len(labels) - lookback + 1
        
        if self.n_samples <= 0:
            raise ValueError(f"数据量 ({len(labels)}) 小于回看窗口 ({lookback})")
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取单个样本
        
        Args:
            idx: 样本索引
        
        Returns:
            (特征序列, 标签) 元组
            - 特征序列形状: (lookback, n_features)
            - 标签形状: (1,)
        """
        # 获取特征序列
        x = self.features[idx:idx + self.lookback]
        
        # 获取对应的标签 (使用窗口最后一个时间点的标签)
        y = self.labels[idx + self.lookback - 1]
        
        return torch.FloatTensor(x), torch.LongTensor([y])


class OrderbookDatasetWithAugmentation(Dataset):
    """
    带数据增强的订单簿数据集
    
    支持噪声注入和时间扰动等增强方法
    """
    
    def __init__(self, 
                 features: np.ndarray, 
                 labels: np.ndarray, 
                 lookback: int,
                 noise_std: float = 0.01,
                 use_augmentation: bool = True):
        """
        初始化数据集
        
        Args:
            features: 特征数组
            labels: 标签数组
            lookback: 回看窗口大小
            noise_std: 噪声标准差
            use_augmentation: 是否使用数据增强
        """
        self.features = features
        self.labels = labels
        self.lookback = lookback
        self.noise_std = noise_std
        self.use_augmentation = use_augmentation
        
        self.n_samples = len(labels) - lookback + 1
        
        if self.n_samples <= 0:
            raise ValueError(f"数据量 ({len(labels)}) 小于回看窗口 ({lookback})")
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.features[idx:idx + self.lookback].copy()
        y = self.labels[idx + self.lookback - 1]
        
        # 应用数据增强
        if self.use_augmentation and self.training:
            # 添加高斯噪声
            noise = np.random.normal(0, self.noise_std, x.shape)
            x = x + noise
        
        return torch.FloatTensor(x), torch.LongTensor([y])
    
    @property
    def training(self) -> bool:
        """返回是否处于训练模式"""
        return getattr(self, '_training', True)
    
    def train(self) -> None:
        """设置为训练模式"""
        self._training = True
    
    def eval(self) -> None:
        """设置为评估模式"""
        self._training = False


def create_data_loaders(data_dict: Dict,
                        lookback: int,
                        batch_size: int = 64,
                        num_workers: int = 0,
                        use_augmentation: bool = False) -> Dict[str, DataLoader]:
    """
    创建训练、验证和测试数据加载器
    
    Args:
        data_dict: 包含 train/val/test 数据的字典
        lookback: 回看窗口大小
        batch_size: 批次大小
        num_workers: 数据加载线程数
        use_augmentation: 是否使用数据增强
    
    Returns:
        包含 DataLoader 的字典
    """
    loaders = {}
    
    for split in ['train', 'val', 'test']:
        if split not in data_dict:
            continue
        
        features = data_dict[split]['features']
        labels = data_dict[split]['labels']
        
        # 创建数据集
        if use_augmentation and split == 'train':
            dataset = OrderbookDatasetWithAugmentation(
                features, labels, lookback, use_augmentation=True
            )
        else:
            dataset = OrderbookDataset(features, labels, lookback)
        
        # 创建数据加载器
        shuffle = (split == 'train')  # 只有训练集需要打乱
        
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=(split == 'train')  # 训练时丢弃不完整的最后一批
        )
    
    # 打印数据加载器信息
    print("\n数据加载器信息:")
    for split, loader in loaders.items():
        print(f"  {split}: {len(loader.dataset)} 个样本, {len(loader)} 个批次")
    
    return loaders


def get_class_weights(labels: np.ndarray, 
                      method: str = 'balanced') -> torch.Tensor:
    """
    计算类别权重用于处理不平衡问题
    
    Args:
        labels: 标签数组
        method: 权重计算方法 ('balanced' 或 'inverse')
    
    Returns:
        类别权重张量
    """
    classes = np.unique(labels)
    n_classes = len(classes)
    n_samples = len(labels)
    
    class_counts = np.array([np.sum(labels == c) for c in classes])
    
    if method == 'balanced':
        # sklearn 的 balanced 方法
        weights = n_samples / (n_classes * class_counts)
    elif method == 'inverse':
        # 简单的逆频率
        weights = 1.0 / class_counts
        weights = weights / weights.sum() * n_classes  # 归一化
    else:
        raise ValueError(f"不支持的权重方法: {method}")
    
    print(f"\n类别权重 ({method}):")
    for c, w in zip(classes, weights):
        print(f"  类别 {c}: {w:.4f}")
    
    return torch.FloatTensor(weights)


if __name__ == "__main__":
    # 测试代码
    print("测试 OrderbookDataset...")
    
    # 模拟数据
    n_samples = 1000
    n_features = 20
    lookback = 60
    
    features = np.random.randn(n_samples, n_features)
    labels = np.random.randint(0, 3, n_samples)
    
    # 创建数据集
    dataset = OrderbookDataset(features, labels, lookback)
    print(f"数据集大小: {len(dataset)}")
    
    # 获取一个样本
    x, y = dataset[0]
    print(f"特征形状: {x.shape}")  # (lookback, n_features)
    print(f"标签形状: {y.shape}")  # (1,)
    
    # 测试 DataLoader
    data_dict = {
        'train': {'features': features[:700], 'labels': labels[:700]},
        'val': {'features': features[700:850], 'labels': labels[700:850]},
        'test': {'features': features[850:], 'labels': labels[850:]}
    }
    
    loaders = create_data_loaders(data_dict, lookback=lookback, batch_size=32)
    
    # 迭代一个批次
    for batch_x, batch_y in loaders['train']:
        print(f"\n训练批次:")
        print(f"  特征形状: {batch_x.shape}")  # (batch_size, lookback, n_features)
        print(f"  标签形状: {batch_y.shape}")  # (batch_size, 1)
        break
    
    # 测试类别权重
    weights = get_class_weights(labels, method='balanced')
