"""
模型训练器
负责训练循环、评估、早停和模型保存
"""

import os
import time
from typing import Dict, Optional, Tuple, List
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False


class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001, mode: str = 'max'):
        """
        初始化早停
        
        Args:
            patience: 容忍的轮数
            min_delta: 最小改进阈值
            mode: 'max' 表示指标越大越好，'min' 表示越小越好
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        """
        检查是否应该停止
        
        Args:
            score: 当前分数
        
        Returns:
            是否应该停止
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


class FocalLoss(nn.Module):
    """
    Focal Loss
    用于处理类别不平衡问题，专注于难分类样本
    """
    
    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None):
        """
        初始化 Focal Loss
        
        Args:
            gamma: 聚焦参数，越大越关注难样本
            alpha: 类别权重
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        return focal_loss.mean()


class Trainer:
    """
    模型训练器
    """
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 config: Dict,
                 device: Optional[str] = None):
        """
        初始化训练器
        
        Args:
            model: PyTorch 模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            config: 训练配置
            device: 计算设备
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # 设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        
        # 优化器
        lr = config.get('learning_rate', 0.001)
        weight_decay = config.get('weight_decay', 0.01)
        
        self.optimizer = AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # 学习率调度器
        scheduler_type = config.get('scheduler', 'plateau')
        if scheduler_type == 'plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=5
            )
        elif scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=config.get('epochs', 100)
            )
        else:
            self.scheduler = None
        
        # 损失函数
        class_weights = config.get('class_weights', None)
        loss_type = config.get('loss_type', 'cross_entropy')
        
        if loss_type == 'focal':
            self.criterion = FocalLoss(gamma=2.0, alpha=class_weights)
        else:
            if class_weights is not None:
                self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
            else:
                self.criterion = nn.CrossEntropyLoss()
        
        # 早停
        patience = config.get('early_stopping_patience', 10)
        self.early_stopping = EarlyStopping(patience=patience, mode='max')
        
        # TensorBoard
        if HAS_TENSORBOARD:
            log_dir = config.get('log_dir', 'outputs/logs')
            self.writer = SummaryWriter(log_dir)
        else:
            self.writer = None
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': []
        }
        
        # 最佳模型
        self.best_val_acc = 0.0
        self.best_model_state = None
    
    def train_epoch(self) -> Tuple[float, float]:
        """
        训练一个 epoch
        
        Returns:
            (平均损失, 准确率)
        """
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training', leave=False)
        
        for batch_x, batch_y in pbar:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.squeeze().to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(batch_x)
            loss = self.criterion(outputs, batch_y)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item() * batch_x.size(0)
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        avg_loss = total_loss / total
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def validate(self, data_loader: Optional[DataLoader] = None) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """
        验证模型
        
        Args:
            data_loader: 数据加载器，默认使用验证集
        
        Returns:
            (平均损失, 准确率, 真实标签, 预测标签)
        """
        if data_loader is None:
            data_loader = self.val_loader
        
        self.model.eval()
        
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.squeeze().to(self.device)
                
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                
                total_loss += loss.item() * batch_x.size(0)
                _, predicted = outputs.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        avg_loss = total_loss / len(all_labels)
        accuracy = (all_preds == all_labels).mean()
        
        return avg_loss, accuracy, all_labels, all_preds
    
    def train(self, epochs: Optional[int] = None) -> Dict:
        """
        完整训练流程
        
        Args:
            epochs: 训练轮数
        
        Returns:
            训练历史
        """
        if epochs is None:
            epochs = self.config.get('epochs', 100)
        
        print(f"\n开始训练...")
        print(f"设备: {self.device}")
        print(f"训练集: {len(self.train_loader.dataset)} 样本")
        print(f"验证集: {len(self.val_loader.dataset)} 样本")
        print(f"Epochs: {epochs}")
        print("-" * 50)
        
        start_time = time.time()
        
        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            
            # 训练
            train_loss, train_acc = self.train_epoch()
            
            # 验证
            val_loss, val_acc, _, _ = self.validate()
            
            # 学习率调度
            current_lr = self.optimizer.param_groups[0]['lr']
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_acc)
                else:
                    self.scheduler.step()
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rate'].append(current_lr)
            
            # TensorBoard 日志
            if self.writer is not None:
                self.writer.add_scalar('Loss/train', train_loss, epoch)
                self.writer.add_scalar('Loss/val', val_loss, epoch)
                self.writer.add_scalar('Accuracy/train', train_acc, epoch)
                self.writer.add_scalar('Accuracy/val', val_acc, epoch)
                self.writer.add_scalar('LearningRate', current_lr, epoch)
            
            # 保存最佳模型
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_model_state = self.model.state_dict().copy()
                best_marker = " *"
            else:
                best_marker = ""
            
            # 打印进度
            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f}, Acc: {train_acc*100:.2f}% | "
                  f"Val Loss: {val_loss:.4f}, Acc: {val_acc*100:.2f}%{best_marker} | "
                  f"LR: {current_lr:.2e} | Time: {epoch_time:.1f}s")
            
            # 早停检查
            if self.early_stopping(val_acc):
                print(f"\n早停触发！在 Epoch {epoch} 停止训练")
                break
        
        total_time = time.time() - start_time
        print("-" * 50)
        print(f"训练完成！总用时: {total_time/60:.1f} 分钟")
        print(f"最佳验证准确率: {self.best_val_acc*100:.2f}%")
        
        # 恢复最佳模型
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        return self.history
    
    def save_checkpoint(self, path: str, include_optimizer: bool = True):
        """
        保存检查点
        
        Args:
            path: 保存路径
            include_optimizer: 是否包含优化器状态
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'best_val_acc': self.best_val_acc,
            'history': self.history
        }
        
        if include_optimizer:
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
            if self.scheduler is not None:
                checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(checkpoint, path)
        print(f"模型已保存到: {path}")
    
    def load_checkpoint(self, path: str, load_optimizer: bool = True):
        """
        加载检查点
        
        Args:
            path: 检查点路径
            load_optimizer: 是否加载优化器状态
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        self.history = checkpoint.get('history', self.history)
        
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if load_optimizer and 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"模型已从 {path} 加载")
        print(f"最佳验证准确率: {self.best_val_acc*100:.2f}%")
    
    def predict(self, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        预测
        
        Args:
            data_loader: 数据加载器
        
        Returns:
            (预测标签, 预测概率)
        """
        self.model.eval()
        
        all_preds = []
        all_probs = []
        
        with torch.no_grad():
            for batch_x, _ in data_loader:
                batch_x = batch_x.to(self.device)
                outputs = self.model(batch_x)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        return np.array(all_preds), np.array(all_probs)


if __name__ == "__main__":
    # 测试代码
    print("测试训练器...")
    
    from lstm_model import LSTMClassifier
    from torch.utils.data import TensorDataset, DataLoader
    
    # 创建模拟数据
    n_samples = 1000
    seq_len = 60
    n_features = 50
    n_classes = 3
    
    X = torch.randn(n_samples, seq_len, n_features)
    y = torch.randint(0, n_classes, (n_samples,))
    
    # 划分数据
    train_size = int(0.7 * n_samples)
    val_size = int(0.15 * n_samples)
    
    train_dataset = TensorDataset(X[:train_size], y[:train_size])
    val_dataset = TensorDataset(X[train_size:train_size+val_size], y[train_size:train_size+val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # 创建模型
    model = LSTMClassifier(
        input_size=n_features,
        hidden_size=64,
        num_layers=1,
        num_classes=n_classes
    )
    
    # 训练配置
    config = {
        'learning_rate': 0.001,
        'epochs': 5,
        'early_stopping_patience': 3
    }
    
    # 创建训练器
    trainer = Trainer(model, train_loader, val_loader, config)
    
    # 训练
    history = trainer.train(epochs=5)
    
    print("\n训练历史:")
    print(f"最终训练损失: {history['train_loss'][-1]:.4f}")
    print(f"最终验证准确率: {history['val_acc'][-1]*100:.2f}%")
