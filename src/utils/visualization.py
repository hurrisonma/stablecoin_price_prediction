"""
可视化模块
绘制训练历史、混淆矩阵等图表
"""

import os
from typing import Dict, List, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def plot_training_history(history: Dict, 
                          save_path: Optional[str] = None,
                          figsize: tuple = (12, 4)):
    """
    绘制训练历史曲线
    
    Args:
        history: 训练历史字典，包含 train_loss, val_loss, train_acc, val_acc
        save_path: 保存路径
        figsize: 图片尺寸
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # 损失曲线
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curve')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 准确率曲线
    axes[1].plot(epochs, [acc * 100 for acc in history['train_acc']], 'b-', label='Train Acc')
    axes[1].plot(epochs, [acc * 100 for acc in history['val_acc']], 'r-', label='Val Acc')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Accuracy Curve')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 学习率曲线
    if 'learning_rate' in history:
        axes[2].plot(epochs, history['learning_rate'], 'g-')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Learning Rate')
        axes[2].set_title('Learning Rate Schedule')
        axes[2].set_yscale('log')
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"训练历史图已保存到: {save_path}")
    
    plt.show()
    plt.close()


def plot_confusion_matrix(y_true: np.ndarray, 
                          y_pred: np.ndarray,
                          class_names: List[str] = ['下降', '不变', '上升'],
                          normalize: bool = True,
                          save_path: Optional[str] = None,
                          figsize: tuple = (8, 6)):
    """
    绘制混淆矩阵热力图
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        class_names: 类别名称
        normalize: 是否归一化
        save_path: 保存路径
        figsize: 图片尺寸
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                ax=ax)
    
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(title)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"混淆矩阵图已保存到: {save_path}")
    
    plt.show()
    plt.close()


def plot_class_distribution(labels: np.ndarray,
                            class_names: List[str] = ['下降', '不变', '上升'],
                            title: str = 'Class Distribution',
                            save_path: Optional[str] = None,
                            figsize: tuple = (8, 5)):
    """
    绘制类别分布柱状图
    
    Args:
        labels: 标签数组
        class_names: 类别名称
        title: 图标题
        save_path: 保存路径
        figsize: 图片尺寸
    """
    unique, counts = np.unique(labels, return_counts=True)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
    bars = ax.bar(range(len(class_names)), [0] * len(class_names), color=colors)
    
    # 填充实际数据
    for i, (u, c) in enumerate(zip(unique, counts)):
        if int(u) < len(class_names):
            bars[int(u)].set_height(c)
    
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_xlabel('Class')
    ax.set_ylabel('Count')
    ax.set_title(title)
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.annotate(f'{int(height)}\n({height/len(labels)*100:.1f}%)',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"类别分布图已保存到: {save_path}")
    
    plt.show()
    plt.close()


def plot_prediction_confidence(y_pred_probs: np.ndarray,
                                y_true: np.ndarray,
                                class_names: List[str] = ['下降', '不变', '上升'],
                                save_path: Optional[str] = None,
                                figsize: tuple = (10, 4)):
    """
    绘制预测置信度分布
    
    Args:
        y_pred_probs: 预测概率矩阵 (n_samples, n_classes)
        y_true: 真实标签
        class_names: 类别名称
        save_path: 保存路径
        figsize: 图片尺寸
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # 正确预测和错误预测的置信度分布
    y_pred = y_pred_probs.argmax(axis=1)
    max_probs = y_pred_probs.max(axis=1)
    
    correct_mask = y_pred == y_true
    
    axes[0].hist(max_probs[correct_mask], bins=20, alpha=0.7, 
                 label='Correct', color='green', density=True)
    axes[0].hist(max_probs[~correct_mask], bins=20, alpha=0.7, 
                 label='Incorrect', color='red', density=True)
    axes[0].set_xlabel('Max Prediction Probability')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Confidence Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 各类别的平均预测概率
    avg_probs_per_class = []
    for i in range(len(class_names)):
        mask = y_true == i
        if mask.any():
            avg_probs_per_class.append(y_pred_probs[mask].mean(axis=0))
        else:
            avg_probs_per_class.append(np.zeros(len(class_names)))
    
    x = np.arange(len(class_names))
    width = 0.25
    
    for i, name in enumerate(class_names):
        if i < len(avg_probs_per_class):
            axes[1].bar(x + i*width, avg_probs_per_class[i], 
                       width, label=f'Actual: {name}')
    
    axes[1].set_xticks(x + width)
    axes[1].set_xticklabels(class_names)
    axes[1].set_xlabel('Predicted Class')
    axes[1].set_ylabel('Average Probability')
    axes[1].set_title('Average Prediction Probabilities')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"置信度分布图已保存到: {save_path}")
    
    plt.show()
    plt.close()


def plot_feature_importance(feature_names: List[str],
                            importance_scores: np.ndarray,
                            top_k: int = 20,
                            save_path: Optional[str] = None,
                            figsize: tuple = (10, 8)):
    """
    绘制特征重要性
    
    Args:
        feature_names: 特征名称列表
        importance_scores: 重要性分数
        top_k: 显示前K个特征
        save_path: 保存路径
        figsize: 图片尺寸
    """
    # 排序
    indices = np.argsort(importance_scores)[::-1][:top_k]
    top_features = [feature_names[i] for i in indices]
    top_scores = importance_scores[indices]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    y_pos = np.arange(len(top_features))
    ax.barh(y_pos, top_scores, color='steelblue')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features)
    ax.invert_yaxis()
    ax.set_xlabel('Importance Score')
    ax.set_title(f'Top {top_k} Feature Importance')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"特征重要性图已保存到: {save_path}")
    
    plt.show()
    plt.close()


def plot_price_with_predictions(timestamps: np.ndarray,
                                 prices: np.ndarray,
                                 y_true: np.ndarray,
                                 y_pred: np.ndarray,
                                 start_idx: int = 0,
                                 end_idx: int = 500,
                                 save_path: Optional[str] = None,
                                 figsize: tuple = (14, 6)):
    """
    绘制价格曲线及预测结果
    
    Args:
        timestamps: 时间戳
        prices: 价格数据
        y_true: 真实标签
        y_pred: 预测标签
        start_idx: 起始索引
        end_idx: 结束索引
        save_path: 保存路径
        figsize: 图片尺寸
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    # 截取数据
    ts = timestamps[start_idx:end_idx]
    px = prices[start_idx:end_idx]
    yt = y_true[start_idx:end_idx]
    yp = y_pred[start_idx:end_idx]
    
    # 价格曲线
    ax1.plot(range(len(px)), px, 'b-', linewidth=0.8)
    ax1.set_ylabel('Price')
    ax1.set_title('Price with Predictions')
    ax1.grid(True, alpha=0.3)
    
    # 预测结果
    colors = {0: 'red', 1: 'gray', 2: 'green'}
    labels = {0: 'Down', 1: 'Neutral', 2: 'Up'}
    
    for i in range(len(yp)):
        if yp[i] == yt[i]:
            ax2.scatter(i, yp[i], c=colors[yp[i]], s=10, alpha=0.6)
        else:
            ax2.scatter(i, yp[i], c='black', s=10, marker='x', alpha=0.8)
    
    ax2.set_ylabel('Prediction')
    ax2.set_xlabel('Time Index')
    ax2.set_yticks([0, 1, 2])
    ax2.set_yticklabels(['Down', 'Neutral', 'Up'])
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"价格预测图已保存到: {save_path}")
    
    plt.show()
    plt.close()


if __name__ == "__main__":
    # 测试代码
    print("测试可视化模块...")
    
    # 模拟数据
    np.random.seed(42)
    
    # 训练历史
    epochs = 50
    history = {
        'train_loss': np.exp(-np.linspace(0, 2, epochs)) + 0.3 + np.random.randn(epochs) * 0.05,
        'val_loss': np.exp(-np.linspace(0, 1.5, epochs)) + 0.4 + np.random.randn(epochs) * 0.08,
        'train_acc': 1 - np.exp(-np.linspace(0, 2, epochs)) * 0.5 + np.random.randn(epochs) * 0.02,
        'val_acc': 1 - np.exp(-np.linspace(0, 1.5, epochs)) * 0.6 + np.random.randn(epochs) * 0.03,
        'learning_rate': [0.001 * (0.95 ** i) for i in range(epochs)]
    }
    
    print("绘制训练历史...")
    plot_training_history(history)
    
    # 混淆矩阵
    y_true = np.random.randint(0, 3, 500)
    y_pred = np.random.randint(0, 3, 500)
    correct_indices = np.random.choice(500, 200, replace=False)
    y_pred[correct_indices] = y_true[correct_indices]
    
    print("\n绘制混淆矩阵...")
    plot_confusion_matrix(y_true, y_pred, normalize=True)
    
    print("\n绘制类别分布...")
    plot_class_distribution(y_true)
