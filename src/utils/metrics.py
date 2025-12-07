"""
评估指标模块
计算分类性能指标
"""

from typing import Dict, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def calculate_metrics(y_true: np.ndarray, 
                      y_pred: np.ndarray,
                      class_names: list = ['下降', '不变', '上升']) -> Dict:
    """
    计算分类评估指标
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        class_names: 类别名称
    
    Returns:
        评估指标字典
    """
    metrics = {}
    
    # 总体指标
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # 各类别指标
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    for i, name in enumerate(class_names):
        if i < len(precision_per_class):
            metrics[f'precision_{name}'] = precision_per_class[i]
            metrics[f'recall_{name}'] = recall_per_class[i]
            metrics[f'f1_{name}'] = f1_per_class[i]
    
    # 混淆矩阵
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    
    return metrics


def print_classification_report(y_true: np.ndarray, 
                                 y_pred: np.ndarray,
                                 class_names: list = ['下降', '不变', '上升']):
    """
    打印分类报告
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        class_names: 类别名称
    """
    print("\n" + "="*60)
    print("分类评估报告")
    print("="*60)
    
    metrics = calculate_metrics(y_true, y_pred, class_names)
    
    # 总体指标
    print("\n【总体指标】")
    print(f"  准确率 (Accuracy):     {metrics['accuracy']*100:.2f}%")
    print(f"  宏平均 F1 (Macro F1):  {metrics['f1_macro']:.4f}")
    print(f"  加权 F1 (Weighted F1): {metrics['f1_weighted']:.4f}")
    
    # 各类别指标
    print("\n【各类别指标】")
    print(f"{'类别':<10} {'Precision':>12} {'Recall':>12} {'F1-Score':>12}")
    print("-" * 48)
    for name in class_names:
        precision = metrics.get(f'precision_{name}', 0)
        recall = metrics.get(f'recall_{name}', 0)
        f1 = metrics.get(f'f1_{name}', 0)
        print(f"{name:<10} {precision:>12.4f} {recall:>12.4f} {f1:>12.4f}")
    
    # 混淆矩阵
    print("\n【混淆矩阵】")
    cm = metrics['confusion_matrix']
    print(f"{'预测':<8}", end='')
    for name in class_names:
        print(f"{name:>8}", end='')
    print("\n" + "实际")
    
    for i, name in enumerate(class_names):
        if i < len(cm):
            print(f"{name:<8}", end='')
            for j in range(len(cm[i])):
                print(f"{cm[i][j]:>8}", end='')
            print()
    
    print("="*60)
    
    return metrics


def get_detailed_report(y_true: np.ndarray, 
                        y_pred: np.ndarray,
                        class_names: list = ['下降', '不变', '上升']) -> str:
    """
    获取详细的分类报告字符串
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        class_names: 类别名称
    
    Returns:
        报告字符串
    """
    return classification_report(y_true, y_pred, target_names=class_names, zero_division=0)


def calculate_class_distribution(labels: np.ndarray,
                                  class_names: list = ['下降', '不变', '上升']) -> Dict:
    """
    计算类别分布
    
    Args:
        labels: 标签数组
        class_names: 类别名称
    
    Returns:
        类别分布字典
    """
    total = len(labels)
    distribution = {}
    
    for i, name in enumerate(class_names):
        count = np.sum(labels == i)
        distribution[name] = {
            'count': int(count),
            'percentage': count / total * 100 if total > 0 else 0
        }
    
    return distribution


def compare_with_baseline(y_true: np.ndarray, 
                          y_pred: np.ndarray,
                          class_names: list = ['下降', '不变', '上升']) -> Dict:
    """
    与随机基线和多数类基线比较
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        class_names: 类别名称
    
    Returns:
        比较结果字典
    """
    # 模型性能
    model_acc = accuracy_score(y_true, y_pred)
    model_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # 随机基线 (均匀分布)
    n_classes = len(class_names)
    random_acc = 1.0 / n_classes
    
    # 多数类基线
    unique, counts = np.unique(y_true, return_counts=True)
    majority_class = unique[np.argmax(counts)]
    majority_acc = np.max(counts) / len(y_true)
    
    # 多数类预测的 F1
    majority_pred = np.full_like(y_true, majority_class)
    majority_f1 = f1_score(y_true, majority_pred, average='macro', zero_division=0)
    
    comparison = {
        'model': {
            'accuracy': model_acc,
            'f1_macro': model_f1
        },
        'random_baseline': {
            'accuracy': random_acc,
            'f1_macro': random_acc  # 随机的 F1 约等于准确率
        },
        'majority_baseline': {
            'accuracy': majority_acc,
            'f1_macro': majority_f1
        },
        'improvement_over_random': {
            'accuracy': model_acc - random_acc,
            'f1_macro': model_f1 - random_acc
        },
        'improvement_over_majority': {
            'accuracy': model_acc - majority_acc,
            'f1_macro': model_f1 - majority_f1
        }
    }
    
    return comparison


def print_baseline_comparison(y_true: np.ndarray, 
                               y_pred: np.ndarray,
                               class_names: list = ['下降', '不变', '上升']):
    """
    打印基线比较
    """
    comparison = compare_with_baseline(y_true, y_pred, class_names)
    
    print("\n【与基线比较】")
    print(f"{'方法':<20} {'Accuracy':>12} {'Macro F1':>12}")
    print("-" * 46)
    print(f"{'模型':<20} {comparison['model']['accuracy']*100:>11.2f}% {comparison['model']['f1_macro']:>12.4f}")
    print(f"{'随机基线':<20} {comparison['random_baseline']['accuracy']*100:>11.2f}% {comparison['random_baseline']['f1_macro']:>12.4f}")
    print(f"{'多数类基线':<20} {comparison['majority_baseline']['accuracy']*100:>11.2f}% {comparison['majority_baseline']['f1_macro']:>12.4f}")
    print("-" * 46)
    print(f"{'提升(vs随机)':<20} {comparison['improvement_over_random']['accuracy']*100:>+11.2f}% {comparison['improvement_over_random']['f1_macro']:>+12.4f}")
    print(f"{'提升(vs多数类)':<20} {comparison['improvement_over_majority']['accuracy']*100:>+11.2f}% {comparison['improvement_over_majority']['f1_macro']:>+12.4f}")


if __name__ == "__main__":
    # 测试代码
    print("测试评估指标模块...")
    
    # 模拟数据
    np.random.seed(42)
    y_true = np.random.randint(0, 3, 1000)
    y_pred = np.random.randint(0, 3, 1000)
    
    # 让预测有一定准确性
    correct_indices = np.random.choice(1000, 400, replace=False)
    y_pred[correct_indices] = y_true[correct_indices]
    
    # 打印分类报告
    print_classification_report(y_true, y_pred)
    
    # 打印基线比较
    print_baseline_comparison(y_true, y_pred)
    
    # 类别分布
    print("\n【类别分布】")
    dist = calculate_class_distribution(y_true)
    for name, info in dist.items():
        print(f"  {name}: {info['count']} ({info['percentage']:.1f}%)")
