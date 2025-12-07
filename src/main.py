"""
稳定币价格预测系统 - 主程序入口
"""

import os
import sys
import argparse
from datetime import datetime
from typing import Dict, Optional

import yaml
import numpy as np
import torch

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.data.loader import load_orderbook_data, validate_data
from src.data.preprocessor import prepare_data
from src.data.dataset import create_data_loaders, get_class_weights
from src.features.feature_engineering import create_features
from src.models.lstm_model import LSTMClassifier, SimpleLSTMClassifier, GRUClassifier
from src.models.transformer_model import TransformerClassifier, TransformerWithCLS
from src.models.trainer import Trainer
from src.utils.metrics import print_classification_report, print_baseline_comparison
from src.utils.visualization import plot_training_history, plot_confusion_matrix


def load_config(config_path: str) -> Dict:
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model(model_type: str, input_size: int, config: Dict) -> torch.nn.Module:
    """
    创建模型
    
    Args:
        model_type: 模型类型 (lstm, lstm_simple, gru, transformer, transformer_cls)
        input_size: 输入特征维度
        config: 模型配置
    
    Returns:
        PyTorch 模型
    """
    model_config = config.get('model', {})
    transformer_config = config.get('transformer', {})
    
    if model_type == 'lstm':
        model = LSTMClassifier(
            input_size=input_size,
            hidden_size=model_config.get('hidden_size', 128),
            num_layers=model_config.get('num_layers', 2),
            num_classes=3,
            dropout=model_config.get('dropout', 0.2),
            use_attention=True
        )
    elif model_type == 'lstm_simple':
        model = SimpleLSTMClassifier(
            input_size=input_size,
            hidden_size=model_config.get('hidden_size', 64),
            num_layers=model_config.get('num_layers', 1),
            num_classes=3,
            dropout=model_config.get('dropout', 0.3)
        )
    elif model_type == 'gru':
        model = GRUClassifier(
            input_size=input_size,
            hidden_size=model_config.get('hidden_size', 128),
            num_layers=model_config.get('num_layers', 2),
            num_classes=3,
            dropout=model_config.get('dropout', 0.2)
        )
    elif model_type == 'transformer':
        model = TransformerClassifier(
            input_size=input_size,
            d_model=transformer_config.get('d_model', 64),
            nhead=transformer_config.get('nhead', 4),
            num_encoder_layers=transformer_config.get('num_encoder_layers', 2),
            dim_feedforward=transformer_config.get('dim_feedforward', 256),
            num_classes=3,
            dropout=model_config.get('dropout', 0.1)
        )
    elif model_type == 'transformer_cls':
        model = TransformerWithCLS(
            input_size=input_size,
            d_model=transformer_config.get('d_model', 64),
            nhead=transformer_config.get('nhead', 4),
            num_encoder_layers=transformer_config.get('num_encoder_layers', 2),
            num_classes=3,
            dropout=model_config.get('dropout', 0.1)
        )
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型: {model_type}")
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    return model


def train(args):
    """训练模型"""
    print("=" * 60)
    print("稳定币价格预测系统 - 训练模式")
    print("=" * 60)
    
    # 加载配置
    config = load_config(args.config)
    
    # 数据参数
    data_config = config.get('data', {})
    features_config = config.get('features', {})
    split_config = config.get('split', {})
    training_config = config.get('training', {})
    
    symbol = data_config.get('symbol', 'USDCUSDT')
    data_dir = os.path.join(PROJECT_ROOT, data_config.get('data_dir', 'data/orderbooks'))
    lookback = features_config.get('lookback_window', 60)
    horizon = features_config.get('prediction_horizon', 5)
    threshold = features_config.get('price_threshold', 0.0001)
    
    # 1. 加载数据
    print(f"\n[1/5] 加载数据: {symbol}")
    df = load_orderbook_data(data_dir, symbol)
    df = validate_data(df)
    
    # 2. 特征工程
    print(f"\n[2/5] 特征工程")
    features = create_features(df)
    
    # 3. 准备数据
    print(f"\n[3/5] 准备训练数据")
    data_dict = prepare_data(
        df, features,
        horizon=horizon,
        threshold=threshold,
        train_ratio=split_config.get('train_ratio', 0.7),
        val_ratio=split_config.get('val_ratio', 0.15)
    )
    
    # 4. 创建数据加载器
    print(f"\n[4/5] 创建数据加载器")
    loaders = create_data_loaders(
        data_dict,
        lookback=lookback,
        batch_size=training_config.get('batch_size', 64)
    )
    
    # 计算类别权重
    class_weights = None
    if training_config.get('class_weights', 'balanced') == 'balanced':
        class_weights = get_class_weights(data_dict['train']['labels'])
    
    # 5. 创建模型
    print(f"\n[5/5] 创建模型")
    input_size = data_dict['train']['features'].shape[1]
    model_type = args.model if args.model else config.get('model', {}).get('type', 'lstm')
    model = create_model(model_type, input_size, config)
    
    # 训练配置
    trainer_config = {
        'learning_rate': training_config.get('learning_rate', 0.001),
        'epochs': args.epochs if args.epochs else training_config.get('epochs', 100),
        'early_stopping_patience': training_config.get('early_stopping_patience', 10),
        'class_weights': class_weights,
        'log_dir': os.path.join(PROJECT_ROOT, config.get('output', {}).get('log_dir', 'outputs/logs'))
    }
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=loaders['train'],
        val_loader=loaders['val'],
        config=trainer_config
    )
    
    # 训练
    history = trainer.train()
    
    # 保存模型
    model_dir = os.path.join(PROJECT_ROOT, config.get('output', {}).get('model_dir', 'outputs/models'))
    os.makedirs(model_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = os.path.join(model_dir, f'{model_type}_{timestamp}.pt')
    best_model_path = os.path.join(model_dir, 'best_model.pt')
    
    trainer.save_checkpoint(model_path)
    trainer.save_checkpoint(best_model_path)
    
    # 绘制训练历史
    plot_path = os.path.join(model_dir, f'{model_type}_{timestamp}_history.png')
    plot_training_history(history, save_path=plot_path)
    
    # 在测试集上评估
    print("\n" + "=" * 60)
    print("测试集评估")
    print("=" * 60)
    
    test_loss, test_acc, y_true, y_pred = trainer.validate(loaders['test'])
    print(f"测试集准确率: {test_acc*100:.2f}%")
    
    print_classification_report(y_true, y_pred)
    print_baseline_comparison(y_true, y_pred)
    
    # 绘制混淆矩阵
    cm_path = os.path.join(model_dir, f'{model_type}_{timestamp}_confusion.png')
    plot_confusion_matrix(y_true, y_pred, save_path=cm_path)
    
    print("\n训练完成!")
    print(f"模型已保存到: {model_path}")
    
    return history


def evaluate(args):
    """评估模型"""
    print("=" * 60)
    print("稳定币价格预测系统 - 评估模式")
    print("=" * 60)
    
    # 加载配置
    config = load_config(args.config)
    
    # 加载数据
    data_config = config.get('data', {})
    features_config = config.get('features', {})
    
    symbol = data_config.get('symbol', 'USDCUSDT')
    data_dir = os.path.join(PROJECT_ROOT, data_config.get('data_dir', 'data/orderbooks'))
    lookback = features_config.get('lookback_window', 60)
    horizon = features_config.get('prediction_horizon', 5)
    threshold = features_config.get('price_threshold', 0.0001)
    
    print(f"\n加载数据: {symbol}")
    df = load_orderbook_data(data_dir, symbol)
    df = validate_data(df)
    
    print(f"\n特征工程")
    features = create_features(df)
    
    print(f"\n准备数据")
    data_dict = prepare_data(df, features, horizon=horizon, threshold=threshold)
    
    loaders = create_data_loaders(data_dict, lookback=lookback, batch_size=64)
    
    # 加载模型
    print(f"\n加载模型: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location='cpu')
    
    # 根据模型路径推断模型类型
    model_type = 'lstm'
    for mt in ['transformer_cls', 'transformer', 'lstm_simple', 'lstm', 'gru']:
        if mt in args.model_path:
            model_type = mt
            break
    
    input_size = data_dict['test']['features'].shape[1]
    model = create_model(model_type, input_size, config)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 创建训练器用于评估
    trainer = Trainer(model, loaders['train'], loaders['val'], {})
    
    # 在测试集上评估
    print("\n" + "=" * 60)
    print("测试集评估")
    print("=" * 60)
    
    test_loss, test_acc, y_true, y_pred = trainer.validate(loaders['test'])
    
    print_classification_report(y_true, y_pred)
    print_baseline_comparison(y_true, y_pred)
    
    # 绘制混淆矩阵
    plot_confusion_matrix(y_true, y_pred)


def predict(args):
    """预测模式"""
    print("预测模式尚未实现")
    # TODO: 实现实时预测功能


def main():
    parser = argparse.ArgumentParser(description='稳定币价格预测系统')
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 训练命令
    train_parser = subparsers.add_parser('train', help='训练模型')
    train_parser.add_argument('--config', type=str, default='config/config.yaml',
                              help='配置文件路径')
    train_parser.add_argument('--model', type=str, 
                              choices=['lstm', 'lstm_simple', 'gru', 'transformer', 'transformer_cls'],
                              help='模型类型')
    train_parser.add_argument('--epochs', type=int, help='训练轮数')
    
    # 评估命令
    eval_parser = subparsers.add_parser('evaluate', help='评估模型')
    eval_parser.add_argument('--config', type=str, default='config/config.yaml',
                             help='配置文件路径')
    eval_parser.add_argument('--model-path', type=str, required=True,
                             help='模型文件路径')
    
    # 预测命令
    pred_parser = subparsers.add_parser('predict', help='预测')
    pred_parser.add_argument('--config', type=str, default='config/config.yaml',
                             help='配置文件路径')
    pred_parser.add_argument('--model-path', type=str, required=True,
                             help='模型文件路径')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train(args)
    elif args.command == 'evaluate':
        evaluate(args)
    elif args.command == 'predict':
        predict(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
