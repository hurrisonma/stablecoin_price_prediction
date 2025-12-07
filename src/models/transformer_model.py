"""
Transformer 分类模型
用于时序数据的三分类预测
"""

import math
import torch
import torch.nn as nn
from typing import Optional


class PositionalEncoding(nn.Module):
    """
    位置编码层
    
    为序列添加位置信息，使模型能够理解时序顺序
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """
        初始化位置编码
        
        Args:
            d_model: 模型维度
            dropout: Dropout 比例
            max_len: 最大序列长度
        """
        super(PositionalEncoding, self).__init__()
        
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        添加位置编码
        
        Args:
            x: 输入张量，形状 (batch_size, seq_len, d_model)
        
        Returns:
            添加位置编码后的张量
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerClassifier(nn.Module):
    """
    Transformer 分类器
    
    架构:
        Input -> Linear Projection -> Positional Encoding -> 
        Transformer Encoder -> Classification Head -> Output
    """
    
    def __init__(self,
                 input_size: int,
                 d_model: int = 64,
                 nhead: int = 4,
                 num_encoder_layers: int = 2,
                 dim_feedforward: int = 256,
                 num_classes: int = 3,
                 dropout: float = 0.1,
                 max_seq_len: int = 200):
        """
        初始化 Transformer 分类器
        
        Args:
            input_size: 输入特征维度
            d_model: 模型维度
            nhead: 注意力头数
            num_encoder_layers: Encoder 层数
            dim_feedforward: 前馈网络维度
            num_classes: 分类类别数
            dropout: Dropout 比例
            max_seq_len: 最大序列长度
        """
        super(TransformerClassifier, self).__init__()
        
        self.d_model = d_model
        
        # 输入投影层
        self.input_projection = nn.Linear(input_size, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_len)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, 
                src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状 (batch_size, seq_len, input_size)
            src_mask: 可选的注意力掩码
        
        Returns:
            输出张量，形状 (batch_size, num_classes)
        """
        # 投影到模型维度
        x = self.input_projection(x) * math.sqrt(self.d_model)
        
        # 添加位置编码
        x = self.pos_encoder(x)
        
        # Transformer 编码
        encoded = self.transformer_encoder(x, src_mask)
        
        # 取最后一个时间步进行分类
        out = encoded[:, -1, :]
        
        # 分类
        logits = self.classifier(out)
        
        return logits
    
    def get_attention_weights(self, x: torch.Tensor) -> list:
        """
        获取各层的注意力权重（用于可视化）
        
        Args:
            x: 输入张量
        
        Returns:
            各层注意力权重列表
        """
        attention_weights = []
        
        x = self.input_projection(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        
        for layer in self.transformer_encoder.layers:
            # 手动调用自注意力获取权重
            attn_output, attn_weights = layer.self_attn(
                x, x, x, need_weights=True
            )
            attention_weights.append(attn_weights.detach())
            
            # 继续前向传播
            x = layer(x)
        
        return attention_weights


class TransformerWithCLS(nn.Module):
    """
    带 CLS Token 的 Transformer 分类器
    
    类似 BERT，使用特殊的 [CLS] token 进行分类
    """
    
    def __init__(self,
                 input_size: int,
                 d_model: int = 64,
                 nhead: int = 4,
                 num_encoder_layers: int = 2,
                 dim_feedforward: int = 256,
                 num_classes: int = 3,
                 dropout: float = 0.1,
                 max_seq_len: int = 200):
        """
        初始化带 CLS 的 Transformer
        """
        super(TransformerWithCLS, self).__init__()
        
        self.d_model = d_model
        
        # CLS token (可学习参数)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # 输入投影
        self.input_projection = nn.Linear(input_size, d_model)
        
        # 位置编码 (多一个位置给 CLS)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_len + 1)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        # 投影输入
        x = self.input_projection(x)
        
        # 添加 CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # 添加位置编码
        x = self.pos_encoder(x)
        
        # Transformer 编码
        encoded = self.transformer_encoder(x)
        
        # 使用 CLS token 进行分类
        cls_output = encoded[:, 0, :]
        
        logits = self.classifier(cls_output)
        
        return logits


class TemporalConvTransformer(nn.Module):
    """
    时域卷积 + Transformer 混合模型
    
    使用 1D 卷积提取局部特征，再用 Transformer 建模全局依赖
    """
    
    def __init__(self,
                 input_size: int,
                 d_model: int = 64,
                 nhead: int = 4,
                 num_encoder_layers: int = 2,
                 num_classes: int = 3,
                 kernel_size: int = 3,
                 dropout: float = 0.1):
        """
        初始化混合模型
        """
        super(TemporalConvTransformer, self).__init__()
        
        # 时域卷积层
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_size, d_model, kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(d_model),
            nn.ReLU()
        )
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, features)
        
        # 转置用于 1D 卷积: (batch, features, seq_len)
        x = x.transpose(1, 2)
        
        # 卷积提取局部特征
        x = self.conv_layers(x)
        
        # 转置回来: (batch, seq_len, d_model)
        x = x.transpose(1, 2)
        
        # 位置编码
        x = self.pos_encoder(x)
        
        # Transformer 编码
        x = self.transformer(x)
        
        # 取最后一个时间步
        out = x[:, -1, :]
        
        # 分类
        logits = self.classifier(out)
        
        return logits


if __name__ == "__main__":
    # 测试代码
    print("测试 Transformer 模型...")
    
    batch_size = 32
    seq_len = 60
    input_size = 50
    num_classes = 3
    
    # 创建测试数据
    x = torch.randn(batch_size, seq_len, input_size)
    
    # 测试 TransformerClassifier
    model = TransformerClassifier(
        input_size=input_size,
        d_model=64,
        nhead=4,
        num_encoder_layers=2,
        dim_feedforward=256,
        num_classes=num_classes,
        dropout=0.1
    )
    
    print(f"\nTransformerClassifier 参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    output = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    
    # 测试 TransformerWithCLS
    cls_model = TransformerWithCLS(
        input_size=input_size,
        d_model=64,
        nhead=4,
        num_encoder_layers=2,
        num_classes=num_classes
    )
    
    print(f"\nTransformerWithCLS 参数量: {sum(p.numel() for p in cls_model.parameters()):,}")
    
    output = cls_model(x)
    print(f"输出形状: {output.shape}")
    
    # 测试 TemporalConvTransformer
    conv_trans_model = TemporalConvTransformer(
        input_size=input_size,
        d_model=64,
        nhead=4,
        num_encoder_layers=2,
        num_classes=num_classes
    )
    
    print(f"\nTemporalConvTransformer 参数量: {sum(p.numel() for p in conv_trans_model.parameters()):,}")
    
    output = conv_trans_model(x)
    print(f"输出形状: {output.shape}")
