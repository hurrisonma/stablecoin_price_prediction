"""
LSTM 分类模型
用于时序数据的三分类预测
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class LSTMClassifier(nn.Module):
    """
    双向 LSTM 分类器，带自注意力机制
    
    架构:
        Input -> BiLSTM -> Self-Attention -> FC -> Output
    """
    
    def __init__(self, 
                 input_size: int,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 num_classes: int = 3,
                 dropout: float = 0.2,
                 use_attention: bool = True):
        """
        初始化 LSTM 分类器
        
        Args:
            input_size: 输入特征维度
            hidden_size: LSTM 隐藏层大小
            num_layers: LSTM 层数
            num_classes: 分类类别数
            dropout: Dropout 比例
            use_attention: 是否使用注意力机制
        """
        super(LSTMClassifier, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_attention = use_attention
        
        # 双向 LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # LSTM 输出维度 (双向所以 *2)
        lstm_output_size = hidden_size * 2
        
        # 自注意力层
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=lstm_output_size,
                num_heads=4,
                dropout=dropout,
                batch_first=True
            )
            self.layer_norm = nn.LayerNorm(lstm_output_size)
        
        # 全连接分类层
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # 设置遗忘门偏置为1
                n = param.size(0)
                param.data[n//4:n//2].fill_(1)
        
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状 (batch_size, seq_len, input_size)
        
        Returns:
            输出张量，形状 (batch_size, num_classes)
        """
        # LSTM 编码
        lstm_out, (h_n, c_n) = self.lstm(x)
        # lstm_out: (batch_size, seq_len, hidden_size * 2)
        
        if self.use_attention:
            # 自注意力
            attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
            # 残差连接 + 层归一化
            out = self.layer_norm(lstm_out + attn_out)
            # 取最后一个时间步
            out = out[:, -1, :]
        else:
            # 直接取最后一个时间步
            out = lstm_out[:, -1, :]
        
        # 分类
        logits = self.classifier(out)
        
        return logits
    
    def get_attention_weights(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """
        获取注意力权重（用于可视化）
        
        Args:
            x: 输入张量
        
        Returns:
            注意力权重张量
        """
        if not self.use_attention:
            return None
        
        lstm_out, _ = self.lstm(x)
        _, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        
        return attn_weights


class SimpleLSTMClassifier(nn.Module):
    """
    简化版 LSTM 分类器（不含注意力）
    
    用于快速实验和基线对比
    """
    
    def __init__(self,
                 input_size: int,
                 hidden_size: int = 64,
                 num_layers: int = 1,
                 num_classes: int = 3,
                 dropout: float = 0.3):
        """
        初始化简化版 LSTM
        
        Args:
            input_size: 输入特征维度
            hidden_size: 隐藏层大小
            num_layers: LSTM 层数
            num_classes: 分类类别数
            dropout: Dropout 比例
        """
        super(SimpleLSTMClassifier, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, (h_n, c_n) = self.lstm(x)
        # 取最后一个时间步
        out = lstm_out[:, -1, :]
        out = self.dropout(out)
        logits = self.fc(out)
        return logits


class GRUClassifier(nn.Module):
    """
    GRU 分类器
    
    相比 LSTM 参数更少，训练更快
    """
    
    def __init__(self,
                 input_size: int,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 num_classes: int = 3,
                 dropout: float = 0.2,
                 bidirectional: bool = True):
        """
        初始化 GRU 分类器
        
        Args:
            input_size: 输入特征维度
            hidden_size: 隐藏层大小
            num_layers: GRU 层数
            num_classes: 分类类别数
            dropout: Dropout 比例
            bidirectional: 是否双向
        """
        super(GRUClassifier, self).__init__()
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        gru_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        self.classifier = nn.Sequential(
            nn.Linear(gru_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gru_out, h_n = self.gru(x)
        out = gru_out[:, -1, :]
        logits = self.classifier(out)
        return logits


if __name__ == "__main__":
    # 测试代码
    print("测试 LSTM 模型...")
    
    batch_size = 32
    seq_len = 60
    input_size = 50
    num_classes = 3
    
    # 创建测试数据
    x = torch.randn(batch_size, seq_len, input_size)
    
    # 测试 LSTMClassifier
    model = LSTMClassifier(
        input_size=input_size,
        hidden_size=128,
        num_layers=2,
        num_classes=num_classes,
        dropout=0.2,
        use_attention=True
    )
    
    print(f"\nLSTMClassifier 参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    output = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    
    # 测试注意力权重
    attn_weights = model.get_attention_weights(x)
    if attn_weights is not None:
        print(f"注意力权重形状: {attn_weights.shape}")
    
    # 测试 SimpleLSTMClassifier
    simple_model = SimpleLSTMClassifier(
        input_size=input_size,
        hidden_size=64,
        num_layers=1,
        num_classes=num_classes
    )
    
    print(f"\nSimpleLSTMClassifier 参数量: {sum(p.numel() for p in simple_model.parameters()):,}")
    
    output = simple_model(x)
    print(f"输出形状: {output.shape}")
    
    # 测试 GRUClassifier
    gru_model = GRUClassifier(
        input_size=input_size,
        hidden_size=128,
        num_layers=2,
        num_classes=num_classes
    )
    
    print(f"\nGRUClassifier 参数量: {sum(p.numel() for p in gru_model.parameters()):,}")
    
    output = gru_model(x)
    print(f"输出形状: {output.shape}")
