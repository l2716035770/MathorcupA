import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


# ===================== 神经算子层 (积分核形式) =====================
class NeuralOperatorLayer(nn.Module):
    def __init__(self, input_dim, dk):
        super().__init__()
        # 定义可学习的核参数化函数 Q, K, V (与Transformer一致)
        self.WQ = nn.Linear(input_dim, dk)  # 查询变换
        self.WK = nn.Linear(input_dim, dk)  # 键变换
        self.WV = nn.Linear(input_dim, dk)  # 值变换
        self.dk = dk  # 缩放因子

    def forward(self, u):
        """
        输入: u [batch, N, input_dim] (N个离散点的函数值)
        输出: (Ku) [batch, N, dk]
        """
        Q = self.WQ(u)  # [B, N, dk]
        K = self.WK(u)  # [B, N, dk]
        V = self.WV(u)  # [B, N, dk]

        # 核函数 k(x_i, y_j) = softmax(Q(x_i)^T K(y_j) / sqrt(dk))
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.dk)  # [B, N, N]
        attn_weights = torch.softmax(attn_scores, dim=-1)  # 沿j维度归一化

        # 积分操作退化为离散求和: (Ku)(x_i) = Σ_j k(x_i, y_j) V(y_j)
        Ku = torch.matmul(attn_weights, V)  # [B, N, dk]
        return Ku, attn_weights


# ===================== Transformer自注意力层 =====================
class TransformerSelfAttention(nn.Module):
    def __init__(self, input_dim, dk):
        super().__init__()
        # 与神经算子共享相同的参数化结构
        self.WQ = nn.Linear(input_dim, dk)
        self.WK = nn.Linear(input_dim, dk)
        self.WV = nn.Linear(input_dim, dk)
        self.dk = dk

    def forward(self, x):
        Q = self.WQ(x)
        K = self.WK(x)
        V = self.WV(x)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.dk)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_weights, V)
        return output, attn_weights


# ===================== 测试数据与可视化 =====================
if __name__ == "__main__":
    torch.manual_seed(42)

    # 生成输入数据 (模拟离散函数采样)
    # u: [batch=1, N=5, input_dim=3] (5个离散点，每个点3维特征)
    u = torch.randn(1, 5, 3)

    # 初始化两个“不同”的模型（实际参数相同）
    neural_op = NeuralOperatorLayer(input_dim=3, dk=4)
    transformer_attn = TransformerSelfAttention(input_dim=3, dk=4)

    # 强制参数一致以验证数学等价性
    transformer_attn.WQ.weight.data = neural_op.WQ.weight.data.clone()
    transformer_attn.WQ.bias.data = neural_op.WQ.bias.data.clone()
    transformer_attn.WK.weight.data = neural_op.WK.weight.data.clone()
    transformer_attn.WK.bias.data = neural_op.WK.bias.data.clone()
    transformer_attn.WV.weight.data = neural_op.WV.weight.data.clone()
    transformer_attn.WV.bias.data = neural_op.WV.bias.data.clone()

    # 前向传播
    Ku_op, attn_weights_op = neural_op(u)
    output_attn, attn_weights_trans = transformer_attn(u)

    # 验证输出一致性
    print("神经算子输出与Transformer输出是否一致:", torch.allclose(Ku_op, output_attn, atol=1e-6))  # 应输出True

    # 可视化注意力权重
    plt.figure(figsize=(12, 5))

    # 神经算子的核权重
    plt.subplot(1, 2, 1)
    plt.imshow(attn_weights_op[0].detach().numpy(), cmap='viridis')
    plt.title("Neural Operator Kernel Weights")
    plt.xlabel("Source Points (j)")
    plt.ylabel("Target Points (i)")
    plt.colorbar()

    # Transformer的注意力权重
    plt.subplot(1, 2, 2)
    plt.imshow(attn_weights_trans[0].detach().numpy(), cmap='viridis')
    plt.title("Transformer Attention Weights")
    plt.xlabel("Key Positions (j)")
    plt.ylabel("Query Positions (i)")
    plt.colorbar()

    plt.tight_layout()
    plt.show()
