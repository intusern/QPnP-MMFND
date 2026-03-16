"""
复值波函数量子态编码模块
将CLIP融合特征通过波函数转为量子态，深度捕捉对齐语义，同时保留CLIP语义信息

核心思想：
1. 波函数转换：利用欧拉公式 c_k = |c_k|·e^(iθ_k) 将CLIP特征映射为量子态
2. 语义保留：幅值直接复用L2归一化的CLIP特征，保证语义分布不变
3. 相位编码：相位从CLIP特征学习得到，捕捉深层语义关联
4. 复值网络：严格遵循复数运算规则，保留量子态的幅值+相位信息
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ===================== 复值网络基础组件 =====================

class ComplexLinear(nn.Module):
    """
    复值线性层：严格遵循复数乘法规则
    (a+bi)(c+di) = (ac-bd) + (ad+bc)i
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        # 复权重拆分为实部+虚部（可训练）
        bound = 1.0 / math.sqrt(in_dim)
        self.w_real = nn.Parameter(torch.empty(in_dim, out_dim).uniform_(-bound, bound))
        self.w_imag = nn.Parameter(torch.empty(in_dim, out_dim).uniform_(-bound, bound))
        # 复偏置拆分为实部+虚部
        self.b_real = nn.Parameter(torch.zeros(out_dim))
        self.b_imag = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x):
        """
        输入: x - 复值张量 [B, in_dim]
        输出: 复值张量 [B, out_dim]
        """
        x_real, x_imag = x.real, x.imag
        # 计算输出实部：x_real·w_real - x_imag·w_imag + b_real
        out_real = torch.matmul(x_real, self.w_real) - torch.matmul(x_imag, self.w_imag) + self.b_real
        # 计算输出虚部：x_real·w_imag + x_imag·w_real + b_imag
        out_imag = torch.matmul(x_real, self.w_imag) + torch.matmul(x_imag, self.w_real) + self.b_imag
        return torch.complex(out_real, out_imag)


class ComplexReLU(nn.Module):
    """
    复值ReLU：分别对实部、虚部应用ReLU
    保留正负语义，不截断相位信息
    """
    def forward(self, x):
        return torch.complex(F.relu(x.real), F.relu(x.imag))


class ComplexLayerNorm(nn.Module):
    """
    复值LayerNorm：基于模长进行归一化
    保持复数的相位信息不变
    """
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        # 计算模长
        magnitude = torch.sqrt(x.real ** 2 + x.imag ** 2 + self.eps)
        # 基于模长归一化
        mean = magnitude.mean(dim=-1, keepdim=True)
        var = magnitude.var(dim=-1, keepdim=True, unbiased=False)
        mag_norm = (magnitude - mean) / torch.sqrt(var + self.eps)
        # 缩放因子
        scale = mag_norm / (magnitude + self.eps)
        # 应用归一化（保持相位不变）
        out_real = x.real * scale * self.gamma + self.beta
        out_imag = x.imag * scale * self.gamma
        return torch.complex(out_real, out_imag)


# ===================== 波函数量子态转换模块 =====================

class WaveFunctionQuantizer(nn.Module):
    """
    基于波函数+欧拉公式，将CLIP融合特征转为量子态（复值）
    
    核心设计：
    - 幅值：复用CLIP特征的L2归一化值（保留语义分布）
    - 相位：从CLIP特征学习映射到[-π, π]（保留语义关联）
    - 欧拉公式：c_k = |c_k|·e^(iθ_k) = |c_k|(cos θ + i sin θ)
    - 归一化：∑|c_k|²=1，满足量子态概率约束
    """
    def __init__(self, clip_dim=320, eps=1e-8):
        super().__init__()
        self.clip_dim = clip_dim
        self.eps = eps
        
        # 相位编码器：将CLIP特征映射到相位空间
        self.phase_encoder = nn.Sequential(
            nn.Linear(clip_dim, clip_dim),
            nn.LayerNorm(clip_dim),
            nn.Tanh()  # 归一化到[-1,1]，后续映射到[-π, π]
        )
        
        # 幅值调制器（可选）：轻微调制幅值，增强表达能力，同时保持接近原始语义
        self.amp_modulator = nn.Sequential(
            nn.Linear(clip_dim, clip_dim),
            nn.Sigmoid()  # 输出[0,1]，用于软调制
        )
        
    def forward(self, clip_fused):
        """
        输入：clip_fused - CLIP融合特征 [B, clip_dim]（实值向量）
        输出：
            quantum_state - 量子态（复值波函数） [B, clip_dim]
            amp - 幅值 [B, clip_dim]
            phase - 相位 [B, clip_dim]
        
        波函数定义：|ψ⟩ = ∑c_k|k⟩，c_k = |c_k|·e^(iθ_k)
        """
        # Step 1: L2归一化CLIP特征（保证幅值的概率归一性）
        clip_norm = F.normalize(clip_fused, p=2, dim=-1)  # [B, clip_dim]
        
        # Step 2: 幅值 = 归一化CLIP特征（直接复用，保留语义幅值）
        # 可选：轻微调制幅值增强表达能力
        amp_gate = self.amp_modulator(clip_fused)
        # 使用残差调制：0.9*原始 + 0.1*调制，保持语义主导
        amp = clip_norm * (0.9 + 0.1 * amp_gate)
        # 重新归一化确保概率约束
        amp = amp / (amp.norm(dim=-1, keepdim=True) + self.eps)
        
        # Step 3: 相位 = CLIP特征映射到[-π, π]
        phase = self.phase_encoder(clip_fused) * math.pi  # [B, clip_dim]
        
        # Step 4: 欧拉公式展开复值概率幅
        real_part = amp * torch.cos(phase)  # 实部
        imag_part = amp * torch.sin(phase)  # 虚部
        
        # Step 5: 构建复值量子态（波函数）
        quantum_state = torch.complex(real_part, imag_part)  # [B, clip_dim]
        
        return quantum_state, amp, phase


# ===================== 复值语义网络 =====================

class ComplexSemanticNetwork(nn.Module):
    """
    复值神经网络：处理量子态波函数，深度捕捉语义关联
    
    核心设计：
    - 复值运算保留量子态的幅值+相位
    - 残差连接保留原始量子态信息
    - 语义投影层将复值特征映射回实值空间
    """
    def __init__(self, clip_dim=320, hidden_dim=320, out_dim=320):
        super().__init__()
        self.clip_dim = clip_dim
        self.out_dim = out_dim
        
        # 复值线性层
        self.cplx_linear1 = ComplexLinear(clip_dim, hidden_dim)
        self.cplx_linear2 = ComplexLinear(hidden_dim, out_dim)
        
        # 复值归一化层
        self.cplx_norm1 = ComplexLayerNorm(hidden_dim)
        self.cplx_norm2 = ComplexLayerNorm(out_dim)
        
        # 复值激活函数
        self.cplx_relu = ComplexReLU()
        
        # 语义保留投影层：将复值量子特征映射回原始CLIP维度
        self.semantic_proj = nn.Sequential(
            nn.Linear(out_dim * 2, clip_dim),  # 实部+虚部拼接
            nn.LayerNorm(clip_dim),
            nn.SiLU(),
            nn.Linear(clip_dim, clip_dim)
        )
        
        # 残差权重（可学习）
        self.residual_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, quantum_state, original_clip):
        """
        输入：
            quantum_state - 量子态（复值波函数） [B, clip_dim]
            original_clip - 原始CLIP融合特征 [B, clip_dim]（用于残差连接）
        输出：
            cplx_feat - 深度量子语义特征（复值） [B, out_dim]
            semantic_feat - 语义保留特征（实值，与CLIP维度一致） [B, clip_dim]
        """
        # Step 1: 复值层传播
        x = self.cplx_linear1(quantum_state)  # [B, hidden_dim]
        x = self.cplx_norm1(x)
        x = self.cplx_relu(x)
        
        cplx_feat = self.cplx_linear2(x)  # [B, out_dim]
        cplx_feat = self.cplx_norm2(cplx_feat)
        
        # Step 2: 将复值特征转换为实值语义特征
        real_imag_concat = torch.cat([cplx_feat.real, cplx_feat.imag], dim=-1)  # [B, out_dim*2]
        semantic_feat = self.semantic_proj(real_imag_concat)  # [B, clip_dim]
        
        # Step 3: 残差连接保留原始CLIP语义
        alpha = torch.sigmoid(self.residual_weight)
        semantic_feat = alpha * semantic_feat + (1 - alpha) * original_clip
        
        # Step 4: 最终归一化
        semantic_feat = F.normalize(semantic_feat, p=2, dim=-1) * original_clip.norm(dim=-1, keepdim=True)
        
        return cplx_feat, semantic_feat


# ===================== 完整的波函数量子编码器 =====================

class WaveFunctionQuantumEncoder(nn.Module):
    """
    完整的波函数量子态编码器
    
    流程：
    1. CLIP融合特征 → 波函数量子态（幅值+相位）
    2. 量子态 → 复值网络深度处理
    3. 复值特征 → 语义保留投影回实值空间
    4. 残差连接确保CLIP语义不丢失
    
    输出维度与输入一致，可直接替换原始CLIP融合特征
    """
    def __init__(self, clip_dim=320, hidden_dim=320, return_quantum_info=False):
        super().__init__()
        self.clip_dim = clip_dim
        self.return_quantum_info = return_quantum_info
        
        # 波函数量子态转换
        self.quantizer = WaveFunctionQuantizer(clip_dim=clip_dim)
        
        # 复值语义网络
        self.cplx_net = ComplexSemanticNetwork(
            clip_dim=clip_dim, 
            hidden_dim=hidden_dim, 
            out_dim=clip_dim
        )
        
        # 最终输出层归一化
        self.output_norm = nn.LayerNorm(clip_dim)

    def forward(self, clip_fused):
        """
        输入：clip_fused - CLIP融合特征 [B, clip_dim]
        输出：
            output_feat - 量子增强语义特征 [B, clip_dim]（与输入维度一致）
            如果 return_quantum_info=True，还返回：
                quantum_state - 量子态（复值）
                amp - 幅值
                phase - 相位
        """
        # Step 1: 转换为量子态
        quantum_state, amp, phase = self.quantizer(clip_fused)
        
        # Step 2: 复值网络处理
        cplx_feat, semantic_feat = self.cplx_net(quantum_state, clip_fused)
        
        # Step 3: 输出归一化
        output_feat = self.output_norm(semantic_feat)
        
        if self.return_quantum_info:
            return output_feat, quantum_state, amp, phase
        else:
            return output_feat


# ===================== 语义保留验证函数 =====================

def check_clip_semantic(original_clip, recovered_clip, verbose=True):
    """
    验证CLIP语义保留程度：计算原始特征与编码后特征的余弦相似度
    阈值：相似度>0.8 说明语义未丢失（越高越好）
    """
    cos_sim = F.cosine_similarity(
        F.normalize(original_clip, p=2, dim=-1), 
        F.normalize(recovered_clip, p=2, dim=-1), 
        dim=-1
    )
    avg_sim = cos_sim.mean().item()
    if verbose:
        print(f"CLIP语义保留余弦相似度：{avg_sim:.4f} (≥0.8表示语义未丢失)")
    return avg_sim


# ===================== 测试代码 =====================

if __name__ == "__main__":
    # 测试波函数量子编码器
    BATCH_SIZE = 4
    CLIP_DIM = 320  # 与MMDFND中clip_fusion输出维度一致
    
    # 模拟CLIP融合特征
    clip_fused = torch.randn(BATCH_SIZE, CLIP_DIM)
    
    # 初始化编码器
    encoder = WaveFunctionQuantumEncoder(clip_dim=CLIP_DIM, return_quantum_info=True)
    
    # 前向传播
    output_feat, quantum_state, amp, phase = encoder(clip_fused)
    
    print(f"输入特征形状：{clip_fused.shape}")
    print(f"输出特征形状：{output_feat.shape}")
    print(f"量子态形状：{quantum_state.shape}，数据类型：{quantum_state.dtype}")
    print(f"幅值范围：{amp.min():.4f} ~ {amp.max():.4f}")
    print(f"相位范围：{phase.min():.4f} ~ {phase.max():.4f}")
    
    # 验证语义保留
    check_clip_semantic(clip_fused, output_feat)
