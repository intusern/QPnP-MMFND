"""
BERT/MAE特征的量子混合态编码器

核心思想：
1. 将BERT文本特征 [batch, 197, 768] 和 MAE图像特征 [batch, 197, 768] 
   分别编码为量子纯态（波函数形式 ψ = A·e^(iθ)）
2. 基于图文纯态构建跨模态混合态（密度矩阵形式 ρ = Σp_i|ψ_i⟩⟨ψ_i|）
3. 通过量子期望值提取融合语义，输出重编码特征

物理意义：
- 纯态：单一确定的量子状态，对应单模态的确定性语义
- 混合态：多个纯态的概率叠加，对应图文语义的不确定性融合
- 密度矩阵：完整描述混合态的统计特性，保留跨模态相关性

与CLIP量子态的区别：
- CLIP量子态：已对齐的图文特征，直接融合编码
- 本模块：未对齐的原始BERT/MAE特征，需要先独立编码再混合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

#1纯态c
class SequenceQuantumPureStateEncoder(nn.Module):
    """
    序列特征 → 量子纯态编码器
    
    输入: [batch, seq_len, feature_dim] (如BERT/MAE的输出)
    输出: 量子纯态 (振幅A, 相位θ, 波函数实部A·cosθ)
    
    处理流程:
    1. 注意力池化: [batch, seq_len, 768] → [batch, 768]
    2. 维度投影: [batch, 768] → [batch, output_dim]
    3. 量子编码: 生成振幅A ∈ [-1,1], 相位θ ∈ [-π, π]
    4. 波函数计算: ψ = A·e^(iθ) = A·cosθ + i·A·sinθ
    """
    
    def __init__(self, input_dim: int = 768, output_dim: int = 320, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 1. 注意力池化层 (学习token重要性权重)
        self.attention_pool = nn.Sequential(
            nn.Linear(input_dim, input_dim // 4),
            nn.Tanh(),
            nn.Linear(input_dim // 4, 1),
        )
        
        # 2. 特征投影层 (768 → output_dim)
        self.feature_proj = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # 3. 振幅编码器: 输出 A ∈ [-1, 1]
        self.amplitude_encoder = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
            nn.Tanh(),  # 限制振幅范围 [-1, 1]
        )
        
        # 4. 相位编码器: 输出 θ ∈ [-π, π]
        self.phase_encoder = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
            nn.Tanh(),  # 输出[-1,1]，后续×π映射到[-π, π]
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, seq_feature: torch.Tensor, attention_mask: torch.Tensor = None):
        """
        Args:
            seq_feature: [batch, seq_len, input_dim] 序列特征
            attention_mask: [batch, seq_len] 注意力掩码 (1=有效, 0=padding)
        
        Returns:
            quantum_state: [batch, output_dim] 量子态实部 (A·cosθ)
            amplitude: [batch, output_dim] 振幅 A ∈ [-1, 1]
            phase: [batch, output_dim] 相位 θ ∈ [-π, π]
            modulus: [batch, output_dim] 模长 |ψ|
            pooled_feature: [batch, output_dim] 池化后的原始特征
        """
        batch, seq_len, _ = seq_feature.shape
        
        # Step 1: 注意力池化
        # 计算每个token的注意力权重
        attn_scores = self.attention_pool(seq_feature).squeeze(-1)  # [B, L]
        
        # 应用注意力掩码
        if attention_mask is not None:
            attn_scores = attn_scores.masked_fill(attention_mask == 0, float('-inf'))
        
        attn_weights = F.softmax(attn_scores, dim=-1).unsqueeze(-1)  # [B, L, 1]
        
        # 加权求和
        pooled = torch.sum(seq_feature * attn_weights, dim=1)  # [B, input_dim]
        
        # Step 2: 特征投影
        projected = self.feature_proj(pooled)  # [B, output_dim]
        
        # Step 3: 量子编码
        amplitude = self.amplitude_encoder(projected)  # [-1, 1]
        phase = self.phase_encoder(projected) * math.pi  # [-π, π]
        
        # Step 4: 波函数计算 (欧拉公式)
        real = amplitude * torch.cos(phase)  # A·cosθ
        imag = amplitude * torch.sin(phase)  # A·sinθ
        modulus = torch.sqrt(real.pow(2) + imag.pow(2) + 1e-8)  # |ψ| = sqrt(Re² + Im²)
        
        quantum_state = modulus  # 输出模长|ψ|，表示概率幅度
        
        return quantum_state, amplitude, phase, modulus, projected

#2混合态
class mixedstate(nn.Module):
    """
    BERT/MAE特征的跨模态量子混合态编码器
    
    核心逻辑:
    1. 分别将BERT文本特征和MAE图像特征编码为量子纯态
    2. 生成多个视角的纯态（rank个），增加表达能力
    3. 构建混合态密度矩阵 ρ = Σp_i|ψ_i⟩⟨ψ_i|
    4. 计算量子期望，输出融合特征
    
    输入:
        text_feature: [batch, 197, 768] BERT输出
        image_feature: [batch, 197, 768] MAE输出
    
    输出:
        融合的量子混合态特征 [batch, output_dim]
    """
    
    def __init__(
        self, 
        input_dim: int = 768,      # BERT/MAE特征维度
        output_dim: int = 320,     # 输出特征维度
        hidden_dim: int = 128,     # 混合态隐藏维度，hillbert存储量子信息，
        rank: int = 4,             # 混合态的纯态数量
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.rank = rank
        
        # 1. 文本量子纯态编码器 (多视角)
        self.text_quantum_encoders = nn.ModuleList([
            SequenceQuantumPureStateEncoder(input_dim, hidden_dim, dropout)
            for _ in range(rank)
        ])
        
        # 2. 图像量子纯态编码器 (多视角)
        self.image_quantum_encoders = nn.ModuleList([
            SequenceQuantumPureStateEncoder(input_dim, hidden_dim, dropout)
            for _ in range(rank)
        ])
        
        # 3. 跨模态纯态编码器 (融合图文信息后编码)
        self.cross_modal_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # 4. 混合态权重学习 (图像、文本、跨模态的混合概率)
        # 总共有 rank*3 =12个纯态 (图像rank个 + 文本rank个 + 跨模态rank个)
        self.mix_weights = nn.Parameter(torch.ones(rank * 3) / (rank * 3))
        
        # 5. 输出读出层
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        
        # 6. 残差连接的投影层 (保留原始语义)
        self.text_residual_proj = nn.Linear(input_dim, output_dim)
        self.image_residual_proj = nn.Linear(input_dim, output_dim)
        
        # 7. 残差权重 (可学习)
        self.residual_weight = nn.Parameter(torch.tensor(0.3))
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def _pure_state_to_density(self, pure_state: torch.Tensor) -> torch.Tensor:
        """
        量子纯态 → 密度矩阵
        
        Args:
            pure_state: [batch, hidden_dim] 量子纯态
        
        Returns:
            density: [batch, hidden_dim, hidden_dim] 密度矩阵 ρ = |ψ⟩⟨ψ|
        """
        # 归一化纯态 (保证量子态合法性)
        norm = pure_state.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        pure_state_normalized = pure_state / norm
        
        # 构建密度矩阵: |ψ⟩⟨ψ|
        psi = pure_state_normalized.unsqueeze(-1)  # [B, H, 1]
        psi_dagger = psi.transpose(-1, -2)  # [B, 1, H]
        density = torch.matmul(psi, psi_dagger)  # [B, H, H]
        
        return density
    
    def forward(
        self, 
        text_feature: torch.Tensor, 
        image_feature: torch.Tensor,
        text_mask: torch.Tensor = None,
        image_mask: torch.Tensor = None
    ) -> dict:
        """
        前向传播: BERT/MAE特征 → 量子混合态 → 融合特征
        
        Args:
            text_feature: [batch, seq_len, 768] BERT输出
            image_feature: [batch, seq_len, 768] MAE输出
            text_mask: [batch, seq_len] 文本注意力掩码
            image_mask: [batch, seq_len] 图像注意力掩码 (通常全1)
        
        Returns:
            output_dict: 包含融合特征和量子态信息的字典
        """
        batch = text_feature.shape[0]
        
        # ============ Step 1: 生成文本量子纯态-[B, rank, H] ============
        text_pure_states = []
        text_amplitudes = []
        text_phases = []
        text_pooled_features = []
        
        for encoder in self.text_quantum_encoders:
            q_state, amp, phase, mod, pooled = encoder(text_feature, text_mask)
            text_pure_states.append(q_state)
            text_amplitudes.append(amp)
            text_phases.append(phase)
            text_pooled_features.append(pooled)
        
        # Stack一维拼接四个纯态: [B, rank, H]
        text_pure_states = torch.stack(text_pure_states, dim=1)
        text_amplitudes = torch.stack(text_amplitudes, dim=1)
        text_phases = torch.stack(text_phases, dim=1)
        
        # ============ Step 2: 生成图像量子纯态-[B, rank, H] ============
        image_pure_states = []
        image_amplitudes = []
        image_phases = []
        image_pooled_features = []
        
        for encoder in self.image_quantum_encoders:
            q_state, amp, phase, mod, pooled = encoder(image_feature, image_mask)
            image_pure_states.append(q_state)
            image_amplitudes.append(amp)
            image_phases.append(phase)
            image_pooled_features.append(pooled)
        
        # Stack: [B, rank, H]
        image_pure_states = torch.stack(image_pure_states, dim=1)
        image_amplitudes = torch.stack(image_amplitudes, dim=1)
        image_phases = torch.stack(image_phases, dim=1)
        
        # ============ Step 3: 生成跨模态量子纯态 -图文融合============
        cross_pure_states = []
        for i in range(self.rank):
            # 拼接图文特征
            concat_feat = torch.cat([
                text_pooled_features[i], 
                image_pooled_features[i]
            ], dim=-1)  # [B, H*2]
            
            # 跨模态融合编码
            cross_state = self.cross_modal_encoder(concat_feat)  # [B, H]
            cross_pure_states.append(cross_state)
        
        cross_pure_states = torch.stack(cross_pure_states, dim=1)  # [B, rank, H]
        
        # ============ Step 4: 构建所有纯态的密度矩阵-[B, rank*3, H, H] ============
        #第一维拼接d
        all_pure_states = torch.cat([
            text_pure_states,   # [B, rank, H]
            image_pure_states,  # [B, rank, H]
            cross_pure_states   # [B, rank, H]
        ], dim=1)  # [B, rank*3, H]
        
        density_matrices = []
        for i in range(self.rank * 3):
            density = self._pure_state_to_density(all_pure_states[:, i, :])
            density_matrices.append(density)
        
        density_matrices = torch.stack(density_matrices, dim=1)  # [B, rank*3, H, H]
        
        # ============ Step 5: 构建混合态密度矩阵-============
        # 混合概率归一化
        mix_probs = F.softmax(self.mix_weights, dim=0)  # [rank*3]
        
        # 扩展维度匹配: [rank*3] → [1, rank*3, 1, 1]
        mix_probs_expand = mix_probs.view(1, -1, 1, 1)
        
        # 混合态: ρ = Σp_i * ρ_i
        mixed_density = torch.sum(density_matrices * mix_probs_expand, dim=1)  # [B, H, H]
        
        # 归一化混合态 (保证迹为1)
        trace = mixed_density.diagonal(dim1=-2, dim2=-1).sum(dim=-1, keepdim=True)
        trace = trace.clamp(min=1e-6)
        mixed_density = mixed_density / trace.unsqueeze(-1)
        
        # ============ Step 6: 计算量子期望 torch.einsum('bh,bhk->bk'） 来特征聚合求得混合信息============
        # 使用所有纯态的平均作为观测态
        avg_pure_state = all_pure_states.mean(dim=1)  # [B, H]
        
        # 量子期望: <ψ|ρ|ψ>
        expectation = torch.einsum('bh,bhk->bk', avg_pure_state, mixed_density)  # [B, H]
        
        # ============ Step 7: 读出层 + 残差连接 ============
        # 混合态特征
        quantum_feature = self.readout(expectation)  # [B, output_dim]
        
        # 残差连接 (保留原始语义)
        text_residual = self.text_residual_proj(text_feature.mean(dim=1))  # [B, output_dim]
        image_residual = self.image_residual_proj(image_feature.mean(dim=1))  # [B, output_dim]
        residual = (text_residual + image_residual) / 2
        
        # 加权融合
        alpha = torch.sigmoid(self.residual_weight)
        fused_feature = alpha * quantum_feature + (1 - alpha) * residual
        
        # ============ 整理输出 ============
        output_dict = {
            # 核心输出
            "fused_feature": fused_feature,           # [B, output_dim] 最终融合特征
            "quantum_feature": quantum_feature,       # [B, output_dim] 纯量子态特征
            
            # 混合态信息
            "mixed_density": mixed_density,           # [B, H, H] 混合态密度矩阵
            "mix_probabilities": mix_probs,           # [rank*3] 混合概率
            
            # 文本量子态
            "text_pure_states": text_pure_states,     # [B, rank, H]
            "text_amplitudes": text_amplitudes,       # [B, rank, H]
            "text_phases": text_phases,               # [B, rank, H]
            
            # 图像量子态
            "image_pure_states": image_pure_states,   # [B, rank, H]
            "image_amplitudes": image_amplitudes,     # [B, rank, H]
            "image_phases": image_phases,             # [B, rank, H]
            
            # 跨模态量子态
            "cross_pure_states": cross_pure_states,   # [B, rank, H]
            
            # 残差权重
            "residual_weight": alpha,                 # 标量
        }
        
        return output_dict

#量子态+混合态实例s
class SimpleBertMaeMixedStateEncoder(nn.Module):
    """
    简化版BERT/MAE混合态编码器
    
    适用于直接替换原有的特征提取流程，输出与原始维度兼容的特征
    
    输入:
        text_feature: [batch, 197, 768] BERT输出
        image_feature: [batch, 197, 768] MAE输出
    
    输出:
        text_quantum: [batch, 768] 文本量子态特征
        image_quantum: [batch, 768] 图像量子态特征
        fused_quantum: [batch, 768] 融合量子态特征
    """
    
    def __init__(
        self, 
        feature_dim: int = 768,
        hidden_dim: int = 256,
        rank: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.rank = rank
        
        # 完整版编码器
        self.mixed_encoder = mixedstate(
            input_dim=feature_dim,
            output_dim=feature_dim,  # 输出维度与输入一致
            hidden_dim=hidden_dim,
            rank=rank,
            dropout=dropout
        )
        
        # 单独的文本/图像量子编码器 (保持原始维度)
        self.text_quantum_encoder = SequenceQuantumPureStateEncoder(
            input_dim=feature_dim,
            output_dim=feature_dim,
            dropout=dropout
        )
        self.image_quantum_encoder = SequenceQuantumPureStateEncoder(
            input_dim=feature_dim,
            output_dim=feature_dim,
            dropout=dropout
        )
    
    def forward(
        self, 
        text_feature: torch.Tensor, 
        image_feature: torch.Tensor,
        text_mask: torch.Tensor = None
    ):
        """
        Args:
            text_feature: [batch, seq_len, 768] BERT输出
            image_feature: [batch, seq_len, 768] MAE输出
            text_mask: [batch, seq_len] 文本掩码
        
        Returns:
            text_quantum: [batch, 768] 文本量子态特征
            image_quantum: [batch, 768] 图像量子态特征
            fused_quantum: [batch, 768] 融合量子态特征
            quantum_info: dict 包含详细量子态信息
        """
        # 1. 单独编码文本量子态
        text_q, text_amp, text_phase, text_mod, _ = self.text_quantum_encoder(
            text_feature, text_mask
        )
        
        # 2. 单独编码图像量子态
        image_q, image_amp, image_phase, image_mod, _ = self.image_quantum_encoder(
            image_feature, None
        )
        
        # 3. 融合编码 (混合态)
        mixed_output = self.mixed_encoder(text_feature, image_feature, text_mask)
        fused_quantum = mixed_output["fused_feature"]
        
        # 4. 整理量子信息
        quantum_info = {
            "text_amplitude": text_amp,
            "text_phase": text_phase,
            "text_modulus": text_mod,
            "image_amplitude": image_amp,
            "image_phase": image_phase,
            "image_modulus": image_mod,
            "mixed_density": mixed_output["mixed_density"],
            "mix_probabilities": mixed_output["mix_probabilities"],
        }
        
        return text_q, image_q, fused_quantum, quantum_info


# ============ 量子干涉融合模块 ============
class QuantumInterferenceFusion(nn.Module):
    """
    基于真实量子态编码的图文量子干涉融合模块
    流程：图文分别编码为量子态 → 量子干涉叠加 → 干涉强度计算 → 特征提取
    
    适用场景:
    1. BERT/MAE序列特征的量子干涉
    2. 生成增强的融合特征，用于下游任务
    """
    def __init__(self, text_input_dim=768, img_input_dim=768, fusion_dim=768, dropout=0.1):
        super().__init__()
        self.fusion_dim = fusion_dim
        
        # 1. 文本量子态编码器（处理序列型文本特征，如BERT输出）
        self.text_quantum_encoder = SequenceQuantumPureStateEncoder(
            input_dim=text_input_dim,
            output_dim=fusion_dim,
            dropout=dropout
        )
        
        # 2. 图像量子态编码器（处理序列型图像特征，如MAE/ViT输出）
        self.img_quantum_encoder = SequenceQuantumPureStateEncoder(
            input_dim=img_input_dim,
            output_dim=fusion_dim,
            dropout=dropout
        )
        
        # 3. 干涉特征提取器（将干涉强度转换为有用特征）
        self.interference_extractor = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
        )
        
        # 4. 残差权重
        self.residual_weight = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, text_seq_feat, img_seq_feat, text_attn_mask=None, img_attn_mask=None):
        """
        Args:
            text_seq_feat: [batch, text_seq_len, text_input_dim] 文本序列特征
            img_seq_feat: [batch, img_seq_len, img_input_dim] 图像序列特征
            text_attn_mask: [batch, text_seq_len] 文本注意力掩码
            img_attn_mask: [batch, img_seq_len] 图像注意力掩码
        
        Returns:
            interference_feature: [batch, fusion_dim] 量子干涉融合特征
            interference_intensity: [batch, fusion_dim] 干涉强度
            text_psi: [batch, fusion_dim] 文本量子态波函数
            img_psi: [batch, fusion_dim] 图像量子态波函数
            quantum_info: dict 详细量子态信息
        """
        # 1. 分别编码文本和图像为量子态（复数量子波函数）
        text_psi, text_amp, text_phase, text_mod, text_pooled = \
            self.text_quantum_encoder(text_seq_feat, text_attn_mask)
        img_psi, img_amp, img_phase, img_mod, img_pooled = \
            self.img_quantum_encoder(img_seq_feat, img_attn_mask)
        
        # 2. 量子干涉叠加（叠加原理：总波函数 = 文本波函数 + 图像波函数）
        # 建设性干涉：增强互补信息；破坏性干涉：抑制冗余噪声
        psi_total = text_psi + img_psi
        
        # 3. 计算干涉后的强度（测量概率幅度，包含干涉项）
        # |ψ_total|² = |ψ_text|² + |ψ_img|² + 2·|ψ_text|·|ψ_img|·cos(θ_text - θ_img)
        interference_intensity = psi_total ** 2  # 已经是实数模长，直接平方
        
        # 4. 提取干涉特征
        interference_feature_pure = self.interference_extractor(interference_intensity)
        
        # 5. 残差连接（保留原始语义信息）
        alpha = torch.sigmoid(self.residual_weight)
        interference_feature = alpha * interference_feature_pure + (1 - alpha) * (text_pooled + img_pooled) / 2
        
        # 6. 整理量子态信息
        quantum_info = {
            "text_amplitude": text_amp,
            "text_phase": text_phase,
            "text_modulus": text_mod,
            "image_amplitude": img_amp,
            "image_phase": img_phase,
            "image_modulus": img_mod,
            "phase_difference": text_phase - img_phase,
            "residual_weight": alpha,
        }
        
        return interference_feature, interference_intensity, text_psi, img_psi, quantum_info
#         super().__init__()
#         # 1. 文本量子态编码器（处理序列型文本特征，如BERT输出）
#         self.text_quantum_encoder = SequenceQuantumPureStateEncoder(
#             input_dim=text_input_dim,
#             output_dim=fusion_dim
#         )
        
#         # 2. 图像量子态编码器（处理序列型图像特征，如MAE/ViT输出；若为全局特征可调整seq_len=1）
#         self.img_quantum_encoder = SequenceQuantumPureStateEncoder(
#             input_dim=img_input_dim,
#             output_dim=fusion_dim
#         )
        
#         # 3. 下游分类器（输入干涉强度特征，输出分类结果）
#         self.classifier = nn.Sequential(
#             nn.Linear(fusion_dim, fusion_dim // 2),
#             nn.LayerNorm(fusion_dim // 2),
#             nn.GELU(),
#             nn.Dropout(0.1),
#             nn.Linear(fusion_dim // 2, num_classes)
#         )
    
#     def forward(self, text_seq_feat, img_seq_feat, text_attn_mask=None, img_attn_mask=None):
#         """
#         Args:
#             text_seq_feat: [batch, text_seq_len, text_input_dim] 文本序列特征
#             img_seq_feat: [batch, img_seq_len, img_input_dim] 图像序列特征
#             text_attn_mask: [batch, text_seq_len] 文本注意力掩码
#             img_attn_mask: [batch, img_seq_len] 图像注意力掩码（若图像无padding可传None）
        
#         Returns:
#             logits: [batch, num_classes] 分类预测结果
#             interference_intensity: [batch, fusion_dim] 干涉后的强度特征
#             text_psi: [batch, fusion_dim] 文本量子态波函数
#             img_psi: [batch, fusion_dim] 图像量子态波函数
#         """
#         # 1. 分别编码文本和图像为量子态（复数量子波函数）
#         text_psi, text_amp, text_phase, text_mod, _ = self.text_quantum_encoder(text_seq_feat, text_attn_mask)
#         img_psi, img_amp, img_phase, img_mod, _ = self.img_quantum_encoder(img_seq_feat, img_attn_mask)
        
#         # 2. 量子干涉叠加（叠加原理：总波函数 = 文本波函数 + 图像波函数）
#         # 建设性干涉：增强互补信息；破坏性干涉：抑制冗余噪声
#         psi_total = text_psi + img_psi
        
#         # 3. 计算干涉后的强度（测量概率幅度，包含干涉项）
#         # |ψ_total|² = |ψ_text|² + |ψ_img|² + 2·|ψ_text|·|ψ_img|·cos(θ_text - θ_img)
#         interference_intensity = torch.abs(psi_total) ** 2
        
#         # 4. 分类预测
#         logits = self.classifier(interference_intensity)
        
#         return logits, interference_intensity, text_psi, img_psi

# # ===================== 测试代码 =====================
# if __name__ == "__main__":
#     print("=" * 60)
#     print("BERT/MAE 量子混合态编码器测试")
#     print("=" * 60)
    
#     # 模拟输入
#     BATCH_SIZE = 4
#     SEQ_LEN = 197
#     FEATURE_DIM = 768
    
#     text_feature = torch.randn(BATCH_SIZE, SEQ_LEN, FEATURE_DIM)
#     image_feature = torch.randn(BATCH_SIZE, SEQ_LEN, FEATURE_DIM)
#     text_mask = torch.ones(BATCH_SIZE, SEQ_LEN)
    
#     # 测试完整版编码器
#     print("\n--- 测试完整版 BertMaeMixedStateEncoder ---")
#     encoder = BertMaeMixedStateEncoder(
#         input_dim=768,
#         output_dim=320,
#         hidden_dim=128,
#         rank=4
#     )
    
#     output = encoder(text_feature, image_feature, text_mask)
    
#     print(f"融合特征形状: {output['fused_feature'].shape}")  # [4, 320]
#     print(f"量子特征形状: {output['quantum_feature'].shape}")  # [4, 320]
#     print(f"混合态密度矩阵形状: {output['mixed_density'].shape}")  # [4, 128, 128]
#     print(f"文本纯态形状: {output['text_pure_states'].shape}")  # [4, 4, 128]
#     print(f"图像纯态形状: {output['image_pure_states'].shape}")  # [4, 4, 128]
#     print(f"混合概率: {output['mix_probabilities'].detach().numpy()}")
#     print(f"残差权重: {output['residual_weight'].item():.4f}")
    
#     # 测试简化版编码器
#     print("\n--- 测试简化版 SimpleBertMaeMixedStateEncoder ---")
#     simple_encoder = SimpleBertMaeMixedStateEncoder(
#         feature_dim=768,
#         hidden_dim=256,
#         rank=2
#     )
    
#     text_q, image_q, fused_q, q_info = simple_encoder(
#         text_feature, image_feature, text_mask
#     )
    
#     print(f"文本量子态形状: {text_q.shape}")   # [4, 768]
#     print(f"图像量子态形状: {image_q.shape}")  # [4, 768]
#     print(f"融合量子态形状: {fused_q.shape}")  # [4, 768]
#     print(f"文本振幅范围: [{q_info['text_amplitude'].min():.4f}, {q_info['text_amplitude'].max():.4f}]")
#     print(f"文本相位范围: [{q_info['text_phase'].min():.4f}, {q_info['text_phase'].max():.4f}]")
    
#     print("\n✓ 所有测试通过！")
