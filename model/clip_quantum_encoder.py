
import torch
import torch.nn as nn
import numpy as np
import math

#1纯态构建
class CLIPQuantumEncoder(nn.Module):
    def __init__(self, input_dim=512, output_dim=320, use_phase_correlation=False):
        super(CLIPQuantumEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_phase_correlation = use_phase_correlation
        
        # ========== 振幅编码网络 ==========
        # 将CLIP特征映射为振幅（可正可负）
        # 振幅编码语义极性: 正=支持/相关, 负=反对/不相关
        self.amplitude_encoder = nn.Sequential(
            nn.Linear(input_dim, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim * 2, output_dim),
            nn.Tanh()  # 输出 [-1, 1]，支持正负振幅
        )
        
        # ========== 相位编码网络 ==========
        # 将CLIP特征映射为相位 θ ∈ [-π, π]
        self.phase_encoder = nn.Sequential(
            nn.Linear(input_dim, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim * 2, output_dim),
            nn.Tanh()  # 输出 [-1, 1], 乘以π得到 [-π, π]
        )
        
        # ========== 可选: 相位关联矩阵 ==========
        # 学习不同维度之间的相位关系
        if use_phase_correlation:
            self.phase_correlation = nn.Parameter(
                torch.randn(output_dim, output_dim) * 0.01
            )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
#量子态
    def forward(self, clip_feature):
        """
        前向传播: CLIP特征 → 量子态
        
        Args:
            clip_feature: [batch, 512] CLIP编码的图像或文本特征
            
        Returns:
            quantum_state: [batch, output_dim] 量子态（实部）
            amplitude: [batch, output_dim] 振幅 A
            phase: [batch, output_dim] 相位 θ
        """
        # ========== 1. 提取振幅 A ∈ [-1, 1] (可正可负) ==========
        amplitude = self.amplitude_encoder(clip_feature)  # [batch, output_dim]
        # Tanh输出 [-1, 1]，支持正负振幅
        # - A > 0: 正向语义（图文一致、支持、真实）
        # - A < 0: 反向语义（图文矛盾、反对、虚假）
        # - |A| 大: 语义强度高
        # - |A| 小: 语义强度低
        
        # ========== 2. 提取相位 θ ∈ [-π, π] ==========
        phase_raw = self.phase_encoder(clip_feature)  # [batch, output_dim] ∈ [-1, 1]
        phase = phase_raw * math.pi  # 缩放到 [-π, π]
        
        # 可选: 添加相位相关性（不同维度的相位相互影响）
        if self.use_phase_correlation:
            phase_corr = torch.matmul(phase, self.phase_correlation)
            phase = phase + 0.1 * phase_corr  # 小幅调整相位
        
        # ========== 3. 构造复数量子态 ψ = A·e^(iθ) ==========
        # 欧拉公式: e^(iθ) = cos(θ) + i·sin(θ)
        real_part = amplitude * torch.cos(phase)  # 实部: A·cos(θ)
        imag_part = amplitude * torch.sin(phase)  # 虚部: A·sin(θ)
        modulus = torch.sqrt(real_part.pow(2) + imag_part.pow(2) + 1e-8)  # |ψ| = sqrt(Re² + Im²)
        
        quantum_state = modulus  # 输出模长|ψ|，表示概率幅度
        
        return quantum_state, amplitude, phase
        # # ========== 4. 输出量子态（使用实部作为主要特征） ==========
        # quantum_state = real_part  # [batch, output_dim]
        
        # return quantum_state, amplitude, phase

##≠2两个纯态融合
class DualCLIPQuantumFusion(nn.Module):
    def __init__(self, clip_dim=512, output_dim=320, fusion_type='concat'):
        super(DualCLIPQuantumFusion, self).__init__()
        
        self.clip_dim = clip_dim
        self.output_dim = output_dim
        self.fusion_type = fusion_type  # 'concat', 'weighted', 'interference'
        
        # ========== 图像量子态编码器 ==========
        self.image_quantum_encoder = CLIPQuantumEncoder(
            input_dim=clip_dim,
            output_dim=output_dim,
            use_phase_correlation=False
        )
        
        # ========== 文本量子态编码器 ==========
        self.text_quantum_encoder = CLIPQuantumEncoder(
            input_dim=clip_dim,
            output_dim=output_dim,
            use_phase_correlation=False
        )
        
        # ========== 量子态融合网络 ==========
        if fusion_type == 'concat':
            # 拼接融合
            self.fusion_net = nn.Sequential(
                nn.Linear(output_dim * 2, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(output_dim, output_dim)
            )
        elif fusion_type == 'weighted':
            # 加权融合
            self.fusion_weight = nn.Parameter(torch.tensor(0.5))
        elif fusion_type == 'interference':
            # 量子干涉融合
            self.interference_net = nn.Sequential(
                nn.Linear(output_dim * 2, output_dim),
                nn.Tanh()
            )
        
    def forward(self, clip_image_feature, clip_text_feature):
        """
        前向传播: 双路CLIP特征 → 量子态融合
        
        Args:
            clip_image_feature: [batch, 512] CLIP图像特征
            clip_text_feature: [batch, 512] CLIP文本特征
            
        Returns:
            fused_quantum_state: [batch, output_dim] 融合后的量子态
            image_amplitude: [batch, output_dim] 图像振幅
            text_amplitude: [batch, output_dim] 文本振幅
            phase_coherence: [batch] 图文相位相干性
        """
        # ========== 1. 分别编码图像和文本为量子态 ==========
        image_quantum, image_amplitude, image_phase = self.image_quantum_encoder(clip_image_feature)
        text_quantum, text_amplitude, text_phase = self.text_quantum_encoder(clip_text_feature)
        
        # ========== 2. 计算相位相干性 ==========
        # 相位差
        phase_diff = image_phase - text_phase  # [batch, output_dim]
        
        # 相位相干性: ⟨cos(Δθ)⟩ ∈ [-1, 1]
        # cos(0) = 1 (完全相干), cos(π) = -1 (完全反相)
        phase_coherence = torch.cos(phase_diff).mean(dim=-1)  # [batch]
        
        # 归一化到 [0, 1]: (cos(Δθ) + 1) / 2
        phase_coherence_normalized = (phase_coherence + 1) / 2
        
        # ========== 3. 量子态融合 ==========
        if self.fusion_type == 'concat':
            # ------------- 新增：维度统一处理（核心修正）-------------
            # 先获取两个张量的维度数
            img_dim = image_quantum.dim()
            txt_dim = text_quantum.dim()
            
            # 情况1：图像2维、文本3维（当前报错场景，优先将文本降为2维，推荐方案）
            if img_dim == 2 and txt_dim == 3:
                # 可选方式1：文本张量均值池化降维（保留全局语义，通用）
                text_quantum_permuted = text_quantum.permute(0, 2, 1)  # [B, L, D] → [B, D, L]
                text_quantum_pooled = F.avg_pool1d(text_quantum_permuted, kernel_size=text_quantum.shape[1])
                text_quantum_2d = text_quantum_pooled.squeeze(dim=-1)  # [B, D, 1] → [B, D]
                
                # 可选方式2：取文本CLS token（若文本是Transformer输出，更简洁）
                # text_quantum_2d = text_quantum[:, 0, :]  # [B, L, D] → [B, D]
                
                # 此时两者均为2维，执行拼接
                concat_quantum = torch.cat([image_quantum, text_quantum_2d], dim=-1)
            
            # 情况2：图像3维、文本2维（反向场景，备用）
            elif img_dim == 3 and txt_dim == 2:
                # 将图像升维匹配文本，或图像降维为2维
                image_quantum_2d = image_quantum[:, 0, :]  # 取图像第0个token降维
                concat_quantum = torch.cat([image_quantum_2d, text_quantum], dim=-1)
            
            # 情况3：两者维度一致（无需处理，原有逻辑）
            else:
                concat_quantum = torch.cat([image_quantum, text_quantum], dim=-1)
            
            # 原有融合网络逻辑不变
            fused_quantum_state = self.fusion_net(concat_quantum)
        # if self.fusion_type == 'concat':
        #     # 拼接融合
        #     concat_quantum = torch.cat([image_quantum, text_quantum], dim=-1)
        #     fused_quantum_state = self.fusion_net(concat_quantum)
            
        elif self.fusion_type == 'weighted':
            # 加权融合（相位相干性作为权重）
            weight = torch.sigmoid(self.fusion_weight)
            fused_quantum_state = weight * image_quantum + (1 - weight) * text_quantum
            
        elif self.fusion_type == 'interference':
            # 量子干涉融合（考虑相位关系）
            # 干涉项: ψ_int = √(A_img·A_txt)·cos(θ_img - θ_txt)
            interference_amplitude = torch.sqrt(image_amplitude * text_amplitude + 1e-8)
            interference_term = interference_amplitude * torch.cos(phase_diff)
            
            # 总态: ψ_fused = ψ_img + ψ_txt + ψ_int
            base_fusion = image_quantum + text_quantum
            concat_features = torch.cat([base_fusion, interference_term], dim=-1)
            fused_quantum_state = self.interference_net(concat_features)
        
        return fused_quantum_state, image_amplitude, text_amplitude, phase_coherence_normalized


# ============ CLIP量子干涉融合模块 ============
class CLIPQuantumInterferenceFusion(nn.Module):
    def __init__(self, clip_dim=512, fusion_dim=320, dropout=0.1):
        super().__init__()
        self.clip_dim = clip_dim
        self.fusion_dim = fusion_dim
        
        # 1. 图像和文本量子态编码器
        self.image_quantum_encoder = CLIPQuantumEncoder(
            input_dim=clip_dim,
            output_dim=fusion_dim,
            use_phase_correlation=True
        )
        self.text_quantum_encoder = CLIPQuantumEncoder(
            input_dim=clip_dim,
            output_dim=fusion_dim,
            use_phase_correlation=True
        )
        
        # 2. 干涉特征提取器
        self.interference_extractor = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
        )
        
        # 3. 残差权重
        self.residual_weight = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, clip_image_feature, clip_text_feature):
        """
        Args:
            clip_image_feature: [batch, clip_dim] CLIP图像特征
            clip_text_feature: [batch, clip_dim] CLIP文本特征
        
        Returns:
            interference_feature: [batch, fusion_dim] 量子干涉融合特征
            interference_intensity: [batch, fusion_dim] 干涉强度
            image_psi: [batch, fusion_dim] 图像量子态
            text_psi: [batch, fusion_dim] 文本量子态
            quantum_info: dict 量子态信息
        """
        # 1. 编码为量子态
        image_psi, image_amp, image_phase = self.image_quantum_encoder(clip_image_feature)
        text_psi, text_amp, text_phase = self.text_quantum_encoder(clip_text_feature)
        
        # 2. 量子干涉叠加
        psi_total = image_psi + text_psi
        
        # 3. 计算干涉强度
        interference_intensity = psi_total ** 2
        
        # 4. 提取干涉特征
        interference_feature_pure = self.interference_extractor(interference_intensity)
        
        # 5. 残差连接
        alpha = torch.sigmoid(self.residual_weight)
        residual = (image_psi + text_psi) / 2
        interference_feature = alpha * interference_feature_pure + (1 - alpha) * residual
        
        # 6. 整理量子信息
        phase_diff = image_phase - text_phase
        phase_coherence = torch.cos(phase_diff).mean(dim=-1)
        
        quantum_info = {
            "image_amplitude": image_amp,
            "image_phase": image_phase,
            "text_amplitude": text_amp,
            "text_phase": text_phase,
            "phase_difference": phase_diff,
            "phase_coherence": phase_coherence,
            "residual_weight": alpha,
        }
        
        return interference_feature, interference_intensity, image_psi, text_psi, quantum_info


#3计算量子相似度

def compute_quantum_fidelity(amp1, phase1, amp2, phase2):
    # 重构复数量子态
    psi1_real = amp1 * torch.cos(phase1)
    psi1_imag = amp1 * torch.sin(phase1)
    
    psi2_real = amp2 * torch.cos(phase2)
    psi2_imag = amp2 * torch.sin(phase2)
    
    # 计算内积 ⟨ψ1|ψ2⟩ = Σ(ψ1*·ψ2)
    # 复数共轭: ψ1* = A1·e^(-iθ1)
    inner_product_real = (psi1_real * psi2_real + psi1_imag * psi2_imag).sum(dim=-1)
    inner_product_imag = (psi1_real * psi2_imag - psi1_imag * psi2_real).sum(dim=-1)
    
    # 保真度 F = |⟨ψ1|ψ2⟩|²
    fidelity = inner_product_real ** 2 + inner_product_imag ** 2
    
    return fidelity

#4冯诺依曼熵
def compute_von_neumann_entropy(amplitude):

    # 概率分布 p_i = |ψ_i|²
    probability = amplitude ** 2
    probability = probability / (probability.sum(dim=-1, keepdim=True) + 1e-8)
    
    # 熵: S = -Σ p_i·log(p_i)
    entropy = -(probability * torch.log(probability + 1e-8)).sum(dim=-1)
    
    return entropy

