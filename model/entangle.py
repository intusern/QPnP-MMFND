"""
量子纠缠对齐模块
建模图文特征的纠缠关联，通过余弦相似度 + 矩阵变换 + 残差连接指导特征对齐
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantumEntanglementAlign(nn.Module):
    """量子纠缠对齐：建模图文特征的纠缠关联，指导特征对齐"""

    def __init__(self, dim, device=None, dtype=torch.float32):
        super().__init__()
        # 1. 移除硬编码512，用传入的dim参数适配输入维度
        # 2. 初始化时指定dtype和device，对齐模型的混合精度配置
        self.entanglement_matrix = nn.Parameter(
            torch.randn(dim, dim, dtype=dtype, device=device)
        )
        self.scale = nn.Parameter(
            torch.ones(1, dtype=dtype, device=device) * 0.1
        )
        self.dim = dim

    def forward(self, text_feat, image_feat):
        """
        输入：文本特征[B, dim]、图像特征[B, dim]
        输出：对齐后的文本/图像特征 [B, dim]
        原理：计算纠缠度（余弦相似度），用纠缠矩阵调整特征分布
        """
        # ========== 关键修复1：统一所有张量的dtype和device ==========
        # 确保输入特征与模型参数 dtype/device 一致（适配混合精度）
        text_feat = text_feat.to(dtype=self.entanglement_matrix.dtype, device=self.entanglement_matrix.device)
        image_feat = image_feat.to(dtype=self.entanglement_matrix.dtype, device=self.entanglement_matrix.device)

        # ========== 关键修复2：计算纠缠度，保证维度广播兼容 ==========
        # 纠缠度：[B] → 扩展为 [B, 1]（而非[B,1,1]），避免后续squeeze出错
        entanglement = F.cosine_similarity(text_feat, image_feat, dim=-1, eps=1e-6)
        entanglement = entanglement.unsqueeze(-1)  # [B] → [B, 1]

        # ========== 关键修复3：矩阵乘法（维度已匹配：[B,dim] @ [dim,dim] = [B,dim]） ==========
        # 纠缠变换：保证所有张量dtype一致
        scale_factor = self.scale * entanglement  # [B, 1]（广播到[B,dim]）
        text_align = text_feat @ self.entanglement_matrix * scale_factor
        image_align = image_feat @ self.entanglement_matrix.T * scale_factor

        # ========== 关键修复4：残差连接（无需squeeze，避免维度丢失） ==========
        # 原squeeze(1)会导致维度错误，现text_align/image_align已为[B,dim]，直接残差
        text_align = text_feat + text_align
        image_align = image_feat + image_align

        return text_align, image_align
