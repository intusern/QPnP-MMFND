"""
M3FEND量子态增强版本 - m3ganshenew.py

基于M3FEND原模型，对以下特征进行量子态和量子干涉增强：
1. emotion_feature (235维) 
2. gate_input_feature (768维，从BERT attention得到)
3. memory_att 相关的特征融合

量子增强核心思想：
- 量子纯态编码：将特征编码为波函数 ψ = A·e^(iθ)
- 量子干涉融合：通过干涉增强特征间的互补信息
- 残差连接：保留原始语义信息

依赖模块：
- clip_quantum_encoder: 量子纯态构建 + 融合 + 干涉
- bert_mae_mixed_state_encoder: BERT/MAE量子态 + 干涉
"""

import os
import torch
from torch.autograd import Variable
import tqdm
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .layers import *
from sklearn.metrics import *
from transformers import BertModel
from transformers import RobertaModel
from utils.utils import data2gpu, Averager, metrics, Recorder, metricsTrueFalse
import logging
import math
from sklearn.cluster import KMeans
import numpy as np
from torch.nn.parameter import Parameter

# 导入量子编码器模块
from .clip_quantum_encoder import DualCLIPQuantumFusion, CLIPQuantumInterferenceFusion
from .bert_mae_mixed_state_encoder import SimpleBertMaeMixedStateEncoder, QuantumInterferenceFusion

model_path = "./pretrained_model/chinese_roberta_wwm_base_ext_pytorch"


def cal_length(x):
    return torch.sqrt(torch.sum(torch.pow(x, 2), dim=1))


def norm(x):
    length = cal_length(x).view(-1, 1)
    x = x / length
    return x


# 设备检测，默认使用GPU 0
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def convert_to_onehot(label, batch_size, num):
    return torch.zeros(batch_size, num).to(device).scatter_(1, label, 1)


# ========== 1D特征量子态编码器（适用于非序列特征如emotion_feature、style_feature） ==========
class FeatureQuantumEncoder(nn.Module):
    """
    1D特征量子态编码器
    
    适用于固定维度的特征向量（非序列特征）
    将特征编码为量子态 ψ = A·e^(iθ)
    
    输入: [batch, input_dim] 
    输出: 量子态实部(模长), 振幅A, 相位θ
    """
    
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 特征投影层
        self.feature_proj = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # 振幅编码器: 输出 A ∈ [-1, 1]
        self.amplitude_encoder = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
            nn.Tanh(),
        )
        
        # 相位编码器: 输出 θ ∈ [-π, π]
        self.phase_encoder = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
            nn.Tanh(),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, feature: torch.Tensor):
        """
        Args:
            feature: [batch, input_dim] 输入特征
        
        Returns:
            quantum_state: [batch, output_dim] 量子态（模长）
            amplitude: [batch, output_dim] 振幅 A ∈ [-1, 1]
            phase: [batch, output_dim] 相位 θ ∈ [-π, π]
            projected: [batch, output_dim] 投影后的特征
        """
        # 特征投影
        projected = self.feature_proj(feature)
        
        # 振幅编码
        amplitude = self.amplitude_encoder(projected)
        
        # 相位编码
        phase = self.phase_encoder(projected) * math.pi
        
        # 波函数计算 (欧拉公式)
        real = amplitude * torch.cos(phase)
        imag = amplitude * torch.sin(phase)
        modulus = torch.sqrt(real.pow(2) + imag.pow(2) + 1e-8)
        
        quantum_state = modulus
        
        return quantum_state, amplitude, phase, projected


# ========== 双特征量子干涉融合模块 ==========
class DualFeatureQuantumInterference(nn.Module):
    """
    双特征量子干涉融合模块
    
    将两个不同维度的特征编码为量子态，通过干涉融合
    适用于 emotion_feature 与 style_feature 的融合
    """
    
    def __init__(self, input_dim1: int, input_dim2: int, fusion_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fusion_dim = fusion_dim
        
        # 特征1量子编码器
        self.encoder1 = FeatureQuantumEncoder(input_dim1, fusion_dim, dropout)
        
        # 特征2量子编码器
        self.encoder2 = FeatureQuantumEncoder(input_dim2, fusion_dim, dropout)
        
        # 干涉特征提取器
        self.interference_extractor = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
        )
        
        # 残差权重
        self.residual_weight = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, feat1: torch.Tensor, feat2: torch.Tensor):
        """
        Args:
            feat1: [batch, input_dim1] 特征1（如emotion_feature）
            feat2: [batch, input_dim2] 特征2（如style_feature）
        
        Returns:
            interference_feature: [batch, fusion_dim] 量子干涉融合特征
            interference_intensity: [batch, fusion_dim] 干涉强度
            psi1: [batch, fusion_dim] 特征1量子态
            psi2: [batch, fusion_dim] 特征2量子态
            quantum_info: dict 量子态信息
        """
        # 分别编码为量子态
        psi1, amp1, phase1, proj1 = self.encoder1(feat1)
        psi2, amp2, phase2, proj2 = self.encoder2(feat2)
        
        # 量子干涉叠加
        psi_total = psi1 + psi2
        
        # 干涉强度
        interference_intensity = psi_total ** 2
        
        # 提取干涉特征
        interference_feature_pure = self.interference_extractor(interference_intensity)
        
        # 残差连接
        alpha = torch.sigmoid(self.residual_weight)
        residual = (proj1 + proj2) / 2
        interference_feature = alpha * interference_feature_pure + (1 - alpha) * residual
        
        # 量子信息
        phase_diff = phase1 - phase2
        phase_coherence = torch.cos(phase_diff).mean(dim=-1)
        
        quantum_info = {
            "amplitude1": amp1,
            "phase1": phase1,
            "amplitude2": amp2,
            "phase2": phase2,
            "phase_difference": phase_diff,
            "phase_coherence": phase_coherence,
            "residual_weight": alpha,
        }
        
        return interference_feature, interference_intensity, psi1, psi2, quantum_info


# ========== 序列特征量子态编码器（适用于gate_input_feature等） ==========
class SequenceQuantumPureStateEncoder(nn.Module):
    """
    序列特征量子态编码器
    
    适用于带序列维度的特征或需要attention池化的特征
    """
    
    def __init__(self, input_dim: int = 768, output_dim: int = 768, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 特征投影层
        self.feature_proj = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # 振幅编码器
        self.amplitude_encoder = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
            nn.Tanh(),
        )
        
        # 相位编码器
        self.phase_encoder = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
            nn.Tanh(),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, feature: torch.Tensor):
        """
        Args:
            feature: [batch, input_dim] 输入特征（已池化的序列特征）
        
        Returns:
            quantum_state: [batch, output_dim] 量子态
            amplitude: [batch, output_dim] 振幅
            phase: [batch, output_dim] 相位
            projected: [batch, output_dim] 投影后的特征
        """
        projected = self.feature_proj(feature)
        amplitude = self.amplitude_encoder(projected)
        phase = self.phase_encoder(projected) * math.pi
        
        real = amplitude * torch.cos(phase)
        imag = amplitude * torch.sin(phase)
        modulus = torch.sqrt(real.pow(2) + imag.pow(2) + 1e-8)
        
        return modulus, amplitude, phase, projected


class MemoryNetwork(torch.nn.Module):
    def __init__(self, input_dim, emb_dim, domain_num, memory_num=10):
        super(MemoryNetwork, self).__init__()
        self.domain_num = domain_num
        self.emb_dim = emb_dim
        self.memory_num = memory_num
        self.tau = 32
        self.topic_fc = torch.nn.Linear(input_dim, emb_dim, bias=False)
        self.domain_fc = torch.nn.Linear(input_dim, emb_dim, bias=False)

        self.domain_memory = dict()

    def forward(self, feature, category):
        feature = norm(feature)
        domain_label = torch.tensor([index for index in category]).view(-1, 1).to(device)
        domain_memory = []
        for i in range(self.domain_num):
            domain_memory.append(self.domain_memory[i])

        sep_domain_embedding = []
        for i in range(self.domain_num):
            topic_att = torch.nn.functional.softmax(torch.mm(self.topic_fc(feature), domain_memory[i].T) * self.tau, dim=1)
            tmp_domain_embedding = torch.mm(topic_att, domain_memory[i])
            sep_domain_embedding.append(tmp_domain_embedding.unsqueeze(1))
        sep_domain_embedding = torch.cat(sep_domain_embedding, 1)

        domain_att = torch.bmm(sep_domain_embedding, self.domain_fc(feature).unsqueeze(2)).squeeze()
        
        domain_att = torch.nn.functional.softmax(domain_att * self.tau, dim=1).unsqueeze(1)

        return domain_att

    def write(self, all_feature, category):
        domain_fea_dict = {}
        domain_set = set(category.cpu().detach().numpy().tolist())
        for i in domain_set:
            domain_fea_dict[i] = []
        for i in range(all_feature.size(0)):
            domain_fea_dict[category[i].item()].append(all_feature[i].view(1, -1))

        for i in domain_set:
            domain_fea_dict[i] = torch.cat(domain_fea_dict[i], 0)
            topic_att = torch.nn.functional.softmax(torch.mm(self.topic_fc(domain_fea_dict[i]), self.domain_memory[i].T) * self.tau, dim=1).unsqueeze(2)
            tmp_fea = domain_fea_dict[i].unsqueeze(1).repeat(1, self.memory_num, 1)
            new_mem = tmp_fea * topic_att
            new_mem = new_mem.mean(dim=0)
            topic_att = torch.mean(topic_att, 0).view(-1, 1)
            self.domain_memory[i] = self.domain_memory[i] - 0.05 * topic_att * self.domain_memory[i] + 0.05 * new_mem


class M3GanSheNewModel(torch.nn.Module):
    """
    M3FEND量子态增强模型
    
    在原M3FEND基础上增加：
    1. emotion_feature + style_feature 的量子干涉融合
    2. gate_input_feature 的量子态编码
    3. domain_embedding 与 general_domain_embedding 的量子干涉融合
    """
    
    def __init__(self, emb_dim, mlp_dims, dropout, semantic_num, emotion_num, style_num, LNN_dim, domain_num, dataset='ch'):
        super(M3GanSheNewModel, self).__init__()
        self.domain_num = domain_num
        self.gamma = 10
        self.memory_num = 10
        self.semantic_num_expert = semantic_num
        self.emotion_num_expert = emotion_num
        self.style_num_expert = style_num
        self.LNN_dim = LNN_dim
        self.dataset = dataset
        print('semantic_num_expert:', self.semantic_num_expert, 'emotion_num_expert:', self.emotion_num_expert, 'style_num_expert:', self.style_num_expert, 'lnn_dim:', self.LNN_dim)
        self.fea_size = 256
        self.emb_dim = emb_dim
        
        if dataset == 'weibo' or dataset == 'ch':
            self.bert = BertModel.from_pretrained(model_path, local_files_only=True)
            self.emotion_dim = 47 * 5  # 235
            self.style_dim = 48
        elif dataset == 'en':
            self.bert = RobertaModel.from_pretrained('roberta-base').requires_grad_(False)
            self.emotion_dim = 38 * 5  # 190
            self.style_dim = 32
        
        feature_kernel = {1: 64, 2: 64, 3: 64, 5: 64, 10: 64}

        content_expert = []
        for i in range(self.semantic_num_expert):
            content_expert.append(cnn_extractor(feature_kernel, emb_dim))
        self.content_expert = nn.ModuleList(content_expert)

        emotion_expert = []
        for i in range(self.emotion_num_expert):
            if dataset == 'ch' or dataset == 'weibo':
                emotion_expert.append(MLP(47 * 5, [256, 320,], dropout, output_layer=False))
            elif dataset == 'en':
                emotion_expert.append(MLP(38 * 5, [256, 320,], dropout, output_layer=False))
        self.emotion_expert = nn.ModuleList(emotion_expert)

        style_expert = []
        for i in range(self.style_num_expert):
            if dataset == 'ch' or dataset == 'weibo':
                style_expert.append(MLP(48, [256, 320,], dropout, output_layer=False))
            elif dataset == 'en':
                style_expert.append(MLP(32, [256, 320,], dropout, output_layer=False))
        self.style_expert = nn.ModuleList(style_expert)

        self.gate = nn.Sequential(nn.Linear(self.emb_dim * 2, mlp_dims[-1]),
                                      nn.ReLU(),
                                      nn.Linear(mlp_dims[-1], self.LNN_dim),
                                      nn.Softmax(dim=1))

        self.attention = MaskAttention(emb_dim)

        # 注册为buffer而不是直接unsqueeze，这样可以随模型移动到正确的设备
        self.register_buffer('weight', torch.Tensor(self.LNN_dim, self.semantic_num_expert + self.emotion_num_expert + self.style_num_expert).unsqueeze(0))
        stdv = 1. / math.sqrt(self.weight.size(2))
        self.weight.data.uniform_(-stdv, stdv)

        if dataset == 'ch' or dataset == 'weibo':
            self.domain_memory = MemoryNetwork(input_dim=self.emb_dim + 47 * 5 + 48, emb_dim=self.emb_dim + 47 * 5 + 48, domain_num=self.domain_num, memory_num=self.memory_num)
        elif dataset == 'en':
            self.domain_memory = MemoryNetwork(input_dim=self.emb_dim + 38 * 5 + 32, emb_dim=self.emb_dim + 38 * 5 + 32, domain_num=self.domain_num, memory_num=self.memory_num)

        self.domain_embedder = nn.Embedding(num_embeddings=self.domain_num, embedding_dim=emb_dim)
        self.all_feature = {}

        self.classifier = MLP(320, mlp_dims, dropout)
        
        # ========== 量子态增强模块 ==========
        
        # 1. emotion_feature + style_feature 量子干涉融合
        # emotion_feature: 235维, style_feature: 48维 -> 融合为统一维度
        self.quantum_unified_dim = 256  # 统一的量子态维度
        self.emotion_style_quantum_fusion = DualFeatureQuantumInterference(
            input_dim1=self.emotion_dim,   # 235
            input_dim2=self.style_dim,     # 48
            fusion_dim=self.quantum_unified_dim,
            dropout=dropout
        )
        
        # 2. gate_input_feature 量子态编码器 (768维 -> 768维)
        self.gate_input_quantum_encoder = SequenceQuantumPureStateEncoder(
            input_dim=self.emb_dim,       # 768
            output_dim=self.emb_dim,      # 768
            dropout=dropout
        )
        
        # 3. domain_embedding 与 general_domain_embedding 量子干涉融合
        # 两者都是768维
        self.domain_quantum_fusion = DualFeatureQuantumInterference(
            input_dim1=self.emb_dim,      # 768
            input_dim2=self.emb_dim,      # 768
            fusion_dim=self.emb_dim,      # 768
            dropout=dropout
        )
        
        # 4. 量子增强后的gate (需要调整输入维度)
        # 原始gate_input = [domain_embedding, general_domain_embedding] = 768*2 = 1536
        # 量子增强后gate_input = [quantum_domain_fusion, gate_input_quantum] = 768 + 768 = 1536
        # 所以维度保持不变
        
        # 5. 量子特征增强层 (将量子特征融合回原始特征空间)
        self.emotion_quantum_enhance = nn.Sequential(
            nn.Linear(self.quantum_unified_dim, self.emotion_dim),
            nn.LayerNorm(self.emotion_dim),
            nn.SiLU(),
            nn.Dropout(dropout)
        )
        
        self.style_quantum_enhance = nn.Sequential(
            nn.Linear(self.quantum_unified_dim, self.style_dim),
            nn.LayerNorm(self.style_dim),
            nn.SiLU(),
            nn.Dropout(dropout)
        )
        
        # 6. 量子增强系数 (可学习)
        self.quantum_alpha = nn.Parameter(torch.tensor(0.4))
        
    def forward(self, **kwargs):
        content = kwargs['content']
        content_masks = kwargs['content_masks']

        content_emotion = kwargs['content_emotion']
        comments_emotion = kwargs['comments_emotion']
        emotion_gap = kwargs['emotion_gap']
        style_feature = kwargs['style_feature']
        emotion_feature = torch.cat([content_emotion, comments_emotion, emotion_gap], dim=1)
        category = kwargs['category']
        
        content_feature = self.bert(content, attention_mask=content_masks)[0]

        gate_input_feature, _ = self.attention(content_feature, content_masks)
        
        # ========== 量子态增强1: emotion_feature + style_feature 量子干涉 ==========
        emotion_style_quantum, emotion_style_intensity, emotion_psi, style_psi, es_quantum_info = \
            self.emotion_style_quantum_fusion(emotion_feature, style_feature)
        
        # 将量子特征增强回原始维度
        emotion_quantum_enhanced = self.emotion_quantum_enhance(emotion_style_quantum)
        style_quantum_enhanced = self.style_quantum_enhance(emotion_style_quantum)
        
        # 残差连接增强原始特征
        alpha = torch.sigmoid(self.quantum_alpha)
        emotion_feature_enhanced = emotion_feature + alpha * emotion_quantum_enhanced
        style_feature_enhanced = style_feature + alpha * style_quantum_enhanced
        
        # ========== 量子态增强2: gate_input_feature 量子态编码 ==========
        gate_input_quantum, gate_input_amp, gate_input_phase, gate_input_proj = \
            self.gate_input_quantum_encoder(gate_input_feature)
        
        # 增强gate_input_feature
        gate_input_feature_enhanced = gate_input_feature + alpha * gate_input_quantum
        
        # ========== Memory Network (使用原始特征，保持稳定性) ==========
        memory_att = self.domain_memory(torch.cat([gate_input_feature, emotion_feature, style_feature], dim=-1), category)
        domain_emb_all = self.domain_embedder(torch.LongTensor(range(self.domain_num)).to(device))
        general_domain_embedding = torch.mm(memory_att.squeeze(1), domain_emb_all)

        idxs = torch.tensor([index for index in category]).view(-1, 1).to(device)
        domain_embedding = self.domain_embedder(idxs).squeeze(1)
        
        # ========== 量子态增强3: domain_embedding 与 general_domain_embedding 量子干涉 ==========
        domain_quantum_fusion, domain_intensity, domain_psi, general_psi, domain_quantum_info = \
            self.domain_quantum_fusion(domain_embedding, general_domain_embedding)
        
        # 构建增强后的gate_input
        # 原始: gate_input = [domain_embedding, general_domain_embedding]
        # 增强: 使用量子干涉融合特征 + 量子态编码的gate_input_feature
        gate_input = torch.cat([domain_quantum_fusion, gate_input_feature_enhanced], dim=-1)
        
        gate_value = self.gate(gate_input).view(content_feature.size(0), 1, self.LNN_dim)

        shared_feature = []
        for i in range(self.semantic_num_expert):
            shared_feature.append(self.content_expert[i](content_feature).unsqueeze(1))

        # 使用增强后的emotion_feature
        for i in range(self.emotion_num_expert):
            shared_feature.append(self.emotion_expert[i](emotion_feature_enhanced).unsqueeze(1))

        # 使用增强后的style_feature
        for i in range(self.style_num_expert):
            shared_feature.append(self.style_expert[i](style_feature_enhanced).unsqueeze(1))

        shared_feature = torch.cat(shared_feature, dim=1)

        embed_x_abs = torch.abs(shared_feature)
        embed_x_afn = torch.add(embed_x_abs, 1e-7)
        embed_x_log = torch.log1p(embed_x_afn)

        lnn_out = torch.matmul(self.weight, embed_x_log)
        lnn_exp = torch.expm1(lnn_out)
        shared_feature = lnn_exp.contiguous().view(-1, self.LNN_dim, 320)

        shared_feature = torch.bmm(gate_value, shared_feature).squeeze()
        
        deep_logits = self.classifier(shared_feature)

        return torch.sigmoid(deep_logits.squeeze(1))

    def save_feature(self, **kwargs):
        content = kwargs['content']
        content_masks = kwargs['content_masks']

        content_emotion = kwargs['content_emotion']
        comments_emotion = kwargs['comments_emotion']
        emotion_gap = kwargs['emotion_gap']
        emotion_feature = torch.cat([content_emotion, comments_emotion, emotion_gap], dim=1)

        style_feature = kwargs['style_feature']

        category = kwargs['category']

        content_feature = self.bert(content, attention_mask=content_masks)[0]
        content_feature, _ = self.attention(content_feature, content_masks)

        all_feature = torch.cat([content_feature, emotion_feature, style_feature], dim=1)
        all_feature = norm(all_feature)

        for index in range(all_feature.size(0)):
            domain = int(category[index].cpu().numpy())
            if not (domain in self.all_feature):
                self.all_feature[domain] = []
            self.all_feature[domain].append(all_feature[index].view(1, -1).cpu().detach().numpy())

    def init_memory(self):
        for domain in self.all_feature:
            all_feature = np.concatenate(self.all_feature[domain])
            kmeans = KMeans(n_clusters=self.memory_num, init='k-means++').fit(all_feature)
            centers = kmeans.cluster_centers_
            centers = torch.from_numpy(centers).to(device)
            self.domain_memory.domain_memory[domain] = centers

    def write(self, **kwargs):
        content = kwargs['content']
        content_masks = kwargs['content_masks']

        content_emotion = kwargs['content_emotion']
        comments_emotion = kwargs['comments_emotion']
        emotion_gap = kwargs['emotion_gap']
        emotion_feature = torch.cat([content_emotion, comments_emotion, emotion_gap], dim=1)

        style_feature = kwargs['style_feature']

        category = kwargs['category']

        content_feature = self.bert(content, attention_mask=content_masks)[0]
        content_feature, _ = self.attention(content_feature, content_masks)

        all_feature = torch.cat([content_feature, emotion_feature, style_feature], dim=1)
        all_feature = norm(all_feature)
        self.domain_memory.write(all_feature, category)


class Trainer():
    def __init__(self,
                 emb_dim,
                 mlp_dims,
                 use_cuda,
                 lr,
                 dropout,
                 train_loader,
                 val_loader,
                 test_loader,
                 category_dict,
                 weight_decay,
                 save_param_dir, 
                 semantic_num,
                 emotion_num,
                 style_num,
                 lnn_dim,
                 early_stop=5,
                 epoches=100,
                 dataset='ch'
                 ):
        self.lr = lr
        self.weight_decay = weight_decay
        self.use_cuda = use_cuda
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.early_stop = early_stop
        self.epoches = epoches
        self.category_dict = category_dict
        self.use_cuda = use_cuda
        self.dataset = dataset

        self.emb_dim = emb_dim
        self.mlp_dims = mlp_dims
        self.dropout = dropout
        self.semantic_num = semantic_num
        self.emotion_num = emotion_num
        self.style_num = style_num
        self.lnn_dim = lnn_dim

        if os.path.exists(save_param_dir):
            self.save_param_dir = save_param_dir
        else:
            self.save_param_dir = save_param_dir
            os.makedirs(save_param_dir)

    def train(self, logger=None, resume=False):
        if(logger):
            logger.info('start training......')

        self.model = M3GanSheNewModel(self.emb_dim, self.mlp_dims, self.dropout, self.semantic_num, self.emotion_num, self.style_num, self.lnn_dim, len(self.category_dict), self.dataset)
        # 统一使用device变量
        self.model = self.model.to(device)
        print(f"Model moved to: {device}")

        # 检查是否存在已训练的模型，如果resume=True则直接加载并测试
        model_path_saved = os.path.join(self.save_param_dir, 'parameter_m3ganshenew.pkl')
        if resume and os.path.exists(model_path_saved):
            print(f"发现已保存的模型: {model_path_saved}，直接加载进行测试...")
            self.model.load_state_dict(torch.load(model_path_saved))
            # 初始化记忆网络
            self.model.train()
            train_data_iter = tqdm.tqdm(self.train_loader)
            for step_n, batch in enumerate(train_data_iter):
                batch_data = data2gpu(batch, self.use_cuda)
                label_pred = self.model.save_feature(**batch_data)
            self.model.init_memory()
            # 直接进行测试
            print("开始进行最后的测试")
            results = self.test(self.test_loader)
            if(logger):
                logger.info("start testing......")
                logger.info("test score: {}\n\n".format(results))
            print("final test: ", results)
            return results, model_path_saved

        loss_fn = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        recorder = Recorder(self.early_stop)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.98)
        self.model.train()
        train_data_iter = tqdm.tqdm(self.train_loader)
        for step_n, batch in enumerate(train_data_iter):
            batch_data = data2gpu(batch, self.use_cuda)
            label_pred = self.model.save_feature(**batch_data)
        self.model.init_memory()
        print('initialization finished')

        for epoch in range(self.epoches):
            self.model.train()
            train_data_iter = tqdm.tqdm(self.train_loader)
            avg_loss = Averager()
            for step_n, batch in enumerate(train_data_iter):
                batch_data = data2gpu(batch, self.use_cuda)
                label = batch_data['label']
                category = batch_data['category']
                optimizer.zero_grad()
                label_pred = self.model(**batch_data)
                loss = loss_fn(label_pred, label.float()) 
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    self.model.write(**batch_data)
                if(scheduler is not None):
                    scheduler.step()
                avg_loss.add(loss.item())
                
            print('Training Epoch {}; Loss {}; '.format(epoch + 1, avg_loss.item()))
            status = '[{0}] lr = {1}; batch_loss = {2}; average_loss = {3}'.format(epoch, str(self.lr), loss.item(), avg_loss.item())
            self.model.train()
            results = self.test(self.val_loader)
            mark = recorder.add(results)
            if mark == 'save':
                torch.save(self.model.state_dict(),
                    os.path.join(self.save_param_dir, 'parameter_m3ganshenew.pkl'))
                self.best_mem = self.model.domain_memory.domain_memory
                best_metric = results['metric']
            elif mark == 'esc':
                break
            else:
                continue
        self.model.load_state_dict(torch.load(os.path.join(self.save_param_dir, 'parameter_m3ganshenew.pkl')))
        self.model.domain_memory.domain_memory = self.best_mem
        print("开始进行最后的测试")
        results = self.test(self.test_loader)
        if(logger):
            logger.info("start testing......")
            logger.info("test score: {}\n\n".format(results))
        print("final test: ", results)
        return results, os.path.join(self.save_param_dir, 'parameter_m3ganshenew.pkl')

    def test(self, dataloader):
        pred = []
        label = []
        category = []
        self.model.eval()
        data_iter = tqdm.tqdm(dataloader)
        for step_n, batch in enumerate(data_iter):
            with torch.no_grad():
                batch_data = data2gpu(batch, self.use_cuda)
                batch_label = batch_data['label']
                batch_category = batch_data['category']
                batch_label_pred = self.model(**batch_data)
                label.extend(batch_label.detach().cpu().numpy().tolist())
                pred.extend(batch_label_pred.detach().cpu().numpy().tolist())
                category.extend(batch_category.detach().cpu().numpy().tolist())
        
        return metricsTrueFalse(label, pred, category, self.category_dict)
