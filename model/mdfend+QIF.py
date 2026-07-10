# MDFEND + 量子纠缠对齐增强版本 (基于M3Entanglement优化)
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
from utils.utils import data2gpu, Averager, metrics, metricsTrueFalse, Recorder
import logging
import math

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# 量子纠缠对齐模块：建模特征间的纠缠关联，引导特征对齐
class QuantumEntanglementAlign(nn.Module):
    """
    量子纠缠对齐模块
    通过纠缠矩阵建模两个特征之间的量子关联，实现特征对齐和增强
    """
    def __init__(self, dim, device=None, dtype=torch.float32):
        super().__init__()
        self.entanglement_matrix = nn.Parameter(
            torch.randn(dim, dim, dtype=dtype, device=device)
        )
        self.scale = nn.Parameter(
            torch.ones(1, dtype=dtype, device=device) * 0.1
        )
        self.dim = dim

    def forward(self, feat1, feat2):
        # 确保输入与参数保持一致的 dtype/device
        feat1 = feat1.to(dtype=self.entanglement_matrix.dtype, device=self.entanglement_matrix.device)
        feat2 = feat2.to(dtype=self.entanglement_matrix.dtype, device=self.entanglement_matrix.device)

        # 计算纠缠度（余弦相似度）
        entanglement = F.cosine_similarity(feat1, feat2, dim=-1, eps=1e-6).unsqueeze(-1)
        scale_factor = self.scale * entanglement  # [B,1] 广播
        
        # 通过纠缠矩阵进行特征变换
        feat1_align = feat1 @ self.entanglement_matrix * scale_factor
        feat2_align = feat2 @ self.entanglement_matrix.T * scale_factor

        # 残差连接
        feat1_align = feat1 + feat1_align
        feat2_align = feat2 + feat2_align
        return feat1_align, feat2_align


class MultiDomainFENDEntanglementModel(torch.nn.Module):
    """
    MDFEND + 量子纠缠增强模型
    在原有MDFEND基础上，增加多层次的量子纠缠对齐机制
    """
    def __init__(self, emb_dim, mlp_dims, domain_num, dropout, dataset):
        super(MultiDomainFENDEntanglementModel, self).__init__()
        self.domain_num = domain_num
        self.gamma = 10
        self.num_expert = 5
        self.fea_size = 256
        self.emb_dim = emb_dim
        self.unified_dim = 256  # 用于纠缠对齐的统一维度
        self.dataset = dataset
        
        if dataset == 'ch' or dataset == 'weibo' or dataset == 'weibo21' or dataset == 'weibo_21':
            # 使用本地预训练模型
            import os as os_module
            base_dir = os_module.path.dirname(os_module.path.dirname(os_module.path.abspath(__file__)))
            model_path = os_module.path.join(base_dir, 'pretrained_model', 'chinese_roberta_wwm_base_ext_pytorch')
            
            if os_module.path.exists(model_path):
                try:
                    self.bert = BertModel.from_pretrained(model_path).requires_grad_(False)
                except Exception as e:
                    print(f"Error loading local model from {model_path}: {e}")
                    raise
            else:
                raise ValueError(f"Local model path not found: {model_path}")
        elif dataset == 'en':
            self.bert = RobertaModel.from_pretrained('roberta-base').requires_grad_(False)
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")
        
        feature_kernel = {1: 64, 2: 64, 3: 64, 5: 64, 10: 64}
        expert = []
        for i in range(self.num_expert):
            expert.append(cnn_extractor(feature_kernel, emb_dim))
        self.expert = nn.ModuleList(expert)

        self.gate = nn.Sequential(nn.Linear(emb_dim * 2, mlp_dims[-1]),
                                      nn.ReLU(),
                                      nn.Linear(mlp_dims[-1], self.num_expert),
                                      nn.Softmax(dim = 1))

        self.attention = MaskAttention(emb_dim)

        self.domain_embedder = nn.Embedding(num_embeddings = self.domain_num, embedding_dim = emb_dim)
        self.specific_extractor = SelfAttentionFeatureExtract(multi_head_num = 1, input_size=emb_dim, output_size=self.fea_size)
        self.classifier = MLP(320, mlp_dims, dropout)
        
        # ==================== 量子纠缠对齐模块 ====================
        # 1. 特征与特征的纠缠对齐（共享特征增强）
        self.quantum_feature_align = QuantumEntanglementAlign(dim=320)
        
        # 2. domain_embedding与特征的纠缠对齐
        self.domain_feature_proj = nn.Linear(emb_dim, self.unified_dim)
        self.attention_feature_proj = nn.Linear(emb_dim, self.unified_dim)
        self.quantum_domain_feature_align = QuantumEntanglementAlign(dim=self.unified_dim)
        self.domain_feature_proj_back = nn.Linear(self.unified_dim, emb_dim)
        self.attention_feature_proj_back = nn.Linear(self.unified_dim, emb_dim)
        
        # 3. 多个专家特征的纠缠增强
        self.expert_feature_aligns = nn.ModuleList([
            QuantumEntanglementAlign(dim=320) for _ in range(self.num_expert - 1)
        ])
        

        
    
    def forward(self, **kwargs):
        inputs = kwargs['content']
        masks = kwargs['content_masks']
        category = kwargs['category']
        init_feature = self.bert(inputs, attention_mask = masks).last_hidden_state
        
        feature, _ = self.attention(init_feature, masks)
        idxs = torch.LongTensor([int(index) for index in category]).view(-1, 1).to(device)
        domain_embedding = self.domain_embedder(idxs).squeeze(1)

        # ==================== 应用量子纠缠对齐 ====================
        # 1. domain_embedding与attention特征的纠缠对齐
        domain_proj = self.domain_feature_proj(domain_embedding)  # [B, unified_dim]
        feature_proj = self.attention_feature_proj(feature)  # [B, unified_dim]
        domain_entangled, feature_entangled = self.quantum_domain_feature_align(domain_proj, feature_proj)
        # 投影回原始维度
        domain_embedding_enhanced = self.domain_feature_proj_back(domain_entangled)  # [B, emb_dim]
        feature_enhanced = self.attention_feature_proj_back(feature_entangled)  # [B, emb_dim]
        
        # 使用纠缠增强后的特征
        domain_embedding = domain_embedding + domain_embedding_enhanced  # 残差连接
        feature = feature + feature_enhanced
        
        # 直接拼接增强后的domain_embedding和feature用于gate输入
        gate_input = torch.cat([domain_embedding, feature], dim=-1)
        gate_value = self.gate(gate_input)

        # 处理多个专家特征
        expert_features = []
        for i in range(self.num_expert):
            tmp_feature = self.expert[i](init_feature)
            expert_features.append(tmp_feature)
        
        # 应用多个专家间的纠缠对齐
        for i in range(len(self.expert_feature_aligns)):
            expert_features[i], expert_features[i+1] = self.expert_feature_aligns[i](
                expert_features[i], expert_features[i+1]
            )
        
        # 加权求和
        shared_feature = 0
        for i in range(self.num_expert):
            shared_feature += (expert_features[i] * gate_value[:, i].unsqueeze(1))

        # 对共享特征进行自纠缠增强
        if shared_feature.dim() == 1:
            shared_feature = shared_feature.unsqueeze(0)
        
        # 创建共享特征的两个版本进行纠缠对齐
        shared_feature_v1 = shared_feature
        shared_feature_v2 = shared_feature * 0.95 + torch.randn_like(shared_feature) * 0.05  # 轻微扰动版本
        shared_feature_entangled_v1, shared_feature_entangled_v2 = self.quantum_feature_align(
            shared_feature_v1, shared_feature_v2
        )
        # 使用增强后的特征
        shared_feature = (shared_feature_entangled_v1 + shared_feature_entangled_v2) / 2
        
        label_pred = self.classifier(shared_feature)

        return torch.sigmoid(label_pred.squeeze(1))


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
                 dataset,
                 semantic_num=5,
                 emotion_num=5,
                 style_num=5,
                 lnn_dim=12,
                 early_stop = 5,
                 epoches = 100
                 ):
        self.lr = lr
        self.weight_decay = weight_decay
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
        
        if os.path.exists(save_param_dir):
            self.save_param_dir = save_param_dir
        else:
            self.save_param_dir = save_param_dir
            os.makedirs(save_param_dir)
        

    def train(self, logger = None):
        if(logger):
            logger.info('start training......')
        self.model = MultiDomainFENDEntanglementModel(self.emb_dim, self.mlp_dims, len(self.category_dict), self.dropout, self.dataset)
        if self.use_cuda:
            self.model = self.model.to(device)
        loss_fn = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        recorder = Recorder(self.early_stop)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 100, gamma = 0.98)
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
                loss =  loss_fn(label_pred, label.float()) 
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if(scheduler is not None):
                    scheduler.step()
                avg_loss.add(loss.item())
                
            print('Training Epoch {}; Loss {}; '.format(epoch + 1, avg_loss.item()))
            status = '[{0}] lr = {1}; batch_loss = {2}; average_loss = {3}'.format(epoch, str(self.lr), loss.item(), avg_loss.item())

            results = self.test(self.val_loader)
            mark = recorder.add(results)
            if mark == 'save':
                torch.save(self.model.state_dict(),
                    os.path.join(self.save_param_dir, 'parameter_mdfend_entanglement.pkl'))
                best_metric = results['metric']
            elif mark == 'esc':
                break
            else:
                continue
        self.model.load_state_dict(torch.load(os.path.join(self.save_param_dir, 'parameter_mdfend_entanglement.pkl')))
        results = self.test(self.test_loader)
        if(logger):
            logger.info("start testing......")
            logger.info("test score: {}\n\n".format(results))
        print(results)
        return results, os.path.join(self.save_param_dir, 'parameter_mdfend_entanglement.pkl')

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
