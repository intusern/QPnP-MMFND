import os
import torch
import tqdm
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .layers import *
from sklearn.metrics import *
from transformers import BertModel
from transformers import RobertaModel
from utils.utils import data2gpu, Averager, metrics, Recorder, metricsTrueFalse
from .bert_mae_mixed_state_encoder import SimpleBertMaeMixedStateEncoder, QuantumInterferenceFusion
import logging

model_path = "./pretrained_model/chinese_roberta_wwm_base_ext_pytorch"


class QuantumEntanglementAlign(nn.Module):
    """
    量子纠缠对齐模块：建模两个特征的纠缠关联，引导特征对齐
    参考DAMM模型的实现
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

        # 计算纠缠度：基于余弦相似度
        entanglement = F.cosine_similarity(feat1, feat2, dim=-1, eps=1e-6).unsqueeze(-1)
        # 自适应缩放因子 
        scale_factor = self.scale * entanglement  # [B,1] broadcasts
        # 应用纠缠矩阵变换
        feat1_align = feat1 @ self.entanglement_matrix * scale_factor
        feat2_align = feat2 @ self.entanglement_matrix.T * scale_factor

        # 残差连接
        feat1_align = feat1 + feat1_align
        feat2_align = feat2 + feat2_align
        return feat1_align, feat2_align


class MLP(torch.nn.Module):

    def __init__(self, input_dim, embed_dims, dropout, output_layer=True):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, embed_dim)``
        """
        return self.mlp(x)


class EDDFNWaveMixGanSheModel(torch.nn.Module):
    def __init__(self, emb_dim, mlp_dims, domain_num, dropout, dataset):
        super(EDDFNWaveMixGanSheModel, self).__init__()
        if dataset == 'weibo' or dataset == 'ch':
            self.bert = BertModel.from_pretrained(model_path, local_files_only=True)
        elif dataset == 'en':
            self.bert = RobertaModel.from_pretrained('roberta-base').requires_grad_(False)
        elif dataset == 'weibo21':
            self.bert = BertModel.from_pretrained(model_path, local_files_only=True)
        else:
            raise ValueError(f'Unsupported dataset for EDDFNWaveMixGanSheModel: {dataset}')

        self.shared_mlp = MLP(emb_dim, mlp_dims, dropout, False)
        self.specific_mlp = torch.nn.ModuleList([MLP(emb_dim, mlp_dims, dropout, False) for i in range(domain_num)])
        self.decoder = MLP(mlp_dims[-1] * 2, (64, emb_dim), dropout, False)
        self.classifier = torch.nn.Linear(2 * mlp_dims[-1], 1)
        self.domain_classifier = nn.Sequential(MLP(mlp_dims[-1], mlp_dims, dropout, False), torch.nn.ReLU(),
                        torch.nn.Linear(mlp_dims[-1], domain_num))
        self.attention = MaskAttention(emb_dim)
        
        # ========== 量子纠缠对齐模块 ==========
        self.quantum_entanglement_align = QuantumEntanglementAlign(mlp_dims[-1])
        self.quantum_alpha = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        
        # ========== 量子混合态编码器 ==========
        # 将BERT序列特征[batch, 197, 768]编码为量子混合态
        self.bert_quantum_encoder = SimpleBertMaeMixedStateEncoder(
            feature_dim=emb_dim,    # BERT特征维度 (768)
            hidden_dim=256,         # 混合态隐藏维度
            rank=2,                 # 混合态的纯态数量
            dropout=dropout
        )
        
        # ========== 量子干涉融合模块 ==========
        # BERT特征的量子干涉模块
        # 这里我们用self.bert的输出作为两个输入进行自干涉
        self.bert_quantum_interference = QuantumInterferenceFusion(
            text_input_dim=emb_dim,
            img_input_dim=emb_dim,
            fusion_dim=emb_dim,
            dropout=dropout
        )
        
        # ========== 量子特征增强层 ==========
        self.bert_quantum_enhance = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.SiLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, alpha = 1, **kwargs):
        inputs = kwargs['content']
        masks = kwargs['content_masks']
        category = kwargs['category']
        bert_feature = self.bert(inputs, attention_mask = masks).last_hidden_state
        # bert_feature: [batch, seq_len, 768]
        
        # ========== BERT特征的量子混合态编码 ==========
        # SimpleBertMaeMixedStateEncoder期望3D张量输入
        bert_quantum_feat, _, bert_mae_fused_quantum, quantum_info = \
            self.bert_quantum_encoder(bert_feature, bert_feature, masks)
        # bert_quantum_feat: [batch, 768] 量子纯态特征
        
        # ========== 应用注意力机制 - 将序列特征压缩为单一表示 ==========
        bert_feature_atn, _ = self.attention(bert_feature, masks)
        # bert_feature_atn: [batch, 768] 注意力加权特征
        
        # ========== 量子干涉融合 - 现在处理2D特征 ==========
        # 注意：QuantumInterferenceFusion的内部实现可能需要2D输入
        # 我们创建虚拟的3D表示用于干涉计算
        bert_feature_expanded = bert_feature_atn.unsqueeze(1)  # [batch, 1, 768] 用于干涉
        bert_interference_feat, bert_interference_intensity, bert_text_psi, bert_image_psi, bert_quantum_info = \
            self.bert_quantum_interference(bert_feature, bert_feature, masks, None)
        # bert_interference_feat: [batch, 768] 干涉融合特征
        # bert_interference_intensity: [batch, 768] 干涉强度
        
        # ========== 量子特征增强：融合量子语义 ==========
        bert_quantum_enhanced = self.bert_quantum_enhance(bert_quantum_feat + bert_interference_intensity)
        # bert_quantum_enhanced: [batch, 768] 增强后的量子特征
        
        # 使用增强后的量子特征和原始attention特征混合
        bert_feature_enhanced = bert_feature_atn + 0.3 * bert_quantum_enhanced
        
        # ========== EDDFN核心逻辑 ==========
        specific_feature = []
        for i in range(bert_feature_enhanced.size(0)):
            specific_feature.append(self.specific_mlp[category[i]](bert_feature_enhanced[i].view(1, -1)))
        specific_feature = torch.cat(specific_feature)
        shared_feature = self.shared_mlp(bert_feature_enhanced)
        
        # 应用量子纠缠对齐 - 对齐shared_feature和specific_feature
        shared_feature_aligned, specific_feature_aligned = self.quantum_entanglement_align(shared_feature, specific_feature)
        
        # 使用learnable quantum_alpha混合原始特征和对齐特征
        quantum_alpha = torch.sigmoid(self.quantum_alpha)  # 限制在[0,1]范围
        shared_feature = quantum_alpha * shared_feature_aligned + (1 - quantum_alpha) * shared_feature
        specific_feature = quantum_alpha * specific_feature_aligned + (1 - quantum_alpha) * specific_feature
        
        feature = torch.cat([shared_feature, specific_feature], 1)
        rec_feature = self.decoder(feature)
        output = self.classifier(feature)

        reverse = ReverseLayerF.apply
        domain_pred = self.domain_classifier(reverse(shared_feature, alpha))

        return torch.sigmoid(output.squeeze(1)), rec_feature, bert_feature_enhanced, domain_pred


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
                 domain_num,
                 dataset,
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
        self.domain_num = domain_num
        
        if os.path.exists(save_param_dir):
            self.save_param_dir = save_param_dir
        else:
            self.save_param_dir = save_param_dir
            os.makedirs(save_param_dir)
        

    def train(self, logger=None):
        print(self.save_param_dir)
        if(logger):
            logger.info("start training......")
        self.model = EDDFNWaveMixGanSheModel(self.emb_dim, self.mlp_dims, self.domain_num, self.dropout, self.dataset)
        if self.use_cuda:
            self.model = self.model.cuda()
        loss_fn = torch.nn.BCELoss()
        loss_mse = torch.nn.MSELoss(reduce=True, size_average=True)
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        recorder = Recorder(self.early_stop)
        for epoch in range(self.epoches):
            self.model.train()
            train_data_iter = tqdm.tqdm(self.train_loader)
            avg_loss = Averager()
            alpha = max(2. / (1. + np.exp(-10 * epoch / self.epoches)) - 1, 1e-1)

            for step_n, batch in enumerate(train_data_iter):
                batch_data = data2gpu(batch, self.use_cuda)
                label = batch_data['label']
                domain_label = batch_data['category'].long()

                optimizer.zero_grad()
                pred, rec_feature, bert_feature, domain_pred = self.model(**batch_data, alpha=alpha)
                loss = loss_fn(pred, label.float()) + loss_mse(rec_feature, bert_feature) + 0.1 * F.nll_loss(F.log_softmax(domain_pred, dim=1), domain_label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                avg_loss.add(loss.item())
            print('Training Epoch {}; Loss {}; '.format(epoch + 1, avg_loss.item()))
            status = "[{0}] lr = {1}; batch_loss = {2}; average_loss = {3}".format(epoch, str(self.lr), loss.item(), avg_loss)

            results = self.test(self.val_loader)
            mark = recorder.add(results)
            if mark == 'save':
                torch.save(self.model.state_dict(),
                    os.path.join(self.save_param_dir, 'parameter_eddfnwavemixganshe.pkl'))
                best_metric = results['metric']
            elif mark == 'esc':
                break
            else:
                continue
        self.model.load_state_dict(torch.load(os.path.join(self.save_param_dir, 'parameter_eddfnwavemixganshe.pkl')))
        results = self.test(self.test_loader)
        if(logger):
            logger.info("start testing......")
            logger.info("test score: {}\n\n".format(results))
        print(results)
        return results, os.path.join(self.save_param_dir, 'parameter_eddfnwavemixganshe.pkl')

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
                batch_pred, _, __, ___ = self.model(**batch_data)

                label.extend(batch_label.detach().cpu().numpy().tolist())
                pred.extend(batch_pred.detach().cpu().numpy().tolist())
                category.extend(batch_category.detach().cpu().numpy().tolist())
        
        return metricsTrueFalse(label, pred, category, self.category_dict)
