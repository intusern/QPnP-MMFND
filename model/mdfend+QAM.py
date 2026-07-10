import os
import torch
import tqdm
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from transformers import RobertaModel

from .layers import *
from .bert_mae_mixed_state_encoder import SimpleBertMaeMixedStateEncoder, QuantumInterferenceFusion
from utils.utils import data2gpu, Averager, metricsTrueFalse, Recorder


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class QuantumEntanglementAlign(nn.Module):
    def __init__(self, dim, device=None, dtype=torch.float32):
        super().__init__()
        self.entanglement_matrix = nn.Parameter(torch.randn(dim, dim, dtype=dtype, device=device))
        self.scale = nn.Parameter(torch.ones(1, dtype=dtype, device=device) * 0.1)

    def forward(self, feat1, feat2):
        feat1 = feat1.to(dtype=self.entanglement_matrix.dtype, device=self.entanglement_matrix.device)
        feat2 = feat2.to(dtype=self.entanglement_matrix.dtype, device=self.entanglement_matrix.device)

        entanglement = F.cosine_similarity(feat1, feat2, dim=-1, eps=1e-6).unsqueeze(-1)
        scale_factor = self.scale * entanglement

        feat1_align = feat1 @ self.entanglement_matrix * scale_factor
        feat2_align = feat2 @ self.entanglement_matrix.T * scale_factor

        return feat1 + feat1_align, feat2 + feat2_align


class MultiDomainFENDWaveMixModel(torch.nn.Module):
    def __init__(self, emb_dim, mlp_dims, domain_num, dropout, dataset):
        super(MultiDomainFENDWaveMixModel, self).__init__()
        self.domain_num = domain_num
        self.num_expert = 5
        self.fea_size = 256
        self.emb_dim = emb_dim
        self.unified_dim = 256

        if dataset in ['ch', 'weibo', 'weibo21', 'weibo_21']:
            local_model_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'pretrained_model',
                'chinese_roberta_wwm_base_ext_pytorch'
            )
            if os.path.isdir(local_model_path):
                self.bert = BertModel.from_pretrained(local_model_path, local_files_only=True).requires_grad_(False)
            else:
                self.bert = BertModel.from_pretrained('hfl/chinese-bert-wwm-ext').requires_grad_(False)
        elif dataset == 'en':
            self.bert = RobertaModel.from_pretrained('roberta-base').requires_grad_(False)
        else:
            raise ValueError(f'Unsupported dataset: {dataset}')

        feature_kernel = {1: 64, 2: 64, 3: 64, 5: 64, 10: 64}
        self.expert = nn.ModuleList([cnn_extractor(feature_kernel, emb_dim) for _ in range(self.num_expert)])

        self.gate = nn.Sequential(
            nn.Linear(emb_dim, mlp_dims[-1]),
            nn.ReLU(),
            nn.Linear(mlp_dims[-1], self.num_expert),
            nn.Softmax(dim=1)
        )

        self.attention = MaskAttention(emb_dim)
        self.domain_embedder = nn.Embedding(num_embeddings=self.domain_num, embedding_dim=emb_dim)
        self.specific_extractor = SelfAttentionFeatureExtract(multi_head_num=1, input_size=emb_dim, output_size=self.fea_size)
        self.classifier = MLP(320, mlp_dims, dropout)

        self.quantum_feature_align = QuantumEntanglementAlign(dim=320)
        self.domain_feature_proj = nn.Linear(emb_dim, self.unified_dim)
        self.attention_feature_proj = nn.Linear(emb_dim, self.unified_dim)
        self.quantum_domain_feature_align = QuantumEntanglementAlign(dim=self.unified_dim)
        self.domain_feature_proj_back = nn.Linear(self.unified_dim, emb_dim)
        self.attention_feature_proj_back = nn.Linear(self.unified_dim, emb_dim)

        self.entangle_fusion = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Wave mixed-state and quantum interference modules
        self.wave_mixed_encoder = SimpleBertMaeMixedStateEncoder(
            feature_dim=emb_dim,
            hidden_dim=256,
            rank=2,
            dropout=dropout,
        )
        self.wave_interference = QuantumInterferenceFusion(
            text_input_dim=emb_dim,
            img_input_dim=emb_dim,
            fusion_dim=emb_dim,
            dropout=dropout,
        )
        self.quantum_enhance = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.quantum_fuse_alpha = nn.Parameter(torch.tensor(0.35, dtype=torch.float32))

    def forward(self, **kwargs):
        inputs = kwargs['content']
        masks = kwargs['content_masks']
        category = kwargs['category']

        init_feature = self.bert(inputs, attention_mask=masks).last_hidden_state

        # 1) Mixed-state quantum encoding on sequence feature
        text_q, image_q, fused_q, _ = self.wave_mixed_encoder(init_feature, init_feature, masks)

        # 2) Quantum interference to capture constructive/destructive relations
        _, interference_intensity, _, _, _ = self.wave_interference(init_feature, init_feature, masks, masks)

        # 3) Residual fusion back to sequence tokens
        quantum_signal = self.quantum_enhance(fused_q + 0.5 * (text_q + image_q) + interference_intensity)
        alpha = torch.sigmoid(self.quantum_fuse_alpha)
        quantum_signal_seq = quantum_signal.unsqueeze(1).expand(-1, init_feature.size(1), -1)
        init_feature = (1 - alpha) * init_feature + alpha * quantum_signal_seq

        feature, _ = self.attention(init_feature, masks)
        idxs = category.long().view(-1, 1).to(feature.device)
        domain_embedding = self.domain_embedder(idxs).squeeze(1)

        domain_proj = self.domain_feature_proj(domain_embedding)
        feature_proj = self.attention_feature_proj(feature)
        domain_entangled, feature_entangled = self.quantum_domain_feature_align(domain_proj, feature_proj)

        domain_embedding = domain_embedding + self.domain_feature_proj_back(domain_entangled)
        feature = feature + self.attention_feature_proj_back(feature_entangled)

        gate_input = self.entangle_fusion(torch.cat([domain_embedding, feature], dim=-1))
        gate_value = self.gate(gate_input)

        shared_feature = 0
        for i in range(self.num_expert):
            tmp_feature = self.expert[i](init_feature)
            shared_feature += (tmp_feature * gate_value[:, i].unsqueeze(1))

        if shared_feature.dim() == 1:
            shared_feature = shared_feature.unsqueeze(0)

        shared_feature_v1 = shared_feature
        shared_feature_v2 = shared_feature * 0.95 + torch.randn_like(shared_feature) * 0.05
        sf1, sf2 = self.quantum_feature_align(shared_feature_v1, shared_feature_v2)
        shared_feature = (sf1 + sf2) / 2

        label_pred = self.classifier(shared_feature)
        return torch.sigmoid(label_pred.squeeze(1))


class Trainer():
    def __init__(
        self,
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
        early_stop=5,
        epoches=100,
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

    def train(self, logger=None):
        if logger:
            logger.info('start training......')

        self.model = MultiDomainFENDWaveMixModel(
            self.emb_dim, self.mlp_dims, len(self.category_dict), self.dropout, self.dataset
        )
        if self.use_cuda:
            self.model = self.model.to(device)

        loss_fn = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        recorder = Recorder(self.early_stop)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.98)

        for epoch in range(self.epoches):
            self.model.train()
            train_data_iter = tqdm.tqdm(self.train_loader)
            avg_loss = Averager()

            for _, batch in enumerate(train_data_iter):
                batch_data = data2gpu(batch, self.use_cuda)
                label = batch_data['label']

                optimizer.zero_grad()
                label_pred = self.model(**batch_data)
                loss = loss_fn(label_pred, label.float())
                loss.backward()
                optimizer.step()

                if scheduler is not None:
                    scheduler.step()

                avg_loss.add(loss.item())

            print('Training Epoch {}; Loss {}; '.format(epoch + 1, avg_loss.item()))

            results = self.test(self.val_loader)
            mark = recorder.add(results)
            if mark == 'save':
                torch.save(self.model.state_dict(), os.path.join(self.save_param_dir, 'parameter_mdfendwavemix.pkl'))
            elif mark == 'esc':
                break

        self.model.load_state_dict(torch.load(os.path.join(self.save_param_dir, 'parameter_mdfendwavemix.pkl')))
        results = self.test(self.test_loader)
        if logger:
            logger.info('start testing......')
            logger.info('test score: {}\n\n'.format(results))
        print(results)
        return results, os.path.join(self.save_param_dir, 'parameter_mdfendwavemix.pkl')

    def test(self, dataloader):
        pred = []
        label = []
        category = []

        self.model.eval()
        data_iter = tqdm.tqdm(dataloader)
        for _, batch in enumerate(data_iter):
            with torch.no_grad():
                batch_data = data2gpu(batch, self.use_cuda)
                batch_label = batch_data['label']
                batch_category = batch_data['category']
                batch_label_pred = self.model(**batch_data)

                label.extend(batch_label.detach().cpu().numpy().tolist())
                pred.extend(batch_label_pred.detach().cpu().numpy().tolist())
                category.extend(batch_category.detach().cpu().numpy().tolist())

        return metricsTrueFalse(label, pred, category, self.category_dict)
