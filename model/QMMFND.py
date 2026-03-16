import os
import tqdm
import torch
from positional_encodings.torch_encodings import PositionalEncoding1D, PositionalEncoding2D, PositionalEncodingPermute3D
from transformers import BertModel
import torch.nn as nn
# from positional_encodings.torch_encodings import PositionalEncoding1D
import models_mae
from utils.utils import data2gpu, Averager, metrics, Recorder, clipdata2gpu
from utils.utils import metricsTrueFalse
from .layers import *
from .pivot import *
from timm.models.vision_transformer import Block
import cn_clip.clip as clip
from cn_clip.clip import load_from_name, available_models
from .wavefunction_quantum import WaveFunctionQuantumEncoder
from .clip_quantum_encoder import DualCLIPQuantumFusion, CLIPQuantumInterferenceFusion  # 量子纯态构建+两者融合+干涉
from .bert_mae_mixed_state_encoder import SimpleBertMaeMixedStateEncoder, QuantumInterferenceFusion  # BERT/MAE量子态+干涉

from .entangle import QuantumEntanglementAlign  # 量子纠缠对齐

class SimpleGate(nn.Module):
    def __init__(self, dim=1):
        super(SimpleGate, self).__init__()
        self.dim = dim

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=self.dim)
        return x1 * x2

class AdaIN(nn.Module):#自适应归一化，调整特征分布，更容易被比较。
    def __init__(self):
        super().__init__()

    def mu(self, x):
        """ Takes a (n,c,h,w) tensor as input and returns the average across
        it's spatial dimensions as (h,w) tensor [See eq. 5 of paper]"""
        return torch.sum(x,(1))/(x.shape[1])

    def sigma(self, x):
        """ Takes a (n,c,h,w) tensor as input and returns the standard deviation
        across it's spatial dimensions as (h,w) tensor [See eq. 6 of paper] Note
        the permutations are required for broadcasting"""
        return torch.sqrt((torch.sum((x.permute([1,0])-self.mu(x)).permute([1,0])**2,(1))+0.000000023)/(x.shape[1]))

    def forward(self, x, mu, sigma):
        """ Takes a content embeding x and a style embeding y and changes
        transforms the mean and standard deviation of the content embedding to
        that of the style. [See eq. 8 of paper] Note the permutations are
        required for broadcasting"""
        # print(mu.shape) # 12
        x_mean = self.mu(x)
        x_std = self.sigma(x)
        x_reduce_mean = x.permute([1, 0]) - x_mean
        x_norm = x_reduce_mean/x_std
        # print(x_mean.shape) # 768, 12
        return (sigma.squeeze(1)*(x_norm + mu.squeeze(1))).permute([1,0])


class MultiDomainPLEFENDModel(torch.nn.Module):
    def __init__(self, emb_dim, mlp_dims, bert, out_channels, dropout):
        super(MultiDomainPLEFENDModel, self).__init__()
        # 1超参
        self.num_expert = 6  # 特定领域专家数
        self.domain_num = 10  # 领域数（类别）- 包含'未分类'
        self.gate_num = 10  # 门控
        self.num_share = 1  # 一个共享专家
        self.unified_dim, self.text_dim = emb_dim, 768  # 统一成768和bert输出维度768
        self.image_dim = 768  # clip、mae输出768
        self.bert = BertModel.from_pretrained(bert).requires_grad_(False)  # 不需要训练，直接用
        feature_kernel = {1: 64, 2: 64, 3: 64, 5: 64, 10: 64}  # 卷积核key：value
        self.text_token_len = 197  # bert本token最大词长
        self.image_token_len = 197  # vit图像token最大词长
        self.quantum_entanglement_align = QuantumEntanglementAlign(512)
        # 文本总专家
        # self.num_expert = 6
        # self.domain_num = 9
        # self.gate_num = 10
        # self.num_share = 1
        # self.unified_dim, self.text_dim = emb_dim, 768
        # self.image_dim = 768
        # 尝试从本地加载BERT，如果失败则从Hugging Face自动下载
        try:
            self.bert = BertModel.from_pretrained(bert).requires_grad_(False)
        except (OSError, ValueError):
            print(f"本地BERT模型未找到（{bert}），从Hugging Face自动下载...")
            
            # 清除可能导致问题的环境变量
            if 'HF_ENDPOINT' in os.environ:
                original_endpoint = os.environ['HF_ENDPOINT']
                del os.environ['HF_ENDPOINT']
                print(f"⚠️  已临时清除HF_ENDPOINT设置: {original_endpoint}")
                print("   直接从Hugging Face官方下载...")
            
            try:
                self.bert = BertModel.from_pretrained('hfl/chinese-roberta-wwm-ext').requires_grad_(False)
                print("✓ BERT模型下载完成")
            except Exception as e:
                print(f"✗ BERT模型下载失败: {e}")
                print("\n解决方案：")
                print("1. 检查网络连接")
                print("2. 清除HF_ENDPOINT环境变量")
                print("3. 运行: 清除镜像并运行.bat")
                raise
        feature_kernel = {1: 64, 2: 64, 3: 64, 5: 64, 10: 64}#卷积核键值对
        self.text_token_len = 197
        self.image_token_len = 197
        # 确保你已经从 layers.py 导入了 MLP_fusion
        # from model.layers import MLP_fusion, ...
        
        # ... (其他代码) ...
        
        # 四类专家，文本 图像-cnn 融合-mlp 最终处理-transformer块
        text_expert_list = []
        for i in range(self.domain_num):
            text_expert = []
            for j in range(self.num_expert):
                # 使用cnn_extractor处理3D序列数据[batch, 197, 768]
                expert = cnn_extractor(emb_dim, feature_kernel)
                text_expert.append(expert)
            text_expert = nn.ModuleList(text_expert)
            text_expert_list.append(text_expert)
        self.text_experts = nn.ModuleList(text_expert_list)
        
        image_expert_list = []
        for i in range(self.domain_num):
            image_expert = []
            for j in range(self.num_expert):
                # 使用cnn_extractor处理3D序列数据 [batch, 197, 768]
                image_expert.append(cnn_extractor(self.image_dim, feature_kernel))
            image_expert = nn.ModuleList(image_expert)
            image_expert_list.append(image_expert)
        self.image_experts = nn.ModuleList(image_expert_list)
        
        fusion_expert_list = []
        for i in range(self.domain_num):
            fusion_expert = []
            for j in range(self.num_expert):
                expert = nn.Sequential(nn.Linear(320, 320),
                                       nn.SiLU(),
                                       #SimpleGate(),
                                       #nn.BatchNorm1d(160),
                                       nn.Linear(320, 320),
                                       )
                fusion_expert.append(expert)
            fusion_expert = nn.ModuleList(fusion_expert)
            fusion_expert_list.append(fusion_expert)
        self.fusion_experts = nn.ModuleList(fusion_expert_list)

        final_expert_list = []
        for i in range(self.domain_num):
            final_expert = []
            for j in range(self.num_expert):
                final_expert.append(Block(dim=320, num_heads=8))
            final_expert = nn.ModuleList(final_expert)
            final_expert_list.append(final_expert)
        self.final_experts = nn.ModuleList(final_expert_list)
        
        # 共享专家数量翻倍12，
        text_share_expert, image_share_expert, fusion_share_expert, final_share_expert = [], [], [], []
        for i in range(self.num_share):
            text_share = []
            image_share = []
            fusion_share = []
            final_share = []
            for j in range(self.num_expert * 2):
                # 使用cnn_extractor处理3D序列数据
                text_share.append(cnn_extractor(emb_dim, feature_kernel))
                image_share.append(cnn_extractor(self.image_dim, feature_kernel))
                
                expert = nn.Sequential(nn.Linear(320, 320),
                                       nn.SiLU(),
                                       nn.Linear(320, 320),
                                       )
                fusion_share.append(expert)
                final_share.append(Block(dim=320, num_heads=8))
                
            text_share = nn.ModuleList(text_share)
            text_share_expert.append(text_share)
            image_share = nn.ModuleList(image_share)
            image_share_expert.append(image_share)
            fusion_share = nn.ModuleList(fusion_share)
            fusion_share_expert.append(fusion_share)
            final_share = nn.ModuleList(final_share)
            final_share_expert.append(final_share)
            
        self.text_share_expert = nn.ModuleList(text_share_expert)
        self.image_share_expert = nn.ModuleList(image_share_expert)
        self.fusion_share_expert = nn.ModuleList(fusion_share_expert)
        self.final_share_expert = nn.ModuleList(final_share_expert)        
# #四类专家，文本 图像-cnn 融合-mlp 最终处理-transformer块
#         text_expert_list = []
#         for i in range(self.domain_num):
#             text_expert = []
#             for j in range(self.num_expert):
#                 text_expert.append(cnn_extractor(emb_dim, feature_kernel))

#             text_expert = nn.ModuleList(text_expert)
#             text_expert_list.append(text_expert)
#         self.text_experts = nn.ModuleList(text_expert_list)

#         image_expert_list = []
#         for i in range(self.domain_num):
#             image_expert = []
#             for j in range(self.num_expert):
#                 image_expert.append(cnn_extractor(self.image_dim, feature_kernel))
#                 #image_expert.append(image_cnn_extractor())
#             image_expert = nn.ModuleList(image_expert)
#             image_expert_list.append(image_expert)
#         self.image_experts = nn.ModuleList(image_expert_list)

#         fusion_expert_list = []
#         for i in range(self.domain_num):
#             fusion_expert = []
#             for j in range(self.num_expert):
#                 expert = nn.Sequential(nn.Linear(320, 320),
#                                        nn.SiLU(),
#                                        #SimpleGate(),
#                                        #nn.BatchNorm1d(160),
#                                        nn.Linear(320, 320),
#                                        )
#                 fusion_expert.append(expert)
#             fusion_expert = nn.ModuleList(fusion_expert)
#             fusion_expert_list.append(fusion_expert)
#         self.fusion_experts = nn.ModuleList(fusion_expert_list)

#         final_expert_list = []
#         for i in range(self.domain_num):
#             final_expert = []
#             for j in range(self.num_expert):
#                 final_expert.append(Block(dim=320, num_heads=8))
#             final_expert = nn.ModuleList(final_expert)
#             final_expert_list.append(final_expert)
#         self.final_experts = nn.ModuleList(final_expert_list)
# #共享专家数量翻倍12，
#         text_share_expert, image_share_expert, fusion_share_expert,final_share_expert = [], [], [],[]
#         for i in range(self.num_share):
#             text_share = []
#             image_share = []
#             fusion_share = []
#             final_share = []
#             for j in range(self.num_expert*2):
#                 text_share.append(cnn_extractor(emb_dim, feature_kernel))
#                 image_share.append(cnn_extractor(self.image_dim, feature_kernel))
#                 #image_share.append(image_cnn_extractor())
#                 expert = nn.Sequential(nn.Linear(320, 320),
#                                        nn.SiLU(),
#                                        #SimpleGate(),
#                                        #nn.BatchNorm1d(160),
#                                        nn.Linear(320, 320),
#                                        )
#                 fusion_share.append(expert)
#                 final_share.append(Block(dim=320, num_heads=8))
#             text_share = nn.ModuleList(text_share)
#             text_share_expert.append(text_share)
#             image_share = nn.ModuleList(image_share)
#             image_share_expert.append(image_share)
#             fusion_share = nn.ModuleList(fusion_share)
#             fusion_share_expert.append(fusion_share)
#             final_share = nn.ModuleList(final_share)
#             final_share_expert.append(final_share)
#         self.text_share_expert = nn.ModuleList(text_share_expert)
#         self.image_share_expert = nn.ModuleList(image_share_expert)
#         self.fusion_share_expert = nn.ModuleList(fusion_share_expert)
#         self.final_share_expert = nn.ModuleList(final_share_expert)
        
#门控网络，输入输出维度不同，mlp
        image_gate_list, text_gate_list, fusion_gate_list, fusion_gate_list0,final_gate_list = [], [], [], [],[]
        for i in range(self.domain_num):
            image_gate = nn.Sequential(nn.Linear(self.unified_dim * 2, self.unified_dim),
                                       nn.SiLU(),
                                       #SimpleGate(),
                                       #nn.BatchNorm1d(int(self.unified_dim / 2)),
                                       nn.Linear(self.unified_dim, self.num_expert * 3),
                                       nn.Dropout(0.1),#过拟合防止
                                       nn.Softmax(dim=1)
                                       )
            text_gate = nn.Sequential(nn.Linear(self.unified_dim * 2, self.unified_dim),
                                      nn.SiLU(),
                                      #SimpleGate(),
                                      #nn.BatchNorm1d(int(self.unified_dim / 2)),
                                      nn.Linear(self.unified_dim, self.num_expert * 3),
                                      nn.Dropout(0.1),
                                      nn.Softmax(dim=1)
                                      )
            fusion_gate = nn.Sequential(nn.Linear(self.unified_dim + 320, self.unified_dim),
                                        nn.SiLU(),
                                        #SimpleGate(),
                                        #nn.BatchNorm1d(int(self.unified_dim / 2)),
                                        nn.Linear(self.unified_dim, self.num_expert * 4),
                                        nn.Dropout(0.1),
                                        nn.Softmax(dim=1)
                                        )
            fusion_gate0 = nn.Sequential(nn.Linear(320, 160),
                                         nn.SiLU(),
                                         #SimpleGate(),
                                         #nn.BatchNorm1d(80),
                                         nn.Linear(160, self.num_expert * 3),
                                         nn.Dropout(0.1),
                                         nn.Softmax(dim=1)
                                         )
            final_gate = nn.Sequential(nn.Linear(1088, 720),
                                        nn.SiLU(),
                                        #SimpleGate(),
                                        #nn.BatchNorm1d(int(self.unified_dim / 2)),
                                        nn.Linear(720, 160),
                                        nn.SiLU(),
                                        nn.Linear(160, self.num_expert * 3),
                                        nn.Dropout(0.1),
                                        nn.Softmax(dim=1)
                                         )
            image_gate_list.append(image_gate)
            text_gate_list.append(text_gate)
            fusion_gate_list.append(fusion_gate)
            fusion_gate_list0.append(fusion_gate0)
            final_gate_list.append(final_gate)
        self.image_gate_list = nn.ModuleList(image_gate_list)
        self.text_gate_list = nn.ModuleList(text_gate_list)
        self.fusion_gate_list = nn.ModuleList(fusion_gate_list)
        self.fusion_gate_list0 = nn.ModuleList(fusion_gate_list0)
        self.final_gate_list = nn.ModuleList(final_gate_list)
#定义不同注意力机制，fusion输入维度翻倍
        #self.text_attention = TokenAttention(self.unified_dim)
        self.text_attention = MaskAttention(self.unified_dim)
        self.image_attention = TokenAttention(self.unified_dim)
        self.fusion_attention = TokenAttention(self.unified_dim * 2)
        self.final_attention = TokenAttention(320)
#！领域嵌入，mlp分类器
        self.domain_embedder = torch.nn.Embedding(num_embeddings=self.domain_num, embedding_dim=emb_dim)

        text_classifier_list = []

        for i in range(self.domain_num):
            text_classifier = MLP(320, mlp_dims, dropout)
            text_classifier_list.append(text_classifier)
        self.text_classifier_list = nn.ModuleList(text_classifier_list)

        image_classifier_list = []

        for i in range(self.domain_num):
            image_classifier = MLP(320, mlp_dims, dropout)
            image_classifier_list.append(image_classifier)
        self.image_classifier_list = nn.ModuleList(image_classifier_list)

        fusion_classifier_list = []

        for i in range(self.domain_num):
            fusion_classifier = MLP(320, mlp_dims, dropout)
            fusion_classifier_list.append(fusion_classifier)
        self.fusion_classifier_list = nn.ModuleList(fusion_classifier_list)

        share_classifier_list = []

        for i in range(self.domain_num):
            share_classifier = MLP(320, mlp_dims, dropout)
            share_classifier_list.append(share_classifier)
        self.share_classifier_list = nn.ModuleList(share_classifier_list)

        dom_classifier_list = []

        for i in range(self.domain_num):
            dom_classifier = MLP(320, mlp_dims, dropout)
            dom_classifier_list.append(dom_classifier)
        self.dom_classifier_list = nn.ModuleList(dom_classifier_list)



        final_classifier_list = []

        for i in range(self.domain_num):
            final_classifier = MLP(320, mlp_dims, dropout)
            final_classifier_list.append(final_classifier)
        self.final_classifier_list = nn.ModuleList(final_classifier_list)

        self.MLP_fusion = MLP_fusion(960, 320, [348], 0.1)
        self.domain_fusion = MLP_fusion(1088, 320, [348], 0.1)
        self.MLP_fusion0 = MLP_fusion(320, 320, [348], 0.1)
        self.clip_fusion = clip_fuion(1024, 320, [348], 0.1)
        
        # 波函数量子态编码器：将CLIP融合特征转为量子态，深度捕捉对齐语义
        # 同时通过残差连接保留原始CLIP语义信息
        self.quantum_encoder = WaveFunctionQuantumEncoder(
            clip_dim=320, 
            hidden_dim=320, 
            return_quantum_info=False
        )

#加载图像特征提取-mae
        self.model_size = "base"
        self.image_model = models_mae.__dict__["mae_vit_{}_patch16".format(self.model_size)](norm_pix_loss=False)
        # 根据是否有GPU自动选择设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_model = self.image_model.to(self.device)
        
        # 检查MAE模型文件是否存在
        mae_path = './mae_pretrain_vit_{}.pth'.format(self.model_size)
        if not os.path.exists(mae_path):
            raise FileNotFoundError(
                f"\n{'='*60}\n"
                f"❌ MAE预训练模型文件未找到！\n"
                f"文件路径: {mae_path}\n\n"
                f"请下载MAE模型：\n"
                f"  方法1: 运行 py download_models.py\n"
                f"  方法2: 手动下载\n"
                f"    下载地址: https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth\n"
                f"    保存到项目根目录，文件名: mae_pretrain_vit_base.pth\n"
                f"{'='*60}"
            )
        checkpoint = torch.load(mae_path, map_location='cpu')
        self.image_model.load_state_dict(checkpoint['model'], strict=False)
        for param in self.image_model.parameters():
            param.requires_grad = False#冻结参数，不训练

        #### mapping MLPs是特征分布更一致
        self.mapping_IS_MLP_mu = nn.Sequential(
            nn.Linear(1, self.unified_dim),
            nn.SiLU(),
            # nn.BatchNorm1d(self.unified_dim),
            nn.Linear(self.unified_dim, 1),
        )
        self.mapping_IS_MLP_sigma = nn.Sequential(
            nn.Linear(1, self.unified_dim),
            nn.SiLU(),
            # nn.BatchNorm1d(self.unified_dim),
            nn.Linear(self.unified_dim,1),
        )
        self.mapping_T_MLP_mu = nn.Sequential(
            nn.Linear(1, self.unified_dim),
            nn.SiLU(),
            # nn.BatchNorm1d(self.unified_dim),
            nn.Linear(self.unified_dim, 1),
        )
        self.mapping_T_MLP_sigma = nn.Sequential(
            nn.Linear(1, self.unified_dim),
            nn.SiLU(),
            # nn.BatchNorm1d(self.unified_dim),
            nn.Linear(self.unified_dim, 1),
        )
        self.mapping_IP_MLP_mu = nn.Sequential(
            nn.Linear(1, self.unified_dim),
            nn.SiLU(),
            # nn.BatchNorm1d(self.unified_dim),
            nn.Linear(self.unified_dim, 1),
        )
        self.mapping_IP_MLP_sigma = nn.Sequential(
            nn.Linear(1, self.unified_dim),
            nn.SiLU(),
            # nn.BatchNorm1d(self.unified_dim),
            nn.Linear(self.unified_dim, 1),
        )
        self.mapping_CC_MLP_mu = nn.Sequential(
            nn.Linear(1, self.unified_dim),
            nn.SiLU(),
            # nn.BatchNorm1d(self.unified_dim),
            nn.Linear(self.unified_dim, 1),
        )
        self.mapping_CC_MLP_sigma = nn.Sequential(
            nn.Linear(1, self.unified_dim),
            nn.SiLU(),
            # nn.BatchNorm1d(self.unified_dim),
            nn.Linear(self.unified_dim, 1),
        )
        self.adaIN = AdaIN()
        self.irrelevant_tensor = []
        for i in range(self.domain_num):
            self.irrelevant_tensor.append(nn.Parameter(torch.ones((1, 320)), requires_grad=True))
#对齐用clip
        # 使用之前定义的device（自动检测CPU/GPU）
        clip_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.ClipModel,_ = load_from_name("ViT-B-16", device=clip_device, download_root='./')
        
        # 初始化CLIP量子态编码器（振幅+相位严格遵循波函数定义）
        self.clip_quantum_fusion = DualCLIPQuantumFusion(
            clip_dim=512,       # CLIP特征维度
            output_dim=320,     # 量子态输出维度
            fusion_type='concat'  # 'concat', 'weighted', 'interference'
        )
        
        # BERT/MAE特征的量子混合态编码器
        # 将序列特征[batch, 197, 768]编码为量子混合态，输出[batch, 768]
        self.bert_mae_quantum_encoder = SimpleBertMaeMixedStateEncoder(
            feature_dim=768,    # BERT/MAE特征维度
            hidden_dim=256,     # 混合态隐藏维度
            rank=2,             # 混合态的纯态数量
            dropout=0.1
        )
        
        # ========== 新增：量子干涉融合模块 ==========
        # 1. CLIP特征的量子干涉模块
        self.clip_quantum_interference = CLIPQuantumInterferenceFusion(
            clip_dim=512,
            fusion_dim=320,
            dropout=0.1
        )
        
        # 2. BERT/MAE特征的量子干涉模块
        self.bert_mae_quantum_interference = QuantumInterferenceFusion(
            text_input_dim=768,
            img_input_dim=768,
            fusion_dim=768,
            dropout=0.1
        )
        
        # 3. 量子特征增强层 (将量子特征融合回原始768维)
        self.text_quantum_enhance = nn.Sequential(
            nn.Linear(768, 768),
            nn.LayerNorm(768),
            nn.SiLU(),
            nn.Dropout(0.1)
        )
        
        self.image_quantum_enhance = nn.Sequential(
            nn.Linear(768, 768),
            nn.LayerNorm(768),
            nn.SiLU(),
            nn.Dropout(0.1)
        )


        #pivot:
        feature_emb_size = 320
        img_emb_size =320
        feature_num = 4
        self.feature_num = 4
        text_emb_size = 320
        #self.n_node = 64
        self.feature_emb_size = 320
        self.emb_size = 320
        self.layers = 12
 #transformer实现长序列特征融合，初始化
        self.transformers = torch.nn.ModuleList([TransformerLayer(feature_emb_size, head_num=4, dropout=0.6,
                                                                  attention_dropout=0,
                                                                  initializer_range=0.02) for _ in
                                                 range(self.layers)])
        self.mlp_img = torch.nn.ModuleList([MLP_trans(img_emb_size, feature_emb_size, dropout=0.6) for _ in
                                                 range(feature_num)])#统一或转换维度

        self.mlp_text = torch.nn.ModuleList([MLP_trans(text_emb_size, feature_emb_size, dropout=0.6) for _ in
                                            range(feature_num)])
        self.pivot_mlp_fusion = torch.nn.ModuleList([MLP_trans(text_emb_size, feature_emb_size, dropout=0.6) for _ in
                                             range(feature_num)])
        self.transformers_list = torch.nn.ModuleList()
        self.mlp_img_list = torch.nn.ModuleList()
        self.mlp_text_list = torch.nn.ModuleList()
        self.pivot_mlp_fusion_list = torch.nn.ModuleList()
#每个领域多transformer层
        for i in range(self.domain_num):
            self.transformers_list.append(torch.nn.ModuleList([TransformerLayer(feature_emb_size, head_num=4, dropout=0.6,
                                                                  attention_dropout=0,
                                                                  initializer_range=0.02) for _ in
                                                 range(self.layers)]))
            self.mlp_img_list.append(torch.nn.ModuleList([MLP_trans(img_emb_size, feature_emb_size, dropout=0.6) for _ in
                                                 range(feature_num)]))
            self.mlp_text_list.append(torch.nn.ModuleList([MLP_trans(text_emb_size, feature_emb_size, dropout=0.6) for _ in
                                            range(feature_num)]))
            self.pivot_mlp_fusion_list.append(torch.nn.ModuleList([MLP_trans(text_emb_size, feature_emb_size, dropout=0.6) for _ in
                                             range(feature_num)]))


        self.active = nn.SiLU()
        self.dropout2 = nn.Dropout(0.2)
        self.mlp_star_f1 = nn.Linear(self.feature_emb_size * 4, self.emb_size)
        self.mlp_star_f2 = nn.Linear(self.emb_size, self.emb_size)
        self.mlp_star_f1_list = torch.nn.ModuleList()
        self.mlp_star_f2_list = torch.nn.ModuleList()
        for i in range(self.domain_num):
            self.mlp_star_f1_list.append(nn.Linear(self.feature_emb_size * 4, self.emb_size))
            self.mlp_star_f2_list.append(nn.Linear(self.emb_size, self.emb_size))
    #图像、文本融合特征第一个初始化，其他拼接
    def fusion_img_text(self, image_emb, text_emb,fusion_emb,mlp_img,mlp_text,mlp_fusion,transformers,mlp_star_f1,mlp_star_f2):
        for img_feature_num in range(0, self.feature_num):
            if img_feature_num == 0:
                img_feature_seq = mlp_img[img_feature_num](image_emb)#维度转换【batch，num0】——【batch，num1】

                img_feature_seq = img_feature_seq.unsqueeze(1)#第1维前面加一个维度，默认1

            else:
                img_feature_seq = torch.cat((img_feature_seq, mlp_img[img_feature_num](image_emb).unsqueeze(1)), 1)

        for text_feature_num in range(0, self.feature_num):
            if text_feature_num == 0:
                text_feature_seq = mlp_text[text_feature_num](text_emb)
                text_feature_seq = text_feature_seq.unsqueeze(1)
            else:
                text_feature_seq = torch.cat((text_feature_seq, mlp_text[text_feature_num](text_emb).unsqueeze(1)), 1)

        for text_feature_num in range(0, self.feature_num):
            if text_feature_num == 0:
                fusion_feature_seq = mlp_fusion[text_feature_num](fusion_emb)
                fusion_feature_seq = fusion_feature_seq.unsqueeze(1)
            else:
                fusion_feature_seq = torch.cat((fusion_feature_seq, mlp_fusion[text_feature_num](fusion_emb).unsqueeze(1)), 1)
        #print(img_feature_seq.shape)
        #print(text_feature_seq.shape)
        #print(fusion_feature_seq.shape)
        #star_emb1 = (img_feature_seq[:, 0, :] + text_feature_seq[:, 0, :] + fusion_feature_seq[:, 0, :]) / 3
        #star_emb2 = (img_feature_seq[:, 1, :] + text_feature_seq[:, 1, :] + fusion_feature_seq[:, 1, :]) / 3
        #star_emb3 = (img_feature_seq[:, 2, :] + text_feature_seq[:, 2, :] + fusion_feature_seq[:, 2, :]) / 3
        #star_emb4 = (img_feature_seq[:, 3, :] + text_feature_seq[:, 3, :] + fusion_feature_seq[:, 3, :]) / 3
        star_emb1 = text_feature_seq[:, 0, :]
        star_emb2 = text_feature_seq[:, 1, :]
        star_emb3 = text_feature_seq[:, 2, :]
        star_emb4 = text_feature_seq[:, 3, :]
#初始化4个文本嵌入


#transformer实现交互——先拼接嵌入和文本/图像序列，在通过残差连接去更新
        for sa_i in range(0, int(self.layers), 3):
            trans_text_item = torch.cat(
                [star_emb1.unsqueeze(1), star_emb2.unsqueeze(1), star_emb3.unsqueeze(1), star_emb4.unsqueeze(1),  text_feature_seq], 1)
            text_output = transformers[sa_i + 2](trans_text_item)

            star_emb1 = (text_output[:, 0, :] + star_emb1)/2
            star_emb2 = (text_output[:, 1, :] + star_emb2)/2
            star_emb3 = (text_output[:, 2, :] + star_emb3)/2
            star_emb4 = (text_output[:, 3, :] + star_emb4)/2
            text_feature_seq = text_output[:, 4:self.feature_num+4, :] + text_feature_seq

            trans_img_item = torch.cat(
                [star_emb1.unsqueeze(1), star_emb2.unsqueeze(1), star_emb3.unsqueeze(1),star_emb4.unsqueeze(1),
                 img_feature_seq], 1)
            img_output = transformers[sa_i+1](trans_img_item)
            star_emb1 = (img_output[:, 0, :] + star_emb1) / 2
            star_emb2 = (img_output[:, 1, :] + star_emb2) / 2
            star_emb3 = (img_output[:, 2, :] + star_emb3) / 2
            star_emb4 = (img_output[:, 3, :] + star_emb4) / 2
            img_feature_seq = img_output[:, 4:self.feature_num + 4, :] + img_feature_seq

            trans_fusion_item = torch.cat(
                [star_emb1.unsqueeze(1), star_emb2.unsqueeze(1), star_emb3.unsqueeze(1),star_emb4.unsqueeze(1),
                 fusion_feature_seq], 1)
            fusion_output = transformers[sa_i](trans_fusion_item)
            star_emb1 = (fusion_output[:, 0, :] + star_emb1) / 2
            star_emb2 = (fusion_output[:, 1, :] + star_emb2) / 2
            star_emb3 = (fusion_output[:, 2, :] + star_emb3) / 2
            star_emb4 = (fusion_output[:, 3, :] + star_emb4) / 2
            fusion_feature_seq = fusion_output[:, 4:self.feature_num + 4, :] + fusion_feature_seq
#四个融合特征生成最终特征
        item_emb_trans = self.dropout2(torch.cat([star_emb1, star_emb2, star_emb3,star_emb4], 1))
        item_emb_trans = self.dropout2(self.active(mlp_star_f1(item_emb_trans)))
        item_emb_trans = self.dropout2(self.active(mlp_star_f2(item_emb_trans)))
        return item_emb_trans

    def forward(self, **kwargs):
        #1提取输入特征
        inputs = kwargs['content']
        masks = kwargs['content_masks']
        category = kwargs['category']
        text_feature = self.bert(inputs, attention_mask=masks)[0]  # ([64, 197, 768])

        image = kwargs['image']
        image_feature = self.image_model.forward_ying(image)  # MAE([64, 197, 768])
        #image_feature = self.bert(inputs, attention_mask=masks)[0]
        
        # ========== BERT/MAE特征的量子混合态编码 ==========
        # 关键修改：不覆盖原始特征，使用新变量名
        text_quantum_feat, image_quantum_feat, bert_mae_fused_quantum, quantum_info = \
            self.bert_mae_quantum_encoder(text_feature, image_feature, masks)
        # text_quantum_feat/image_quantum_feat: [batch, 768] 量子纯态特征
        
        # ========== 新增：BERT/MAE量子干涉 ==========
        # 通过量子干涉增强图文特征融合
        bert_mae_interference_feat, bert_mae_interference_intensity, bert_text_psi, bert_image_psi, bert_quantum_info = \
            self.bert_mae_quantum_interference(text_feature, image_feature, masks, None)
        # bert_mae_interference_feat: [batch, 768] 干涉融合特征
        # bert_mae_interference_intensity: [batch, 768] 干涉强度（体现图文相关性）
        
        # ========== 量子特征增强：将量子语义融合回原始特征 ==========
        # 1. 增强文本特征：原始特征 + 量子纯态 + 量子干涉强度
        text_quantum_enhanced = self.text_quantum_enhance(text_quantum_feat + bert_mae_interference_intensity)
        # 将2D量子特征[batch, 768]扩展到3D序列维度[batch, 197, 768]
        text_quantum_enhanced = text_quantum_enhanced.unsqueeze(1).expand(-1, text_feature.size(1), -1)
        text_feature = text_feature + 0.4 * text_quantum_enhanced  # 残差连接，保留原始信息
        
        # 2. 增强图像特征：原始特征 + 量子纯态 + 量子干涉强度  
        image_quantum_enhanced = self.image_quantum_enhance(image_quantum_feat + bert_mae_interference_intensity)
        # 将2D量子特征[batch, 768]扩展到3D序列维度[batch, 197, 768]
        image_quantum_enhanced = image_quantum_enhanced.unsqueeze(1).expand(-1, image_feature.size(1), -1)
        image_feature = image_feature + 0.4 * image_quantum_enhanced  # 残差连接，保留原始信息

        clip_image = kwargs['clip_image']
        clip_text = kwargs['clip_text']

        # ========== CLIP特征提取 ==========
        with torch.no_grad():
            clip_image_feature_raw = self.ClipModel.encode_image(clip_image)# ([64, 512])
            clip_text_feature_raw = self.ClipModel.encode_text(clip_text)  # ([64, 512])
            # L2归一化：便于余弦相似度计算
            clip_image_feature_raw /= clip_image_feature_raw.norm(dim=-1, keepdim=True)
            clip_text_feature_raw /= clip_text_feature_raw.norm(dim=-1, keepdim=True)
        
        # ========== CLIP量子态编码（纯态特征） ==========
        # 将CLIP图文特征编码为量子态 ψ = A·e^(iθ)，深度捕捉对齐语义
        clip_quantum_feat, clip_image_amplitude, clip_text_amplitude, clip_phase_coherence = \
            self.clip_quantum_fusion(clip_image_feature_raw.float(), clip_text_feature_raw.float())
        # clip_quantum_feat: [batch, 320] 量子态融合特征（振幅+相位编码）
        
        # ========== CLIP量子干涉增强 ==========
        # 通过量子干涉捕捉图文特征的深层关联
        clip_interference_feat, clip_interference_intensity, clip_image_psi, clip_text_psi, clip_quantum_info = \
            self.clip_quantum_interference(clip_image_feature_raw.float(), clip_text_feature_raw.float())
        # clip_interference_feat: [batch, 320] 干涉融合特征
        # clip_interference_intensity: [batch, 320] 干涉强度（体现图文相关性）
        
        # ========== 量子特征增强CLIP原始特征 ==========
        # 1. 保留原始CLIP特征 + 量子纯态 + 干涉强度（最大程度保留语义）
        # 将[batch, 320]的量子特征映射回[batch, 512]以匹配CLIP维度
        clip_quantum_expand = torch.cat([clip_quantum_feat, clip_interference_intensity], dim=-1)  # [batch, 640]
        clip_quantum_to_512 = nn.Linear(640, 512).to(self.device)(clip_quantum_expand)  # [batch, 512]
        
        # 2. 残差增强：原始特征 + 量子增强
        clip_image_feature = clip_image_feature_raw + 0.5 * clip_quantum_to_512
        clip_text_feature = clip_text_feature_raw + 0.5 * clip_quantum_to_512
        
        # 3. 最终融合特征（用于后续处理）
        fusion_feature = clip_quantum_feat + 0.5 * clip_interference_feat  # [batch, 320]
        #量子纠缠l
        clip_text_feature, clip_image_feature = self.quantum_entanglement_align(clip_text_feature, clip_image_feature)
        clip_fusion_feature = torch.cat((clip_image_feature, clip_text_feature), dim=-1)  # torch.Size([64, 1024])
        clip_fusion_feature = self.clip_fusion(clip_fusion_feature.float())  # torch.Size([64, 320])
     
       
        
#2注意力池化
        #text_atn_feature, _ = self.text_attention(text_feature)  # ([64, 768])
        text_atn_feature = self.text_attention(text_feature,masks)
        image_atn_feature, _ = self.image_attention(image_feature)

        fusion_atn_feature = self.MLP_fusion0(fusion_feature)  # 已是融合特征，直接降维
        # print("image_atn_feature", image_atn_feature.size())
#3领域嵌入与门控输入
        idxs = torch.tensor([index for index in category]).view(-1, 1).to(self.device)
        domain_embedding = self.domain_embedder(idxs).squeeze(1)  ##([32, 768])
        text_gate_input = torch.cat([domain_embedding, text_atn_feature], dim=-1)  # ([64, 1536])
        image_gate_input = torch.cat([domain_embedding, image_atn_feature], dim=-1)#各自注意力之后的特征和领域嵌入拼接送到门控
        fusion_gate_input = torch.cat([domain_embedding, fusion_atn_feature], dim=-1)
#4门控输出
        text_gate_out_list = []
        for i in range(self.domain_num):
            gate_out = self.text_gate_list[i](text_gate_input)
            text_gate_out_list.append(gate_out)
        self.text_gate_out_list = text_gate_out_list
        # self.text_gate_out_list = nn.ModuleList(text_gate_out_list)

        image_gate_out_list = []
        for i in range(self.domain_num):
            gate_out = self.image_gate_list[i](image_gate_input)
            image_gate_out_list.append(gate_out)
        self.image_gate_out_list = image_gate_out_list

        fusion_gate_out_list = []
        for i in range(self.domain_num):
            gate_out = self.fusion_gate_list[i](fusion_gate_input)
            fusion_gate_out_list.append(gate_out)
        self.fusion_gate_out_list = fusion_gate_out_list

#5各个专家网络计算——输出作为权值进行累加
        # text
        text_gate_expert_value = []#特域加共享域输出
        text_gate_spacial_expert_value = []
        text_gate_share_expert_value = []
        for i in range(self.domain_num):
            gate_expert = 0
            gate_spacial_expert = 0
            gate_share_expert = 0
            for j in range(self.num_expert):#特域-累加t*g+=x
                tmp_expert = self.text_experts[i][j](text_feature)  # ([64, 320])
                gate_expert += (tmp_expert * text_gate_out_list[i][:, j].unsqueeze(1))  ##([64, 320]*[64, 1])
                gate_spacial_expert += (tmp_expert * text_gate_out_list[i][:, j].unsqueeze(1))
            for j in range(self.num_expert*2):
                tmp_expert = self.text_share_expert[0][j](text_feature)
                gate_expert += (tmp_expert * text_gate_out_list[i][:, (self.num_expert+j)].unsqueeze(1))
                gate_share_expert += (tmp_expert * text_gate_out_list[i][:, (self.num_expert+j)].unsqueeze(1))
            #print("gate_expert",gate_expert.size()) ([64, 320])
            text_gate_expert_value.append(gate_expert)
            text_gate_spacial_expert_value.append(gate_spacial_expert)
            text_gate_share_expert_value.append(gate_share_expert)
#6图像专家网络计算
        image_gate_expert_value = []
        image_gate_spacial_expert_value = []
        image_gate_share_expert_value = []
        for i in range(self.domain_num):
            gate_expert = 0
            gate_spacial_expert = 0
            gate_share_expert = 0
            for j in range(self.num_expert):
                tmp_expert = self.image_experts[i][j](image_feature)  # ([64, 320])
                gate_expert += (tmp_expert * image_gate_out_list[i][:, j].unsqueeze(1))  ##([64, 320]*[64, 1])
                gate_spacial_expert += (tmp_expert * image_gate_out_list[i][:, j].unsqueeze(1))
            for j in range(self.num_expert*2):
                tmp_expert = self.image_share_expert[0][j](image_feature)
                gate_expert += (tmp_expert * image_gate_out_list[i][:, (self.num_expert+j)].unsqueeze(1))
                gate_share_expert += (tmp_expert * image_gate_out_list[i][:, (self.num_expert+j)].unsqueeze(1))
            # print("gate_expert",gate_expert.size()) ([64, 320])
            image_gate_expert_value.append(gate_expert)
            image_gate_spacial_expert_value.append(gate_spacial_expert)
            image_gate_share_expert_value.append(gate_share_expert)

        #clip_fusion_feature
        #fusion

        text = text_gate_share_expert_value[0]
        image = image_gate_share_expert_value[0]
        fusion_share_feature = torch.cat((fusion_feature,text, image), dim=-1)
#7融合特征专家网络计算
        fusion_share_feature = self.MLP_fusion(fusion_share_feature)
        #fusion_share_feature = self.MLP_fusion(fusion_share_feature)
        #fusion_share_feature = clip_fusion_feature
        fusion_gate_input0 = self.domain_fusion(torch.cat([domain_embedding, fusion_share_feature], dim=-1))
        fusion_gate_out_list0 = []
        for k in range(self.domain_num):
            gate_out = self.fusion_gate_list0[k](fusion_gate_input0)
            fusion_gate_out_list0.append(gate_out)
        self.fusion_gate_out_list0 = fusion_gate_out_list0


        fusion_gate_expert_value0 = []
        fusion_gate_spacial_expert_value0 = []
        fusion_gate_share_expert_value0 = []
        for m in range(self.domain_num):
            share_gate_expert0 = 0
            gate_spacial_expert = 0
            gate_share_expert = 0
            for n in range(self.num_expert):
                fusion_tmp_expert0 = self.fusion_experts[m][n](fusion_share_feature)
                share_gate_expert0 += (fusion_tmp_expert0 * self.fusion_gate_out_list0[m][:, n].unsqueeze(1))
                gate_spacial_expert += (fusion_tmp_expert0 * self.fusion_gate_out_list0[m][:, n].unsqueeze(1))
            for n in range(self.num_expert * 2):
                fusion_tmp_expert0 = self.fusion_share_expert[0][n](fusion_share_feature)
                share_gate_expert0 += (fusion_tmp_expert0 * self.fusion_gate_out_list0[m][:, (self.num_expert + n)].unsqueeze(1))
                gate_share_expert += (fusion_tmp_expert0 * self.fusion_gate_out_list0[m][:, (self.num_expert + n)].unsqueeze(1))
            fusion_gate_expert_value0.append(share_gate_expert0)
            fusion_gate_spacial_expert_value0.append(gate_spacial_expert)
            fusion_gate_share_expert_value0.append(gate_share_expert)

##continue
#8预测输出
        #test
        text_only_output = []
        text_label_pred = []
        final_text_feature = []
        for i in range(self.domain_num):
            # label_pred.append(torch.sigmoid(self.text_classifier_list[i](text_gate_expert_value[i]).squeeze(1)))
            final_text_feature.append(text_gate_expert_value[i])
            text_class = self.text_classifier_list[i](text_gate_expert_value[i]).squeeze(1)
            text_only_output.append(text_class)
            pre = torch.sigmoid(text_class)
            text_label_pred.append(pre)
        text_label_pred_list = []
        text_label_pred_avg = 0
        for i in range(self.domain_num):#按区域拼接
            text_label_pred_list.append(text_label_pred[i][idxs.squeeze() == i])
            text_label_pred_avg += text_label_pred[i]
        text_label_pred_avg = text_label_pred_avg / self.domain_num
        text_label_pred_list = torch.cat((text_label_pred_list[0], text_label_pred_list[1], text_label_pred_list[2], text_label_pred_list[3],
                                     text_label_pred_list[4], text_label_pred_list[5], text_label_pred_list[6], text_label_pred_list[7], text_label_pred_list[8], text_label_pred_list[9]))
        #image
        image_only_output = []
        image_label_pred = []
        final_image_feature = []
        for i in range(self.domain_num):
            # label_pred.append(torch.sigmoid(self.text_classifier_list[i](text_gate_expert_value[i]).squeeze(1)))
            final_image_feature.append(image_gate_expert_value[i])
            image_class = self.image_classifier_list[i](image_gate_expert_value[i]).squeeze(1)
            image_only_output.append(image_class)
            pre = torch.sigmoid(image_class)
            image_label_pred.append(pre)
        image_label_pred_list = []
        image_label_pred_avg = 0
        for i in range(self.domain_num):
            image_label_pred_list.append(image_label_pred[i][idxs.squeeze() == i])
            image_label_pred_avg += image_label_pred[i]
        image_label_pred_avg = image_label_pred_avg / self.domain_num

        image_label_pred_list = torch.cat((image_label_pred_list[0], image_label_pred_list[1], image_label_pred_list[2], image_label_pred_list[3],
                                     image_label_pred_list[4], image_label_pred_list[5], image_label_pred_list[6], image_label_pred_list[7], image_label_pred_list[8], image_label_pred_list[9]))
        # fusion
        fusion_only_output = []
        fusion_label_pred = []
        final_fusion_feature = []
        for i in range(self.domain_num):
            # label_pred.append(torch.sigmoid(self.text_classifier_list[i](text_gate_expert_value[i]).squeeze(1)))
            final_fusion_feature.append(fusion_gate_expert_value0[i])
            fusion_class = self.fusion_classifier_list[i](fusion_gate_expert_value0[i]).squeeze(1)
            fusion_only_output.append(fusion_class)
            pre = torch.sigmoid(fusion_class)
            fusion_label_pred.append(pre)
        fusion_label_pred_list = []
        fusion_label_pred_avg = 0
        for i in range(self.domain_num):
            fusion_label_pred_list.append(fusion_label_pred[i][idxs.squeeze() == i])
            fusion_label_pred_avg += fusion_label_pred[i]
        fusion_label_pred_avg = fusion_label_pred_avg / self.domain_num
        fusion_label_pred_list = torch.cat(
            (fusion_label_pred_list[0], fusion_label_pred_list[1], fusion_label_pred_list[2], fusion_label_pred_list[3],
             fusion_label_pred_list[4], fusion_label_pred_list[5], fusion_label_pred_list[6],
             fusion_label_pred_list[7],fusion_label_pred_list[8], fusion_label_pred_list[9]))
        # pivot fusion
        text_gate_share_expert_value = text_gate_share_expert_value[0]
        image_gate_share_expert_value = image_gate_share_expert_value[0]
        fusion_gate_share_expert_value = fusion_gate_share_expert_value0[0]
#9all domain共享知识
        cross_knowledge = self.fusion_img_text(image_gate_share_expert_value, text_gate_share_expert_value, fusion_gate_share_expert_value,self.mlp_img,self.mlp_text,self.pivot_mlp_fusion,self.transformers,self.mlp_star_f1,self.mlp_star_f2)
        domain_special_list = []
 #领域内共享知识
        for i in range(self.domain_num):
            text_spacial_knowledge = text_gate_spacial_expert_value[i]
            image_spacial_knowledge = image_gate_spacial_expert_value[i]
            fusion_spacial_knowledge = fusion_gate_spacial_expert_value0[i]
            domain_knowledge = self.fusion_img_text(image_spacial_knowledge, text_spacial_knowledge,
                                                   fusion_spacial_knowledge,self.mlp_img_list[i],self.mlp_text_list[i],self.pivot_mlp_fusion_list[i],self.transformers_list[i],self.mlp_star_f1_list[i],self.mlp_star_f2_list[i])

            domain_special_list.append(domain_knowledge)
#10作者利用自适应实例标准化(ADAIN)方法进行重新训练公式，特定领域的知识和跨领域知识
        dom_mu = []
        share_mu = []
        dom_sigma = []
        share_sigma = []
        dom_score_list = []
        share_score_list = []
        for i in range(self.domain_num):
            # label_pred.append(torch.sigmoid(self.text_classifier_list[i](text_gate_expert_value[i]).squeeze(1)))
            image_class = self.dom_classifier_list[i](domain_special_list[i]).squeeze(1)
            dom_score_list.append(image_class)
        for i in range(self.domain_num):
            # label_pred.append(torch.sigmoid(self.text_classifier_list[i](text_gate_expert_value[i]).squeeze(1)))
            image_class = self.share_classifier_list[i](cross_knowledge).squeeze(1)
            share_score_list.append(image_class)

        for i in range(self.domain_num):
            dom_mu.append(self.mapping_IS_MLP_mu(torch.sigmoid(dom_score_list[i]).clone().detach().view(-1,1)))
            share_mu.append(self.mapping_T_MLP_mu(torch.sigmoid(share_score_list[i]).clone().detach().view(-1,1)))
            dom_sigma.append(self.mapping_IS_MLP_sigma(torch.sigmoid(dom_score_list[i]).clone().detach().view(-1,1)))
            share_sigma.append(self.mapping_T_MLP_sigma(torch.sigmoid(share_score_list[i]).clone().detach().view(-1,1)))

        concat_feature_list = []
        for i in range(self.domain_num):#
   #10adain重新调整输出
            final_dom_feature0 = self.adaIN(domain_special_list[i],dom_mu[i],dom_sigma[i])
            final_share_feature0 = self.adaIN(cross_knowledge, share_mu[i],share_sigma[i])
            concat_feature_main_biased = torch.stack((final_dom_feature0,
                                                      final_share_feature0,
                                                      ), dim=1)#([64, 2, 320])
            concat_feature_list.append(concat_feature_main_biased)
    #11最终门控预测，专家计算
        final_gate_out_list = []
        for i in range(self.domain_num):
            fusion_tempfeat_main_task, _ = self.final_attention(concat_feature_list[i])
            final_gate_input = torch.cat([domain_embedding, fusion_tempfeat_main_task], dim=-1)
            final_gate_out = self.final_gate_list[i](final_gate_input)
            final_gate_out_list.append(final_gate_out)

        final_gate_expert_value = []
        for i in range(self.domain_num):
            gate_expert = 0
            for j in range(self.num_expert):
                tmp_expert = self.final_experts[i][j](concat_feature_list[i])  # [64, 4, 320]
                tmp_expert = tmp_expert[:,0]
                gate_expert += (tmp_expert * final_gate_out_list[i][:, j].unsqueeze(1))  ##([64, 320]*[64, 1])
            for j in range(self.num_expert*2):
                tmp_expert = self.final_share_expert[0][j](concat_feature_list[i])
                tmp_expert = tmp_expert[:, 0]
                gate_expert += (tmp_expert * final_gate_out_list[i][:, (self.num_expert+j)].unsqueeze(1))
            # print("gate_expert",gate_expert.size()) ([64, 320])
            final_gate_expert_value.append(gate_expert)

        #final
        final_label_pred = []
        for i in range(self.domain_num):
            # label_pred.append(torch.sigmoid(self.text_classifier_list[i](text_gate_expert_value[i]).squeeze(1)))
            pre = torch.sigmoid(self.final_classifier_list[i](final_gate_expert_value[i]).squeeze(1))
            final_label_pred.append(pre)
        final_label_pred_list = []
        final_label_pred_avg = 0
        for i in range(self.domain_num):
            final_label_pred_list.append(final_label_pred[i][idxs.squeeze() == i])
            final_label_pred_avg += final_label_pred[i]
        final_label_pred_avg = final_label_pred_avg / self.domain_num
        final_label_pred_list = torch.cat((final_label_pred_list[0], final_label_pred_list[1], final_label_pred_list[2], final_label_pred_list[3],
                                     final_label_pred_list[4], final_label_pred_list[5], final_label_pred_list[6], final_label_pred_list[7],final_label_pred_list[8], final_label_pred_list[9]))



        #return final_label_pred_list, final_label_pred_avg,fusion_label_pred_list, fusion_label_pred_avg,image_label_pred_list, image_label_pred_avg,text_label_pred_list, text_label_pred_avg
        return final_label_pred_list,fusion_label_pred_list,image_label_pred_list,text_label_pred_list


class Trainer():
    def __init__(self,
                 emb_dim,
                 mlp_dims,
                 bert,
                 use_cuda,
                 lr,
                 dropout,
                 train_loader,
                 val_loader,
                 test_loader,
                 category_dict,
                 weight_decay,
                 save_param_dir,
                 loss_weight=[1, 0.006, 0.009, 5e-5],
                 early_stop=5,
                 epoches=100
                 ):
        self.lr = lr
        self.weight_decay = weight_decay
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.early_stop = early_stop
        self.epoches = epoches
        self.category_dict = category_dict
        self.loss_weight = loss_weight
        self.use_cuda = use_cuda

        self.emb_dim = emb_dim
        self.mlp_dims = mlp_dims
        self.bert = bert
        self.dropout = dropout
        if not os.path.exists(save_param_dir):
            os.makedirs(save_param_dir)
        self.save_param_dir = save_param_dir

    def train(self):
        self.model = MultiDomainPLEFENDModel(self.emb_dim, self.mlp_dims, self.bert, 320, self.dropout)
        if self.use_cuda:
            self.model = self.model.cuda()
        loss_fn = torch.nn.BCELoss()#二分类loss
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.98)#学习率调度
        recorder = Recorder(self.early_stop)#记录器，用于早停
        for epoch in range(self.epoches):
            self.model.train()
            train_data_iter = tqdm.tqdm(self.train_loader)
            avg_loss = Averager()
            for step_n, batch in enumerate(train_data_iter):
                batch_data = clipdata2gpu(batch)
                label = batch_data['label']
                category = batch_data['category']
                idxs = torch.tensor([index for index in category]).view(-1, 1)
                batch_label = torch.cat((label[idxs.squeeze() == 0], label[idxs.squeeze() == 1],
                                         label[idxs.squeeze() == 2], label[idxs.squeeze() == 3],
                                         label[idxs.squeeze() == 4], label[idxs.squeeze() == 5],
                                         label[idxs.squeeze() == 6], label[idxs.squeeze() == 7],label[idxs.squeeze() == 8], label[idxs.squeeze() == 9]))
#类别标签
#四种预测结果
                final_label_pred_list,fusion_label_pred_list,image_label_pred_list,text_label_pred_list = self.model(**batch_data)
                loss0 = loss_fn(final_label_pred_list, batch_label.float())
                loss1 = loss_fn(fusion_label_pred_list, batch_label.float())
                loss2 = loss_fn(image_label_pred_list, batch_label.float())
                loss3 = loss_fn(text_label_pred_list, batch_label.float())
                loss = 0.7*loss0+0.1*loss1+0.1*loss2+0.1*loss3
#总损失
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if (scheduler is not None):
                    scheduler.step()
                avg_loss.add(loss.item())
            print('Training Epoch {}; Loss {}; '.format(epoch + 1, avg_loss.item()))
            results0, results1, results2, results3 = self.test(self.val_loader)
            mark = recorder.add(results0)
#mark判断早停还是保存模型
            if mark == 'save':
                torch.save(self.model.state_dict(),
                           os.path.join(self.save_param_dir, 'parameter_mmdfnd.pkl'))
            elif mark == 'esc':
                break
            else:
                continue
  #记录最优参数
        self.model.load_state_dict(torch.load(os.path.join(self.save_param_dir, 'parameter_mmdfnd.pkl')))
  #测试集验证
        results0,results1,results2,results3 = self.test(self.test_loader)
        print(results0)
        return results0, os.path.join(self.save_param_dir, 'parameter_clip111.pkl')

    def test(self, dataloader):
        pred0 = []
        pred1 = []
        pred2 = []
        pred3 = []
        label1 = []
        category = []
        self.model.eval()#设置为评估模式
        data_iter = tqdm.tqdm(dataloader)
        for step_n, batch in enumerate(data_iter):
            with torch.no_grad():
                batch_data = clipdata2gpu(batch)
                label = batch_data['label']
                batch_category = batch_data['category']
                final_label_pred_list,fusion_label_pred_list,image_label_pred_list,text_label_pred_list= self.model(**batch_data)

                idxs = torch.tensor([index for index in batch_category]).view(-1, 1)
                batch_label_pred0 = final_label_pred_list
                batch_label_pred1 = fusion_label_pred_list
                batch_label_pred2 = image_label_pred_list
                batch_label_pred3 = text_label_pred_list

#分类拼接标签
                batch_label = torch.cat((label[idxs.squeeze() == 0], label[idxs.squeeze() == 1],
                                         label[idxs.squeeze() == 2], label[idxs.squeeze() == 3],
                                         label[idxs.squeeze() == 4], label[idxs.squeeze() == 5],
                                         label[idxs.squeeze() == 6], label[idxs.squeeze() == 7],label[idxs.squeeze() == 8], label[idxs.squeeze() == 9]))
                batch_category = torch.sort(batch_category).values
                label1.extend(batch_label.detach().cpu().numpy().tolist())
                pred0.extend(batch_label_pred0.detach().cpu().numpy().tolist())
                pred1.extend(batch_label_pred1.detach().cpu().numpy().tolist())
                pred2.extend(batch_label_pred2.detach().cpu().numpy().tolist())
                pred3.extend(batch_label_pred3.detach().cpu().numpy().tolist())
                category.extend(batch_category.detach().cpu().numpy().tolist())

        return metricsTrueFalse(label1, pred0, category, self.category_dict),metricsTrueFalse(label1, pred1, category, self.category_dict),metricsTrueFalse(label1, pred2, category, self.category_dict),metricsTrueFalse(label1, pred3, category, self.category_dict)
