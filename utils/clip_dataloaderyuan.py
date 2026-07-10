# -*-codeing = utf-8 -*-
# Quantum-inspired Multimodal Multi-domain Fake News Detection
import pickle
import cn_clip.clip as clip
from cn_clip.clip import load_from_name, available_models
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer
import torch
import pandas as pd
from torchvision import datasets, models, transforms
import os
import numpy as np
from PIL import Image


def read_image():
    """读取图片并转换为tensor，返回图片名到tensor的映射"""
    image_list = {}
    file_list = ['data/nonrumor_images/', 'data/rumor_images/']

    # 定义图片预处理
    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 遍历图片路径
    for path in file_list:
        # 确保路径存在
        if not os.path.exists(path):
            print(f"警告：图片路径 {path} 不存在，跳过")
            continue

        for filename in os.listdir(path):
            try:
                # 读取并处理图片
                img_path = os.path.join(path, filename)
                im = Image.open(img_path).convert('RGB')
                im = data_transforms(im)
                # 统一图片名格式（小写，去后缀）
                img_key = filename.split('/')[-1].split(".")[0].lower()
                image_list[img_key] = im
            except Exception as e:
                print(f"处理图片失败 {filename}: {str(e)}")

    print(f"成功加载图片数量: {len(image_list)}")
    return image_list


def _init_fn(worker_id):
    """DataLoader worker初始化函数，固定随机种子"""
    np.random.seed(2024)


def read_pkl(path):
    """读取pkl文件"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"pkl文件不存在: {path}")
    with open(path, "rb") as f:
        t = pickle.load(f)
    return t


def df_filter(df_data):
    """过滤掉类别为'无法确定'的行"""
    # 先检查列是否存在
    if 'category' not in df_data.columns:
        print("警告：数据中无category列，跳过过滤")
        return df_data
    # 过滤无法确定的类别
    df_data = df_data[df_data['category'] != '无法确定']
    # 重置索引
    df_data = df_data.reset_index(drop=True)
    return df_data


def word2input(texts, vocab_file, max_len):
    # 加载分词器
    try:
        tokenizer = BertTokenizer(vocab_file=vocab_file)
    except (ValueError, FileNotFoundError):
        print("本地vocab.txt未找到，从Hugging Face自动下载中文BERT模型...")

        # 清除可能导致问题的环境变量
        if 'HF_ENDPOINT' in os.environ:
            original_endpoint = os.environ['HF_ENDPOINT']
            del os.environ['HF_ENDPOINT']
            print(f"⚠️  已临时清除HF_ENDPOINT设置: {original_endpoint}")

        try:
            tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
            print("✓ 模型下载完成")
        except Exception as e:
            print(f"✗ 下载失败: {e}")
            print("\n解决方案：")
            print("1. 检查网络连接")
            print("2. 清除HF_ENDPOINT环境变量")
            print("3. 或手动下载模型文件")
            raise

    token_ids = []
    # 遍历文本列表，处理每个文本
    for i, text in enumerate(texts):
        # ========== 核心修复：处理nan/空值/非字符串 ==========
        # 1. 处理nan值
        if pd.isna(text):
            text = ""
        # 2. 确保是字符串类型
        elif not isinstance(text, str):
            text = str(text)
        # 3. 去除首尾空白字符
        text = text.strip()

        # ========== 分词处理 ==========
        encoded = tokenizer.encode(
            text,
            max_length=max_len,
            add_special_tokens=True,
            padding='max_length',
            truncation=True,  # 截断超长文本
            return_tensors=None  # 不返回tensor，避免维度问题
        )
        token_ids.append(encoded)

    # 转换为tensor并生成attention mask
    token_ids = torch.tensor(token_ids)
    masks = torch.zeros(token_ids.size(), dtype=torch.float32)
    for i, token in enumerate(token_ids):
        masks[i] = (token != 0).float()  # mask: 1表示有效token，0表示padding

    return token_ids, masks


class bert_data():
    """BERT数据加载器类"""

    def __init__(self, max_len, batch_size, vocab_file, category_dict, num_workers=2):
        self.max_len = max_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.vocab_file = vocab_file
        self.category_dict = category_dict
        # 检查类别字典是否为空
        if not isinstance(self.category_dict, dict) or len(self.category_dict) == 0:
            raise ValueError("category_dict必须是非空字典")

    def load_data(self, path, imagepath, clipimagepath, shuffle, text_only=False):
        # ========== 1. 加载并清洗文本数据 ==========
        # 读取数据
        if path.endswith('.xls') or path.endswith('.xlsx'):
            self.data = pd.read_excel(path)
        else:
            self.data = pd.read_csv(path, encoding='utf-8')

        # 数据清洗
        self.data['_original_idx'] = range(len(self.data))
        self.data = df_filter(self.data)  # 过滤无法确定的类别
        # 填充/清洗content列的nan值
        if 'content' in self.data.columns:
            self.data['content'] = self.data['content'].fillna("").astype(str).str.strip()
            # 删除空文本行
            # self.data = self.data[self.data['content'] != ""].reset_index(drop=True)
            self.data = self.data[self.data['content'] != ""]
        else:
            raise ValueError("数据中必须包含content列")

        # 检查数据是否为空
        if len(self.data) == 0:
            raise ValueError("数据清洗后为空，请检查数据源")

        # ========== 2. 加载CLIP模型（仅用于tokenize文本） ==========
        valid_indices = self.data['_original_idx'].tolist()
        self.data = self.data.drop(columns=['_original_idx']).reset_index(drop=True)
        print(f"数据清洗完成：原始 {len(valid_indices) + (self.data.shape[0] - len(valid_indices))} 条 -> 有效 {len(valid_indices)} 条")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        clipmodel, _ = load_from_name("ViT-B-16", device=device, download_root='./')
        clipmodel.eval()  # 设置为评估模式

        # ========== 3. 处理文本数据 ==========
        content = self.data['content'].astype('object').to_numpy()
        # 分词处理
        token_ids, masks = word2input(content, self.vocab_file, self.max_len)

        # ========== 4. 处理标签和类别 ==========
        # 标签处理
        if 'label' not in self.data.columns:
            raise ValueError("数据中必须包含label列")
        label = torch.tensor(self.data['label'].astype(int).to_numpy(), dtype=torch.long)

        # 类别处理（修复KeyError）
        if 'category' not in self.data.columns:
            raise ValueError("数据中必须包含category列")
        # 先检查所有类别是否在字典中
        unknown_categories = [c for c in self.data['category'].unique() if c not in self.category_dict]
        if unknown_categories:
            print(f"警告：发现未定义的类别 {unknown_categories}，自动映射为0")
            # 补充未知类别到字典
            for c in unknown_categories:
                self.category_dict[c] = 0
        # 转换类别为数字
        category = torch.tensor(
            self.data['category'].apply(lambda c: self.category_dict.get(c, 0)).to_numpy(),
            dtype=torch.long
        )

        # ========== 5. 加载图片数据 ==========
        try:
            ordered_image = read_pkl(imagepath)
            clip_image = read_pkl(clipimagepath)
        except Exception as e:
            raise ValueError(f"加载图片pkl文件失败: {str(e)}")

        # 确保图片数据维度匹配
        expected_size = len(self.data)
        if ordered_image.shape[0] != expected_size:
            # 截断或填充图片数据以匹配文本数据
            if ordered_image.shape[0] > expected_size:
                ordered_image = ordered_image[:expected_size]
                clip_image = clip_image[:expected_size]
                print(f"图片数据过长，已截断至 {expected_size} 条")
            else:
                # 填充空图片（不推荐，仅应急）
                pad_size = expected_size - ordered_image.shape[0]
                pad_image = torch.zeros((pad_size,) + ordered_image.shape[1:])
                ordered_image = torch.cat([ordered_image, pad_image], dim=0)
                clip_image = torch.cat([clip_image, torch.zeros((pad_size,) + clip_image.shape[1:])], dim=0)
                print(f"图片数据过短，已填充 {pad_size} 条空数据")

        # ========== 6. CLIP文本编码 ==========
        # 处理nan/空文本
        clip_content = [t if (not pd.isna(t) and isinstance(t, str)) else "" for t in content]
        try:
            # 新版本CLIP
            clip_text = clip.tokenize(clip_content, truncate=True)
        except TypeError:
            # 旧版本CLIP兼容
            print("⚠️  检测到旧版本CLIP，使用兼容模式截断文本")
            truncated_content = [text[:200] if len(text) > 200 else text for text in clip_content]
            clip_text = clip.tokenize(truncated_content)

        # ========== 7. 构建DataLoader ==========
        # 检查所有tensor维度
        print(f"\n数据维度检查:")
        print(f"  文本token: {token_ids.shape}")
        print(f"  注意力掩码: {masks.shape}")
        print(f"  标签: {label.shape}")
        print(f"  类别: {category.shape}")
        print(f"  图片数据: {ordered_image.shape}")
        print(f"  CLIP图片: {clip_image.shape}")
        print(f"  CLIP文本: {clip_text.shape}")

        # 构建数据集
        datasets = TensorDataset(
            token_ids,
            masks,
            label,
            category,
            ordered_image,
            clip_image,
            clip_text
        )

        # 配置DataLoader参数（Windows兼容）
        use_pin_memory = torch.cuda.is_available() and self.num_workers > 0
        # Windows系统下num_workers设为0避免报错
        if os.name == 'nt' and self.num_workers > 0:
            print("警告：Windows系统下num_workers设为0以避免多进程错误")
            self.num_workers = 0

        dataloader = DataLoader(
            dataset=datasets,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=use_pin_memory,
            shuffle=shuffle,
            worker_init_fn=_init_fn if self.num_workers > 0 else None
        )

        return dataloader
