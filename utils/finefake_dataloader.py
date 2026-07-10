# -*- coding: utf-8 -*-
import pickle
import cn_clip.clip as clip
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer
import torch
import pandas as pd
import os
import numpy as np


def _init_fn(worker_id):
    """DataLoader worker 初始化函数，固定随机种子"""
    np.random.seed(2024)


def read_pkl(path):
    """读取 pkl 文件"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"pkl 文件不存在: {path}")
    with open(path, "rb") as f:
        t = pickle.load(f)
    return t


def word2input(texts, vocab_file, max_len):
    """
    文本分词处理（支持中英文 BERT）
    :param texts: 文本列表
    :param vocab_file: 分词器词典文件路径或模型名
    :param max_len: 最大序列长度
    :return: token_ids, masks (torch.tensor)
    """
    tokenizer = None

    # 1. 尝试从 vocab_file 加载
    if vocab_file and os.path.isfile(vocab_file):
        try:
            tokenizer = BertTokenizer(vocab_file=vocab_file)
        except Exception:
            pass

    # 2. 尝试从目录加载（from_pretrained）
    if tokenizer is None and vocab_file and os.path.isdir(vocab_file):
        try:
            tokenizer = BertTokenizer.from_pretrained(vocab_file)
        except Exception:
            pass

    # 3. 尝试将 vocab_file 当作模型名（如 'bert-base-uncased'）
    if tokenizer is None and vocab_file:
        try:
            tokenizer = BertTokenizer.from_pretrained(vocab_file)
        except Exception:
            pass

    # 4. 回退到默认英文 BERT
    if tokenizer is None:
        print("⚠️  无法加载指定的分词器，使用 bert-base-uncased ...")
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    token_ids = []
    for i, text in enumerate(texts):
        # 处理 nan/空值
        if pd.isna(text):
            text = ""
        elif not isinstance(text, str):
            text = str(text)
        text = text.strip()

        encoded = tokenizer.encode(
            text,
            max_length=max_len,
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            return_tensors=None
        )
        token_ids.append(encoded)

    token_ids = torch.tensor(token_ids)
    masks = torch.zeros(token_ids.size(), dtype=torch.float32)
    for i, token in enumerate(token_ids):
        masks[i] = (token != 0).float()

    return token_ids, masks


class bert_data():
    """FineFake 数据加载器"""

    def __init__(self, max_len, batch_size, vocab_file, category_dict, num_workers=0):
        self.max_len = max_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.vocab_file = vocab_file
        self.category_dict = category_dict

    def load_data(self, path, imagepath, clipimagepath, shuffle, text_only=False):
        """
        加载数据并构建 DataLoader
        :param path: CSV 文件路径
        :param imagepath: MAE 图片 pkl 路径
        :param clipimagepath: CLIP 图片 pkl 路径
        :param shuffle: 是否打乱
        :return: DataLoader
        """
        # ========== 1. 加载文本数据 ==========
        if path.endswith('.xls') or path.endswith('.xlsx'):
            self.data = pd.read_excel(path)
        else:
            self.data = pd.read_csv(path, encoding='utf-8')

        original_len = len(self.data)
        print(f"\n[FineFake DataLoader] 加载数据: {path}")
        print(f"  原始数据: {original_len} 条")

        # ========== 2. 数据清洗 ==========
        # FineFake 的文本列名为 'text'
        text_col = 'text'
        if text_col not in self.data.columns:
            # 兼容：如果有 'content' 列则使用
            if 'content' in self.data.columns:
                text_col = 'content'
            else:
                raise ValueError(f"数据中必须包含 'text' 或 'content' 列，当前列: {self.data.columns.tolist()}")

        self.data[text_col] = self.data[text_col].fillna("").astype(str).str.strip()
        self.data = self.data[self.data[text_col] != ""]
        self.data = self.data.reset_index(drop=True)
        print(f"  清洗后数据: {len(self.data)} 条")

        # ========== 3. 处理文本 ==========
        content = self.data[text_col].astype('object').to_numpy()
        token_ids, masks = word2input(content, self.vocab_file, self.max_len)

        # ========== 4. 处理标签 ==========
        if 'label' not in self.data.columns:
            raise ValueError("数据中必须包含 'label' 列")
        label = torch.tensor(self.data['label'].astype(int).to_numpy(), dtype=torch.long)

        # ========== 5. 处理话题/领域 ==========
        # FineFake 的话题列名为 'topic'
        topic_col = 'topic'
        if topic_col not in self.data.columns:
            if 'category' in self.data.columns:
                topic_col = 'category'
            else:
                raise ValueError(f"数据中必须包含 'topic' 或 'category' 列")

        def map_category(cat):
            if cat in self.category_dict:
                return self.category_dict[cat]
            # 尝试部分匹配
            for key in self.category_dict:
                if str(cat).lower() in key.lower() or key.lower() in str(cat).lower():
                    return self.category_dict[key]
            # 默认返回 0
            return 0

        # 检查未知类别
        known = set(self.category_dict.keys())
        unknown = [c for c in self.data[topic_col].unique() if c not in known]
        if unknown:
            print(f"  ⚠️  发现未知话题: {unknown}，将尝试自动映射")

        category = torch.tensor(
            self.data[topic_col].apply(map_category).to_numpy(),
            dtype=torch.long
        )

        # ========== 6. 加载图片 pkl ==========
        try:
            ordered_image = read_pkl(imagepath)
            clip_image = read_pkl(clipimagepath)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"图片 pkl 文件不存在: {e}\n"
                f"请先运行 finefake_preprocess.py 生成图片数据"
            )

        # ========== 7. 数据对齐（处理数量不匹配） ==========
        n_text = len(self.data)
        n_mae = ordered_image.shape[0]
        n_clip = clip_image.shape[0]
        n_samples = min(n_text, n_mae, n_clip)

        if not (n_text == n_mae == n_clip):
            print(f"  ⚠️  数据数量不匹配: 文本 {n_text}, MAE {n_mae}, CLIP {n_clip}")
            print(f"      使用前 {n_samples} 条")

        ordered_image = ordered_image[:n_samples]
        clip_image = clip_image[:n_samples]
        token_ids = token_ids[:n_samples]
        masks = masks[:n_samples]
        label = label[:n_samples]
        category = category[:n_samples]
        content = content[:n_samples]

        # ========== 8. CLIP 文本编码 ==========
        clip_content = [t if (not pd.isna(t) and isinstance(t, str)) else "" for t in content]
        try:
            clip_text = clip.tokenize(clip_content, truncate=True)
        except TypeError:
            # 旧版本 CLIP 兼容
            truncated_content = [text[:200] if len(text) > 200 else text for text in clip_content]
            clip_text = clip.tokenize(truncated_content)

        # ========== 9. 构建 DataLoader ==========
        print(f"\n  数据维度:")
        print(f"    文本 token:   {token_ids.shape}")
        print(f"    注意力掩码:   {masks.shape}")
        print(f"    标签:         {label.shape}")
        print(f"    话题:         {category.shape}")
        print(f"    MAE 图片:     {ordered_image.shape}")
        print(f"    CLIP 图片:    {clip_image.shape}")
        print(f"    CLIP 文本:    {clip_text.shape}")

        datasets = TensorDataset(
            token_ids,
            masks,
            label,
            category,
            ordered_image,
            clip_image,
            clip_text
        )

        # Windows 兼容
        use_pin_memory = torch.cuda.is_available() and self.num_workers > 0
        if os.name == 'nt' and self.num_workers > 0:
            print("  ⚠️  Windows 系统下 num_workers 设为 0")
            self.num_workers = 0

        dataloader = DataLoader(
            dataset=datasets,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=use_pin_memory,
            shuffle=shuffle,
            worker_init_fn=_init_fn if self.num_workers > 0 else None
        )

        print(f"  DataLoader 已构建: {len(dataloader)} 个 batch")
        return dataloader
