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
    image_list = {}
    file_list = ['data/nonrumor_images/', 'data/rumor_images/']
    for path in file_list:
        data_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        for i, filename in enumerate(os.listdir(path)):  # assuming gif

            # print(filename)
            try:
                im = Image.open(path + filename).convert('RGB')
                im = data_transforms(im)
                #im = 1
                image_list[filename.split('/')[-1].split(".")[0].lower()] = im
            except:
                print("wrong"+filename)
    print("image length " + str(len(image_list)))
    #print("image names are " + str(image_list.keys()))
    return image_list

def _init_fn(worker_id):
    np.random.seed(2024)

def read_pkl(path):
    with open(path,"rb")as f:
        t = pickle.load(f)
    return t
def df_filter(df_data):
    df_data = df_data[df_data['category'] != '无法确定']
    return df_data

def word2input(texts, vocab_file, max_len):
    """文本分词处理，修复nan值问题"""
    tokenizer = BertTokenizer(vocab_file=vocab_file)
    token_ids = []
    for i, text in enumerate(texts):
        # 处理nan/空值
        if pd.isna(text):
            text = ""
        elif not isinstance(text, str):
            text = str(text)
        text = text.strip()
        
        token_ids.append(tokenizer.encode(
            text, 
            max_length=max_len, 
            add_special_tokens=True, 
            padding='max_length',
            truncation=True
        ))
    token_ids = torch.tensor(token_ids)
    masks = torch.zeros(token_ids.size())
    for i, token in enumerate(token_ids):
        masks[i] = (token != 0)
    return token_ids, masks

class bert_data():
    def __init__(self, max_len, batch_size, vocab_file, category_dict, num_workers=2):
        self.max_len = max_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.vocab_file = vocab_file
        self.category_dict = category_dict

    def load_data(self, path, imagepath, clipimagepath, shuffle, text_only=False):
        """加载数据，处理图片/文本数量不匹配问题"""
        # ========== 1. 加载文本数据 ==========
        self.data = pd.read_excel(path)
        original_len = len(self.data)
        print(f"原始数据: {original_len} 条")
        
        # 处理content列的nan值
        if 'content' in self.data.columns:
            self.data['content'] = self.data['content'].fillna("").astype(str).str.strip()
            # 过滤空文本
            self.data = self.data[self.data['content'] != ""]
        
        # 过滤无法确定的类别
        if 'category' in self.data.columns:
            self.data = self.data[self.data['category'] != '无法确定']
        
        self.data = self.data.reset_index(drop=True)
        print(f"过滤后数据: {len(self.data)} 条")
        
        # ========== 2. 加载CLIP模型 ==========
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            clipmodel, _ = load_from_name("ViT-B-16", device=device, download_root='./')
        except Exception as e:
            print(f"警告: CLIP模型加载失败 ({e})，继续使用预处理的CLIP特征...")
            clipmodel = None
        
        # ========== 3. 处理文本数据 ==========
        content = self.data['content'].astype('object').to_numpy()
        token_ids, masks = word2input(content, self.vocab_file, self.max_len)
        
        # ========== 4. 处理标签和类别 ==========
        label = torch.tensor(self.data['label'].astype(int).to_numpy(), dtype=torch.long)
        
        if 'category' not in self.data.columns:
            print("警告：数据中无category列，使用默认类别0")
            category = torch.zeros(len(self.data), dtype=torch.long)
        else:
            # 类别名称映射表（处理数据中的简化类别名）
            # 将数据中的类别映射到模型期望的类别
            category_mapping = {
                '社会': '社会生活',
                '生活': '社会生活', 
                '娱乐': '文体娱乐',
                '科技': '科技',
                '军事': '军事',
                '教育': '教育考试',
                '政治': '政治',
                '健康': '医药健康',
                '医药': '医药健康',
                '财经': '财经商业',
                '商业': '财经商业',
                '灾难': '灾难事故',
                '事故': '灾难事故',
                '文体': '文体娱乐',
            }
            
            # 映射类别名称
            def map_category(cat):
                # 1. 直接匹配
                if cat in self.category_dict:
                    return self.category_dict[cat]
                # 2. 通过映射表匹配
                if cat in category_mapping:
                    mapped = category_mapping[cat]
                    if mapped in self.category_dict:
                        return self.category_dict[mapped]
                # 3. 部分匹配（类别名包含在字典键中，或字典键包含在类别名中）
                for key in self.category_dict:
                    if cat in key or key in cat:
                        return self.category_dict[key]
                # 4. 都匹配不上，返回0
                return 0
            
            # 检查未知类别
            unknown_categories = [c for c in self.data['category'].unique() if c not in self.category_dict]
            if unknown_categories:
                print(f"警告：发现未定义的类别 {unknown_categories}，尝试智能映射...")
                for c in unknown_categories:
                    mapped_idx = map_category(c)
                    # 找到映射到的类别名
                    mapped_name = [k for k, v in self.category_dict.items() if v == mapped_idx]
                    mapped_name = mapped_name[0] if mapped_name else f"索引{mapped_idx}"
                    print(f"  '{c}' -> '{mapped_name}' (索引 {mapped_idx})")
            
            category = torch.tensor(
                self.data['category'].apply(map_category).to_numpy(),
                dtype=torch.long
            )
        
        # ========== 5. 加载图片数据 ==========
        ordered_image = pickle.load(open(imagepath, 'rb'))
        if isinstance(ordered_image, torch.Tensor):
            ordered_image = ordered_image.cpu()
        clip_image = pickle.load(open(clipimagepath, 'rb'))
        if isinstance(clip_image, torch.Tensor):
            clip_image = clip_image.cpu()
        
        # ========== 6. 处理图片/文本数量不匹配 ==========
        # 重要：pkl文件是按顺序生成的（0, 1, 2, ...），不是按原始行号
        # 所以我们只需要取 min(所有数据) 条数据
        n_ordered_image = ordered_image.shape[0]
        n_clip_image = clip_image.shape[0]
        n_texts = len(self.data)
        
        print(f"图片数据: {n_ordered_image} 条, CLIP图片: {n_clip_image} 条, 文本数据: {n_texts} 条")
        
        # 取所有数据的最小值作为最终样本数
        n_samples = min(n_ordered_image, n_clip_image, n_texts)
        
        if n_samples == 0:
            raise ValueError("没有有效数据！")
        
        if not (n_ordered_image == n_clip_image == n_texts):
            print(f"⚠️  数据数量不匹配，使用前 {n_samples} 条数据")
        
        # 截取数据到相同长度
        ordered_image = ordered_image[:n_samples]
        clip_image = clip_image[:n_samples]
        token_ids = token_ids[:n_samples]
        masks = masks[:n_samples]
        label = label[:n_samples]
        category = category[:n_samples]
        content = content[:n_samples]
        
        print(f"最终数据: {n_samples} 条")
        
        # ========== 7. CLIP文本编码 ==========
        clip_content = [t if (not pd.isna(t) and isinstance(t, str)) else "" for t in content]
        try:
            clip_text = clip.tokenize(clip_content, truncate=True)
        except TypeError:
            truncated_content = [text[:200] if len(text) > 200 else text for text in clip_content]
            clip_text = clip.tokenize(truncated_content)
        
        # ========== 8. 验证维度 ==========
        print(f"\n数据维度检查:")
        print(f"  token_ids: {token_ids.shape}")
        print(f"  masks: {masks.shape}")
        print(f"  label: {label.shape}")
        print(f"  category: {category.shape}")
        print(f"  ordered_image: {ordered_image.shape}")
        print(f"  clip_image: {clip_image.shape}")
        print(f"  clip_text: {clip_text.shape}")
        
        # ========== 9. 构建DataLoader ==========
        datasets = TensorDataset(
            token_ids,
            masks,
            label,
            category,
            ordered_image,
            clip_image,
            clip_text
        )
        
        use_pin_memory = torch.cuda.is_available() and self.num_workers > 0
        
        dataloader = DataLoader(
            dataset=datasets,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=use_pin_memory,
            shuffle=shuffle,
            worker_init_fn=_init_fn if self.num_workers > 0 else None
        )
        return dataloader
