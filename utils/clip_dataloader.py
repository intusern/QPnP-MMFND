# -*-coding = utf-8 -*-
# @Time : 2023-12-11 22:56
# @Author : 童宇
# @File : clip_dataloader.py
# @Software :
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
    """
    文本分词处理，修复nan值、空值问题
    :param texts: 文本列表（可能含nan/空值）
    :param vocab_file: 分词器词典文件路径
    :param max_len: 最大序列长度
    :return: token_ids, masks (torch.tensor)
    """
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
        """
        加载数据并构建DataLoader
        :param path: 文本数据路径（csv/xlsx）
        :param imagepath: 图片pkl文件路径
        :param clipimagepath: CLIP图片特征pkl路径
        :param shuffle: 是否打乱数据
        :param text_only: 是否仅使用文本数据
        :return: DataLoader
        """
        # ========== 1. 加载并清洗文本数据 ==========
        # 读取数据
        if path.endswith('.xls') or path.endswith('.xlsx'):
            self.data = pd.read_excel(path)
        else:
            self.data = pd.read_csv(path, encoding='utf-8')
        
        original_len = len(self.data)
        print(f"原始数据: {original_len} 条")

        # 数据清洗
        self.data = df_filter(self.data)  # 过滤无法确定的类别
        # 填充/清洗content列的nan值
        if 'content' in self.data.columns:
            self.data['content'] = self.data['content'].fillna("").astype(str).str.strip()
            # 删除空文本行
            self.data = self.data[self.data['content'] != ""]
        else:
            raise ValueError("数据中必须包含content列")

        # 检查数据是否为空
        if len(self.data) == 0:
            raise ValueError("数据清洗后为空，请检查数据源")
        
        self.data = self.data.reset_index(drop=True)
        print(f"数据清洗完成: {original_len} 条 -> {len(self.data)} 条")

        # ========== 2. 加载CLIP模型（仅用于tokenize文本） ==========
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

        # 类别处理（智能映射类别名称）
        if 'category' not in self.data.columns:
            raise ValueError("数据中必须包含category列")
        
        # 类别名称映射表（处理数据中的简化类别名）
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
            # weibo数据集的类别
            '经济': '经济',
            '国际': '国际',
            '未分类': '未分类',
        }
        
        # 智能映射类别名称
        def map_category(cat):
            # 1. 直接匹配
            if cat in self.category_dict:
                return self.category_dict[cat]
            # 2. 通过映射表匹配
            if cat in category_mapping:
                mapped = category_mapping[cat]
                if mapped in self.category_dict:
                    return self.category_dict[mapped]
            # 3. 部分匹配
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
                mapped_name = [k for k, v in self.category_dict.items() if v == mapped_idx]
                mapped_name = mapped_name[0] if mapped_name else f"索引{mapped_idx}"
                print(f"  '{c}' -> '{mapped_name}' (索引 {mapped_idx})")
        
        # 转换类别为数字
        category = torch.tensor(
            self.data['category'].apply(map_category).to_numpy(),
            dtype=torch.long
        )

        # ========== 5. 加载图片数据 ==========
        try:
            ordered_image = read_pkl(imagepath)
            clip_image = read_pkl(clipimagepath)
        except Exception as e:
            raise ValueError(f"加载图片pkl文件失败: {str(e)}")

        # ========== 6. 处理图片/文本数量不匹配 ==========
        # 重要：pkl文件是按顺序生成的（0, 1, 2, ...），不是按原始CSV行号
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

# # -*-codeing = utf-8 -*-
# # @Time : 2023-12-1122:56
# # @Author : 童宇
# # @File : dataloader.py
# # @software :
# import pickle
# import cn_clip.clip as clip
# from cn_clip.clip import load_from_name, available_models
# from torch.utils.data import TensorDataset, DataLoader
# from transformers import BertTokenizer
# import torch
# import pandas as pd
# from torchvision import datasets, models, transforms
# import os
# import numpy as np
# from PIL import Image
#
# def read_image():
#     image_list = {}
#     file_list = ['data/nonrumor_images/', 'data/rumor_images/']
#     for path in file_list:
#         data_transforms = transforms.Compose([
#             transforms.Resize(256),
#             transforms.CenterCrop(224),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#             ])
#
#         for i, filename in enumerate(os.listdir(path)):  # assuming gif
#
#             # print(filename)
#             try:
#                 im = Image.open(path + filename).convert('RGB')
#                 im = data_transforms(im)
#                 #im = 1
#                 image_list[filename.split('/')[-1].split(".")[0].lower()] = im
#             except:
#                 print("wrong"+filename)
#     #print("image length " + str(len(image_list)))
#     #print("image names are " + str(image_list.keys()))
#     return image_list
#
# def _init_fn(worker_id):
#     np.random.seed(2024)
#
# def read_pkl(path):
#     with open(path,"rb")as f:
#         t = pickle.load(f)
#     return t
# def df_filter(df_data):
#     df_data = df_data[df_data['category'] != '无法确定']
#     return df_data
#
# def word2input(texts,vocab_file,max_len):
#     # 尝试从本地加载，如果失败则从Hugging Face自动下载
#     try:
#         tokenizer = BertTokenizer(vocab_file=vocab_file)
#     except (ValueError, FileNotFoundError):
#         print("本地vocab.txt未找到，从Hugging Face自动下载中文BERT模型...")
#
#         # 清除可能导致问题的环境变量
#         import os
#         if 'HF_ENDPOINT' in os.environ:
#             original_endpoint = os.environ['HF_ENDPOINT']
#             del os.environ['HF_ENDPOINT']
#             print(f"⚠️  已临时清除HF_ENDPOINT设置: {original_endpoint}")
#             print("   直接从Hugging Face官方下载...")
#
#         try:
#             tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
#             print("✓ 模型下载完成")
#         except Exception as e:
#             print(f"✗ 下载失败: {e}")
#             print("\n解决方案：")
#             print("1. 检查网络连接")
#             print("2. 清除HF_ENDPOINT环境变量")
#             print("3. 或手动下载模型文件")
#             raise
#
#     token_ids =[]
#     for i,text in enumerate(texts):
#         token_ids.append(tokenizer.encode(text, max_length=max_len, add_special_tokens=True, padding='max_length',
#                              truncation=True))
#     token_ids = torch.tensor(token_ids)
#     masks = torch.zeros(token_ids.size())
#     for i,token in enumerate(token_ids):
#         masks[i] = (token != 0)
#     return token_ids,masks
#
# class bert_data():
#     def __init__(self,max_len, batch_size, vocab_file, category_dict, num_workers=2):
#         self.max_len = max_len
#         self.batch_size = batch_size
#         self.num_workers = num_workers
#         self.vocab_file = vocab_file
#         self.category_dict = category_dict
#
#     def load_data(self,path,imagepath,clipimagepath,shuffle,text_only = False):
#         # 根据文件扩展名选择读取方法
#         if path.endswith('.xls') or path.endswith('.xlsx'):
#             self.data = pd.read_excel(path)
#         else:
#             self.data = pd.read_csv(path,encoding='utf-8')
#         device = "cuda" if torch.cuda.is_available() else "cpu"
#         clipmodel, _ = load_from_name("ViT-B-16", device=device, download_root='./')
#         content = self.data['content'].astype('object').to_numpy()
#         label = torch.tensor(self.data['label'].astype('object').astype(int).to_numpy())
#         category = torch.tensor(self.data['category'].astype('object').apply(lambda c: self.category_dict[c]).to_numpy())
#         token_ids, masks = word2input(content,self.vocab_file,self.max_len)
#         ordered_image = pickle.load(open(imagepath,'rb'))
#         clip_image = pickle.load(open(clipimagepath, 'rb'))
#
#         # CLIP tokenize - 兼容不同版本
#         try:
#             # 尝试新版本的truncate参数
#             clip_text = clip.tokenize(content, truncate=True)
#         except TypeError:
#             # 旧版本不支持truncate，手动截断文本
#             print("⚠️  检测到旧版本CLIP，使用兼容模式...")
#             # CLIP默认最大长度是77，手动截断长文本
#             truncated_content = [text[:200] if len(text) > 200 else text for text in content]
#             clip_text = clip.tokenize(truncated_content)
#
#         # 调试信息：检查所有tensor的大小
#         print(f"数据大小检查:")
#         print(f"  token_ids: {token_ids.shape}")
#         print(f"  masks: {masks.shape}")
#         print(f"  label: {label.shape}")
#         print(f"  category: {category.shape}")
#         print(f"  ordered_image: {ordered_image.shape}")
#         print(f"  clip_image: {clip_image.shape}")
#         print(f"  clip_text: {clip_text.shape}")
#
#         # 确保所有tensor的第一维大小相同
#         expected_size = token_ids.size(0)
#         if not all([
#             masks.size(0) == expected_size,
#             label.size(0) == expected_size,
#             category.size(0) == expected_size,
#             ordered_image.size(0) == expected_size,
#             clip_image.size(0) == expected_size,
#             clip_text.size(0) == expected_size
#         ]):
#             print("\n⚠️  错误：tensor大小不匹配！")
#             print(f"CSV文件记录数: {len(self.data)}")
#             print(f"预期样本数: {expected_size}")
#
#             # 找出大小不匹配的tensor
#             if ordered_image.size(0) != expected_size:
#                 print(f"  ✗ ordered_image 大小不匹配: {ordered_image.size(0)} vs {expected_size}")
#             if clip_image.size(0) != expected_size:
#                 print(f"  ✗ clip_image 大小不匹配: {clip_image.size(0)} vs {expected_size}")
#
#             raise ValueError(f"数据大小不匹配。请检查pickle文件是否与CSV文件对应。")
#
#         datasets =TensorDataset(token_ids,
#                                 masks,
#                                 label,
#                                 category,
#                                 ordered_image,
#                                 clip_image,
#                                 clip_text
#         )
#         # Windows或CPU运行时，pin_memory应该为False
#         # torch已在文件顶部导入，不需要重复导入
#         use_pin_memory = torch.cuda.is_available() and self.num_workers > 0
#
#         dataloader = DataLoader(
#             dataset = datasets,
#             batch_size = self.batch_size,
#             num_workers = self.num_workers,
#             pin_memory = use_pin_memory,
#             shuffle = shuffle,
#             worker_init_fn = _init_fn if self.num_workers > 0 else None
#         )
#         return dataloader
