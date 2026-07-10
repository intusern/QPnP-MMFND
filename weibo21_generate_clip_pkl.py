#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Quantum-inspired Multimodal Multi-domain Fake News Detection
"""
为Weibo_21数据集生成CLIP特征PKL缓存文件
生成: train_clip_loader.pkl, val_clip_loader.pkl, test_clip_loader.pkl
"""

import pickle
import pandas as pd
import numpy as np
import torch
import os
from PIL import Image
import cn_clip.clip as clip
from cn_clip.clip import load_from_name

def read_image_clip():
    """使用CLIP模型读取并预处理Weibo_21数据集的图片"""
    image_list = {}
    file_list = ['Weibo_21/nonrumor_images/', 'Weibo_21/rumor_images/']
    
    # 加载本地CLIP模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    print("加载本地CLIP模型...")
    
    # 检查本地模型文件
    model_path = './clip_cn_vit-b-16.pt'
    if not os.path.exists(model_path):
        print(f"警告: 本地模型文件不存在 {model_path}")
        print("尝试在线下载...")
        model, preprocess = load_from_name("ViT-B-16", device=device, download_root='./')
    else:
        # 从本地加载模型
        import cn_clip.clip as clip
        model, preprocess = clip.load_from_name("ViT-B-16", device=device, download_root='./')
    print("✓ CLIP模型加载完成")
    
    print("\n开始读取图片...")
    for path in file_list:
        if not os.path.exists(path):
            print(f"警告: 目录不存在 {path}")
            continue
            
        files = os.listdir(path)
        print(f"处理 {path}: {len(files)} 个文件")
        
        for i, filename in enumerate(files):
            try:
                im = Image.open(path + filename)
                im = preprocess(im).unsqueeze(0).to(device)
                image_list[filename.split('/')[-1].split(".")[0].lower()] = im
                
                if (i + 1) % 500 == 0:
                    print(f"  已处理 {i + 1}/{len(files)} 个图片")
            except Exception as e:
                print(f"  错误: {filename} - {e}")
    
    print(f"✓ 图片读取完成，总计: {len(image_list)} 张")
    return image_list

def generate_clip_pkl_for_split(xlsx_path, output_pkl, image_dict):
    """为指定的数据集划分生成CLIP PKL文件"""
    print(f"\n处理: {xlsx_path}")
    
    # 读取XLSX
    data = pd.read_excel(xlsx_path)
    print(f"  数据条数: {len(data)}")
    
    ordered_image = []
    matched_count = 0
    missing_count = 0
    
    # 创建占位符图像（CLIP预处理后的形状：3x224x224）
    device = "cuda" if torch.cuda.is_available() else "cpu"
    placeholder_image = torch.zeros(3, 224, 224).to(device).unsqueeze(0)
    
    # 处理每条数据
    for i, row in data.iterrows():
        image_id = ""
        found_image = False
        
        # 尝试匹配图片（Weibo_21使用image列）
        if pd.notna(row['image']) and row['image']:
            for img_url in str(row['image']).split('|'):
                image_id = img_url.split("/")[-1].split(".")[0].lower()
                if image_id in image_dict:
                    ordered_image.append(image_dict[image_id])
                    matched_count += 1
                    found_image = True
                    break
        
        # 如果没有找到图片，添加占位符
        if not found_image:
            ordered_image.append(placeholder_image)
            missing_count += 1
        
        if (i + 1) % 500 == 0:
            print(f"  已处理 {i + 1}/{len(data)} 条数据")
    
    print(f"  匹配到图片: {matched_count}/{len(data)} 条")
    print(f"  使用占位符: {missing_count}/{len(data)} 条")
    
    # 转换为tensor
    if ordered_image:
        ordered_image = torch.cat(ordered_image, dim=0)
        print(f"  Tensor形状: {ordered_image.shape}")
        print(f"  预期形状: ({len(data)}, 3, 224, 224)")
        
        # 验证大小
        if ordered_image.size(0) != len(data):
            raise ValueError(f"图像数量({ordered_image.size(0)})与XLSX行数({len(data)})不匹配!")
    else:
        print("  警告: 没有匹配到任何图片!")
        ordered_image = torch.tensor([])
    
    # 保存PKL
    os.makedirs(os.path.dirname(output_pkl) if os.path.dirname(output_pkl) else '.', exist_ok=True)
    with open(output_pkl, 'wb') as file:
        pickle.dump(ordered_image, file)
    
    print(f"✓ 已保存: {output_pkl}")
    return matched_count

def main():
    print("="*60)
    print("Weibo_21数据集 - 生成CLIP特征PKL文件")
    print("="*60)
    
    # 读取所有图片（使用CLIP预处理）
    image_dict = read_image_clip()
    
    if not image_dict:
        print("\n✗ 错误: 没有读取到任何图片，请检查图片目录!")
        return
    
    # 生成三个CLIP PKL文件
    print("\n" + "="*60)
    print("开始生成CLIP PKL文件...")
    print("="*60)
    
    train_matched = generate_clip_pkl_for_split(
        'Weibo_21/train_datasets.xlsx',
        'Weibo_21/train_clip_loader.pkl',
        image_dict
    )
    
    val_matched = generate_clip_pkl_for_split(
        'Weibo_21/val_datasets.xlsx',
        'Weibo_21/val_clip_loader.pkl',
        image_dict
    )
    
    test_matched = generate_clip_pkl_for_split(
        'Weibo_21/test_datasets.xlsx',
        'Weibo_21/test_clip_loader.pkl',
        image_dict
    )
    
    # 总结
    print("\n" + "="*60)
    print("生成完成!")
    print("="*60)
    print(f"✓ Weibo_21/train_clip_loader.pkl - {train_matched} 条数据")
    print(f"✓ Weibo_21/val_clip_loader.pkl - {val_matched} 条数据")
    print(f"✓ Weibo_21/test_clip_loader.pkl - {test_matched} 条数据")
    print("="*60)

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()
