# -*- coding: utf-8 -*-
"""
QMMFND Web System 配置文件
Configuration for QMMFND Web System
"""

import os
from datetime import timedelta

# ========== Flask配置 ==========
class Config:
    """基础配置"""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    DEBUG = False
    TESTING = False
    
    # ========== 数据库配置 ==========
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ECHO = False
    
    # SQLite (默认)
    SQLALCHEMY_DATABASE_URI = 'sqlite:///qmmfnd.db'
    
    # MySQL (可选)
    # SQLALCHEMY_DATABASE_URI = 'mysql+pymysql://user:password@localhost:3306/qmmfnd'
    
    # PostgreSQL (可选)
    # SQLALCHEMY_DATABASE_URI = 'postgresql://user:password@localhost:5432/qmmfnd'
    
    # ========== JWT配置 ==========
    JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY') or 'jwt-secret-key-change-in-production'
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(days=30)
    
    # ========== CORS配置 ==========
    CORS_ORIGINS = ['*']  # 生产环境应该限制具体的域名
    
    # ========== 日志配置 ==========
    LOG_LEVEL = 'INFO'
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'


class DevelopmentConfig(Config):
    """开发配置"""
    DEBUG = True
    TESTING = False
    SQLALCHEMY_ECHO = True
    LOG_LEVEL = 'DEBUG'


class TestingConfig(Config):
    """测试配置"""
    DEBUG = True
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'  # 使用内存数据库
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=1)


class ProductionConfig(Config):
    """生产配置"""
    DEBUG = False
    TESTING = False
    
    # 生产环境必须使用强密钥
    SECRET_KEY = os.environ.get('SECRET_KEY')
    if not SECRET_KEY:
        raise ValueError("生产环境必须设置SECRET_KEY环境变量")
    
    JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY')
    if not JWT_SECRET_KEY:
        raise ValueError("生产环境必须设置JWT_SECRET_KEY环境变量")
    
    # 限制CORS来源
    CORS_ORIGINS = [
        os.environ.get('FRONTEND_URL', 'https://example.com')
    ]


# ========== QMMFND模型配置 ==========
class ModelConfig:
    """模型配置"""
    MODEL_NAME = 'QMMFND-v1.0'
    
    # 文本模型
    BERT_MODEL = './pretrained_model/chinese_roberta_wwm_base_ext_pytorch'
    BERT_MAX_LENGTH = 197
    BERT_EMB_DIM = 768
    
    # 图像模型
    MAE_MODEL = './mae_pretrain_vit_base.pth'
    MAE_OUTPUT_DIM = 768
    
    # CLIP模型
    CLIP_MODEL = './clip_cn_vit-b-16.pt'
    CLIP_OUTPUT_DIM = 512
    
    # 量子编码
    QUANTUM_OUTPUT_DIM = 320
    
    # PLE配置
    NUM_EXPERT = 6
    DOMAIN_NUM = 10
    NUM_SHARE = 1
    
    # 检测阈值
    CONFIDENCE_THRESHOLD = 0.5
    
    # GPU配置
    USE_CUDA = True
    DEVICE = 'cuda:0'


# ========== 系统配置 ==========
class SystemConfig:
    """系统配置"""
    # API配置
    API_TIMEOUT = 30  # 秒
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    
    # 分页配置
    DEFAULT_PAGE_SIZE = 20
    MAX_PAGE_SIZE = 100
    
    # 缓存配置
    CACHE_TYPE = 'simple'
    CACHE_DEFAULT_TIMEOUT = 300
    
    # 数据保留天数
    DATA_RETENTION_DAYS = 365
    
    # 并发配置
    MAX_WORKERS = 4
    TIMEOUT_SECONDS = 60


# ========== 环境选择 ==========
def get_config(env=None):
    """根据环境获取配置"""
    if env is None:
        env = os.environ.get('FLASK_ENV', 'development')
    
    config_map = {
        'development': DevelopmentConfig,
        'testing': TestingConfig,
        'production': ProductionConfig
    }
    
    return config_map.get(env, DevelopmentConfig)


# ========== 默认导出 ==========
current_config = get_config()
model_config = ModelConfig()
system_config = SystemConfig()
