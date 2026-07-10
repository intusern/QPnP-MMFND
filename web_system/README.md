# QMMFND 虚假新闻检测系统 - Web可视化前端

## 📋 项目概述

这是QMMFND（量子启发多模态多域虚假新闻检测）系统的完整Web解决方案，包括：
- ✨ 专业的Web界面（Vue.js + Element UI）
- 🔒 完整的用户认证系统（JWT安全认证）
- 📊 实时数据看板和可视化分析
- 🔍 一键智能新闻检测功能
- ⚛️ 量子模型流程可视化
- 👥 多角色权限管理（管理员/操作员/分析师）

## 🎯 核心功能

### 1️⃣ 系统登录界面
- ✅ 账号密码验证
- ✅ JWT安全认证（30天有效期）
- ✅ 多角色权限登录（管理员 / 操作员 / 分析师）
- ✅ 简洁专业的登录交互设计
- ✅ 内置演示账户

### 2️⃣ 系统主界面・数据看板
- 📈 核心数据卡片（总检测数 / 真实新闻 / 虚假新闻 / 虚假率 / 平均置信度）
- 📊 检测趋势分析图表（基于ECharts）
- 🎯 快捷操作入口
- ⏱️ 实时数据监控（30秒自动刷新）

### 3️⃣ 模型检测界面
- 📝 新闻文本输入
- 🔗 URL链接粘贴
- 🚀 一键智能检测功能
- 📈 可信度评分（百分比显示）
- 🎨 检测结果直观展示
- 🔄 特征可视化（雷达图）
- 📋 检测历史记录

### 4️⃣ 量子多模态模型可视化
- 📊 完整的处理流程可视化
  - 文本输入 → BERT编码
  - 图像编码 → CLIP融合  
  - 量子编码 → 纠缠对齐
  - PLE融合 → 分类器 → 结果输出
- ⚛️ 量子核心模块说明
  - ComplexLinear（复值线性层）
  - ComplexReLU（复值激活）
  - QuantumEntanglement（量子纠缠）
- 📈 系统架构图（基于ECharts）

## 🛠️ 技术栈

### 前端
- **Vue.js 2.7** - 前端框架
- **Element UI** - UI组件库
- **ECharts 5** - 数据可视化
- **Axios** - HTTP客户端
- **原生HTML5/CSS3** - 页面结构和样式

### 后端
- **Flask 2.3** - Web框架
- **Flask-SQLAlchemy** - ORM
- **Flask-JWT-Extended** - JWT认证
- **SQLite** - 数据库（默认）
- **Python 3.7+** - 编程语言

## 📦 文件结构

```
web_system/
├── backend/                    # 后端应用
│   ├── app.py                 # Flask主应用
│   ├── init_db.py             # 数据库初始化
│   └── requirements.txt        # Python依赖
├── frontend/                   # 前端应用
│   ├── index.html             # 主HTML文件
│   ├── js/
│   │   ├── api.js            # API调用模块
│   │   ├── components.js     # Vue组件定义
│   │   └── app.js            # 主应用逻辑
│   └── css/
│       ├── style.css         # 主样式
│       └── responsive.css    # 响应式设计
├── run.bat                    # Windows启动脚本
├── run.sh                     # Linux/Mac启动脚本
└── README.md                  # 本文件
```

## 🚀 快速开始

### 方式一：Windows用户

1. **双击运行启动脚本**
   ```
   双击 run.bat 文件
   ```
   脚本会自动完成：
   - ✅ 检查Python环境
   - ✅ 创建虚拟环境
   - ✅ 安装依赖包
   - ✅ 初始化数据库
   - ✅ 启动Flask服务

2. **打开前端应用**
   ```
   在浏览器中打开: file://web_system/frontend/index.html
   或直接访问: http://localhost:5000
   ```

3. **使用演示账户登录**
   ```
   账户1: admin / 123456 (管理员)
   账户2: operator / 123456 (操作员) 
   账户3: analyst / 123456 (分析师)
   ```

### 方式二：Linux/Mac用户

1. **给脚本添加执行权限**
   ```bash
   chmod +x run.sh
   ```

2. **运行启动脚本**
   ```bash
   ./run.sh
   ```

### 方式三：手动启动

1. **安装Python依赖**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate
   pip install -r backend/requirements.txt
   
   # Linux/Mac
   python3 -m venv venv
   source venv/bin/activate
   pip install -r backend/requirements.txt
   ```

2. **初始化数据库**
   ```bash
   python backend/init_db.py
   ```

3. **启动后端服务**
   ```bash
   cd backend
   python app.py
   ```

4. **打开前端应用**
   - 在浏览器打开: `frontend/index.html`
   - 或访问API文档: `http://localhost:5000/api/health`

## 🔑 用户角色说明

| 角色 | 权限 | 用途 |
|------|------|------|
| 管理员 (admin) | 完全权限 | 系统管理和配置 |
| 操作员 (operator) | 检测和查看 | 执行新闻检测 |
| 分析师 (analyst) | 只读 | 分析检测结果 |

## 📊 API 接口说明

### 认证相关
```
POST /api/auth/register          - 用户注册
POST /api/auth/login            - 用户登录
GET  /api/auth/me              - 获取当前用户信息
```

### 数据看板
```
GET  /api/dashboard/stats      - 获取统计数据
GET  /api/dashboard/trend      - 获取检测趋势
```

### 新闻检测
```
POST /api/detection/analyze     - 分析新闻
GET  /api/detection/history    - 获取检测历史
```

### 可视化
```
GET  /api/visualization/model-flow        - 获取模型流程
GET  /api/visualization/quantum-components - 获取量子组件
```

### 系统
```
GET  /api/logs                 - 获取操作日志
GET  /api/health               - 健康检查
```

## 🔒 安全特性

- ✅ JWT令牌认证（30天有效期）
- ✅ 密码加密存储（werkzeug.security）
- ✅ CORS跨域保护
- ✅ 角色权限控制
- ✅ SQL注入防护（SQLAlchemy ORM）
- ✅ 操作日志记录

## 🎨 界面特性

- ✨ 现代化设计风格
- 📱 完全响应式适配（桌面/平板/手机）
- 🌙 深色模式支持
- ⚡ 流畅动画效果
- ♿ 无障碍访问支持

## 📈 数据可视化

### 检测趋势图
- 实时显示真实/虚假新闻趋势
- 支持自定义时间范围
- 交互式数据点提示

### 特征可视化
- 雷达图展示特征分布
- 包含文本/图像/语义等5个维度

### 系统架构图
- 网络图展示模型流程
- 支持拖拽和缩放

## ⚙️ 配置说明

### 后端配置 (backend/app.py)
```python
# 数据库
SQLALCHEMY_DATABASE_URI = 'sqlite:///qmmfnd.db'

# JWT认证
JWT_SECRET_KEY = 'your-secret-key-change-in-production'
JWT_ACCESS_TOKEN_EXPIRES = timedelta(days=30)

# Flask设置
DEBUG = True
HOST = '0.0.0.0'
PORT = 5000
```

### 前端配置 (frontend/js/api.js)
```javascript
const API_BASE_URL = 'http://localhost:5000/api';
```

## 🐛 常见问题

### Q1: 启动后提示"无法连接到后端"
**解决方案：**
1. 确保Flask服务在5000端口运行
2. 检查防火墙设置
3. 确保前端和后端在同一网络

### Q2: 登录失败
**解决方案：**
1. 确保数据库已初始化 (`python backend/init_db.py`)
2. 使用演示账户 (admin/123456)
3. 检查后端日志

### Q3: 图表不显示
**解决方案：**
1. 检查浏览器控制台错误
2. 确保ECharts库已正确加载
3. 尝试刷新页面

### Q4: 如何修改JWT密钥
**解决方案：**
编辑 `backend/app.py`：
```python
app.config['JWT_SECRET_KEY'] = 'your-new-secret-key'
```

### Q5: 如何使用MySQL替代SQLite
**解决方案：**
1. 安装 `pip install PyMySQL`
2. 修改 `app.py` 中的数据库URI：
```python
SQLALCHEMY_DATABASE_URI = 'mysql+pymysql://user:password@localhost/qmmfnd'
```

## 📝 演示数据

初始化脚本会自动创建：
- ✅ 3个演示用户（admin、operator、analyst）
- ✅ 最近30天的检测数据
- ✅ 系统操作日志

## 🔄 API调用流程

```
用户登录
  ↓
获取JWT令牌
  ↓
请求API端点（携带令牌）
  ↓
验证令牌
  ↓
执行业务逻辑
  ↓
返回数据
```

## 🎓 与QMMFND模型集成

目前系统返回模拟数据。要集成真实的QMMFND模型：

1. **在 `backend/app.py` 中导入模型**
   ```python
   from model.QMMFND import MultiDomainPLEFENDModel
   import torch
   ```

2. **在 `/api/detection/analyze` 路由中加载模型**
   ```python
   model = MultiDomainPLEFENDModel(...)
   model.load_state_dict(torch.load('model.pth'))
   
   # 执行推理
   prediction, confidence = model.predict(content)
   ```

3. **返回真实检测结果**

## 📞 技术支持

- 📧 问题反馈
- 🐛 Bug报告
- 💡 功能建议

## 📄 许可证

本项目采用学术使用许可。

## 🚀 部署到产环境

### 使用Gunicorn部署

```bash
# 安装
pip install gunicorn

# 启动
cd backend
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### 使用Nginx反向代理

```nginx
server {
    listen 80;
    server_name example.com;
    
    location /api {
        proxy_pass http://localhost:5000;
    }
    
    location / {
        root /path/to/frontend;
        try_files $uri $uri/ /index.html;
    }
}
```

## 🎯 后续计划

- [ ] 集成真实QMMFND模型
- [ ] 支持MySQL/PostgreSQL
- [ ] 添加数据导出功能
- [ ] 支持批量检测
- [ ] 添加用户管理后台
- [ ] 支持模型微调
- [ ] 添加API文档页面

---

**祝您使用愉快！** 🎉

如有问题，请查看项目文档或提交Issue。
