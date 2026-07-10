#!/bin/bash
# ========== QMMFND Web System 启动脚本 (Linux/Mac) ==========

echo ""
echo "========================================"
echo "  QMMFND Web System 启动"
echo "========================================"
echo ""

# 检查Python
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: 未找到 Python 3，请先安装 Python 3.7+")
    echo "下载地址: https://www.python.org/downloads/"
    exit 1
fi

echo "✅ Python 已安装: $(python3 --version)"

# 创建虚拟环境
if [ ! -d "venv" ]; then
    echo "📦 创建虚拟环境..."
    python3 -m venv venv
    echo "✅ 虚拟环境创建完成"
fi

# 激活虚拟环境
echo "🔄 激活虚拟环境..."
source venv/bin/activate

# 升级pip
echo "📦 升级pip..."
pip install --upgrade pip setuptools wheel -q

# 安装依赖
echo "📥 安装依赖包..."
if ! pip install -r backend/requirements.txt; then
    echo "❌ 依赖安装失败，请检查网络连接"
    exit 1
fi
echo "✅ 依赖安装完成"

# 检查Flask是否已安装
python -c "import flask; print('✅ Flask已安装')" || {
    echo "❌ Flask安装失败，重试..."
    pip install Flask==2.3.3 Flask-CORS==4.0.0 Flask-SQLAlchemy==3.0.5 Flask-JWT-Extended==4.5.2
}

# 初始化数据库
echo "🗄️  初始化数据库..."
python backend/init_db.py

# 启动Flask应用
echo ""
echo "========================================"
echo "  启动 Flask 后端服务..."
echo "========================================"
echo ""
echo "📍 API 地址: http://localhost:5000"
echo "📍 前端地址: file://$(pwd)/frontend/index.html"
echo ""
echo "注意: 请保持此窗口打开"
echo "     前端应用需要此服务运行"
echo ""

cd backend
python app.py
