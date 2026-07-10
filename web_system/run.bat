@echo off
REM QMMFND Web System Startup Script for Windows

setlocal enabledelayedexpansion
chcp 65001 >nul 2>&1

echo.
echo ========================================
echo   QMMFND Web System Startup
echo ========================================
echo.

REM Check Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python not found. Please install Python 3.7+
    echo Download: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo Python installation confirmed

REM Create virtual environment
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    echo Virtual environment created
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

if errorlevel 1 (
    echo Warning: Virtual environment activation may have failed, continuing...
)

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip setuptools wheel -q

REM Install dependencies
echo Installing dependencies...
pip install -r backend\requirements.txt

if errorlevel 1 (
    echo Warning: Dependency installation may have failed
    echo Attempting individual package installation...
    pip install Flask==2.3.3 --no-cache-dir
    pip install Flask-CORS==4.0.0 --no-cache-dir
    pip install Flask-SQLAlchemy==3.0.5 --no-cache-dir
    pip install Flask-JWT-Extended==4.5.2 --no-cache-dir
    pip install Werkzeug==2.3.7 --no-cache-dir
    pip install PyJWT==2.12.1 --no-cache-dir
    pip install SQLAlchemy==1.3.24 --no-cache-dir
)

echo Dependencies installed

REM Initialize database
echo Initializing database...
cd backend
python init_db.py

if errorlevel 1 (
    echo Warning: Database initialization completed with warnings
)

cd ..

REM Start Flask application
echo.
echo ========================================
echo   Starting Flask Backend Service
echo ========================================
echo.
echo API Address: http://localhost:5000
echo.
echo Demo Accounts:
echo   - admin / 123456 (Administrator)
echo   - operator / 123456 (Operator)
echo   - analyst / 123456 (Analyst)
echo.
echo Frontend: Open frontend/index.html in browser
echo.
echo Note: Keep this window open
echo.

cd backend
python app.py
echo      前端应用需要此服务运行
echo.

cd backend
python app.py

pause
