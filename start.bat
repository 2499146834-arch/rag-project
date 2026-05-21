@echo off
title Agentic RAG System
echo ============================================
echo   Agentic RAG -- Starting all services
echo ============================================

set PYTHON=D:\Qwen 2.5 7B\env\python.exe
set PROJECT=D:\rag-project

:: Check Docker
docker ps >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARN] Docker not running, starting Docker Desktop...
    start "" "C:\Program Files\Docker\Docker\Docker Desktop.exe"
    :wait_docker
    ping -n 5 127.0.0.1 >nul
    docker ps >nul 2>&1
    if %errorlevel% neq 0 goto wait_docker
)

:: Start Qdrant
docker ps | findstr rag_qdrant >nul 2>&1
if %errorlevel% neq 0 (
    echo [*] Starting Qdrant...
    cd /d %PROJECT%
    docker-compose up -d qdrant
    ping -n 4 127.0.0.1 >nul
)

:: Kill any old processes on port 8501
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8501.*LISTENING') do (
    echo [*] Killing old process on port 8501 (PID %%a)
    taskkill /F /PID %%a >nul 2>&1
    ping -n 2 127.0.0.1 >nul
)

:: Start FastAPI
echo [*] Starting FastAPI on port 8000...
start "RAG-API" /min cmd /c "%PYTHON% %PROJECT%\scripts\app.py --mode api"

:: Start Streamlit
echo [*] Starting Streamlit on port 8501...
start "RAG-UI" /min cmd /c "%PYTHON% -m streamlit run %PROJECT%\scripts\streamlit_app.py --server.port 8501 --server.headless true"

:: Wait for Streamlit to load models (BGE embedding takes ~20s)
echo [*] Waiting for Streamlit to load models (~20s)...
ping -n 25 127.0.0.1 >nul

:: Open browser
echo [*] Opening browser...
start "" "http://localhost:8501"
echo.
echo ============================================
echo   Streamlit: http://localhost:8501
echo   API Docs:  http://localhost:8000/docs
echo ============================================
echo.
