@echo off
title Agentic RAG System
echo ============================================
echo   Agentic RAG -- Starting all services
echo ============================================

:: Check Docker
docker ps >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARN] Docker not running, starting Docker Desktop...
    start "" "C:\Program Files\Docker\Docker\Docker Desktop.exe"
    echo Waiting for Docker to start...
    :wait_docker
    ping -n 5 127.0.0.1 >nul
    docker ps >nul 2>&1
    if %errorlevel% neq 0 goto wait_docker
)

:: Start Qdrant
docker ps | findstr rag_qdrant >nul 2>&1
if %errorlevel% neq 0 (
    echo [*] Starting Qdrant...
    cd /d D:\rag-project
    docker-compose up -d qdrant
    ping -n 3 127.0.0.1 >nul
)

:: Start API
curl -s http://localhost:8000/health >nul 2>&1
if %errorlevel% neq 0 (
    echo [*] Starting FastAPI...
    start "RAG-API" "D:\Qwen 2.5 7B\env\python.exe" D:\rag-project\scripts\app.py --mode api
    ping -n 3 127.0.0.1 >nul
)

:: Start Streamlit
echo [*] Starting Streamlit UI...
start "RAG-UI" "D:\Qwen 2.5 7B\env\python.exe" D:\rag-project\scripts\app.py --mode ui

:: Open browser
echo [*] Opening browser...
start "" "http://localhost:8501"
echo Done!
echo.
echo   Streamlit: http://localhost:8501
echo   API Docs: http://localhost:8000/docs
echo.
