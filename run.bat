@echo off
echo ============================================
echo  BreastAI - Breast Cancer Detection SaaS
echo ============================================

REM Start backend in a new window
echo [1/2] Starting Backend API (FastAPI)...
start "BreastAI Backend" cmd /k "cd /d %~dp0 && python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000"

REM Wait 3 seconds for backend to start
timeout /t 3 /nobreak > nul

REM Start frontend in a new window
echo [2/2] Starting Frontend (React)...
start "BreastAI Frontend" cmd /k "cd /d %~dp0\frontend && npm start"

echo.
echo ============================================
echo Backend:  http://localhost:8000
echo Frontend: http://localhost:3000
echo API Docs: http://localhost:8000/docs
echo ============================================
echo Both servers starting in new windows...
pause
