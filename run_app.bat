@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
title Whisper BG Transcriber

echo ======================================================
echo        Whisper BG Transcriber (OpenAI Whisper)
echo ======================================================
echo.
echo Стартиране на локалното приложение...
echo Адрес: http://127.0.0.1:8000
echo.

:: Проверка дали Python е наличен
where python >nul 2>&1
if %errorlevel% neq 0 (
    if not exist "C:\Users\yanik\miniconda3\python.exe" (
        echo [ГРЕШКА] Python не е намерен на вашия компютър!
        echo Моля, първо кликнете два пъти върху: install.bat
        echo.
        pause
        exit /b 1
    )
)

:: Изчакваме 2 секунди и отваряме браузъра автоматично
start "" cmd /c "timeout /t 2 /nobreak >nul & start http://127.0.0.1:8000"

:: Стартираме FastAPI сървъра с автоматично презареждане при промени
if exist "C:\Users\yanik\miniconda3\python.exe" (
    "C:\Users\yanik\miniconda3\python.exe" -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
) else (
    python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
)

pause
