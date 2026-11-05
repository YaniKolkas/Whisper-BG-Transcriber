@echo off
chcp 65001 >nul
title Инсталиране на Whisper BG Transcriber

echo ======================================================
echo    Инсталиране на Whisper BG Transcriber за Windows
echo ======================================================
echo.

:: Проверка дали Python е инсталиран
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [ГРЕШКА] Python не е намерен на вашия компютър!
    echo.
    echo Моля, изтеглете и инсталирайте безплатно Python от:
    echo https://www.python.org/downloads/
    echo.
    echo *************************************************************
    echo ВАЖНО: По време на инсталацията на първия екран сложете
    echo отметка на кутийката:
    echo.
    echo    [X] "Add python.exe to PATH"   (Добави Python към PATH)
    echo.
    echo След това стартирайте този файл отново.
    echo *************************************************************
    echo.
    pause
    exit /b 1
)

echo [1/2] Намерен Python:
python --version
echo.
echo [2/2] Инсталиране на необходимите библиотеки...
echo Това може да отнеме 1-2 минути в зависимост от интернет връзката...
echo.

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

if %errorlevel% equ 0 (
    echo.
    echo ======================================================
    echo   ГОТОВО! Всичко е инсталирано успешно!
    echo ======================================================
    echo.
    echo Сега можете да стартирате приложението чрез:
    echo     run_app.bat
    echo.
) else (
    echo.
    echo [ГРЕШКА] Възникна проблем при инсталиране на библиотеките.
    echo Моля, проверете вашата интернет връзка и опитайте отново.
    echo.
)

pause
