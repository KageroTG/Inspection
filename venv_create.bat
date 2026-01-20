@echo off
title Python 3.11 Full Project Setup

:: 1. Check for Python 3.11
echo Checking for Python 3.11...
py -3.11 --version >nul 2>&1

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Python 3.11 was not found. 
    echo Please ensure it is installed and the "py" launcher is available.
    pause
    exit /b
)

:: 2. Create the Virtual Environment
echo.
echo [1/3] Creating virtual environment: .venv...
py -3.11 -m venv .venv

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Failed to create virtual environment.
    pause
    exit /b
)

:: 3. Upgrade Pip
echo [2/3] Upgrading pip...
.venv\Scripts\python.exe -m pip install --upgrade pip

:: 4. Check and Install Requirements
echo.
if exist requirements.txt (
    echo [3/3] requirements.txt found! Installing dependencies...
    .venv\Scripts\python.exe -m pip install -r requirements.txt
) else (
    echo [3/3] No requirements.txt found. Skipping dependency install.
)

echo.
echo --------------------------------------------------
echo SETUP COMPLETE
echo.
echo To activate your environment:
echo   .venv\Scripts\activate
echo --------------------------------------------------
pause