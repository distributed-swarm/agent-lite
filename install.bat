@echo off
REM Agent-Lite USB Installer
REM Run this as Administrator

echo ========================================
echo Agent-Lite Installer
echo ========================================
echo.

REM Check for admin rights
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo ERROR: This script must be run as Administrator!
    echo Right-click install.bat and select "Run as administrator"
    pause
    exit /b 1
)

echo [1/5] Checking Python installation...
python --version >nul 2>&1
if %errorLevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH!
    echo Please install Python 3.8+ from python.org
    pause
    exit /b 1
)
python --version
echo.

echo [2/5] Installing Python dependencies...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install pywin32
if %errorLevel% neq 0 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)
echo.

echo [3/5] Configuring controller URL...
set /p CONTROLLER_URL="Enter controller URL (e.g., http://192.168.1.100:8080): "
if "%CONTROLLER_URL%"=="" (
    set CONTROLLER_URL=http://controller:8080
)

REM Write config to a local .env file
echo CONTROLLER_URL=%CONTROLLER_URL% > agent.env
echo Configuration saved to agent.env
echo.

echo [4/5] Installing Windows Service...
python Service.py install
if %errorLevel% neq 0 (
    echo ERROR: Failed to install service
    pause
    exit /b 1
)
echo.

echo [5/5] Starting Agent-Lite service...
net start AgentLite
if %errorLevel% neq 0 (
    echo WARNING: Service failed to start automatically
    echo Check logs at: %USERPROFILE%\AppData\Local\AgentLite\service.log
)
echo.

echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo Service Status:
sc query AgentLite
echo.
echo View logs:
echo   type %USERPROFILE%\AppData\Local\AgentLite\service.log
echo.
echo Manage service:
echo   net stop AgentLite    - Stop the service
echo   net start AgentLite   - Start the service
echo   python Service.py remove - Uninstall
echo.
pause
