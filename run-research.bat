@echo off
chcp 65001 >nul
cd /d "%~dp0"
".venv\Scripts\python.exe" -X utf8 run.py --output results\replay-v1
if errorlevel 1 (
  echo 回放失败，请查看上方错误说明。
  pause
  exit /b 1
)
echo 已生成 results\replay-v1\report.html，可双击该文件查看。
pause
