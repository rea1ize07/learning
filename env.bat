@echo off
REM ============================================
REM  F:\AI 深度学习环境 —— 一键激活
REM  用法：命令行执行 F:\AI\env.bat
REM       或在 PyCharm 里把解释器指向下面的 PY
REM ============================================

set "PY=F:\AI\envs\dl\Scripts\python.exe"

REM 缓存全部指向 F 盘，避免撑爆 C 盘
set "PIP_CACHE_DIR=F:\AI\cache\pip"
set "HF_HOME=F:\AI\cache\hf"
set "TORCH_HOME=F:\AI\cache\torch"

echo.
echo  [F:\AI] 环境已激活
echo    Python : %PY%
echo    PIP_CACHE_DIR = %PIP_CACHE_DIR%
echo    HF_HOME       = %HF_HOME%
echo    TORCH_HOME    = %TORCH_HOME%
echo.
echo  进入子 shell（exit 退出）...
echo.

cmd /k
