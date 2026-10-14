@echo off
REM ============================================
REM  D:\learning 深度学习环境 —— 一键激活
REM  用法：命令行执行 D:\learning\env.bat
REM       或在编辑器里把解释器指向下面的 PY
REM ============================================

set "PY=D:\learning\envs\dl\Scripts\python.exe"

REM 缓存全部指向工作区所在盘，避免撑爆 C 盘
set "PIP_CACHE_DIR=D:\learning\cache\pip"
set "HF_HOME=D:\learning\cache\hf"
set "TORCH_HOME=D:\learning\cache\torch"

echo.
echo  [D:\learning] 环境已激活
echo    Python : %PY%
echo    PIP_CACHE_DIR = %PIP_CACHE_DIR%
echo    HF_HOME       = %HF_HOME%
echo    TORCH_HOME    = %TORCH_HOME%
echo.
echo  注意：本机无 NVIDIA 显卡，PyTorch 为 CPU 版
echo.
echo  进入子 shell（exit 退出）...
echo.

cmd /k
