# F:\AI — 深度学习学习工作区

创建于 2026-09-13。所有 AI 相关的环境、数据、代码、论文统一放这里，不再散落到 C/D 盘。

---

## 目录结构

```
F:\AI\
├── README.md              本文件：目录说明 + 学习路径 + 进度追踪
├── env.bat                一键激活虚拟环境（Windows 双击/命令行运行）
├── envs\
│   └── dl\                主虚拟环境（Python 3.11.2 + PyTorch CUDA）
├── cache\                 所有包/模型缓存，防止撑爆 C 盘
│   ├── pip\               pip 下载缓存
│   ├── hf\                HuggingFace 模型缓存（Whisper / wav2vec 等）
│   └── torch\             torch.hub 缓存
├── datasets\              数据集
│   ├── audio\             语音/音频（KWS、VoiceBank-DEMAND、LibriSpeech…）
│   ├── vision\            图像（MNIST、CIFAR、MSTAR…）
│   └── timeseries\        时序/信号（RadioML…）
├── projects\              代码，按学习关卡编号
│   ├── 00-micrograd\              关卡1 手写自动微分
│   ├── 01-cnn-from-scratch\       关卡2 手写卷积
│   ├── 02-attention-from-scratch\ 关卡3 手写 attention
│   ├── 03-gpt-from-scratch\       关卡3 GPT 从零
│   ├── 04-conformer\              关卡4 Conformer
│   └── 99-sandbox\                随手试验用
├── checkpoints\           模型权重（.pt / .pth / .safetensors）
├── papers\                论文 PDF（配合 Zotero 用）
└── notes\                 笔记、推导、实验结果记录
```

---

## 环境信息

| 项 | 值 |
|---|---|
| 虚拟环境 | `F:\AI\envs\dl` |
| Python | 3.11.2（来自 `C:\Python311`） |
| 解释器路径 | `F:\AI\envs\dl\Scripts\python.exe` |
| GPU | NVIDIA GeForce RTX 3060 Laptop, 6GB |
| CUDA wheel | cu124 |
| IDE | PyCharm Community 2022.3.1（`F:\PyCharm Community Edition 2022.3.1`） |
| 系统 Python | `C:\Python311`（裸装，**不要往这里 pip install**） |

### 缓存重定向（已通过 setx 永久设置）

```
PIP_CACHE_DIR = F:\AI\cache\pip
HF_HOME       = F:\AI\cache\hf
TORCH_HOME    = F:\AI\cache\torch
```

**Why:** C 盘只有约 58GB 可用，pip 缓存、HuggingFace 模型（单个动辄 1–3GB）默认全塞 C 盘。
新开的终端会自动带上这些变量；老终端需重开或手动 export。

### PyCharm 配置解释器

`File → Settings → Project → Python Interpreter → Add Interpreter → Existing`
选：`F:\AI\envs\dl\Scripts\python.exe`

**不要**选 `C:\Python311` 或 Python 3.8，那两个是裸环境/旧环境。

---

## 6GB 显存的能力边界

- ✅ 能训：KWS、语音增强（DCCRN / Conv-TasNet 小配置）、whisper-tiny/base 微调、
  char-level GPT、CIFAR 级 CNN
- ❌ 不能训：7B 级全参微调、大 batch ViT
- 必备技巧：AMP 混合精度 + 梯度累积
- 兜底算力：Kaggle（每周 30h 免费 GPU）、Colab T4、学校服务器

---

## 四关卡进度

| # | 内容 | 状态 | 完成日期 | 关键产物 |
|---|---|---|---|---|
| 1 | MLP / 手写自动微分（micrograd） | ☐ 未开始 | | `micrograd.py` + XOR 训练脚本 |
| 2 | CNN / 手写卷积（conv2d 前向+反向） | ☐ 未开始 | | `conv_numpy.py` 通过 gradcheck |
| 3 | Transformer / 手写 attention + GPT | ☐ 未开始 | | `attention.py` + 能训的 `gpt.py` |
| 4 | Conformer（卷积+attention 融合） | ☐ 未开始 | | Conformer block 实现 + 结构图 |
| 5 | 语音实战（KWS 或语音增强） | ☐ 未开始 | | 跑通 baseline + 复现指标 |

### 每关的检验标准（做不到就不算过）

- **关卡1**：白板写出 ∂L/∂W 的完整链式推导；解释计算图、`requires_grad`、backward 为何要拓扑排序
- **关卡2**：解释「两个 3×3 堆叠」为何优于「一个 5×5」；算任意层感受野；白板推卷积反向传播
- **关卡3**：白板写 `Attention(Q,K,V)` 并解释每一项；解释为何除 √d_k；解释 causal mask 的实现
- **关卡4**：画出 Conformer block 完整结构图，说清每个模块去掉后的影响

---

## 常用命令

```bash
# 激活环境（Git Bash）
source /f/AI/envs/dl/Scripts/activate

# 或 Windows cmd / PowerShell
F:\AI\env.bat

# 检查 GPU 是否可用
python -c "import torch;print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 看当前 venv 装了哪些包
python -m pip list

# 导出环境快照（备份用，放 notes\ 下）
python -m pip freeze > /f/AI/notes/requirements-$(date +%Y%m%d).txt
```

---

## 待办 / 注意

- [ ] PyCharm 2022.3.1 是 2022 年的版本，能用但偏旧；官方 Community 版免费，后续可考虑升级
- [ ] 大数据集（LibriSpeech 完整版 ~60GB、WenetSpeech ~100GB+）下载前先确认 F 盘余量
- [ ] F 盘根目录还有 `pythonProject1`~`pythonProject5` 等 PyCharm 遗留项目，确认无用后可归档
- [ ] 旧环境 `C:\Users\王灿\AppData\Local\Programs\Python\Python38` 是本科遗留（配 `F:\python库` 里的 cp38 whl），如无引用可卸载回收 C 盘空间
