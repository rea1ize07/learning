# D:\learning — 深度学习学习工作区

创建于 2026-09-13。所有 AI 相关的环境、数据、代码、论文统一放这里。

> **2026-09-20 本机迁移记录**：仓库从笔记本迁到本机，工作区路径 `F:\AI` → `D:\learning`；
> 本机**无 NVIDIA 显卡**（AMD Radeon 核显），PyTorch 改用 **CPU 版**。详见下方「环境信息」。

---

## 目录结构

```
D:\learning\
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
├── papers\                论文 PDF（9 份，约 51MB，已被 .gitignore 排除，不入库）
│   ├── The Last AI Built by Humans...pdf   RSI 综述（2026-09，起点文献）
│   ├── 2306.04469 - Model-Based Deep Learning.pdf   ⭐ 信号处理×深度学习，主场
│   ├── 2303.18223 / 2012.12556 / 2009.06732         地基线（LLM / ViT / 高效 Transformer）
│   ├── 2404.14387 / 2507.21046 / 2508.07407         自演化线（LLM / Agent ×2）
│   └── 大语言模型（赵鑫等，人大，高教社2024）.pdf      中文教材，391 页
└── notes\                 笔记、推导、实验结果记录
    ├── 学习路径-视频版.md           四站视频路径（神经网络→CNN→Transformer→LLM）
    ├── 大语言模型-配套视频清单.md    B 站 23 集 8.8h 官方课，含 BV 号与时长
    └── surveys\                     9 篇综述的中文笔记 + README 索引
        ├── README.md                阅读顺序与阅读提醒（先看这个）
        ├── 00-大语言模型-赵鑫-学习导览.md
        ├── 01-llm-survey … 04-model-based-deep-learning
        └── 05-self-evolution … 08-rsi-last-ai-built-by-humans
```

---

## 环境信息

| 项 | 值 |
|---|---|
| 虚拟环境 | `D:\learning\envs\dl` |
| Python | 3.14.7（来自 `C:\Python314`） |
| 解释器路径 | `D:\learning\envs\dl\Scripts\python.exe` |
| GPU | **无 NVIDIA 显卡**（AMD Radeon 核显），仅 CPU |
| PyTorch | `2.14.0+cpu`（CPU 版） |
| IDE | 未安装 |
| 系统 Python | `C:\Python314`（裸装，**不要往这里 pip install**，其 site-packages 不可写） |

> **与笔记本的差异**：笔记本是 RTX 3060 Laptop 6GB + CUDA（cu124）+ Python 3.11.2。
> 本机是 CPU-only，`torch.cuda.is_available()` 恒为 `False`。

### 缓存重定向

```
PIP_CACHE_DIR = D:\learning\cache\pip
HF_HOME       = D:\learning\cache\hf
TORCH_HOME    = D:\learning\cache\torch
```

**Why:** pip 缓存、HuggingFace 模型（单个动辄 1–3GB）默认全塞 C 盘。

**本机现状**：这些变量由 `env.bat` 在激活时设置，**尚未**用 `setx` 永久写入注册表。
如果需要新开的终端自动带上，执行一次：

```cmd
setx PIP_CACHE_DIR D:\learning\cache\pip
setx HF_HOME       D:\learning\cache\hf
setx TORCH_HOME    D:\learning\cache\torch
```

### 编辑器配置解释器

本机未装 PyCharm。若之后装 VS Code / PyCharm，解释器一律指向：

```
D:\learning\envs\dl\Scripts\python.exe
```

**不要**选 `C:\Python314`（裸环境，且其 site-packages 不可写）。

---

## 算力边界（本机 CPU-only）

本机无 NVIDIA 显卡（AMD Radeon 核显），PyTorch 跑 CPU，`torch.cuda.is_available()` 恒为 `False`。

- ✅ **CPU 完全够用**：`micrograd`、手写 conv / attention、小张量数值验证、小模型短跑
- ⚠️ **CPU 很慢（能用，但按小时计）**：CIFAR 级 CNN、char-level GPT
- ❌ **本机做不了**：任何 CUDA 依赖的实验、whisper 微调、7B 级全参微调、大 batch ViT
- **重活一律扔云端**：Kaggle（每周 30h 免费 GPU）、Colab T4、学校服务器

> **笔记本（RTX 3060 Laptop 6GB）另有一套边界**：能训 KWS、语音增强（DCCRN / Conv-TasNet 小配置）、
> whisper-tiny/base 微调、char-level GPT、CIFAR 级 CNN；不能训 7B 全参与大 batch ViT；
> 必备技巧是 AMP 混合精度 + 梯度累积。**这些在 CPU 上都不适用**，CPU 上瓶颈是纯算力。
>
> 迁移思路：**本机学习、写代码、调试 → 云端 GPU 跑重训练**。

---

## 学习路径（两条线，别混）

**A. 视频优先线（先建直觉，当前主线）** —— 完整文档 `notes\学习路径-视频版.md`

| 站 | 主题 | 时长 | 资源 |
|---|---|---|---|
| 0 | 神经网络在干什么 | 1.5h | 3b1b 深度学习 Ch1–4 |
| 1 | CNN | 4h | 3b1b 卷积 + 李宏毅 + 李沐 D2L |
| 2 | Transformer | 4h | 3b1b Ch5–6 + 李宏毅 attention |
| 3 | Conformer / 语音实战 | — | 汇合点，对应 `projects\04-conformer` |
| 4 | **大语言模型全流程**（预训练→微调→对齐→解码→RAG→智能体） | 8.8h | 人大赵鑫 × Datawhale 官方视频 |

第 4 站细节：B 站 23 集合集 <https://space.bilibili.com/431850986/lists/4940683>，
章节与教材一一对应，剧集表见 `notes\大语言模型-配套视频清单.md`。
⚠️ 另有同名**付费**课（49 元），内容一样，别买；⚠️ 第 1 集是发布会（62min），可跳。

**B. 从零手写线（更陡，等有基础再回头）** —— 即下面的四关卡。

**C. 文献线** —— 9 篇综述的中文笔记在 `notes\surveys\`，先读 `notes\surveys\README.md` 看顺序。
信通背景的入口是 **04 Model-Based Deep Learning**（deep unfolding：ISTA→LISTA、PGD→DetNet、
卡尔曼→KalmanNet），单卡可做实验，适合研一先出结果。

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
source /d/learning/envs/dl/Scripts/activate

# 或 Windows cmd / PowerShell
D:\learning\env.bat

# 检查 torch（本机 CPU-only，cuda 恒为 False）
python -c "import torch;print(torch.__version__, torch.cuda.is_available())"

# 看当前 venv 装了哪些包
python -m pip list

# 导出环境快照（备份用，放 notes\ 下）
python -m pip freeze > /d/learning/notes/requirements-$(date +%Y%m%d).txt
```

---

## 待办 / 注意

- [ ] 本机未装 IDE；如需，装 VS Code 或 PyCharm，解释器指向 `D:\learning\envs\dl\Scripts\python.exe`
- [ ] 缓存变量目前只在 `env.bat` 里设置，未 `setx` 永久写入；需要永久生效可执行上面那三条 `setx`
- [ ] 大数据集（LibriSpeech 完整版 ~60GB、WenetSpeech ~100GB+）下载前先确认磁盘余量
- [ ] 本机 Python 3.14 较新，部分老库可能还没有 cp314 wheel，装包遇到问题留意这一点
