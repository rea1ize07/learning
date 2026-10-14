"""
LeNet 训练脚本
==============
在 Fashion-MNIST 上训练 LeNet，并把模型保存到 models/lenet/。

用法（在仓库根目录 D:/learning 下执行）：
  D:/learning/envs/dl/Scripts/python.exe lenet/train.py

做四件事：
  1. 加载 Fashion-MNIST 数据
  2. 训练若干轮，每轮打印测试准确率
  3. 把模型保存到 models/lenet/
  4. 对比训练前后第一个卷积层的变化，直观看到"卷积核学会了什么"
"""

import sys
from pathlib import Path

import torch

# --- 让脚本能找到仓库根目录的 LeNet.py ---
# 运行 lenet/train.py 时，Python 默认只会在 lenet/ 这个目录里找模块，
# 所以要手动把上一级目录（仓库根目录）加进模块搜索路径。
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import LeNet

# 模型统一放在 models/lenet/，和代码目录分开
MODEL_DIR = ROOT / 'models' / 'lenet'


def show_kernels(w, title):
    """字符化显示卷积核：用符号表示每个位置的权重大小。"""
    print(title)
    for k in range(w.shape[0]):
        print(f'  卷积核 {k}:')
        for row in w[k, 0].tolist():
            line = '    '
            for v in row:
                if v > 0.08:
                    line += ' #'          # 明显正
                elif v < -0.08:
                    line += ' -'          # 明显负
                else:
                    line += ' .'          # 接近 0
            print(line)
    print()


def main():
    # ---------------- 训练配置 ----------------
    batch_size = 256     # 每个批次的样本数
    num_epochs = 10      # 训练轮数
    lr = 0.9             # 学习率。d2l 教材用 0.9；实测 0.1 时 10 轮只能到 0.634
    device = 'cpu'       # 本机无 NVIDIA 显卡

    conv1 = LeNet.net[1]                     # 第一个卷积层
    before = conv1.weight.detach().clone()   # 记下训练前的卷积核

    print('=' * 60)
    print('第 1 步：加载 Fashion-MNIST 数据')
    print('=' * 60)
    train_iter, test_iter = LeNet.load_data(batch_size)
    print('数据就绪\n')

    print('=' * 60)
    print(f'第 2 步：训练 {num_epochs} 轮（lr={lr}）')
    print('=' * 60)
    LeNet.train(LeNet.net, train_iter, test_iter, num_epochs, lr, device,
                save_dir=str(MODEL_DIR))

    after = conv1.weight.detach()

    print()
    print('=' * 60)
    print('第 3 步：第一个卷积层（训练前 vs 训练后）')
    print('=' * 60)
    show_kernels(before, '【训练前】第一个卷积层的 6 个卷积核：')
    show_kernels(after, '【训练后】第一个卷积层的 6 个卷积核：')
    print(f'卷积核平均变化量: {(after - before).abs().mean().item():.6f}')


if __name__ == '__main__':
    main()
