"""
查看 LeNet 卷积核的工具
=======================
加载一个训练好的模型，显示两层卷积核的形状和数值图案。

用法（在仓库根目录 D:/learning 下执行）：
  # 自动找 models/lenet/ 下最新的模型
  D:/learning/envs/dl/Scripts/python.exe lenet/show_kernels.py

  # 指定某个模型文件
  D:/learning/envs/dl/Scripts/python.exe lenet/show_kernels.py models/lenet/xxx.pth

怎么判断训练得好不好：
  卷积核出现「正负交替的条纹」 = 学到了边缘检测器，训练到位
  卷积核全是同一种符号、糊成一团 = 训练不足（梯度太小，没学动）
"""

import sys
from pathlib import Path

import torch

# --- 让脚本能找到仓库根目录的 LeNet.py ---
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import LeNet

MODEL_DIR = ROOT / 'models' / 'lenet'


# 数值 -> 符号 的映射：让人眼能看出卷积核的"花纹"
def to_symbol(v):
    if v > 0.25:
        return ' #'      # 明显正
    if v > 0.08:
        return ' +'      # 略正
    if v > -0.08:
        return ' .'      # 接近 0
    if v > -0.25:
        return ' -'      # 略负
    return ' @'          # 明显负


def find_default_model():
    """没指定文件时，优先用 best 模型，否则用最新的那个 .pth。"""
    preferred = MODEL_DIR / 'lenet_fashionmnist_best.pth'
    if preferred.exists():
        return preferred

    candidates = sorted(MODEL_DIR.glob('*.pth'),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def show_conv_layer(net, index, name):
    """显示网络里第 index 层的卷积核。"""
    conv = net[index]

    # 不是卷积层就跳过（这样 index 可以直接用 Sequential 里的编号）
    if not isinstance(conv, torch.nn.Conv2d):
        return

    w = conv.weight.detach()          # 形状 (输出通道, 输入通道, 核高, 核宽)
    b = conv.bias.detach()

    print('=' * 62)
    print(f'{name}  （Sequential 里的第 {index} 层）')
    print('=' * 62)
    print(f'  卷积核张量形状: {tuple(w.shape)}')
    print(f'    = {w.shape[0]} 个卷积核 × {w.shape[1]} 个输入通道 '
          f'× {w.shape[2]}×{w.shape[3]}')
    print(f'  可训练参数: 权重 {w.numel()} + 偏置 {b.numel()} '
          f'= {w.numel() + b.numel()}')
    print()

    for k in range(w.shape[0]):
        print(f'  卷积核 {k}   （偏置 bias = {b[k].item():+.4f}）')
        for c in range(w.shape[1]):
            if w.shape[1] > 1:
                print(f'    [输入通道 {c}]')
            for row in w[k, c].tolist():
                print('    ' + ''.join(to_symbol(v) for v in row))
        print()


def main():
    # 命令行给了路径就用给的，否则自动找
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
    else:
        path = find_default_model()
        if path is None:
            print(f'models/lenet/ 里没有找到 .pth 模型文件。')
            print(f'先运行：python lenet/train.py')
            return

    print(f'加载模型: {path}\n')
    net = LeNet.load_model(str(path))
    print()

    show_conv_layer(net, 1, '第一个卷积层')     # Conv2d(1, 6, 5)
    show_conv_layer(net, 4, '第二个卷积层')     # Conv2d(6, 16, 5)

    print('=' * 62)
    print('符号说明:  # 明显正   + 略正   . 接近0   - 略负   @ 明显负')
    print('训练充分的卷积核通常有正负交替条纹（边缘检测器）；')
    print('全是同一种符号则说明训练不足。')
    print('=' * 62)


if __name__ == '__main__':
    main()
