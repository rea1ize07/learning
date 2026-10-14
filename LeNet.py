"""
LeNet 服装图像分类（Fashion-MNIST）完整训练脚本
========================================
LeNet 是 1998 年 LeCun 提出的经典卷积神经网络。
本脚本用它在 Fashion-MNIST 数据集上做 10 类服装分类。
本脚本包含三部分：
  1. 网络定义：5 层结构（2 个卷积层 + 3 个全连接层）
  2. 数据加载：Fashion-MNIST 数据集（28x28 灰度图，10 个类别）
  3. 训练 + 评估：完整的训练循环

关于 Fashion-MNIST：
  和 MNIST 同样尺寸（28x28、10 类），但图片是衣服而不是数字，
  分类难度更高（MNIST 太简单，随便训训就 99%）。
  10 个类别：T恤/裤子/套头衫/连衣裙/外套/凉鞋/衬衫/运动鞋/包/短靴

运行方式：
  D:/learning/envs/dl/Scripts/python.exe LeNet.py
"""

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms


# =====================================================================
# 一、网络定义
# =====================================================================

class Reshape(torch.nn.Module):
    """自定义层：只改变张量形状，不改变数值。

    作用：把展平后的向量 (batch, 784) 还原成 2D 图像 (batch, 1, 28, 28)，
    因为后面的卷积层 nn.Conv2d 要求输入必须是四维的 (批量, 通道, 高, 宽)。
    784 = 1 * 28 * 28，元素总数不变，所以这个变换是合法的。
    """
    def forward(self, x):
        # view(-1, 1, 28, 28) 四个维度含义：
        #   -1    -> 自动推断批大小，保持样本数不变
        #   1     -> 通道数（灰度图只有 1 个通道）
        #   28,28 -> 图像的高和宽
        return x.view(-1, 1, 28, 28)


def create_net():
    """构建一个全新的 LeNet 网络（权重随机初始化）。

    写成函数而不是直接建一个对象，是为了：
      1. 加载已训练的模型时，可以新建一个"空网络"再把参数灌进去
      2. 想同时持有多个网络（比如对比两个不同配置）时，各建各的
    """
    # 用 Sequential 把各层按顺序串起来，数据依次流过每一层
    return nn.Sequential(
        # 第 1 层：把 784 维向量还原成 28x28 单通道图像
        Reshape(),
        # 第 2 层：卷积，1 通道 -> 6 通道，5x5 卷积核，padding=2 使尺寸保持 28x28 不变
        nn.Conv2d(1, 6, kernel_size=5, padding=2),
        nn.Sigmoid(),                    # 激活函数（LeNet 原版用 Sigmoid，现代多用 ReLU）
        # 第 3 层：平均池化，2x2 窗口、步长 2，尺寸减半 28x28 -> 14x14
        nn.AvgPool2d(2, stride=2),
        # 第 4 层：卷积，6 通道 -> 16 通道，无 padding，尺寸 14x14 -> 10x10
        nn.Conv2d(6, 16, kernel_size=5),
        nn.Sigmoid(),
        # 第 5 层：平均池化，10x10 -> 5x5
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Flatten(),                    # 展平 16 通道 5x5 共 400 个值 -> (batch, 400)
        # 第 6 层：全连接 400 -> 120
        nn.Linear(16*5*5, 120),
        nn.Sigmoid(),
        # 第 7 层：全连接 120 -> 84
        nn.Linear(120, 84),
        nn.Sigmoid(),
        # 第 8 层（输出）：全连接 84 -> 10，输出 10 个类别的 logits（未归一化分数）
        nn.Linear(84, 10))


# 模块级默认网络，供 run_train.py 等外部脚本直接取用
net = create_net()


# =====================================================================
# 二、数据加载
# =====================================================================

def load_data(batch_size):
    """加载 Fashion-MNIST 数据集并包装成 DataLoader。

    参数：
        batch_size: 每个批次（batch）的样本数量

    返回：
        train_iter: 训练集迭代器，每次取出一批训练样本
        test_iter : 测试集迭代器，每次取出一批测试样本

    说明：
        transforms.Compose 把多个预处理步骤按顺序组合：
        1. ToTensor(): 把 PIL 图像转成张量，形状 (1,28,28)，并把像素值从 [0,255] 缩放到 [0,1]
        2. Lambda: 把张量展平成 (784,)，这样进入网络后由第一层 Reshape 负责还原成 28x28
    """
    transform = transforms.Compose([
        transforms.ToTensor(),                   # PIL 图像 -> (1,28,28) 张量，值缩放到 [0,1]
        transforms.Lambda(lambda x: x.view(-1)), # 展平成 (784,)，让 Reshape 层去还原
    ])

    # train=True 是训练集（6 万张），train=False 是测试集（1 万张）
    # download=True 表示本地没有时自动下载（官方源国内连不上，
    # 这里的数据已预先从百度镜像下好放在 ./data/FashionMNIST/raw）
    train_set = torchvision.datasets.FashionMNIST(root='./data', train=True,
                                                  download=True, transform=transform)
    test_set = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                                 download=True, transform=transform)

    # DataLoader 负责分批、打乱顺序、多线程读取
    # shuffle=True 表示每个 epoch 随机打乱训练集顺序，防止模型记住固定顺序
    train_iter = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_iter = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_iter, test_iter


# =====================================================================
# 三、评估
# =====================================================================

def evaluate(net, data_iter, device):
    """在给定的数据集上计算分类准确率。

    参数：
        net      : 要评估的模型
        data_iter: 数据迭代器（通常传测试集）
        device   : 运行设备（'cpu' 或 'cuda'）

    返回：
        correct / total: 准确率，范围 [0, 1]，1 表示全部预测正确

    说明：
        torch.no_grad() 会关闭梯度计算，评估时不需要反向传播，
        这样可以节省内存、加快速度。
        net.eval() 会把模型切换到评估模式（对含 Dropout/BN 的模型重要，
        对本网络虽无实质影响，但属于标准写法）。
    """
    net.eval()
    correct, total = 0, 0
    with torch.no_grad():                        # 评估阶段不计算梯度，省内存、更快
        for X, y in data_iter:
            X, y = X.to(device), y.to(device)    # 把数据搬到对应设备
            y_hat = net(X)                       # 前向传播，得到 (batch, 10) 的 logits
            # argmax(dim=1) 取每行分数最大的类别作为预测结果，再和真实标签 y 比较
            correct += (y_hat.argmax(dim=1) == y).sum().item()
            total += y.size(0)                   # 累加本批样本数
    return correct / total


# =====================================================================
# 四、训练
# =====================================================================

def train(net, train_iter, test_iter, num_epochs, lr, device,
          save_dir='models/lenet', model_name='lenet_fashionmnist',
          save_every=10):
    """训练模型，报告准确率，并把训练好的参数保存到磁盘。

    参数：
        net       : 要训练的模型
        train_iter: 训练集迭代器
        test_iter : 测试集迭代器
        num_epochs: 训练轮数（把整个训练集完整过一遍叫 1 个 epoch）
        lr        : 学习率，控制每次参数更新的步长
        device    : 运行设备（'cpu' 或 'cuda'）
        save_dir  : 模型保存目录，默认 models/lenet
        model_name: 文件名前缀，默认 lenet_fashionmnist
        save_every: 每多少轮存一份存档，默认 10

    返回：
        best_acc: 训练过程中在测试集上达到的最高准确率
    """
    # --- 权重初始化 ---
    # LeNet 用 Sigmoid 激活，若初始权重不当（过大/过小）容易让 Sigmoid 饱和、
    # 梯度消失，导致训练不动。Xavier 初始化能根据前后层神经元数自动调整方差，比较稳妥。
    def init_weights(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)): # 只对带权重的层初始化
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)                        # 递归遍历网络的每一层并应用上面的函数

    net.to(device)                                 # 把模型参数搬到设备上

    # --- 损失函数与优化器 ---
    # CrossEntropyLoss 内部已包含 softmax + 交叉熵，所以网络末尾不用再加 softmax
    loss = nn.CrossEntropyLoss()
    # SGD 随机梯度下降，是 LeNet 原版使用的优化器
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)

    # --- 准备保存目录 ---
    os.makedirs(save_dir, exist_ok=True)           # 目录不存在就创建，已存在不报错
    best_acc = 0.0                                 # 记录历史最高准确率

    # --- 训练循环 ---
    for epoch in range(num_epochs):
        net.train()                                # 切换到训练模式
        for X, y in train_iter:                    # 遍历每个批次
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()                  # 清空上一轮的梯度（否则会累加）
            l = loss(net(X), y)                    # 前向传播 + 计算损失
            l.backward()                           # 反向传播，计算各参数梯度
            optimizer.step()                       # 按梯度更新参数

        # 每个 epoch 结束，在测试集上评估一次，观察训练效果
        acc = evaluate(net, test_iter, device)
        print(f'epoch {epoch+1:2d}, test acc {acc:.3f}')

        # 定期存一份"存档"：文件名带 epoch 和准确率，方便回头对比不同阶段的模型
        if (epoch + 1) % save_every == 0:
            path = os.path.join(save_dir,
                                f'{model_name}_epoch{epoch+1}_acc{acc:.3f}.pth')
            torch.save(net.state_dict(), path)     # 只存参数（state_dict），不存代码
            print(f'          -> 已存档 {path}')

        # 只要准确率刷新了历史纪录，就单独存一份"最佳模型"
        if acc > best_acc:
            best_acc = acc
            path = os.path.join(save_dir, f'{model_name}_best.pth')
            torch.save(net.state_dict(), path)
            print(f'          -> 最佳模型已更新 (acc {acc:.3f})')

    print(f'\n训练结束，最高测试准确率 {best_acc:.3f}')
    print(f'最佳模型已保存到 {os.path.join(save_dir, model_name + "_best.pth")}')
    return best_acc


# =====================================================================
# 五、加载已训练的模型
# =====================================================================

def load_model(path, device='cpu'):
    """从磁盘加载训练好的参数，返回一个可以直接用来推理的网络。

    参数：
        path  : 模型文件路径（train 保存的 .pth 文件）
        device: 运行设备

    返回：
        net: 已装载好参数、处于评估模式的网络

    说明：
        必须先 create_net() 建一个结构完全相同的空网络，再把参数灌进去。
        网络结构一旦改了（层数、每层大小），旧文件就对不上了，会报"键不匹配"。
    """
    net = create_net()                             # 新建一个随机初始化的网络
    # map_location 保证在 CPU 上也能加载 GPU 训练出来的模型
    net.load_state_dict(torch.load(path, map_location=device))
    net.to(device)
    net.eval()                                     # 评估模式：关掉训练相关的行为
    print(f'已加载模型 {path}')
    return net


# =====================================================================
# 六、程序入口
# =====================================================================

if __name__ == '__main__':
    batch_size = 256     # 每个批次的样本数，越大越省时间但占内存越多
    lr = 0.1             # 学习率，过大会震荡、过小收敛慢
    num_epochs = 30      # 训练轮数；实测 10 轮只到 63%，多训几轮效果更好
    device = 'cpu'       # 本机无 NVIDIA 显卡，只能用 CPU（有 GPU 则改成 'cuda'）

    train_iter, test_iter = load_data(batch_size)
    train(net, train_iter, test_iter, num_epochs, lr, device)

    # 训练完的模型存在 models/lenet/ 目录下，之后想直接用（不用重新训练）：
    #   net = load_model('models/lenet/lenet_fashionmnist_best.pth')
    #   acc = evaluate(net, test_iter, device)   # 直接评估
    #   y   = net(x)                             # 或者直接做预测
