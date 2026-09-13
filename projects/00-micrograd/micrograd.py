# -*- coding: utf-8 -*-
"""
micrograd.py —— 手写自动微分引擎（关卡1 · 第一步）

目标：不借助任何框架，用纯 Python 实现一个标量级的自动微分（autograd）引擎。
PyTorch 的 `loss.backward()` 内部就是这个东西，只是它跑在张量上、并且用 C++/CUDA 实现。

核心心智模型（借用信号处理的直觉）：
    - 每个 Value 是一个「节点」，前向运算把节点连成一张有向无环图（DAG）——
      这张图就是信号流图（signal flow graph）。
    - 前向传播 = 信号从输入流向输出。
    - 反向传播 = 梯度的「信号」从输出反向流回输入，每经过一个节点就乘上该节点的局部增益。
      这正是一条链式法则。
    - 一个节点可能被多个下游节点使用（fan-out），所以梯度是**累加**的——
      就像同一条信号沿多条路径到达同一点时叠加。

用法：
    from micrograd import Value

    a = Value(2.0); b = Value(-3.0)
    L = a * b + 1.0
    L.backward()          # 反向传播
    print(a.grad)         # dL/da
"""

import math


class Value:
    """标量节点。

    Attributes:
        data:   前向值（标量）
        grad:   损失对它的梯度 dL/d(self)，由 backward() 填充
        label:  可读名字，画计算图时用
    """

    # __slots__ 让属性固定在几个槽里，省内存也防止手滑拼错属性名
    __slots__ = ('data', 'grad', '_backward', '_prev', '_op', 'label')

    def __init__(self, data, _children=(), _op='', label=''):
        self.data = float(data)
        self.grad = 0.0

        # _backward 是该节点的「局部反向函数」：把 out.grad 换算成各输入的梯度增量。
        # 叶子节点没有输入，所以是空函数。
        self._backward = lambda: None

        # _prev 记录本节点由哪些节点算出来 —— 计算图的边
        self._prev = set(_children)

        # _op 记录产生本节点的运算类型，仅用于画图/调试
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    # ------------------------------------------------------------------
    # 基础运算：只需要把 + * tanh(以及 exp/pow) 这几个「原子」写对，
    # 其余运算（减、除、负数、relu……）都能由它们组合出来。
    # ------------------------------------------------------------------

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            # y = a + b  →  dy/da = 1, dy/db = 1
            # 加法是「梯度路由器」：把上游梯度原封不动地分发给两个输入。
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            # y = a * b  →  dy/da = b, dy/db = a
            # 乘法是「梯度交换器」：把上游梯度乘上对方的当前值再发过去。
            # 注意：局部梯度用的是前向时的值，所以前向的中间结果必须留着。
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, k):
        assert isinstance(k, (int, float)), "目前只支持常数幂"
        out = Value(self.data ** k, (self,), f'**{k}')

        def _backward():
            # y = x^k  →  dy/dx = k * x^(k-1)
            self.grad += k * (self.data ** (k - 1)) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        x = self.data
        t = math.tanh(x)
        out = Value(t, (self,), 'tanh')

        def _backward():
            # y = tanh(x)  →  dy/dx = 1 - tanh²(x) = 1 - y²
            # 用输出 y 自己算导数，省一次 tanh 调用（前向已算过）。
            self.grad += (1 - t ** 2) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), 'exp')

        def _backward():
            # y = e^x  →  dy/dx = e^x = y
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Value(0.0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            # y = max(0, x)  →  dy/dx = 1 if x>0 else 0
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out

    # ------------------------------------------------------------------
    # 派生运算：直接用上面的原子拼出来，不用手写新的导数。
    # ------------------------------------------------------------------

    def __neg__(self):                 # -a
        return self * -1

    def __sub__(self, other):          # a - b = a + (-b)
        return self + (-other)

    def __truediv__(self, other):      # a / b = a * b^(-1)
        return self * other ** -1

    # 反向运算符：让 2 * x、1 + x 这种「常数在左」的写法也能工作
    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return other + (-self)

    def __rtruediv__(self, other):
        return other * self ** -1

    # ------------------------------------------------------------------
    # 反向传播
    # ------------------------------------------------------------------

    def backward(self):
        """从本节点出发做反向自动微分，把 dL/dx 写进每个相关节点的 .grad。

        关键点：必须按**拓扑序的逆序**处理节点。
        原因：某节点的梯度是「所有下游路径贡献之和」，只有当它所有的下游节点
        都已完成反向之后，它自己的 grad 才累积完整，才能继续往上游传。
        乱序处理会导致某些路径的贡献漏掉。
        """
        # 1) 拓扑排序（DFS 后序：先访问所有子节点，再把自己压栈）
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # 2) 逆序传播，起点是 dL/dL = 1
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()


# ======================================================================
# 自检 1：手算验证 —— 用一个能口算的小例子对答案
# ======================================================================
def demo_hand_check():
    print("\n" + "=" * 62)
    print("自检 1：手算验证")
    print("=" * 62)

    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    c = Value(10.0, label='c')
    e = a * b; e.label = 'e'
    d = e + c; d.label = 'd'
    f = Value(-2.0, label='f')
    L = d * f; L.label = 'L'

    # 手算：L = (a*b + c) * f = (2*(-3) + 10) * (-2) = 4 * (-2) = -8
    # dL/df = d = 4
    # dL/dd = f = -2
    # dL/dc = dL/dd * dd/dc = -2 * 1 = -2
    # dL/de = dL/dd * dd/de = -2 * 1 = -2
    # dL/da = dL/de * de/da = -2 * b = -2 * (-3) = 6
    # dL/db = dL/de * de/db = -2 * a = -2 * 2 = -4
    expected = {'a': 6.0, 'b': -4.0, 'c': -2.0, 'e': -2.0, 'd': -2.0, 'f': 4.0, 'L': 1.0}

    L.backward()

    print(f"L = {L.data}   （期望 -8）")
    ok = abs(L.data - (-8.0)) < 1e-9
    for name, node in [('a', a), ('b', b), ('c', c), ('e', e),
                       ('d', d), ('f', f), ('L', L)]:
        exp = expected[name]
        good = abs(node.grad - exp) < 1e-9
        ok = ok and good
        print(f"  {name}.grad = {node.grad:8.4f}   期望 {exp:8.4f}   {'OK' if good else 'FAIL'}")

    print("结果：" + ("全部通过 ✅" if ok else "有不符 ❌"))
    return ok


# ======================================================================
# 自检 2：数值梯度校验（gradcheck）
# ======================================================================
def numerical_grad(f, xs, i, h=1e-6):
    """中心差分估偏导：(f(x+h) - f(x-h)) / (2h)，误差 O(h²)。"""
    old = xs[i].data
    xs[i].data = old + h
    y1 = f().data
    xs[i].data = old - h
    y2 = f().data
    xs[i].data = old          # 复原，别污染后续
    return (y1 - y2) / (2 * h)


def analytic_grad(f, xs):
    """清空梯度 → 前向 → 反向，取回解析梯度。"""
    for x in xs:
        x.grad = 0.0
    out = f()
    out.backward()
    return [x.grad for x in xs], out.data


def demo_gradcheck(seed=42):
    """拿一个含 tanh / 幂 / 乘加混合的表达式，逐个输入比解析梯度 vs 数值梯度。"""
    import random
    print("\n" + "=" * 62)
    print("自检 2：数值梯度校验（中心差分 vs 反向传播）")
    print("=" * 62)

    rng = random.Random(seed)
    xs = [Value(rng.uniform(-2, 2), label=f'x{i}') for i in range(5)]

    def f():
        # 一个刻意拧巴的表达式，覆盖 + - * / ** tanh 各条反向路径
        y = (xs[0].tanh() * xs[1] + xs[2] ** 2) / (xs[3].tanh() + 1.5)
        y = y - xs[4] * xs[0]
        y = y + (xs[1] * xs[2]).tanh()
        return y

    an, y = analytic_grad(f, xs)
    print(f"f(x) = {y:.8f}\n")
    print(f"{'输入':<6}{'解析梯度':>14}{'数值梯度':>14}{'相对误差':>14}")
    print("-" * 62)

    ok = True
    for i, x in enumerate(xs):
        ng = numerical_grad(f, xs, i)
        denom = max(1e-8, abs(an[i]) + abs(ng))
        rel = abs(an[i] - ng) / denom
        good = rel < 1e-5
        ok = ok and good
        print(f"x{i:<5}{an[i]:>14.8f}{ng:>14.8f}{rel:>14.2e}   {'OK' if good else 'FAIL'}")

    print("\n结果：" + ("解析梯度与数值梯度一致 ✅" if ok else "存在偏差 ❌"))
    return ok


# ======================================================================
# 可视化：把计算图画成 Graphviz DOT（可选，装了 graphviz 才有用）
# ======================================================================
def trace(root):
    """遍历图，收集所有节点和边。"""
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges


def draw_dot(root, out_path=None):
    """生成计算图的 DOT 文本；传 out_path 则同时写文件。"""
    nodes, edges = trace(root)
    lines = ['digraph G {', '  rankdir=LR;',
             '  node [shape=record, fontname="Consolas"];']
    for n in nodes:
        uid = str(id(n))
        shape = 'box' if n._op else 'ellipse'
        fill = '#dbeafe' if n._op else '#dcfce7'
        label = f"{{ {n.label or ' '} | data {n.data:.4f} | grad {n.grad:.4f} }}"
        lines.append(f'  "{uid}" [label="{label}", shape={shape}, '
                     f'style=filled, fillcolor="{fill}"];')
    for a, b in edges:
        lines.append(f'  "{id(a)}" -> "{id(b)}";')
    lines.append('}')
    dot = '\n'.join(lines)
    if out_path:
        with open(out_path, 'w', encoding='utf-8') as fh:
            fh.write(dot)
        print(f"计算图 DOT 已写入：{out_path}")
    return dot


if __name__ == '__main__':
    r1 = demo_hand_check()
    r2 = demo_gradcheck()
    print("\n" + "=" * 62)
    print("关卡1 第一步：" + ("引擎可用 ✅" if (r1 and r2) else "未通过 ❌"))
    print("=" * 62)
