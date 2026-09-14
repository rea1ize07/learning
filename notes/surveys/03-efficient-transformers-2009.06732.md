# 高效 Transformer 综述
> 原标题：Efficient Transformers: A Survey

| | |
|---|---|
| **arXiv** | 2009.06732（v3，2022-03-14；v1 约 2020-08，v2 于 2021-12 大幅修订） |
| **作者/机构** | Yi Tay、Mostafa Dehghani、Dara Bahri、Donald Metzler（Google Research / Brain team） |
| **年份 / 篇幅** | 初稿 2020，最后更新 2022.3；原 PDF 39 页，正文 5 章 |
| **类型** | 综述（分类学 + 逐模型点评） |

**一句话总结**：把为绕开 self-attention 的 O(N²) 瓶颈而涌现的几十个"X-former"模型，按技术手段归成九大类，并逐个体检其复杂度与可用性。

## 1. 它想解决什么问题
self-attention 要对序列中所有 token 两两算相关性，仅 QKᵀ 一步就消耗 N² 时间和显存，长文档、图像、视频、蛋白质序列因此难以处理。2020 年前后"X-former"爆发式涌现（作者称仅 6 个月就有近十个效率模型），各说各话，既追不上也难比较。本文为此建立统一分类学并横向点评代价与适用场景。

## 2. 核心框架（重点）
3.1 节提出九类分类学，是全文最值钱的部分（总览见 Table 1 与 Figure 2）：

1. **Fixed Patterns（固定模式）**：直接稀疏化 attention 矩阵。Blockwise/chunking（Blockwise Transformer、Image Transformer）把序列切块，N² 降到 B² 量级（B≪N）；Strided/dilated（Sparse Transformer、Longformer）只在固定间隔 attend；Compressed（Compressed Attention）用带步长卷积池化缩短序列。
2. **Combination of Patterns（模式组合）**：拼多种访问模式提升覆盖率。Sparse Transformer 一半 head 做 local、一半做 strided；Axial Transformer 沿高维张量单轴依次做 attention。
3. **Learnable Patterns（可学习模式）**：让"谁看谁"由数据决定。Reformer 用 LSH 局部敏感哈希分桶，O(N log N)；Routing Transformer 用 online k-means 聚类，O(n^1.5)；Sinkhorn Transformer 学排序。三者本质都是重排序列再做块状 attention。
4. **Neural Memory（神经记忆）**：引入可学习的全局记忆/诱导点，源自 Set Transformer 的 inducing points，ETC 与 Longformer 的 global token 沿用。
5. **Low-Rank（低秩）**：假设 N×N attention 矩阵低秩。Linformer 把 K、V 在**长度维**投影到 k（N→k），显存降到 O(n)；Synthesizer 的 factorized 变体、Low-Rank Transformer 同族。
6. **Kernels（核方法）**：把 attention 视作核函数，靠矩阵乘法结合律绕开显式 N×N 矩阵。Performer 用 FAVOR（正交随机特征），O(Md+Nd+MN)；Linear Transformer 用 φ(x)=elu(x)+1，端到端 O(Nd²)；RFA 同类。
7. **Recurrence（循环）**：不稀疏化，用段级循环连接块。Transformer-XL 提出 segment-level recurrence，Compressive Transformer 再加压缩记忆。
8. **Downsampling（降采样）**：直接降分辨率。Perceiver（O(N)）、Funnel Transformer（O(n(k+m))）、Swin、Charformer、TokenLearner、Luna。作者特别澄清 Nyströmformer 看似低秩/核方法，实为"landmarks 做带步长池化"的降采样。
9. **Sparse Models & Conditional Computation（稀疏/MoE）**：不针对 attention，而是稀疏激活参数。Switch 用 top-1 路由，GShard 用 group-level top-2 门控，另有 GLaM、ST-MoE-32B。

作者强调边界不锐利：模型常是多类杂交（Long Short Transformer = 低秩 + 固定窗），2021 年后流行"两阶段 attention"。

## 3. 主要章节脉络
- **§2 背景**：标准结构、self-attention 公式、三种使用模式（encoder-only / decoder-only / encoder-decoder），并指出 FFN 约占一半算力。
- **§3.1 分类学**：上述九类。
- **§3.2 逐模型走查**（3.2.1–3.2.18）：按大致时间顺序，从 Memory Compressed、Image Transformer、Set Transformer、Longformer、ETC、BigBird，到 Routing、Reformer、Sinkhorn、Linformer、Performer、Linear Transformer、Transformer-XL、Compressive、Sparse/MoE，每节给复杂度与局限。
- **§4 讨论**：4.1 评测现状、4.2 设计趋势、4.3 正交效率手段、4.4 一年回顾与未来方向。**§5 结论**。

## 4. 关键结论与数字
- 基线：dense attention 为 **O(N²)**；FFN 约占每层一半 FLOPs。
- 正文明确给出的复杂度：Reformer **O(N log N)**、Routing Transformer **O(n^1.5)**、Linformer 显存 **O(n)**、BigBird **O(n)**、Longformer **O(n²)→O(nk)**（k 为窗口）、ETC **O(n_g²+n_g·N)**、Performer **O(Md+Nd+MN)**、Linear Transformer **O(Nd²)**、Image Transformer **O(n·m)**。
- 参数代价：Reformer 让 Q、K 共享，QKV 变换开销降三分之一；Longformer 的 global attention 使 attention 层参数翻倍；Sinkhorn 的排序网络额外交 2d² 参数。
- 速度：Linear Transformer 声称推理最快提 **3 个数量级**；block pruning 可让 Transformer 快 **2.4 倍**。
- 评测：Long Range Arena（LRA）首次统一评测 **10 个** xformer 变体，但只覆盖 encoder-only，不含需 causal mask 的生成任务。

## 5. 局限与争议
- **"efficient" 名不副实**：作者明说高效 attention 实践里可能**更慢**；线性 attention 在序列较短时根本没有速度或显存收益，且常以吞吐换线性复杂度。
- **causal mask 是硬伤**：Performer 的单向实现无法高效掩码，自回归训练需从左到右扫描，可能比 vanilla 慢好几倍，甚至"慢到无法训练"；Linformer 的长度维投影难以维持因果掩码；ETC、BigBird 干脆不能自回归解码，只能 encoder-only。
- **评测混乱**：各论文自选 benchmark 与模型规模，还常与预训练混在一起，增益难归因；机器翻译因序列不长而鲜被评测；作者直言"该用哪个高效 block 仍是谜"。
- **没有赢家**：截至 2021 年底 vanilla 二次复杂度 Transformer 仍占主导，无任何 xformer 成为默认标准；线性 attention 在常见 benchmark 上难有竞争力。
- **工程代价**：ETC 的多种 attention 方向、Longformer/Swin 需定制 CUDA kernel（TPU 上基本不可用），难以充当即插即用替代品。
- **理想模型**：应同时解决二次显存、保持通用性、不以速度换显存、支持 causal masking、实现简洁。作者无法指认哪个模型做到了，并反问"真正的 xformer，还会是 Transformer 吗？"

（文中未明确：正文提到的 Appendix 变更日志不在提取文本中；Transformer-XL / Compressive 的复杂度表达式正文未给出。）

## 6. 对我的价值
本文内核是一句话：**如何近似或重写一个 O(N²) 的算子**——这正是信号处理里"快速算法"的老命题换了张皮，是现有基础最能直接迁移的一篇。

- **Kernels ↔ 核方法与随机特征**：Performer 的 FAVOR 用正交随机特征逼近 softmax 核，Linear Transformer 用 φ(q)ᵀφ(k) 替代 exp(qᵀk/d)，即"把相似度换成可分离核"，与随机过程里用随机投影逼近核函数同源。
- **Low-Rank ↔ 低秩近似/子空间**：Linformer 的长度维投影就是给 attention 矩阵降秩，和 SVD、压缩感知同族；Nyström 方法本就是数值线性代数里用 landmarks 近似大矩阵的经典技巧。
- **Fixed Patterns ↔ 稀疏化/多速率采样**：dilated window（Longformer 的膨胀窗）在语音图像里就是多分辨率分析与膨胀卷积。
- **Learnable Patterns ↔ 聚类/矢量量化**：online k-means 与 LSH 分桶都是"先量化再算"的老套路。
- **警惕**："理论复杂度≠实测延迟"。作者强调 O(N) 模型可能比 O(N²) 慢，决定性因素是并行度、访存与 causal mask 的实现——这和 FFT 的常数因子与硬件访存问题同构。

**能否接科研**：能，但要挑角度。"再提一个 xformer 变体"是红海且评价体系混乱。更可行的是：(1) 把通信/信号领域的长序列问题（长帧信道估计、超长 IQ 序列建模、稀疏感知）拿来验证某类高效 attention 的真实省时省显存效果；(2) 做"复杂度 vs 实测延迟"的基准分析，这正是本文点出的空白；(3) 顺作者指的方向走 S4 / 结构化状态空间（Structured State Spaces）——它们和线性时不变系统、状态空间模型几乎同一套语言，迁移成本最低，也是本文最看好的一支。

**何时读**：不建议现在读。等按"神经网络→CNN→Transformer"走完、能手推 QKᵀV 的维度与 causal mask 形状之后再来。它的价值不是教你注意力，而是让你知道"注意力可被近似和重写"；放在学完 Transformer、准备选方向之前最划算。

## 7. 怎么读（可操作）
- **必读**：§2.4（成本从哪来，含"FFN 占一半算力"）；**§3.1 分类学**（全文精华，20 分钟可读完）；§4.1 评测现状与 **§4.4 一年回顾**（作者最诚实的判断）；Table 1 与 Figure 2 当索引图用。
- **精读三个代表**：§3.2.10 Reformer（LSH + 可逆层）、§3.2.12 Linformer（低秩）、§3.2.13 Performer（核/随机特征）——覆盖三条主线。其余 15 小节只看每节末尾的 "Parameter and Memory Complexity" 与 "Restrictions" 两段。
- **可跳**：§2.1–2.3、§2.5（标准 Transformer 复习，视频里已学）；§3.2.1–3.2.9 的历史模型细节（只取复杂度数字）；§4.3 正交手段（权值共享/量化/蒸馏/NAS/adapter，当名词表扫一眼）；§3.2.18 稀疏模型。
- **注意**：Table 1 转文本后列错位（复杂度与模型名对不上），数字以正文各节为准，别直接抄表。
- **预计时间**：第一遍（§3.1 + §4 + Table 1）1–1.5 小时；三个精读模型各 30–40 分钟；整体吃透 4–5 小时，宜分两次。
- **前置知识**：self-attention 流程与复杂度来源、矩阵乘法结合律（理解 kernel 方法的关键）、SVD/低秩近似直觉、哈希与 k-means 直觉。随机特征/核方法若没学过，先补一点 RKHS 或 random Fourier features——这是从信号处理跨过来的最短路径。
