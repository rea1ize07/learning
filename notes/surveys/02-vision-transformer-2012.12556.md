# 视觉 Transformer 综述

> 原标题：A Survey on Visual Transformer

| | |
|---|---|
| **arXiv** | 2012.12556（v6，2023-07-10 定稿） |
| **作者/机构** | Kai Han、Yunhe Wang 等 13 人；华为诺亚方舟实验室（主力）+ 北京大学 EECS + 悉尼大学；投稿 IEEE TPAMI |
| **年份 / 篇幅** | 2020 首版，v6 定稿 2023 年 7 月；约 23 页、308 条参考文献 |
| **类型** | 综述（按应用场景分类） |

**一句话总结**：按视觉任务场景给视觉 Transformer 建立分类学——backbone、高/中层视觉、低层视觉、视频、多模态、高效压缩——逐类列出代表模型与优劣。

## 1. 它想解决什么问题

Transformer 源自 NLP（2017 Vaswani），BERT、GPT-3 证明了它的表征能力；2020 年 ViT 把图像切成 patch 序列送进纯 transformer encoder，在多个基准上追平甚至超过 CNN。痛点是：新模型爆发式增长、难以跟踪；且 transformer 缺少 CNN 的 vision-specific inductive bias（translation equivariance、locality），在中小数据集上反而不如同规模 ResNet。本文按应用场景做分类综述并给出挑战与方向，且明确说明只收录"代表性工作（early / pioneering / novel / inspiring）"。

## 2. 核心框架（重点）

一级维度是**应用场景**（Table 1 总表），四类 + 两个横切块：

**① Backbone（表征学习）**。内部先按"卷积 ↔ attention"谱系（Fig 4）分档：纯 CNN（ResNet 等）→ 只在 CNN 上加 attention（SENet/NLNet/GCNet）→ CNN+Transformer 混合（BoTNet/CeiT）→ 纯 Transformer（ViT/PVT/TNT/Swin）。纯 Transformer 再分三条改进路线：

- **增强 locality**：TNT 把 patch 再切 sub-patch 做 transformer-in-transformer；Swin 用 shifted window 做跨窗连接；Twins/CAT 逐层交替局部与全局注意力；Shuffle Transformer 用空间 shuffle 替代移位窗口；T2T 做局部特征聚合。
- **改进 self-attention 计算**：DeepViT 跨 head 通信；KVT 的 k-NN attention 只算 top-k 相似 token；XCiT 在通道维而非 token 维做 attention。
- **架构设计**：金字塔结构（PVT、Swin、PiT）、two-stream、U-net，以及 NAS（Scaling-ViT、ViTAS、AutoFormer、GLiT）。

第三块是**自监督预训练**：生成式（iGPT 像素自回归 + BERT mask；后续 MAE、SimMIM、BEiT）与对比式（MoCo v3）两条线。

**② High/Mid-level Vision**：检测（DETR 系"集合预测"路线 vs 把 transformer 当 backbone 的路线）、分割（panoptic / instance / semantic / 医学）、姿态估计（手部与人体）、其他（行人检测、车道线、scene graph、tracking、ReID、点云）。

**③ Low-level Vision**（输入输出都是图像）：图像生成（GAN 路线 TransGAN/ViTGAN；自回归路线 Image Transformer、Taming Transformer、DALL-E、CogView）、图像处理（TTSR 参考超分、IPT 多任务预训练）。

**④ Video Processing**：高层（动作识别、群组活动、检索、视频目标检测、多任务）+ 低层（帧合成 ConvTransformer、视频 inpainting STTN）。

**两个横切块**：Multi-modal（CLIP、DALL-E、UniT 等）与 Efficient Transformer（pruning & decomposition、knowledge distillation、quantization、compact architecture design / NAS，以及把 self-attention 从 O(N²) 降到 O(N) 的 linear attention 与稀疏图近似）。

附录 A1 把 self-attention 写成一般 **non-local filtering** 形式；A2 回顾 NLP transformer；A3 梳理 CV 中的纯 self-attention 模块。

## 3. 主要章节脉络

- §1 引言：NLP→CV 迁移背景 + 发展时间线（Fig 1）。
- §2 Transformer 形式化：self-attention 四步、multi-head、FFN、residual + layer norm、positional encoding、decoder 输出层。
- §3 主体：3.1 backbone、3.2 高-中层、3.3 低层、3.4 视频、3.5 多模态、3.6 高效 transformer。
- §4 结论与讨论：挑战（4.1）+ 未来方向（4.2）；附录 A1–A3；参考文献。

## 4. 关键结论与数字

- **ViT**：中小数据（ImageNet）比同规模 ResNet 低几个百分点；用 1400 万–3 亿张图预训练后超过 inductive bias，JFT-300M 上 ImageNet 88.36%、VTAB 19 任务 77.16%。
- **DeiT-B**：86M 参数，仅用 ImageNet，top-1 83.1%；用 CNN teacher 蒸馏到 84.40%（优于 transformer teacher）。
- **检测**：DETR 42.0 AP / 500 epochs / 28 FPS（Faster R-CNN+FPN 为 42.0 AP / 109 epochs / 26 FPS）；Deformable DETR 46.2 AP / 50 epochs，训练成本降至 1/10、推理快 1.6×；Swin-T+ATSS 47.2 AP；Swin 骨干比 ResNet-50 涨约 4 box AP。
- **CLIP**：4 亿 image-text pair，zero-shot ImageNet-1K top-1 76.2%，并在 30 个 CV 基准上验证迁移能力。
- **生成**：DALL-E 12B 参数、330 万 text-image pair；GPT-3 175B、45 TB 语料。
- **低层**：IPT 去噪提升约 2 dB；LSTR 比 PolyLaneNet 精度 +2.82%、FPS 3.65×、参数 1/5；TransTrack 在 MOT17/MOT20 的 MOTA 为 74.5% / 64.5%。
- **效率**：ViT-base 单图 18 GFLOPs，GhostNet 约 600 MFLOPs 即达到相近性能；Table 4 列 TinyBERT（4 层/14.5M/×9.4）、MobileBERT（25.3M/×4.0）等压缩结果。

## 5. 局限与争议

作者自承的开放问题：**CV 专用架构缺位**（ViT 基本照搬 NLP transformer）；**泛化与鲁棒性**（缺 inductive bias、依赖大数据；ViT backbone 直接用于检测未超过 CNN；对抗鲁棒性未解）；**可解释性**（为何在视觉任务上有效，目前只有直觉式分析，over-parameterization 可能是一条线索）；**效率**（模型过大，为 NLP 设计的压缩方法未必适用于 CV）；**CNN vs transformer 之争**（小数据 CNN 好、大数据 transformer 好），以及"只靠全连接层 + 海量数据能否取得满意结果"。

（文中未明确各类基准 SOTA 的截止时间；2023 年之后的扩散 transformer、VLM 等未覆盖。）

## 6. 对我的价值

最直接的接口是**附录 A1**：文中把 self-attention 严格写成 non-local filtering，yi = (1/C(xi)) Σ_j f(xi,xj) g(xj)，并指出它与 CV 中的 non-local means 去噪、non-local neural networks 同源。对信号处理背景来说，这就是"权重由数据学出来的非局部加权滤波/核平滑"，softmax 归一化即核回归的归一化因子——你已有的滤波语言可以直接接上 attention。

**低层视觉就是你熟悉的任务**：超分、去噪、去雨（IPT 去噪 +2 dB）、参考超分（TTSR），都是采样/重建/滤波问题的神经化版本。**复杂度视角**：把 self-attention 从 O(N²) 压到 O(N) 的 linear attention（核特征映射近似）与稀疏图近似，本质是"用近似与稀疏化换效率"，这种 trade-off 与随机过程、估计理论里的思路同构。

**科研接点**：文中未涉及通信与无线信号任务（文中未明确）；但若想做端侧部署或"把模型做小做快"，本文的分类学与 Table 4 是现成起手式，作者团队（华为诺亚方舟）本身就以高效模型见长——这是我基于文中信息的延伸判断，非原文结论。

**时机**：现在别读。先用 3b1b / 李宏毅建立 attention 与 Transformer 的直觉，能自己讲清 Q/K/V、multi-head、positional encoding 后再回来。另需注意 v6 定稿于 2023 年 7 月，覆盖的是 ViT 时代（ViT/DeiT/Swin/DETR），应把它当"backbone 分类地图"而非"最新进展"。

## 7. 怎么读（可操作）

- **必读**：§2（Transformer 形式化，对齐术语）；§3.1（backbone——Fig 4 的卷积-注意力谱系、三条改进路线、自监督两条线，全篇最值钱）；§4.1 + 4.2（挑战与方向，最省时间的领域地图）；附录 **A1**（non-local filtering 推导，信通背景最该看）。Table 1 与 Table 2 当索引查。
- **可跳**：§3.2–3.4 的模型细节（检测/分割/姿态/视频）先扫标题、用时回查；附录 A2（NLP 回顾，视频已学过）；Table 3/4 的逐行数字；308 条参考文献。
- **预计时间**：主线 3–4 小时（§2 约 40 min，§3.1 约 90 min，§4 约 40 min，A1 约 20 min）；通读全篇再加约 2 小时。
- **前置知识**：self-attention / multi-head / positional encoding 的基本概念（李宏毅或李沐任一视频即可）；CNN 的 inductive bias（locality、translation equivariance）。不需要自己实现过 transformer。
