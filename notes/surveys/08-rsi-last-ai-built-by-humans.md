# 人类建造的最后一个 AI：走向真正的递归自我改进
> 原标题：The Last AI Built by Humans: Toward Genuine Recursive Self-Improvement

| | |
|---|---|
| **arXiv** | 2609.11873v1 [cs.LG] |
| **作者/机构** | Yi Duan, Ying Liu, Zirui Tang, Xuanhe Zhou, Fan Wu 等（上海交通大学、Theseus Labs、清华大学、字节跳动、ModelBest、小红书、Humanlaya、上海 AI Lab 等） |
| **年份 / 篇幅** | 2026 年 9 月；约 75 页（正文 53 页 + 参考文献 + 附录） |
| **类型** | 综述 + roadmap（作者自述 autonomy-centered roadmap） |

**一句话总结**：提出 B0–L5 自治等级框架，用 HCI 指标量化能力缺口，论证 RSI 的关键是让 AI 接管改进回路中的决策。

## 1. 它想解决什么问题

前沿模型开发成本在多维度膨胀（Kimi K3 / Qwen3.8-Max 参数达 2.8T / 2.4T）。作者归纳出三大 development burden：**基座训练资源密集**（GDPval 用 1320 个任务、约 9240 专家工时）；**反馈与学习环境难以扩展**（DeepSeek-V3.2 后训练算力超预训练成本的 10%）；**部署后的反复适配**（agentic 负载 token 约为普通对话的 4 倍、多智能体 15 倍）。RSI 主张把这部分外部协调变成系统自身的持久能力。

## 2. 核心框架（重点）

作者以"改进回路"（improvement loop）而非算法为分析单位，按**责任从人到 AI 的转移量**分级。回路部件：AI system、system state、experience、target、improver、strategy、verifier、improvement、successor。每级问三件事：回路在哪闭合、什么被继承、哪些决定仍属人类。

**B0（非 RSI 参照级）**：改进限于当前任务，可反复修订、验证、挑选输出，但**不留持久系统状态**——"输出改变而系统不变"。**B0→L1 的边界是 persistence**。

**L1 执行自治**：人类定目标、更新流程与验收标准，AI 只执行，产物持久进入后续工作流。代表：FineWeb-Edu、EDIT、REPO、Nemotron-4、Meta NCCL Agentic Debugging、OpenAI HealthBench。本级按 AI 开发流水线分六层（数据 / 训练方法 / 训练平台 / 评测安全 / 部署优化 / 应用系统）。

**L2 策略自治**：目标、任务边界、评估标准**仍外部固定**，但 AI 用评测反馈诊断弱点、自行决定**该怎么改**——人类从"提出每个改进"转为"约束搜索空间"。代表：Self-Harness、GEPA、MPO、Promptbreeder、ADAS、AFlow、AgentSquare、AutoKernel。**L1→L2 转移 strategy selection**。

**L3 经验获取自治**：系统还决定**下一轮该学什么**（learner-conditioned future experience acquisition）——演化中 learner 的证据须影响"选、生成、寻找哪些经验"，且持久更新反哺后续获取决策。代表：SIMA 2、VOYAGER、SEAgent、AZR、SSP、STP、R-Zero、PSV。**L2→L3 转移未来学习议程的控制权**。

**L4 环境适配自治**：用**持续部署交互**反馈决定 agent 如何持久适配，管记忆、技能、harness、代码、（可选）权重。三种机制：trajectory distillation（ReasoningBank、Metis）、agent system revision（HarnessDev）、selective retention（HDSO 配对评测门）。代表还有 PANDO、ASPIRE、Tax AI。**L3→L4 边界**：从"选什么经验"变为"哪些经验后果该持久化"；目标、访问边界与发布权仍外部治理。

**L5 递归继承自治**：AI 持久修改**负责未来改进的机制**并在后续轮次使用它；可编辑对象是 improver、successor evaluator、search policy 或研究政策。回路在"被修订机制反过来支配后续 successor 的生成 / 评估 / 选择"时闭合。代表：STOP（improver 以自身源码为优化目标）、Gödel Agent、DGM、HyperAgents、RQGM、A-Evolve-Training、AIDE2。人类仍保留 mission、受保护评测与资源权限。**L4→L5 转移 recursive inheritance**。

**HCI**：`HCI = 100 × (s − F_b,0)/(100 − F_b,0)`，F_b,0 为该 benchmark 首年模型的 90 分位分。0 = 首年前沿，100 = 满分。来源权重为官方表 3、独立通用 harness 2.5、合并报告 2、模型方自报 1（自报再乘 0.75）。

## 3. 主要章节脉络

- **§1 引言**：三大负担、RSI 定义、三大难题、五级框架、四域、工业证据。
- **§2 背景**：2.1 用 HCI 展示能力不均衡（393 条观测、10 个域）；2.2 回路解剖 + 与 continual learning / AutoML / agentic AI 的对照（表 1）；2.3 证据范围。
- **§3 五级自治**：3.1 B0 → 3.6 L5，每级给回路特征式与 "Finding" 小结框；3.7 跨级综合给相邻级边界。
- **§4 应用域**：S1 科学 / S2 具身 / S3 软件工程 / S4 医疗。
- **§5 工业实践**：Theseus、Lark、Humanlaya、ModelBest、腾讯混元 Hyra、Agent-Native Research Lab。
- **§6 八大方向**、**§7 结论**；附录 A（491 篇分布图）、附录 B（工业系统表）。

## 4. 关键结论与数字

**HCI（2026）**：高等数学 86.4、研究生科学 85.8、宽泛知识 77.2；法律 64.5、多模态 62.2、前沿学术广度 60.4。交互类明显落后：搜索 / 终端 56.8、软件工程 52.6、工具 agent 39.9（网络安全最高 91.9）。工具 agent 一年内从 8.2 跃至 39.9，仍最低。外推 `R_d = 100 − 0.22(100 − T_d,2026)`：网络安全→98.2、搜索 / 终端→90.5、软件工程→89.6、工具 agent→86.8。

**文献分布（491 篇）**：L1 215（43.8%）、L2 155（31.6%）、L3 64（13.0%）、L4 28（5.7%）、L5 29（5.9%）——证据密度随等级骤降。

**场景水平（图 9）**：科学 = strong L2 / 早期 L3；具身 = L2–L3 主战场、simulated L4；软件工程 = mature L2、emerging L3、bounded structural L5；医疗 = mature L2、有限 L3-like、simulated L4。

**实测**：A-Evolve-Training 在 30B Nemotron 四轮自主后训练，外部分 0.80→0.86（人类最佳 0.87）；DGM 在 SWE-bench 子集 20%→50%；RQGM 在 held-out Polyglot 71.7% vs HGM-H 69.9%；STOP 第四代 improver 在 5 个迁移任务上全面超 seed。工业：Theseus 重构环境使 rubric 分 +18.65–39.67 个百分点；Lark 自动评估器与人类一致率约 84%；Humanlaya V0→V4 关键缺陷率 9.0%→3.7%、人工时间 48→27 分钟/任务；ModelBest ForgeTrain 从空目录 8 小时匹敌 Megatron-LM v0.15；腾讯 Hyra 报 nanochat BPB 0.9015 vs 0.9109。

## 5. 局限与争议

1. **三个反复出现的难题**。**Safe inheritance**：持久化不等于持续收益——Gödel Agent 100 次 MGSM 试验中 14% 低于初始策略。**Autonomy attribution**：能生成更好候选不等于改进了"发现候选的方式"——DGM 的提升是真的，但 archive 维护与父代选择仍在自修改之外。**Reliable verification**：反复访问评估器会奖励钻空子——Anthropic 报告随机种子挑选与测试标签提取；RQGM 以 epoch 内冻结评估器 + 独立 anchor 应对。
2. **structural vs effective L5**：structural 只证明"AI 主导的变更被继承并支配了后续一轮改进"；effective 才要求该机制在**可比预算与独立评估**下产出更好 successor。定级只针对被考察的机制：单改任务代码而改进流程不变则证据不足，反之演化出的 successor evaluator 即使任务代码未变也可构成 structural L5。
3. **更高自治不等于更好的改进流程**：greater autonomy、durable capability gains、a more effective improvement process 是三件事。
4. **证据强度**：端到端 L5 集中在 bounded prototypes；"可比资源下统计可靠的跨代累积"仍未被建立（HyperAgents 200 轮无显著优势；AIDE2 未见显著效率优势）。工业数字多为公司自报。

## 6. 对我的价值

**先说老实话：和信号处理 / 通信原理关系不大**——verifier 落在软件测试、benchmark 与专家评审上，不涉及信道估计、检测、编码或随机过程建模。

可偷的是方法论：**① 用新的分析单位重排已有文献本身就是贡献**。这篇不提新算法，核心动作是把 491 篇论文从"改了什么"重排成"**哪些决策权从人转移给了 AI**"；研一找选题的启发是，拥挤方向上与其做增量方法，不如找新维度重组已有工作。**② structural vs effective 的区分可复用**：把"结构上做到"与"公平预算下真有效"拆开，等于预置了审稿人会问的问题；matched-budget 对照、冻结机制后做迁移测试可直接抄。**③ 构造可操作指标是低成本产出**——HCI 的构造（首年前沿作基线、按来源加权）本质是实验设计问题。

**何时读**：不是入门材料，几乎无公式。建议在看完 3b1b / 李宏毅、建立完神经网络–CNN–Transformer 直觉、开始选题时读，当领域地图而非技术教程——它让你看清哪些格子已拥挤（§4）。

## 7. 怎么读（可操作）

- **必读**：§1.1–1.4（骨架）；**图 1、图 2**（五级回路结构，最快建立直觉）；**§3.7 Cross-Level Synthesis**（相邻级边界，一页内，性价比最高）；§3.6.1–3.6.3 + §3.6.5（L5 代表系统与评测要求）；§4 图 9（四场景定位）；§6 八方向标题与每段首句；§7。
- **可跳**：§2.1 统计细节；§3.2–3.5 的系统罗列（挑一个域精读）；§4.1–4.4 系统枚举与 "How far are we" 框；§5 六个工业案例（挑 Theseus、Humanlaya，其余看图注）；参考文献；附录 A / B。
- **预计时间**：必读路线约 1.5 小时；§3 全级 + §4 全场景约 3–4 小时。
- **前置知识**：无硬门槛。需知道 LLM / agent 基本词汇（prompt、RLHF / GRPO、harness、benchmark、reward hacking）与 continual learning / AutoML 的大致概念（表 1 本身就在解释这些关系）。信号处理背景不构成障碍。
