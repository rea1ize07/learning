# 自演化 AI Agent 综述：打通 foundation model 与终身 agentic 系统的新范式

> 原标题：A Comprehensive Survey of Self-Evolving AI Agents: A New Paradigm Bridging Foundation Models and Lifelong Agentic Systems

| | |
|---|---|
| **arXiv** | 2508.07407v2（2025-08-31） |
| **作者/机构** | Jinyuan Fang、Yanwen Peng、Xi Zhang 等 15 人，Zaiqiao Meng 通讯；University of Glasgow、Sheffield、MBZUAI、NUS、Cambridge、UCL、Aberdeen、Leiden |
| **年份 / 篇幅** | 2025 年 8 月；正文 9 节，原 PDF 约 55 页 |
| **类型** | 综述（survey），分类学导向 |
| **配套仓库** | https://github.com/EvoAgentX/Awesome-Self-Evolving-Agents |

**一句话总结**：把"部署后自动变好的 agent"拆成四组件反馈闭环，逐组件盘点 2023–2025 年的优化方法。

## 1. 它想解决什么问题

LLM agent（单/多）已能做代码生成、科研、网页导航，但**几乎都依赖人工配置**：prompt、工具链、协作拓扑写死，部署即静止。真实环境却在变——用户意图偏移、工具更新、新算法涌现，人工重配既慢又难扩展。作者据此提出 **self-evolving AI agents** 范式：与环境交互、持续优化内部组件，在保证安全的前提下适应变化。定位上，现有 agent 综述只覆盖架构或单点能力（planning、memory、collaboration、evaluation），**"自我演化"主线尚未被系统覆盖**。

## 2. 核心框架（重点）

**三定律 + 四范式 + 四组件是全文骨架。**

**Three Laws of Self-Evolving AI Agents**（层级约束，后者不得违反前者）：I. **Endure**——任何修改都必须保持安全稳定；II. **Excel**——不得降低已有任务性能；III. **Evolve**——在前两条之下自主优化内部组件。

**范式演进四阶段**：**MOP**（静态语料预训练后冻结）→ **MOA**（SFT / LoRA / RLHF 等部署后适配）→ **MAO**（多 agent 消息交换与 debate，但**不改模型参数**）→ **MASE**（基于环境反馈与 meta-reward，终身演化 prompt、memory、工具策略乃至交互拓扑）。

**统一框架（§3）**：把自演化抽象为迭代闭环，形式化为 `A* = arg max_{A∈S} O(A; I)`，含四组件：

1. **System Inputs（I）**——任务设定。*Task-Level*：`I = {T, D_train}`（任务描述 + 训练集），无标注时用 LLM 合成代理数据；*Instance-Level*：更细，`I = {x, y, C}`。
2. **Agent System（A）**——被优化对象，可单可多。含 backbone LLM、prompt、memory、tool、拓扑；多数工作只优化其中一个，近期出现多组件联合优化。
3. **Environment**——提供运行上下文并**产生反馈**。场景各异（代码任务是编译器与测试用例，科研是文献库）；反馈即 accuracy、F1、success rate；无 ground truth 时用 **LLM-based evaluator** 给 proxy metric 或文本反馈。
4. **Optimisers（P）**——闭环核心，由 **search space（S）**（能探索哪些配置：prompt、工具选择，直到连续 LLM 参数与架构结构）× **optimisation algorithm（H）**（rule-based heuristics、gradient descent、Bayesian、MCTS、RL、evolutionary）共同定义；它还会反向合成训练样本扩充 System Inputs。作者点名 **EvoAgentX** 是首个落地该框架的开源系统。

**可演化组件（§4–5）：**

- **Foundation model**：训练式（STaR、DeepSeek-R1、Absolute Zero）或 test-time（best-of-N、Tree-of-Thoughts、process reward model）。
- **Prompt**：edit-based（GRIPS）、generative（OPRO、MIPRO 用 Bayesian）、text gradient（TextGrad 把文本反馈当"自动微分"）、evolutionary（EvoPrompt）。
- **Memory**：短时（摘要压缩、MemoryBank 借遗忘曲线、Reflexion）；长时（RAG、Mem0、HippoRAG、A-MEM）。
- **Tools**：训练式（ToolLLM、ReTool）、推理时（EASYTOOL、ToolChain）、**工具创建**（CREATOR、Alita）。
- **Workflow / topology**：code-level（AFlow 用 MCTS、ScoreFlow 用 Score-DPO 做连续空间梯度优化）；communication-graph（GPTSwarm 松弛连边成连续概率、G-Designer 用 VGAE 生成图、AgentPrune/AGP 剪枝省 token）；unified（ADAS 把三者全写成 Python 代码共同演化、EvoAgent 演化文本配置、MaAS 学 agentic supernet）。
- **Inter-agent communication**：结构化输出（JSON/XML）vs 自然语言（歧义、低效）；协议 **A2A**（水平、点对点）、**MCP**（垂直、agent↔工具）、**ANP**（带 DID 的去中心化）、**Agora**（元协议）。

## 3. 主要章节脉络

- §1–§2 痛点与四范式、贡献；单/多 agent 基础与 lifelong 愿景
- §3 概念框架（四组件）
- §4 单 agent 优化：LLM behaviour / prompt / memory / tool
- §5 多 agent 优化：手动工作流 → prompt / topology / unified / backbone
- §6 领域优化（生物医学、编程、金融法律）
- §7 评测（benchmark、LLM-as-a-Judge、Agent-as-a-Judge）+ §7.3 安全对齐
- §8 挑战（按三定律）+ 未来方向；§9 结论

## 4. 关键结论与数字

以定性论述为主，**定量结果很少**。少数具体数字：OPTIMA 在信息交换密集任务上取得 **2.8× 性能提升且 token 成本不到 10%**；平行工作 Gao et al. (2025b) 用 what/when/how to evolve 三维度组织，作者强调自身优势在"统一概念框架"。作者明确表态：当前系统**远未达到**安全、鲁棒、开放式自演化的能力，该范式目前只是被 agent evolution/optimisation 技术部分逼近的长期目标。系统名密度高，可当方法索引（Absolute Zero、TextGrad、AFlow、MaAS、AgentHarm、SWE-bench 等）。

## 5. 局限与争议

作者自己承认的（§7.3 + §8）：

- **Endure**：多数流程把任务指标置于安全之上，忽视非预期行为与隐私；演化中的 agent 与假设"模型静态"的法规（EU AI Act、GDPR）不兼容，需要 evolution-aware 审计与可证明安全的 sandbox。reward model 存在数据稀缺与反馈不一致，导致**优化不稳定甚至发散**。
- **Excel**：医学、法律等领域无可靠 ground truth，难构造可信反馈；大规模 MAS 的算力/延迟与性能 trade-off 未解；**优化出的 prompt 与 topology 迁移性差**。
- **Evolve**：优化算法基本 text-only，缺多模态与空间环境下的世界模型；工具多被当作固定集合，缺少自主发现与"与 agent 共演化"。
- **评测**：安全评测多是 **snapshot-based**（单时间点），而 MASE 需要生命周期级 longitudinal benchmark，作者称其"开放且紧迫"。

## 6. 对我的价值

**与你读过的 RSI 综述的关系**：那篇《Toward Genuine Recursive Self-Improvement》（arXiv 2609.11873，2026-09）更宏观前沿——提出 HCI 指标，把 RSI 拆成"改进-执行自主 → 改进策略自主 → 经验获取自主 → 环境适应自主 → 递归元改进"五级路线图；它回答"**AI 能自我改进到什么程度、路径是什么**"。本文（2025-08）回答"**今天已有哪些能跑的具体技术**"。关系是：**RSI 给目标和分级，本文给零件和工具箱**（本文 §3.5 Optimiser / §4.3 memory / §3.4 Environment 分别对应其"改进策略自主""经验获取自主""环境适应自主"）。**建议读法**：你已读过 RSI，把本文当零件清单回头查即可，不必顺读。

**Awesome-Self-Evolving-Agents 值得跟吗**：值得。它是本文配套仓库（EvoAgentX 组织下，与文中点名的首个开源框架同源），比 PDF 更保鲜。

**能不能接科研**：能，与信通背景有天然接口。(1) **inter-agent communication 本身就是通信问题**——结构化 vs 自然语言的 message format、token 开销 vs 信息保真度就是**带宽-失真权衡**，OPTIMA 的"2.8× / <10% token"正是一个可优化的率失真目标，且尚无系统的信息论刻画。(2) **Optimiser 的 search space × algorithm 就是优化/估计问题**，MCTS/RL/evolutionary 的取舍可用随机过程直觉直接读。(3) **优化结果迁移性差**（§8.1.2）是明牌开放问题，等价于"跨任务泛化"，易做出可量化课题。低成本切入点：挑 §7 某个 benchmark，复现一类通信/拓扑优化方法，做效率-效果联合评测。

## 7. 怎么读（可操作）

- **必读**：**§3 全节（3.1–3.5）**——四组件框架，全文最有复述价值，配 Figure 3 一遍就懂；**§1 的三定律 + 四范式**（约 2 页，建立坐标系）；**§8 全节**（约 2 页，等于选题指南）；**§7.3**（短而密）。有余力再加 **§5.2 topology 与 unified optimisation**。
- **可跳**：§2（属前置常识）；§4.1 训练式细节；§6 全部（都是 §4–5 的迁移，看小标题即可）；参考文献。
- **预计时间**：按上述"必读 + 重点"约 **3–4 小时**；只求框架感（§1+§3+§8）**约 1 小时**；全文精读 8–10 小时，不建议。
- **前置知识**：只需"LLM agent 有 prompt / memory / tools / multi-agent 这套组件"的直觉，**不要求**懂 RL 细节；§5 的 MCTS、DPO、GRPO 等名词不认识可先跳过。你的信号处理与随机过程基础在 §3.5（search space 与 algorithm）和 §4.1.2（test-time scaling）会派上用场。
