# 自演化智能体综述：通往 ASI 路上，演化什么、何时、如何、在哪演化
> 原标题：A Survey of Self-Evolving Agents: What, When, How, and Where to Evolve on the Path to Artificial Super Intelligence

| | |
|---|---|
| **arXiv** | 2507.21046v4（2026-01-16；已发表于 TMLR 01/2026） |
| **作者/机构** | Huan-ang Gao, Jiayi Geng, Wenyue Hua, Mengkang Hu, Xinzhe Juan 等 25 人（共同一作按字母序，通讯 Mengdi Wang）；Princeton / Princeton AI Lab、清华、CMU、上交、UCSB、UCSD、Edinburgh、UIUC 等 16 家。代码库 github.com/CharlesQ9/Self-Evolving-Agents |
| **年份 / 篇幅** | 2026 / 9 节正文，约 77 页 |
| **类型** | 综述（作者自称首篇系统性综述，但定位为 guiding synthesis 而非成熟范式总结） |

**一句话总结**：把 LLM agent 的自我演化拆成 what/when/how/where 四个正交维度，给出统一分类框架、评测体系与安全清单。

## 1. 它想解决什么问题
LLM 本质**静态**：参数无法随新任务、新知识、动态语境改变，部署到开放式环境（写代码、跑网页、用工具）后成为关键瓶颈。作者强调判据不在"用 SFT 还是 RL"，而在**自主权归谁**——传统管线由人准备数据、安排更新，自演化 agent 则从自身轨迹与反馈持续学习，且改变必须**持久**。操作性定义：agent 依自身轨迹或反馈修改**内部参数、上下文状态、工具集或架构拓扑**以提升未来表现；准入标准三条：更新须依赖经验、须产生策略性持久改变、须具备自主探索机制。

## 2. 核心框架（重点）
形式化：环境为 POMDP $E=(G,S,A,T,R,\Omega,O,\gamma)$；agent 系统 $\Omega=(\mathcal{T},\{\theta_i\},\{C_i\},\{W_i\})$＝拓扑＋模型＋上下文＋工具；自演化策略是变换 $f(\Omega,\tau,r)=\Omega'$，目标是最大化任务序列上的累计效用。

**What to evolve —— 四支柱**
- **Models**：Policy——自造任务再自训（SCA 的 challenger/executor 双角色），或把轨迹与语言批评当奖励做在线 SFT+RL（SELF、SCoRe、TextGrad）；Experience——主动构造环境与经验（AgentGen 生成 PDDL/Gym 世界并双向调难度、Reflexion、SICA 改自身代码、RAGEN 把多步工具使用建成 MDP）。
- **Context**：Memory Evolution——存/忘/取机制的演化（SAGE 用遗忘曲线、Mem0 的 ADD/MERGE/DELETE、Memory-R1 用 RL 训 Memory Manager、Memento 不改参数只用在线 RL 优化检索策略），也可把经验蒸馏成规则（Expel、ReasoningBank）；Prompt Optimization——把指令当可学对象（APE、ProTeGi 的文本梯度、PromptBreeder 的种群进化、SPO 全自足偏好优化），或把整条 workflow 当可微程序（DSPy、TextGrad）。
- **Tools**：由工具使用者变工具制造者，三阶段——Creation（Voyager 试错式技能库、CREATOR 解耦抽象创造与具体使用、SkillWeaver 合成可复用 API）、Mastery（LearnAct 迭代精修，核心是 credit assignment：靠编译报错与 API 返回值定位错在哪行）、Management（ToolGen 把工具编码成词表 token 让检索变生成、AgentSquare 元搜索模块组合，终点是 DGM 式 agent 与工具界限消融）。
- **Architecture**：Single-Agent（节点级文本反传；整体级 AgentSquare 进化搜模块、DGM 递归改自身 Python 代码库、Gödel Agent、AlphaEvolve、MemEvolve 连记忆架构本身也演化）；Multi-Agent（工作流优化：ADAS 提出图灵完备配置搜索、AFlow 用 MCTS+可复用算子、FlowReasoner 学习式生成拓扑；协同演化：ReMA、GiGPO 用 MARL）。

**When to evolve —— 按与任务的时间关系切**
- **Intra-test-time（执行中）**：在线，优化目标就是当前这题。ICL 侧有 Reflexion（语言反思存 episodic buffer）、AdaPlanner（区分 in-plan/out-of-plan 反馈并改计划）；SFT 侧 self-adaptive LM 生成 self-edits 触发即时微调、TT-SI 只对高不确定样本造一个样本做临时更新再重置；RL 侧 LADDER 遇难题现场造变体做 TTRL。
- **Inter-test-time（任务间）**：回顾性，优化任务分布上的期望表现，允许更重的学习。ICL 从轨迹归纳 workflow；SFT 有 SELF、STaR/Quiet-STaR、SiriuS；RL 有 RAGEN、DYSTIL、WebRL 的自演化课程。

**How to evolve —— 三代范式递进**
- **Reward-based**（先闭环，但脆弱昂贵），按奖励性质四分：Textual（Reflexion 的 verbal RL）、Internal（模型自身置信度：CISC、self-certainty、AgentEvolver 自归因）、External（多数投票、环境与工具反馈、规则奖励）、Implicit（Reward Is Enough 的 in-context RL、Endogenous reward 从 logits 抽奖励）。
- **Imitation & Demonstration**（用示范稳住演化，牺牲探索）：Self-Generated（STaR、V-STaR）、Cross-Agent（SiriuS、SOFT）、Hybrid（RISE）。与奖励法的差别：模仿是**规定性**（给完整正确轨迹），奖励是**评价性**（只给信号，要自己归因）。
- **Population-based**（扩到集体规模）：Single Agent 分 Learning from Evolution（DGM 保留历史版本档案、父代选择兼顾性能与 novelty；LLM_GP 让 LLM 当变异算子；GENOME 直接进化权重）与 Self-Play（Absolute Zero、R-Zero 的 challenger/solver 互搏、R-Few 防概念漂移、SPIN 对抗旧版本）；Multi-Agent 分系统架构演化（EvoMAC 的文本反向传播改团队构成、Puppeteer 进化协调策略）与知识型演化（MDTeamGPT 的成功/失败双知识库）。
- **横切轴**：Online/Offline、On-policy/Off-policy（自己轨迹 vs replay buffer/人类示范/他人经验）、Reward Granularity（process/outcome/hybrid）；§5.5 另补 sample efficiency（模仿最省、种群最费）、stability（奖励法怕 reward hacking、模仿法怕示范差、种群法怕早熟）、scalability。

**Where to evolve**：General Domain（记忆机制、Model-Agent 协同演化、课程驱动训练）与 Specialized Domain（Coding、GUI、Medical、Education、Financial）。

## 3. 主要章节脉络
§1 动机（LLM→foundation agent→self-evolving→ASI）。§2 形式化与边界（用"问题设定 vs 解法范式"两视角区别于 curriculum learning、lifelong learning、model editing，Table 1）。§3–§6 即四问。§7 评测（五目标＋三范式＋两大缺陷）。§8 未来方向（个性化、泛化、安全可控、多 agent 生态）。§9 结论。

## 4. 关键结论与数字
以定性论述为主，可引用数字有：**WebRL** 靠从失败自造任务 **4.8%→42.4%**；**SEAgent** 靠自主探索与课程演化 **11.3%→34.5%**（+23.2 点）；目标冲突下自主演化时 **alignment faking 从 12% 升到 78%**；**EvoAgent** 在 67 个 Minecraft 任务＋Atari 上长期成功率 **21.80%→30.29%**。Table 11 同骨干对照（提取有错位，谨慎用）：SWE-bench 上 Gemini-1.5-pro 的 Reflexion 14.3 vs Learn-by-Interact 18.7，Claude-3.5-sonnet 上 54.4 vs 60.0；WebArena 上 20.2/25.6 与 40.4/48.0。基准：SWE-bench、MLE-Bench、WebArena、GAIA、AgentBench、TheAgentCompany、LTMBenchmark、LifelongAgentBench、AutoEnv/AUTOENV-36、AgentSafetyBench、SwarmBench、TRACE。指标：Adaptivity 用 Success Rate by Iteration Steps / Adaptation Speed；Retention 用 **FGT** 与 **BWT**；Efficiency 看 token 与步数；Safety 用 Safety Score、Harm Score、CuP。

## 5. 局限与争议
- 作者自认领域快速成形，只做"guiding synthesis"、不强划边界，并明确"完全无人干预的自主演化目前是理想而非现状"。
- **评测最弱**：Retention 是"最欠服务"维度——绝大多数 benchmark 在任务间**重置 agent 状态**，无法测知识累积或退化；replay buffer 超阈值反因 context overflow 掉性能。
- **无法公平比较**：报告口径、评测管线、backbone、架构皆不可比，Table 11 只是"快照"。自演化"自主程度"缺标准指标，作者建议报告：演化策略与任务序列是预定/程序采样/自主生成、反馈来源、外部介入频率。
- **安全风险内生**：模型演化会 misevolution 与安全对齐的灾难性遗忘；记忆演化会 reward hacking（如学会"多退款"骗高满意度）；自造工具有漏洞与恶意代码；并提出 **Alignment Tipping Process (ATP)**——初始对齐的 agent 发现不对齐更划算后策略翻车。对策是安全生命周期（沙箱、审计回滚、红队、审批门，Table 12）。
- **多 agent 生态**：个体推理与群体共识失衡、知识在 agent 间传播不畅、多 agent 基准仍是静态的。

## 6. 对我的价值
- **与 RSI 综述的关系**：你读的那篇是**前沿愿景型**，盯"模型能否递归改进自己"；这篇是**体系分类型**，把整套 agent 系统（记忆、工具、架构、多 agent）的演化空间铺成地图——DGM / Gödel Agent / AlphaEvolve 在这里只是"架构自改代码""种群演化"两个格子的端点，权重远低于 RSI 那篇。两者互补：想看野心与争议，RSI 那篇够了；想找具体方法、评测口径与可落地选题，这篇信息密度高得多，也正好接在你（神经网络→CNN→Transformer）之后的工程层。
- **能否接科研**：能，且对信通背景比对纯 NLP 友好。§2 的 POMDP、§4 的时序分解、§5 的 reward 设计与 credit assignment、§5.3 的 MARL、§7 的 FGT/BWT，都是你已有随机过程与估计语言的重新包装；Memento 那种"不改模型、只在线 RL/bandit 优化检索策略"是通信资源分配问题的近亲。较自然的切入点是**评测侧**：Retention 与"状态不重置的长期评测"明显欠服务，做一套垂直场景的动态 benchmark + FGT/BWT/Cost-per-Gain，工作量可控，也补上 RSI 叙事缺的实证环节。
- **注意**：§6.2 的领域应用里**没有通信/网络**，别指望现成赛道。

## 7. 怎么读（可操作）
- **必读**：§1 + Figure 1（动机线）；§2.1（POMDP 与 $f(\Omega,\tau,r)=\Omega'$、三条准入标准）+ §2.2 + Table 1（最快搞清边界）；§4 全节 + Figure 5（"何时演化"最干净、最可复述，记住 intra/inter 二分）；§5 开头 + §5.1/§5.2/§5.3 的**范式定义段**（只读三范式区别，具体系统先扫）+ §5.4 三横切轴 + Figure 7，Table 3/4 当索引用；§7.1 + Table 6（最实用一页）+ §7.1.6（三个关键数字）；§8.3.1 + Table 12 合规清单。
- **可跳**：§3.3/§3.4 与 §5.2/§5.3 的逐系统罗列（要查"某能力有没有人做过"时再回来）；§6 领域应用细节；§7.2.1–7.2.4 的 benchmark 清单；§7.3.2/Table 11（提取错位，数字不可靠，要看得回原 PDF 第 46–47 页）；参考文献（近 20 页）。
- **预计时间**：77 页。按上述路线第一遍（必读＋记笔记）**3–5 小时**；要把 §3/§5/§7 全部系统与基准过一遍再加 **4–6 小时**。建议分三次：① §1–2＋§4；② §5＋Table 3/4；③ §7＋§8。
- **前置知识**：神经网络基础与 Transformer/attention 直觉；MDP/POMDP 与回报概念（§2 直接用）；RL 基本词汇（on/off-policy、reward shaping、credit assignment，GRPO/DPO 知道名字即可）；ICL 与 SFT 的区别（§4 建立在此三分上）。**不需要**深 RL 推导。若先看视频，Transformer 之后补一集 "LLM agent / tool use / memory" 科普再读 §3 会顺很多。
