# 大语言模型综述
> 原标题：A Survey of Large Language Models

| | |
|---|---|
| **arXiv** | 2303.18223（v19，最新更新 2026 年 3 月；初版 2023-03-31） |
| **作者/机构** | Wayne Xin Zhao、Kun Zhou、Junyi Li 等 22 人；中国人民大学高瓴人工智能学院 / 信息学院，Jian-Yun Nie（蒙特利尔大学） |
| **年份 / 篇幅** | 2023 首发，持续更新至 2026；对应 PDF 约 144 页 |
| **类型** | 综述（survey），配套 GitHub：RUCAIBox/LLMSurvey |

**一句话总结**：把 LLM 拆成「预训练 → 后训练适配 → 使用 → 能力评测」四段主线，逐段梳理数据、架构、训练与评测的主流方法与关键数字。

## 1. 它想解决什么问题

语言模型经历四代演进：统计语言模型（n-gram）→ 神经语言模型（word2vec、ELMo）→ 预训练语言模型（BERT/GPT）→ LLM，核心变化是任务求解能力扩张为"通用任务求解器"。痛点是 PLM 已被反复综述而 LLM 缺乏系统回顾；且二者有三点本质差异：(1) LLM 出现 emergent abilities；(2) 主要靠 prompting 接口使用而非改参数；(3) 研究与工程边界模糊——分布式训练、数据清洗本身就是核心贡献。此外 LLM 为何有效、如何对齐人类价值，都还没有讲清楚。

## 2. 核心框架（重点）

全书骨架是一条**四段式流水线**（第 4–7 章），另有两个解释工具（2.1 节）：

**两个理论工具**
- **Scaling law**：KM law（Kaplan 2020）给出 L(N)∝N^-0.076、L(D)∝D^-0.095、L(C)∝C^-0.050；Chinchilla（Hoffmann 2022，E=1.69, A=406.4, B=410.7, α=0.34, β=0.28）主张算力固定时模型与数据应**等比例放大**——同算力下 70B 的 Chinchilla 胜过 280B 的 Gopher。另区分 predictable scaling（小模型预测大模型）与 inverse scaling（loss 降但任务变差）。
- **Emergent abilities**：小模型没有、大模型才出现的能力，分三类——in-context learning、instruction following、step-by-step reasoning（CoT），各有具体门槛。

**四段流水线**
1. **Pre-training（第 4 章）**：数据 → 架构 → 训练。
2. **Post-training（第 5 章）**：instruction tuning + RLHF 对齐 + 参数高效适配（Adapter / Prefix Tuning / Prompt Tuning / LoRA）。
3. **Utilization（第 6 章）**：prompting → ICL → CoT → planning（分解、生成计划、获取反馈）。
4. **Capacity evaluation（第 7 章）**：basic ability（语言生成/知识利用/复杂推理）+ advanced ability（人类对齐/环境交互/工具操纵）+ benchmark。

**架构维度**（4.2.1）：encoder-decoder（T5/BART，LLM 中很少）、causal decoder（GPT 系主流）、prefix decoder（GLM-130B）；可用 MoE 稀疏扩展；非 Transformer 路线为 SSM 系（Mamba、RetNet、RWKV）。

## 3. 主要章节脉络

- **第 1–2 章**：上面的背景 + 五项关键技术 + GPT 系演进（GPT-1 确立 decoder-only + 预训练微调 → GPT-2 无监督多任务 → GPT-3 涌现 ICL → InstructGPT/ChatGPT 走 RLHF）。
- **第 3 章 资源**：公开 checkpoint 与 API、预训练语料、微调数据集、训练与推理库（DeepSpeed、Megatron-LM、vLLM）。
- **第 4 章 预训练**：数据（来源 + 四类过滤）、架构、预训练任务（prefix LM、FIM、MoE 负载均衡）、训练（3D 并行、ZeRO、重启动绕过 loss spike）。
- **第 5 章 后训练**：指令数据三法（格式化 NLP 数据集 / 真实用户对话 / 合成数据，Self-Instruct 仅需 175 条种子，Orca 扩到 5M 条）；RLHF 三组件（LM + reward model + PPO，含 KL 惩罚）；PEFT 四法。
- **第 6 章 使用**：prompt 设计原则；ICL 三要素（示例选择、格式、顺序）与机理假说；CoT；planning。
- **第 7 章 评测**：能力分层（语言生成、代码合成、问答、知识推理、人类对齐、环境交互、工具操纵）+ 数据集表 + 实测。
- **第 8 章 应用**：研究社区（NLP、IR、推荐、多模态）与行业（医疗、金融、科研、法律、教育）。
- **第 9 章 进阶专题**：长上下文、LLM Agent、训练/推理优化、模型压缩、RAG、幻觉、long CoT。
- **第 10 章 结论与未来方向**。

## 4. 关键结论与数字

- LLM 参数门槛：本书取**大于 10B**（无公认标准）。
- 涌现能力阈值：GPT-3 的 ICL 在 **13B** 上于算术任务出现（但 175B 仍做不好波斯语 QA）；instruction following 在 LaMDA-PT 达 **68B** 才显著超过未调优版本；CoT 对 PaLM/LaMDA 需 **>60B** 才有效；PaLM 在 MMLU、BBH、TyDiQA、MGSM 四项上至少需 **62B**。
- ChatGPT 发布后，arXiv 中含 "large language model" 的论文从 **0.40 篇/天涨到 8.58 篇/天**。
- 模型规模：GPT-3 175B / 300B tokens，PaLM 540B，LLaMA-3.1 405B / 15T tokens，Qwen2.5 达 18T tokens；BLOOM 覆盖 46 种语言。
- 实测（Table 16）：GSM8k 上 ChatGPT **78.47** vs Davinci002 49.96；HumanEval 上 ChatGPT **79.88** vs Claude 51.22；HotpotQA 上 Claude 2 **36.40** vs Davinci002 26.00。结论：开源小模型在数学推理、环境交互、工具操纵上普遍不佳。
- 代码：LLM 在 Codeforces 上进入**用户前 28%**；评测用 pass@k。
- 长上下文：LLaMA-2-7B/13B 仅用**数十亿 token** 长文本继续预训练即可达 100K+ 有效上下文。
- RLHF：InstructGPT 用 175B GPT-3 + **6B** reward model；LLaMA 2 迭代了 **5 个版本**。

## 5. 局限与争议

- **涌现能力存疑**：书中记录争论——有观点认为涌现源于评测指标不连续（换连续指标后曲线变平滑），作者则反驳用户对成败的感知本就不连续。
- **原理不清**：为什么涌现、为什么好 prompt 有效、ICL/CoT 的机理，都缺乏原理解释。
- **不可复现**：LLM 主要由工业界训练，数据清洗细节未公开，算力昂贵难做系统消融。
- **架构单一**：现有工作几乎只押 decoder-only Transformer。
- **长上下文未解决**：即使 128K/200K 窗口，模型仍"不能很好处理窗口内信息"。
- **RLHF 成本高**：依赖专业标注员，昂贵难规模化，已有工作用 LLM 辅助标注（RLAIF）缓解。
- **安全**：幻觉是概率建模本质的固有倾向；也可能被诱导生成有害、偏见内容。
- 作者在 CODA 中自认综述"远非完美，很可能遗漏重要文献"。

## 6. 对我的价值

**和信号/通信的接口是现成的**：2.1 节把 scaling law 写成幂律拟合，loss 以 nats 计并分解为 irreducible loss + reducible loss（KL 散度），评测用 perplexity——与你学过的信息论、随机过程中的熵/互信息/率失真是同一套数学；Chinchilla 的"算力固定时模型与数据等比例放大"，就是"给定预算下的最优资源分配"（功率-带宽折中）。RoPE 位置编码的基频缩放/截断（9.1 节）本质是频域波长设计；推理阶段的 memory-bound vs compute-bound 分析（9.4 节）是能接的工程侧。

**能不能接科研**：能，但要选交叉口。最顺的是 **semantic communication / 通信网络智能化**（LLM 做语义编解码、协议理解、网络运维 agent，9.2 节的 memory–planning–execution 框架可直接套）、**RAG 用于通信标准与论文库**（9.6 节）、**LLM Agent 做资源调度**。纯 NLP 不划算；第 7、9 章偏"评测"与"推理优化"的门槛低于预训练，适合研一上手。

**该在哪个阶段读**：不是入门第一本。按你的路径（神经网络 → CNN → Transformer），应在**看完 Transformer 视频、能讲清 self-attention 的 QKV 与 multi-head 之后**再读，当"地图"用——它不教你写 attention，但告诉你这之后领域长什么样。第二遍跑过 LoRA 微调后再读第 5、9 章。

## 7. 怎么读（可操作）

- **必读**：2.1 节（scaling law + emergent abilities，全书理论地基）；4.2.1（三类架构，配合 Table 5 的超参看）；5.1 与 5.2.3（指令微调数据构造 + RLHF 三组件）；6.2 ICL 与 6.3 CoT（最常用的两个技术）；第 10 章。
- **选读**：第 3 章当目录查（选模型/库时翻 Table 1）；7.4.2 的实测结论；Table 14 备查；第 8 章按行业检索。
- **可跳**：4.1 的数据预处理流水线（Pile/CommonCrawl）、4.3 的分布式训练实现、9.3 的训练显存估算、9.5 量化的各位宽推导——除非你后面真去训或部署。
- **预计时间**：必读约 6–8 小时（2.1 节独自 2 小时，配 Kaplan/Chinchilla 原论文读）；全书通读 20 小时以上，不建议。
- **前置知识**：Transformer（self-attention、multi-head、decoder-only 与 encoder-decoder 的区别）、交叉熵与 KL 散度、梯度下降。不需要信号处理之外的新数学；有 RL 基础（PPO、policy gradient）对 5.2 节帮助很大。
