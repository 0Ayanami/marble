# Introduction
**Byzantine risks in MAS**
malicious agents & behaviour:
- memory poison, prompt injection
- long horizon attacks
- contradicting or colluding agents in MAS 
**Challeges facing popular MAS collaboration schemes**
Current Multi-agent Discussion(MAD), Role-based methods:
- MADs lack of memory management and attack resistant mechanism
- Role-based methods have single-point failure risk
**Inspired by CP-WBFT(AAAI26), *Consensus-based* collaboration in MAS has potential**
- Shortcoming of CP-WBFT: relies on self-reported confidence score, which is susceptible to adversarial manipulation, making it impractical in real-world MAS scenarios
- Need more theoratical analysis and upgraded adaptation in MAS
**Our contribution**
- We propose xxx, a novel consensus based collaboration framework in MAS, with high Byzantine roubustness
- We propose Smart Quorum, a dynamic, pluggable protocol based on agent voting weights, which enable swift convergence in various consensus schemes
- We improve fault tolerance limit in MAS, f/n compared to classic 1/3 byzantine limit in distributed systems
- Our approach has better performance in MAS tasks than MAD&Role-based methods, and is more resistant to attacks, even more complicated long-horizon ones 

# Threat Model
we consider $f$ Byzantine agents in a $n$ agents system ($f<n$)
**Adversaries' goal**: to manipulate the multi-agent system to derail the completion of benign tasks or trigger malicious actions aligned with their intent. The attacker can achieve this by influencing other agents through deliberate interactions, or exploiting specialized role assumptions, thereby causing system-wide malicious effects.
**Adversaries'  knowledge**: the attackers know the roles and tools accessible to individual agents and the communication topology in MAS, which enable them to chose proper strategy to achieve their goals.
**Adversaries'  ability**: the attackers may: (i) inject malicious content at the prompt or environment level, (ii) compromise one or more agents via adversarial system prompts, or (iii) add tools with malicious intent into the agent’s toolkit. These capabilities enable attacks across three surfaces in the multi-agent system: prompt-level, agent-level, and environment-level.
attacks can be single-step or multi-step, even long-horizon with deliberately designed setup.

# Methodology
**核心思路**：设计memory poposal作为多agent记忆交互、验证、记录的标准格式，同时基于agent智能设计配套的真实性、合理性、价值性以及是否为恶意的验证算法；在此基础上针对传统分布式系统中的经典共识过程进行优化改进，在agent投票权重的动态评估，以及共识达成的最低投票条件（Quorum ）方面做出优化设计；预期实现效果包括多智能体系统对各类攻击的有效抵抗，协作任务的以及对传统经典算法的效能提升。
## Memory proposal and verification
### Memory proposal

受ReAct模式启发，agent在完成了完整的一轮类react循环后，参照以下格式区分Header、Body、Verification三个模块，整合记录关键信息并向MAS提交memory proposal请求：
- **Header**：proposal id, task id, timestamp, proposing agent signature, parent proposal list, message body hash, proposal summary
- **Body**（以下字段按实际情况填写，部分内容可以留空）:
  - Thoughts：thoughts abstract(思考路径关键信息摘要), key decision points & decision results(涉及到的主要决策点信息和决策结果摘要)
  - Action：action list(执行的操作列表，如action_1: api function call...; action_2: web sesearch with keywords...; action_3: interaction with agent...)
  - Data：data list(任务相关的关键信息/数据列表，agent本地检索到的提供关键信息摘要，公开渠道获取的提供关键信息摘要和访问链接等)
  - Observations：result list(当前取得的主要结果或观测情况列表，如result_1: complete subtask_i...; result_2: fetched data from url...)
- **Verification**：self verification(记录agent提交proposal时自检的各维度评分)，multi-agent verification(共识过程结束后，多agent检验的各维度评分加权平均值)，consensus result(参与投票agent数量/总agent数量，得票权重/总权重，共识结果：pass/fail)
### Proposal verification
1）某memory proposal提出后，分布式系统中的其他agent针对该proposal进行基于大模型的多维度验证。设计四个验证维度，每个验证维度包含若干验证要点，每个维度只需模型判定“是/否”结果，并用1/0二元指标标记，由此形成verification vector。
- Veracity 真实性验证：主要面向Data和Observation字段，针对给出的事实性信息进行真实性、准确性判断。全部条件通过记为1，否则0。
- Rationality 合理性验证：主要面向Thoughts和Action字段，针对思维链决策链判断合理性，动作执行（工具选择、执行逻辑）合理性等。全部条件通过记为1，否则0。
- Value 价值性验证：主要面向Data和Observation字段，判断与主线任务是否相关，对其他agent工作是否有支撑作用、是否包含有价值的新信息而非旧情况的简单重复等。全部条件通过记为1，否则0。
- Security 安全性验证：对几个字段进行综合判定，判断是否出现了常见的拜占庭模式（poison、injection、hallucination等）。无显著安全风险记为1，否则0。
2）针对verification vector设计每个验证维度的归一化权重比例，形成权重向量，适配于不同的场景需求，如强调真实性安全性，或者强调agent行为合理性价值性等，可以选取不同的权重比例组合；最终对所验证的memory proposal加权整合单一proposal confidence分数（0~1）；通过设定系统级confidence阈值，收到请求的agent判定当前proposal能否通过验证。
**不同的维度权重和不同的阈值设定，作为可调节的参数，在后续实验中验证对整体系统的BFT能力影响。**
## Voting Weight Assignment for Agents 

1）从两个维度量化定义参与agent的**质量/行为指标**：
- **vc(Verified Confidence)**: agent以往发起的memory proposal质量。量化方式为前文所述proposal confidence的平均分数，取值范围（0,1）
- **hc(Historical Confidence)**: agent以往参与consensus的投票质量。量化方式为当前agent参与投票的意见和最终共识结果一致的比例，取值范围（0,1）
*假设*：高质量的proposal和voting更有可能来自于诚实的agent，反之则为拜占庭参与者

2）**拜占庭分类器(质量函数)** 构建
对VC和HC两个指标分别设定权重系数$\alpha,\beta$，构造线性质量函数： $$q=\alpha\cdot vc+\beta\cdot hc$$
对于系数取值的确定，**两种处理思路**：
- 采用预设值/经验值，如设定$\alpha=0.5,\beta=0.5$，或$\alpha=0.6,\beta=0.4$ 等若干组，作为可调节参数，通过实验验证BFT能力影响，并确定最佳取值。
- 采用基于线性判别分析(Fisher)的方法计算：取一定数量的MAS共识过程训练样本（采用相关dataset/benchmark模拟已知的拜占庭agent行为），通过最大化类间差距且最小化类内差距的方式推算最优$\alpha,\beta$ 取值。
基于质量函数和前述假设，$0\leq q\leq 1$，设定质量阈值$\theta$，可以构造拜占庭agent分类器（大于等于阈值为诚实，反之为恶意）。

3）**投票权重放大器(权重函数)** 构建
考虑到agent所用支撑LLM的能力对投票质量形成乘数效应，同时最大化诚实和恶意agent投票权重之比。定义**模型能力系数$\lambda$**: agent本身判断力、分析能力、认知能力。量化方式为agent所用支撑LLM在各benchmark上的评分加权平均值，取值范围\[1,10]；定义**权重放大系数**$\gamma$，与上述$\theta$ 阈值一起形成指数级放大效应。
选取指数形式为放大函数，通过放大系数作用，得到agent投票权重：$$w=\lambda\cdot e^{\gamma(q-\theta)}$$
对于筛选为拜占庭agent的能力系数，**两种处理思路**：
- 可以考虑直接取最小值$\lambda=1$
- 不更改$\lambda$ 取值，主要是考虑到质量函数筛选存在误差，且agent并非一直进行作恶行为，在某些交互中也可以提供诚实的投票反馈。

进一步得到诚实和恶意agent投票权重平均值之比： $$\rho=\frac{\bar w_H}{\bar w_F}$$
 以及拜占庭节点容错的理论上界：$$f < \frac{n \cdot \rho}{\rho + 2}$$
## Smart Quorum & Decision Mechanism
基于前述理论分析，在具有不同投票权重的多agent参与的共识过程中，满足共识要求的最少收集选票数($qc>2n/3$)转换为收集到的最少投票权重值之和($qc^*$)要求，推荐取 Quorum 为安全下界的略上方：
$$qc^* = \frac{W_H + 2W_F}{2} + \epsilon$$
以此替换经典共识机制（如hotstuff、pbft、raft）中每一轮次投票的QC条件。
收集到满足要求的$qc^*$之后，可采用**权重的简单多数原则**，来决定本轮投票/共识的结果。

# Evaluation
## Experiment setup
### Baseline methods
**MAD**: 
- **Autoagent/CrewAI** Groupchat
- TBD

**Role-based**:
- **Autoagent/CrewAI** Maganetic one
- TBD

**CP-WBFT**: self-reported confidence set to 1

**CP-WBFT+MAD**:

**CP-WBFT+Role**:

### Proposed method
**MAD + Consensus**:

**Role + Consensus**:

### Benchmark & dataset
**MARBLE** to evaluate collaboration, tasks:
- **research** - agents with complementary research profiles co-author a new proposal on a chosen topic
- **minecraft** - agents to collaboratively construct structures in a shared environment
- **database error analysis** - five agents, each specializing in diagnosing a distinct root cause of system inconsistencies
- **coding challenge** - collective problem-solving and software module development

**TAMAS** to evaluate resistance to byzantine behaviours in MAS:
*promot based*
- **Dricet Prompt Injection** - explicitly modifying the user query with a malicious instruction
- **Impersonation** - modifies the user query by appending a statement that falsely attributes the request to a trusted or authoritative figure
*environment based*
- **Indirect Prompt Injection** - introducing adversarial content into the environment or intermediary observations
*agent based*
- **Single Agent Compromise** - Byzantine agent directly produces inconsistent, or nonsensical outputs.
- **Colluding Agents** - one or more agents within the multi-agent system are adversarial and deliberately coordinate to manipulate the system’s behavior toward an outcome desired by the attacker
- **Contradicting Agents** - a subset of agents which have similar functionalities, intentionally provide conflicting or misleading information to disrupt the overall system performance

**AgentLAB** to evaluate robustness of resisting long-horizon attacks in MAS
- **Intent hijacking** exploits multi-turn user-agent interactions to progressively erode the agent’s safety guardrails and deceive it into executing the malicious task $T^∗$
- **Tool chaining** exploits the fact that malicious tasks can often be accomplished by composing individually benign tool calls
- **Objective drifting** exploits the susceptibility of LLM agents to gradual environmental influence over extended interactions
- **Task injection** extends conventional indirect prompt injection to the longhorizon setting, causing the agent to execute the malicious task T ∗ alongside the benign task T through multi-turn injections into the agent’s observations
- **Memory poisoning** targets agents augmented with external memory

### Experiment groups
#### LLM backbone: 
 - **weak byzantine** vs **strong honest** agents
 - **weak byzantine** vs **weak honest** agents
 - **strong byzantine** vs **strong honest** agents
 - **strong byzantine** vs **weak honest** agents
 - **different** LLM for different agents
#### f in n:
- n=5, f=0,1,2,3,4
- n=7, f=0,1,2,3,4,5,6

### Ablation study
- Without proposal verification
- Without weight assignment
- Without smart quorum

### Metrics
#### Effectiveness
- **KPI(ratio of accomplished milestones $n_j$ to the total number of $M$)** in MARBLE
- **PNA(Performance under No Attack)** in TAMAS, AgentLAB
- **ASR(Attack Success Rate)** in TAMAS, AgentLAB
- **f-tolerance** in n agent setting
- **Improvement** to baselines
#### Efficiency
- task completion **time** / extra time cost
- interaction / message **count** /extra msg count
- **QC** evolvement 
- **voting agents count** evolvement


# Appendix

## Consensus in Multi-agent System: Theoretical Analysis

完整分析参考[[smart_quorum_theoretical_analysis]]
引出后续两个关键问题：
1）如何为MAS中的不同agent设定合理的投票权重，以最大化诚实和恶意agent质量平均值之比 $\rho=\frac{w_H}{w_F}$， 并动态调整
2）如何基于非均衡、动态的权重值，优化经典共识过程
这两个问题分别对应到后两个小节内容
## Hyper-parameters analysis
文中涉及的参数设定影响分析
## Proof of Our Consensus
正确性
活性
## Prompts
### MAS setup (environment, task, tool, collaboration, etc..)
### Memory proposal abstraction (milstones, output format..)
### Proposal verification & voting (rating score..)
