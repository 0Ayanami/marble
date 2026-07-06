# Introduction
**Byzantine risks in MAS**
malicious agents & behaviour:
- memory poison, prompt injection
- long horizon attacks
- other contradicting or colluding attacks in MAS
**Challeges facing popular MAS collaboration schemes**
Current Multi-agent Discussion(MAD), Role-based methods:
- MADs lack of memory management and attack resistancy mechanism
- Role-based methods have single-point failure
**Inspired by CP-WBFT(AAAI26), *Consensus-based* collaboration in MAS has potential**
- Shortcoming of CP-WBFT: relies on self-reported confidence score, which is susceptible to adversarial manipulation, making it impractical in real-world MAS scenarios
- Need more theoratical analysis and upgraded adaptation in MAS
**Our contribution**
- We propose xxx, a novel consensus based collaboration framework in MAS, with high Byzantine roubustness
- We propose Smart Quorum, a dynamic, pluggable protocol based on agent voting weghts, which enable swift convergence in various consensus schemes
- We improve fault tolerance limit in MAS, f/n compared to classic 1/3 byzantine limit in distributed systems
- Our approach has better performance in MAS tasks than MAD&Role-based methods, and is more resistant to attacks, even more complicated long-horizon ones 

# Threat Model
we consider $f$ Byzantine agents in a $n$ agents system ($f<n$)
**Adversaries' goal**: to manipulate the multi-agent system to derail the completion of benign tasks or trigger malicious actions aligned with their intent. The attacker can achieve this by influencing other agents through deliberate interactions, or exploiting specialized role assumptions, thereby causing system-wide malicious effects.
**Adversaries'  knowledge**: the attackers know the roles and tools accessible to individual agents and the communication topology in MAS, which enable them to chose proper strategy to achieve their goals.
**Adversaries'  ability**: the attacks may: (i) inject malicious content at the prompt or environment level, (ii) compromise one or more agents via adversarial system prompts, or (iii) add tools with malicious intent into the agent’s toolkit. These capabilities enable attacks across three surfaces in the multi-agent system: prompt-level, agent-level, and environment-level.

# Methodology
## Memory proposal and verification

## Voting weight assignment for agent

## Smart Quorum and proposal consensus 


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
