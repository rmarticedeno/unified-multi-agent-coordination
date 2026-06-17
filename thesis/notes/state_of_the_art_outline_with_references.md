# State-of-the-Art Outline with References

**Thesis title:** *Unified Multi-Agent Coordination: Bridging Large Language Models and Symbolic AI for Autonomous Systems*  
**Purpose:** outline only, with citation keys and a reference list for later thesis writing.  
**Citation style:** Markdown + Pandoc-style citation keys, e.g., `[@wooldridge1995intelligent]`.

---

## 1. Framing the Research Problem

### 1.1. Reclaiming the computer-science meaning of “agent”
- Agent as autonomous computational entity situated in an environment. [@wooldridge1995intelligent; @russell2021artificial; @jennings2000agent]
- Core agent properties: autonomy, reactivity, proactiveness, social ability. [@wooldridge1995intelligent]
- Agent vs object, service, chatbot, workflow, and LLM wrapper. [@shoham1993agent; @weiss2013multiagent]

### 1.2. Thesis motivation: linguistic agents inside full agentic systems
- LLMs as linguistic reasoning, communication, and translation modules. [@brown2020language; @wei2022chain; @yao2023react]
- Agents as executable autonomous software components, not only LLM applications. [@wooldridge2009introduction; @russell2021artificial]
- Need to bridge natural-language coordination with formal MAS coordination. [@finin1994kqml; @fipa2002acl; @ehtesham2025survey]

### 1.3. Research gap
- Modern LLM-agent literature often uses “agent” loosely compared with classical MAS. [@wang2024survey; @guo2024llmmas]
- Classical MAS provides formalisms for autonomy, protocols, commitments, planning, and verification. [@wooldridge1995intelligent; @rao1995bdi; @singh1999commitments]
- Unified coordination remains open for heterogeneous symbolic agents, linguistic agents, tool agents, and autonomous system components. [@ehtesham2025survey; @tran2025multiagent]

---

## 2. Classical Foundations of Autonomous Agents

### 2.1. Rational and intelligent agents
- Rational agents and environment interaction. [@russell2021artificial]
- Physical symbol systems and symbolic agency. [@newell1976computer]
- Situatedness and embodied interaction. [@brooks1986robust]

### 2.2. Agent properties and taxonomies
- Weak and strong notions of agency. [@wooldridge1995intelligent]
- Reactive, deliberative, hybrid, and cognitive agents. [@brooks1986robust; @arkin1998behavior; @wooldridge2009introduction]
- Autonomous agents as control systems. [@maes1995artificial; @franklin1997agent]

### 2.3. Agent-oriented programming
- Agent-oriented programming as a paradigm. [@shoham1993agent]
- Agents as entities with beliefs, commitments, capabilities, and communication. [@shoham1993agent]
- Programming languages and platforms: AgentSpeak, Jason, JADE, GOAL, JaCaMo. [@rao1996agentspeak; @bordini2007programming; @bellifemine2007developing; @hindriks2009programming; @boissier2013multi]

### 2.4. BDI and practical reasoning
- Philosophical roots of intention and practical reasoning. [@bratman1987intention]
- BDI model: beliefs, desires, intentions, plans. [@rao1991modeling; @rao1995bdi]
- Procedural Reasoning System and implemented BDI architectures. [@georgeff1987procedural; @georgeff1999belief]

---

## 3. Multi-Agent Systems as Distributed Autonomous Computation

### 3.1. Distributed artificial intelligence and MAS
- Historical development of distributed AI. [@bond1988readings; @ferber1999multi]
- Multi-agent systems as societies of autonomous entities. [@weiss2013multiagent; @wooldridge2009introduction]
- Cooperation, competition, coordination, and organization. [@jennings1996coordination; @jennings2000agent]

### 3.2. Coordination theory
- Coordination as management of dependencies among activities. [@malone1994interdisciplinary]
- Centralized, decentralized, hierarchical, peer-to-peer, and market-based coordination. [@jennings1996coordination; @weiss2013multiagent]
- Coordination protocols vs coordination mechanisms vs coordination infrastructures. [@ferber1999multi; @jennings2001automated]

### 3.3. Task allocation and resource allocation
- Contract Net Protocol. [@smith1980contract]
- Auction-based and market-based allocation. [@sandholm1993implementation; @wellman1993market]
- Coalition formation and capability composition. [@shehory1998methods]

### 3.4. Shared-state and indirect coordination
- Blackboard systems. [@nii1986blackboard; @corkill1991blackboard]
- Tuple spaces and Linda-style coordination. [@gelernter1985generative]
- Stigmergy and environment-mediated coordination. [@theraulaz1999brief]

### 3.5. Distributed constraint reasoning
- Distributed Constraint Satisfaction Problems. [@yokoo1998distributed]
- Distributed Constraint Optimization Problems. [@modi2005adopt; @fioretto2018distributed]
- Relevance to multi-agent planning, scheduling, and resource coordination. [@fioretto2018distributed]

### 3.6. Social, organizational, and normative MAS
- Commitments and social semantics. [@singh1999commitments]
- Norms, institutions, and electronic organizations. [@dignum2004model; @esteva2004electronic]
- Agreement technologies: negotiation, argumentation, trust, and commitments. [@sierra1997model; @rahwan2009argumentation]

---

## 4. Agent Communication Languages and Protocols

### 4.1. Speech-act foundations
- Speech acts as theoretical basis for agent communication. [@austin1962how; @searle1969speech]
- Communicative acts, performatives, intentions, and effects. [@cohen1990intention; @fipa2002acl]

### 4.2. Classical Agent Communication Languages
- KQML. [@finin1994kqml; @labrou1997proposal]
- FIPA-ACL. [@fipa2002acl]
- Content languages, ontologies, and semantic interoperability. [@gruber1993translation; @baader2003description]

### 4.3. Interaction protocols
- Request, query, inform, propose, accept, reject. [@fipa2002acl]
- Contract Net and negotiation protocols. [@smith1980contract; @jennings2001automated]
- Protocol states, commitments, and conversation policies. [@singh1999commitments; @yolum2002flexible]

### 4.4. Transition from structured ACL to natural-language coordination
- Natural-language communication as flexible but ambiguous coordination. [@winograd1972understanding; @yao2023react]
- Need for grounding natural-language messages into symbolic communicative acts. [@gruber1993translation; @fipa2002acl; @ehtesham2025survey]

---

## 5. Symbolic AI Foundations for Coordination and Control

### 5.1. Knowledge representation and reasoning
- Logic-based AI and common-sense reasoning. [@mccarthy1969some; @reiter2001knowledge]
- Ontologies and description logics. [@gruber1993translation; @baader2003description]
- Knowledge graphs and semantic integration. [@hogan2021knowledge]

### 5.2. Symbolic planning
- STRIPS and classical planning. [@fikes1971strips; @nilsson1980principles]
- Automated planning foundations. [@ghallab2004automated]
- PDDL and planning-domain interoperability. [@mcdermott1998pddl; @fox2003pddl2]

### 5.3. Hierarchical planning and task decomposition
- HTN planning. [@erol1994htn; @nau2003shop2]
- Symbolic task decomposition for agent workflows. [@ghallab2004automated; @georgievski2015htn]

### 5.4. Rule-based and non-monotonic reasoning
- Logic programming and answer set programming. [@lifschitz2008answer; @gelfond1991classical]
- Rule engines and policy-based execution. [@reiter2001knowledge]

### 5.5. Symbolic control architectures
- Finite-state machines. [@russell2021artificial]
- Behavior trees. [@colledanchise2018behavior]
- Goal-Oriented Action Planning. [@orkin2006three]

---

## 6. Neuro-Symbolic AI and Hybrid Reasoning

### 6.1. Neuro-symbolic AI as integration paradigm
- Neural perception and language understanding plus symbolic reasoning and control. [@besold2017neural; @garcez2019neural; @hitzler2022neuro]
- Hybrid architectures for explainability, compositionality, and verification. [@garcez2019neural; @hitzler2022neuro]

### 6.2. LLM-to-symbolic translation
- Natural language to formal plans, programs, constraints, and logical forms. [@liu2023llmp; @valmeekam2023planbench]
- LLMs as translators from ambiguous human language to structured agent actions. [@yao2023react; @schick2023toolformer]

### 6.3. Symbolic-to-LLM augmentation
- External planners, solvers, knowledge graphs, and rule systems as tools for LLM agents. [@liu2023llmp; @schick2023toolformer; @mialon2023augmented]
- Retrieval-augmented and tool-augmented reasoning. [@lewis2020retrieval; @karpas2022mrkl; @mialon2023augmented]

### 6.4. Hybrid execution loop
- LLM proposes or explains. [@yao2023react]
- Symbolic planner verifies or refines. [@liu2023llmp; @valmeekam2023planbench]
- Autonomous agent executes in the environment. [@russell2021artificial; @wooldridge2009introduction]
- Runtime monitor checks constraints and safety. [@fisher2013verifying; @dennis2016practical]

---

## 7. Large Language Models as Components of Autonomous Agents

### 7.1. LLM foundations relevant to agents
- Transformer architecture. [@vaswani2017attention]
- Large-scale language modeling. [@brown2020language]
- Instruction tuning and alignment. [@ouyang2022training]

### 7.2. Reasoning and acting
- Chain-of-thought prompting. [@wei2022chain]
- ReAct: interleaving reasoning and environment actions. [@yao2023react]
- Reflexion and verbal feedback loops. [@shinn2023reflexion]

### 7.3. Tool use
- Toolformer and self-supervised tool use. [@schick2023toolformer]
- MRKL systems and modular tool composition. [@karpas2022mrkl]
- Augmented language models. [@mialon2023augmented]

### 7.4. Memory and self-improvement
- Episodic memory and reflection in generative agents. [@park2023generative]
- Skill libraries and lifelong learning. [@wang2023voyager]
- Reflection as language-level reinforcement. [@shinn2023reflexion]

### 7.5. LLMs as planners and embodied controllers
- Language models as zero-shot planners. [@huang2022language]
- Inner Monologue and embodied reasoning. [@huang2022inner]
- LLMs with external symbolic planning. [@liu2023llmp; @valmeekam2023planbench]

---

## 8. LLM-Based Multi-Agent Systems

### 8.1. LLM-MAS surveys and taxonomies
- Survey of LLM-based autonomous agents. [@wang2024survey]
- Survey of LLM-based multi-agent systems. [@guo2024llmmas]
- Survey of multi-agent collaboration mechanisms with LLMs. [@tran2025multiagent]

### 8.2. Conversational and role-based multi-agent systems
- CAMEL role-playing communicative agents. [@li2023camel]
- AutoGen multi-agent conversation framework. [@wu2023autogen]
- MetaGPT and standardized operating procedures. [@hong2023metagpt]
- ChatDev and software-development agent societies. [@qian2023communicative]

### 8.3. Generative societies and simulations
- Generative agents and social simulation. [@park2023generative]
- Emergent behaviors in LLM agent societies. [@li2023camel; @park2023generative]

### 8.4. Weaknesses of current LLM-MAS approaches
- Cascading hallucination and inconsistent communication. [@hong2023metagpt; @valmeekam2023planbench]
- Weak formal guarantees and limited protocol compliance. [@ehtesham2025survey]
- Evaluation gaps in long-horizon, heterogeneous, and safety-critical settings. [@liu2023agentbench; @zhou2023webarena; @xie2024osworld]

---

## 9. Modern Agent Interoperability Protocols

### 9.1. Tool and context interoperability
- Model Context Protocol as a standard interface between LLM applications and external tools/data. [@anthropic2024mcp; @mcp2025spec]

### 9.2. Agent-to-agent interoperability
- Agent2Agent Protocol for interoperability among independent agents. [@google2025a2a; @a2aproject2025]
- Agent Communication Protocol for framework-independent agent messaging. [@ibm2025acp]
- Agent Network Protocol for decentralized discovery and agent networking. [@chang2025anp; @anp2025]

### 9.3. Protocol comparison dimensions
- Discovery mechanism. [@ehtesham2025survey]
- Communication pattern. [@ehtesham2025survey]
- Identity and security model. [@ehtesham2025survey; @hou2025mcpsecurity]
- Task delegation and capability description. [@google2025a2a; @a2aproject2025]

### 9.4. Relationship to classical ACLs
- FIPA/KQML: semantic communicative acts. [@finin1994kqml; @fipa2002acl]
- MCP/A2A/ACP/ANP: practical web-scale interoperability for LLM agents. [@ehtesham2025survey]
- Research need: semantic bridge between natural language, protocol messages, and formal agent acts. [@singh1999commitments; @fipa2002acl]

---

## 10. Unified Coordination Architecture for Linguistic and Symbolic Agents

### 10.1. Agent abstraction layer
- Agent identity. [@wooldridge1995intelligent]
- Capabilities. [@shoham1993agent; @google2025a2a]
- Goals and tasks. [@rao1995bdi; @ghallab2004automated]
- Beliefs/world state. [@rao1991modeling; @reiter2001knowledge]
- Plans and policies. [@ghallab2004automated; @mcdermott1998pddl]
- Tools/actions. [@schick2023toolformer; @anthropic2024mcp]
- Communication interface. [@finin1994kqml; @fipa2002acl; @ehtesham2025survey]

### 10.2. Linguistic coordination layer
- Natural-language task negotiation. [@li2023camel; @wu2023autogen]
- Intent interpretation. [@brown2020language; @ouyang2022training]
- Explanation and human-agent communication. [@park2023generative; @yao2023react]
- Translation between natural language and protocol-level actions. [@ehtesham2025survey]

### 10.3. Symbolic coordination layer
- Formal state and task representation. [@reiter2001knowledge; @ghallab2004automated]
- Planning and constraint validation. [@liu2023llmp; @valmeekam2023planbench]
- Protocol state tracking. [@fipa2002acl; @singh1999commitments]
- Commitment tracking. [@singh1999commitments]
- Runtime monitoring. [@fisher2013verifying; @dennis2016practical]

### 10.4. Execution layer
- Tool execution and API invocation. [@schick2023toolformer; @anthropic2024mcp]
- Software agents, robotic agents, web agents, and service agents. [@russell2021artificial; @zhou2023webarena; @xie2024osworld]
- Feedback from environment to planner, LLM, memory, and monitor. [@yao2023react; @shinn2023reflexion]

### 10.5. Interoperability bridge
- Natural-language message → communicative act. [@austin1962how; @searle1969speech; @fipa2002acl]
- LLM-generated plan → symbolic plan. [@liu2023llmp; @valmeekam2023planbench]
- Agent capability → service/protocol descriptor. [@google2025a2a; @a2aproject2025]
- Protocol message → commitment state. [@singh1999commitments; @yolum2002flexible]

---

## 11. Safety, Verification, and Governance

### 11.1. Verification of autonomous agents
- Model checking and formal verification of agent programs. [@bordini2006model; @fisher2013verifying]
- Verification of BDI-style agents. [@dennis2016practical]
- Runtime verification and monitoring. [@fisher2013verifying; @dennis2016practical]

### 11.2. LLM-agent safety risks
- Hallucination and ungrounded action. [@brown2020language; @valmeekam2023planbench]
- Tool misuse and unsafe execution. [@schick2023toolformer; @hou2025mcpsecurity]
- Prompt injection and indirect instruction attacks. [@greshake2023not]
- Multi-agent error amplification. [@hong2023metagpt; @guo2024llmmas]

### 11.3. Governance requirements for unified agents
- Capability scoping. [@google2025a2a; @anthropic2024mcp]
- Authentication and authorization. [@hou2025mcpsecurity]
- Policy enforcement before action execution. [@fisher2013verifying]
- Auditable traces across language, protocol, symbolic reasoning, and execution. [@singh1999commitments; @fipa2002acl]

---

## 12. Evaluation and Benchmarks

### 12.1. Classical MAS evaluation dimensions
- Goal achievement. [@weiss2013multiagent]
- Coordination efficiency. [@malone1994interdisciplinary]
- Scalability. [@jennings2000agent]
- Robustness and fault tolerance. [@ferber1999multi]
- Formal correctness. [@fisher2013verifying]

### 12.2. LLM-agent evaluation dimensions
- Interactive task completion. [@liu2023agentbench]
- Long-horizon web task performance. [@zhou2023webarena]
- General assistant tool-use and reasoning. [@mialon2023gaia]
- Real computer-use task execution. [@xie2024osworld]
- Software engineering task solving. [@jimenez2024swebench]

### 12.3. Environments and benchmarks
- ALFWorld. [@shridhar2021alfworld]
- WebShop. [@yao2022webshop]
- AgentBench. [@liu2023agentbench]
- WebArena. [@zhou2023webarena]
- GAIA. [@mialon2023gaia]
- SWE-bench. [@jimenez2024swebench]
- OSWorld. [@xie2024osworld]

### 12.4. Evaluation gap for the thesis
- Need benchmarks for mixed LLM-symbolic-MAS systems. [@ehtesham2025survey; @tran2025multiagent]
- Need metrics for linguistic-to-symbolic translation fidelity. [@fipa2002acl; @liu2023llmp]
- Need metrics for protocol compliance, safe delegation, and verified execution. [@singh1999commitments; @fisher2013verifying]

---

## 13. Open Challenges and Research Opportunities

### 13.1. Conceptual challenges
- Ambiguous definition of “agent” in current LLM discourse. [@wooldridge1995intelligent; @wang2024survey]
- Boundary between agent, workflow, tool, service, and model. [@shoham1993agent; @anthropic2024mcp]

### 13.2. Technical challenges
- Grounding language into executable actions. [@winograd1972understanding; @yao2023react]
- Maintaining symbolic state consistency. [@reiter2001knowledge; @ghallab2004automated]
- Long-horizon planning and replanning. [@valmeekam2023planbench; @huang2022language]
- Reliable tool invocation. [@schick2023toolformer; @anthropic2024mcp]

### 13.3. Coordination challenges
- Heterogeneous agents with different capabilities and formalisms. [@ehtesham2025survey]
- Mixed natural-language and formal protocol communication. [@finin1994kqml; @fipa2002acl; @li2023camel]
- Dynamic task allocation and negotiation. [@smith1980contract; @sandholm1993implementation]

### 13.4. Verification challenges
- Verifying LLM-mediated decisions. [@fisher2013verifying; @valmeekam2023planbench]
- Auditing natural-language communication. [@fipa2002acl; @singh1999commitments]
- Runtime control of autonomous execution. [@dennis2016practical; @hou2025mcpsecurity]

### 13.5. Expected thesis positioning
- A unified model where LLM-based linguistic agents are embedded inside classical autonomous-agent architectures. [@wooldridge1995intelligent; @rao1995bdi; @yao2023react]
- A coordination bridge between natural-language interaction and symbolic/protocol-level execution. [@fipa2002acl; @ehtesham2025survey]
- A formal or semi-formal execution pipeline with validation, monitoring, and traceability. [@fisher2013verifying; @liu2023llmp]

---

## 14. Suggested State-of-the-Art Chapter Structure

1. Introduction: why the term “agent” must be grounded in computer science. [@wooldridge1995intelligent; @shoham1993agent]
2. Classical autonomous agents and MAS. [@wooldridge2009introduction; @weiss2013multiagent]
3. Symbolic AI for reasoning, planning, and coordination. [@reiter2001knowledge; @ghallab2004automated]
4. Communication languages and coordination protocols. [@finin1994kqml; @fipa2002acl]
5. LLM-based agents and linguistic coordination. [@yao2023react; @wang2024survey]
6. LLM-based multi-agent systems. [@li2023camel; @wu2023autogen; @hong2023metagpt; @guo2024llmmas]
7. Neuro-symbolic and hybrid architectures. [@garcez2019neural; @liu2023llmp]
8. Modern interoperability protocols. [@anthropic2024mcp; @google2025a2a; @ehtesham2025survey]
9. Safety, verification, and governance. [@fisher2013verifying; @hou2025mcpsecurity]
10. Evaluation, benchmarks, and open gaps. [@liu2023agentbench; @zhou2023webarena; @xie2024osworld]
11. Research gap and thesis contribution. [@wooldridge1995intelligent; @ehtesham2025survey]

---

# References

[@a2aproject2025] A2A Project. (2025). *Agent2Agent Protocol*. GitHub repository. https://github.com/a2aproject/A2A

[@anp2025] Agent Network Protocol Project. (2025). *Agent Network Protocol*. https://agent-network-protocol.com/

[@anthropic2024mcp] Anthropic. (2024). *Introducing the Model Context Protocol*. https://www.anthropic.com/news/model-context-protocol

[@arkin1998behavior] Arkin, R. C. (1998). *Behavior-Based Robotics*. MIT Press.

[@austin1962how] Austin, J. L. (1962). *How to Do Things with Words*. Oxford University Press.

[@baader2003description] Baader, F., Calvanese, D., McGuinness, D. L., Nardi, D., & Patel-Schneider, P. F. (Eds.). (2003). *The Description Logic Handbook: Theory, Implementation, and Applications*. Cambridge University Press.

[@bellifemine2007developing] Bellifemine, F. L., Caire, G., & Greenwood, D. (2007). *Developing Multi-Agent Systems with JADE*. Wiley.

[@besold2017neural] Besold, T. R., d’Avila Garcez, A. S., Bader, S., Bowman, H., Domingos, P., Hitzler, P., Kühnberger, K.-U., Lamb, L. C., Lowd, D., Lima, P. M. V., de Penning, L., Pinkas, G., Poon, H., & Zaverucha, G. (2017). Neural-symbolic learning and reasoning: A survey and interpretation. *arXiv:1711.03902*.

[@boissier2013multi] Boissier, O., Bordini, R. H., Hübner, J. F., Ricci, A., & Santi, A. (2013). Multi-agent oriented programming with JaCaMo. *Science of Computer Programming*, 78(6), 747–761.

[@bond1988readings] Bond, A. H., & Gasser, L. (Eds.). (1988). *Readings in Distributed Artificial Intelligence*. Morgan Kaufmann.

[@bordini2006model] Bordini, R. H., Fisher, M., Visser, W., & Wooldridge, M. (2006). Verifying multi-agent programs by model checking. *Autonomous Agents and Multi-Agent Systems*, 12, 239–256.

[@bordini2007programming] Bordini, R. H., Hübner, J. F., & Wooldridge, M. (2007). *Programming Multi-Agent Systems in AgentSpeak Using Jason*. Wiley.

[@bratman1987intention] Bratman, M. E. (1987). *Intention, Plans, and Practical Reason*. Harvard University Press.

[@brooks1986robust] Brooks, R. A. (1986). A robust layered control system for a mobile robot. *IEEE Journal on Robotics and Automation*, 2(1), 14–23.

[@brown2020language] Brown, T. B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P., Neelakantan, A., et al. (2020). Language models are few-shot learners. *Advances in Neural Information Processing Systems*, 33, 1877–1901.

[@chang2025anp] Chang, G., Lin, E., Yuan, C., Cai, R., Chen, B., Xie, X., & Zhang, Y. (2025). *Agent Network Protocol Technical White Paper*. arXiv:2508.00007.

[@cohen1990intention] Cohen, P. R., & Levesque, H. J. (1990). Intention is choice with commitment. *Artificial Intelligence*, 42(2–3), 213–261.

[@colledanchise2018behavior] Colledanchise, M., & Ögren, P. (2018). *Behavior Trees in Robotics and AI: An Introduction*. CRC Press.

[@corkill1991blackboard] Corkill, D. D. (1991). Blackboard systems. *AI Expert*, 6(9), 40–47.

[@dennis2016practical] Dennis, L. A., Fisher, M., Slavkovik, M., & Webster, M. (2016). Formal verification of ethical choices in autonomous systems. *Robotics and Autonomous Systems*, 77, 1–14.

[@dignum2004model] Dignum, V. (2004). *A Model for Organizational Interaction: Based on Agents, Founded in Logic*. SIKS Dissertation Series.

[@ehtesham2025survey] Ehtesham, A., Singh, A., Gupta, G. K., & Kumar, S. (2025). *A survey of agent interoperability protocols: Model Context Protocol (MCP), Agent Communication Protocol (ACP), Agent-to-Agent Protocol (A2A), and Agent Network Protocol (ANP)*. arXiv:2505.02279.

[@erol1994htn] Erol, K., Hendler, J., & Nau, D. S. (1994). HTN planning: Complexity and expressivity. *Proceedings of the Twelfth National Conference on Artificial Intelligence*, 1123–1128.

[@esteva2004electronic] Esteva, M., Rodríguez-Aguilar, J. A., Sierra, C., Garcia, P., & Arcos, J. L. (2004). On the formal specification of electronic institutions. In *Agent Mediated Electronic Commerce*.

[@ferber1999multi] Ferber, J. (1999). *Multi-Agent Systems: An Introduction to Distributed Artificial Intelligence*. Addison-Wesley.

[@fikes1971strips] Fikes, R. E., & Nilsson, N. J. (1971). STRIPS: A new approach to the application of theorem proving to problem solving. *Artificial Intelligence*, 2(3–4), 189–208.

[@finin1994kqml] Finin, T., Fritzson, R., McKay, D., & McEntire, R. (1994). KQML as an agent communication language. *Proceedings of the Third International Conference on Information and Knowledge Management*, 456–463.

[@fioretto2018distributed] Fioretto, F., Pontelli, E., & Yeoh, W. (2018). Distributed constraint optimization problems and applications: A survey. *Journal of Artificial Intelligence Research*, 61, 623–698.

[@fipa2002acl] Foundation for Intelligent Physical Agents. (2002). *FIPA Communicative Act Library Specification*. http://www.fipa.org/specs/fipa00037/

[@fisher2013verifying] Fisher, M., Dennis, L. A., & Webster, M. (2013). Verifying autonomous systems. *Communications of the ACM*, 56(9), 84–93.

[@fox2003pddl2] Fox, M., & Long, D. (2003). PDDL2.1: An extension to PDDL for expressing temporal planning domains. *Journal of Artificial Intelligence Research*, 20, 61–124.

[@franklin1997agent] Franklin, S., & Graesser, A. (1997). Is it an agent, or just a program? A taxonomy for autonomous agents. In *Intelligent Agents III: Agent Theories, Architectures, and Languages*.

[@garcez2019neural] d’Avila Garcez, A. S., Gori, M., Lamb, L. C., Serafini, L., Spranger, M., & Tran, S. N. (2019). Neural-symbolic computing: An effective methodology for principled integration of machine learning and reasoning. *Journal of Applied Logics*, 6(4), 611–632.

[@gelfond1991classical] Gelfond, M., & Lifschitz, V. (1991). Classical negation in logic programs and disjunctive databases. *New Generation Computing*, 9, 365–385.

[@gelernter1985generative] Gelernter, D. (1985). Generative communication in Linda. *ACM Transactions on Programming Languages and Systems*, 7(1), 80–112.

[@georgeff1987procedural] Georgeff, M. P., & Lansky, A. L. (1987). Reactive reasoning and planning. *Proceedings of the Sixth National Conference on Artificial Intelligence*, 677–682.

[@georgeff1999belief] Georgeff, M., Pell, B., Pollack, M., Tambe, M., & Wooldridge, M. (1999). The belief-desire-intention model of agency. In *Intelligent Agents V: Agent Theories, Architectures, and Languages*.

[@georgievski2015htn] Georgievski, I., & Aiello, M. (2015). HTN planning: Overview, comparison, and beyond. *Artificial Intelligence*, 222, 124–156.

[@ghallab2004automated] Ghallab, M., Nau, D., & Traverso, P. (2004). *Automated Planning: Theory and Practice*. Morgan Kaufmann.

[@google2025a2a] Google Developers Blog. (2025). *Announcing the Agent2Agent Protocol (A2A)*. https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/

[@greshake2023not] Greshake, K., Abdelnabi, S., Mishra, S., Endres, C., Holz, T., & Fritz, M. (2023). Not what you’ve signed up for: Compromising real-world LLM-integrated applications with indirect prompt injection. *arXiv:2302.12173*.

[@gruber1993translation] Gruber, T. R. (1993). A translation approach to portable ontology specifications. *Knowledge Acquisition*, 5(2), 199–220.

[@guo2024llmmas] Guo, T., Chen, X., Wang, Y., Chang, R., Pei, S., Chawla, N. V., Wiest, O., & Zhang, X. (2024). *Large Language Model based Multi-Agents: A Survey of Progress and Challenges*. arXiv:2402.01680.

[@hindriks2009programming] Hindriks, K. V. (2009). Programming rational agents in GOAL. In *Multi-Agent Programming: Languages, Tools and Applications*. Springer.

[@hitzler2022neuro] Hitzler, P., Sarker, M. K., & Eberhart, A. (Eds.). (2022). *Neuro-Symbolic Artificial Intelligence: The State of the Art*. IOS Press.

[@hogan2021knowledge] Hogan, A., Blomqvist, E., Cochez, M., d’Amato, C., de Melo, G., Gutierrez, C., Kirrane, S., et al. (2021). Knowledge graphs. *ACM Computing Surveys*, 54(4), 1–37.

[@hong2023metagpt] Hong, S., Zhuge, M., Chen, J., Zheng, X., Cheng, Y., Zhang, C., Wang, J., et al. (2023). *MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework*. arXiv:2308.00352.

[@hou2025mcpsecurity] Hou, X., Zhao, Y., Wang, S., & Wang, H. (2025). *Model Context Protocol (MCP): Landscape, Security Threats, and Future Research Directions*. arXiv:2503.23278.

[@huang2022inner] Huang, W., Xia, F., Xiao, T., Chan, H., Liang, J., Florence, P., Zeng, A., et al. (2022). *Inner Monologue: Embodied Reasoning through Planning with Language Models*. arXiv:2207.05608.

[@huang2022language] Huang, W., Abbeel, P., Pathak, D., & Mordatch, I. (2022). *Language Models as Zero-Shot Planners: Extracting Actionable Knowledge for Embodied Agents*. arXiv:2201.07207.

[@ibm2025acp] IBM Research. (2025). *Agent Communication Protocol*. https://research.ibm.com/projects/agent-communication-protocol

[@jennings1996coordination] Jennings, N. R. (1996). Coordination techniques for distributed artificial intelligence. In *Foundations of Distributed Artificial Intelligence*. Wiley.

[@jennings2000agent] Jennings, N. R. (2000). On agent-based software engineering. *Artificial Intelligence*, 117(2), 277–296.

[@jennings2001automated] Jennings, N. R., Faratin, P., Lomuscio, A. R., Parsons, S., Wooldridge, M., & Sierra, C. (2001). Automated negotiation: Prospects, methods and challenges. *Group Decision and Negotiation*, 10, 199–215.

[@jimenez2024swebench] Jimenez, C. E., Yang, J., Wettig, A., Yao, S., Pei, K., Press, O., & Narasimhan, K. (2024). SWE-bench: Can language models resolve real-world GitHub issues? *International Conference on Learning Representations*.

[@karpas2022mrkl] Karpas, E., Abend, O., Belinkov, Y., Lenz, B., Lieber, O., Ratner, N., Shoham, Y., et al. (2022). *MRKL Systems: A Modular, Neuro-Symbolic Architecture That Combines Large Language Models, External Knowledge Sources and Discrete Reasoning*. arXiv:2205.00445.

[@labrou1997proposal] Labrou, Y., & Finin, T. (1997). A proposal for a new KQML specification. *Technical Report TR CS-97-03*, University of Maryland Baltimore County.

[@lewis2020retrieval] Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., et al. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *Advances in Neural Information Processing Systems*, 33, 9459–9474.

[@li2023camel] Li, G., Hammoud, H. A. A. K., Itani, H., Khizbullin, D., & Ghanem, B. (2023). *CAMEL: Communicative Agents for “Mind” Exploration of Large Language Model Society*. arXiv:2303.17760.

[@lifschitz2008answer] Lifschitz, V. (2008). What is answer set programming? *Proceedings of the AAAI Conference on Artificial Intelligence*, 22(1), 1594–1597.

[@liu2023agentbench] Liu, X., Yu, H., Zhang, H., Xu, Y., Lei, X., Lai, H., Gu, Y., et al. (2023). *AgentBench: Evaluating LLMs as Agents*. arXiv:2308.03688.

[@liu2023llmp] Liu, B., Jiang, Y., Zhang, X., Liu, Q., Zhang, S., Biswas, J., & Stone, P. (2023). *LLM+P: Empowering Large Language Models with Optimal Planning Proficiency*. arXiv:2304.11477.

[@maes1995artificial] Maes, P. (1995). Artificial life meets entertainment: Lifelike autonomous agents. *Communications of the ACM*, 38(11), 108–114.

[@malone1994interdisciplinary] Malone, T. W., & Crowston, K. (1994). The interdisciplinary study of coordination. *ACM Computing Surveys*, 26(1), 87–119.

[@mccarthy1969some] McCarthy, J., & Hayes, P. J. (1969). Some philosophical problems from the standpoint of artificial intelligence. In *Machine Intelligence 4*.

[@mcdermott1998pddl] McDermott, D., Ghallab, M., Howe, A., Knoblock, C., Ram, A., Veloso, M., Weld, D., & Wilkins, D. (1998). *PDDL—The Planning Domain Definition Language*. Technical Report CVC TR-98-003/DCS TR-1165, Yale Center for Computational Vision and Control.

[@mcp2025spec] Model Context Protocol. (2025). *Specification: 2025-06-18*. https://modelcontextprotocol.io/specification/2025-06-18

[@mialon2023augmented] Mialon, G., Dessì, R., Lomeli, M., Nalmpantis, C., Pasunuru, R., Raileanu, R., Rozière, B., Schick, T., Dwivedi-Yu, J., Celikyilmaz, A., Grave, E., LeCun, Y., & Scialom, T. (2023). Augmented language models: A survey. *Transactions on Machine Learning Research*.

[@mialon2023gaia] Mialon, G., Fourrier, C., Swift, C., Wolf, T., LeCun, Y., & Scialom, T. (2023). *GAIA: A Benchmark for General AI Assistants*. arXiv:2311.12983.

[@modi2005adopt] Modi, P. J., Shen, W.-M., Tambe, M., & Yokoo, M. (2005). ADOPT: Asynchronous distributed constraint optimization with quality guarantees. *Artificial Intelligence*, 161(1–2), 149–180.

[@nau2003shop2] Nau, D., Au, T.-C., Ilghami, O., Kuter, U., Murdock, J. W., Wu, D., & Yaman, F. (2003). SHOP2: An HTN planning system. *Journal of Artificial Intelligence Research*, 20, 379–404.

[@newell1976computer] Newell, A., & Simon, H. A. (1976). Computer science as empirical inquiry: Symbols and search. *Communications of the ACM*, 19(3), 113–126.

[@nii1986blackboard] Nii, H. P. (1986). Blackboard systems: The blackboard model of problem solving and the evolution of blackboard architectures. *AI Magazine*, 7(2), 38–53.

[@nilsson1980principles] Nilsson, N. J. (1980). *Principles of Artificial Intelligence*. Tioga Publishing.

[@orkin2006three] Orkin, J. (2006). Three states and a plan: The AI of F.E.A.R. *Game Developers Conference*.

[@ouyang2022training] Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., Zhang, C., et al. (2022). Training language models to follow instructions with human feedback. *Advances in Neural Information Processing Systems*, 35, 27730–27744.

[@park2023generative] Park, J. S., O’Brien, J. C., Cai, C. J., Morris, M. R., Liang, P., & Bernstein, M. S. (2023). Generative agents: Interactive simulacra of human behavior. *Proceedings of the 36th Annual ACM Symposium on User Interface Software and Technology*.

[@qian2023communicative] Qian, C., Cong, X., Yang, C., Chen, W., Su, Y., Xu, J., Liu, Z., & Sun, M. (2023). *Communicative Agents for Software Development*. arXiv:2307.07924.

[@rahwan2009argumentation] Rahwan, I., & Simari, G. R. (Eds.). (2009). *Argumentation in Artificial Intelligence*. Springer.

[@rao1991modeling] Rao, A. S., & Georgeff, M. P. (1991). Modeling rational agents within a BDI-architecture. *Proceedings of the Second International Conference on Principles of Knowledge Representation and Reasoning*, 473–484.

[@rao1995bdi] Rao, A. S., & Georgeff, M. P. (1995). BDI agents: From theory to practice. *Proceedings of the First International Conference on Multi-Agent Systems*, 312–319.

[@rao1996agentspeak] Rao, A. S. (1996). AgentSpeak(L): BDI agents speak out in a logical computable language. *Agents Breaking Away*, 42–55.

[@reiter2001knowledge] Reiter, R. (2001). *Knowledge in Action: Logical Foundations for Specifying and Implementing Dynamical Systems*. MIT Press.

[@russell2021artificial] Russell, S., & Norvig, P. (2021). *Artificial Intelligence: A Modern Approach* (4th ed.). Pearson.

[@sandholm1993implementation] Sandholm, T. W. (1993). An implementation of the Contract Net Protocol based on marginal cost calculations. *Proceedings of the Eleventh National Conference on Artificial Intelligence*, 256–262.

[@schick2023toolformer] Schick, T., Dwivedi-Yu, J., Dessì, R., Raileanu, R., Lomeli, M., Zettlemoyer, L., Cancedda, N., & Scialom, T. (2023). Toolformer: Language models can teach themselves to use tools. *Advances in Neural Information Processing Systems*, 36.

[@searle1969speech] Searle, J. R. (1969). *Speech Acts: An Essay in the Philosophy of Language*. Cambridge University Press.

[@shehory1998methods] Shehory, O., & Kraus, S. (1998). Methods for task allocation via agent coalition formation. *Artificial Intelligence*, 101(1–2), 165–200.

[@shinn2023reflexion] Shinn, N., Cassano, F., Berman, E., Gopinath, A., Narasimhan, K., & Yao, S. (2023). *Reflexion: Language Agents with Verbal Reinforcement Learning*. arXiv:2303.11366.

[@shoham1993agent] Shoham, Y. (1993). Agent-oriented programming. *Artificial Intelligence*, 60(1), 51–92.

[@shridhar2021alfworld] Shridhar, M., Yuan, X., Côté, M.-A., Bisk, Y., Trischler, A., & Hausknecht, M. (2021). ALFWorld: Aligning text and embodied environments for interactive learning. *International Conference on Learning Representations*.

[@sierra1997model] Sierra, C., Faratin, P., & Jennings, N. R. (1997). A service-oriented negotiation model between autonomous agents. *Proceedings of the 8th European Workshop on Modelling Autonomous Agents in a Multi-Agent World*.

[@singh1999commitments] Singh, M. P. (1999). An ontology for commitments in multiagent systems. *Artificial Intelligence and Law*, 7, 97–113.

[@smith1980contract] Smith, R. G. (1980). The Contract Net Protocol: High-level communication and control in a distributed problem solver. *IEEE Transactions on Computers*, C-29(12), 1104–1113.

[@theraulaz1999brief] Theraulaz, G., & Bonabeau, E. (1999). A brief history of stigmergy. *Artificial Life*, 5(2), 97–116.

[@tran2025multiagent] Tran, K. T., et al. (2025). *Multi-Agent Collaboration Mechanisms: A Survey of LLMs*. arXiv:2501.06322.

[@valmeekam2023planbench] Valmeekam, K., Marquez, M., Sreedharan, S., & Kambhampati, S. (2023). *On the Planning Abilities of Large Language Models: A Critical Investigation*. arXiv:2305.15771.

[@vaswani2017attention] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. *Advances in Neural Information Processing Systems*, 30.

[@wang2023voyager] Wang, G., Xie, Y., Jiang, Y., Mandlekar, A., Xiao, C., Zhu, Y., Fan, L., & Anandkumar, A. (2023). *Voyager: An Open-Ended Embodied Agent with Large Language Models*. arXiv:2305.16291.

[@wang2024survey] Wang, L., Ma, C., Feng, X., Zhang, Z., Yang, H., Zhang, J., Chen, Z., et al. (2024). A survey on large language model based autonomous agents. *Frontiers of Computer Science*, 18, 186345.

[@wei2022chain] Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F., Chi, E., Le, Q. V., & Zhou, D. (2022). Chain-of-thought prompting elicits reasoning in large language models. *Advances in Neural Information Processing Systems*, 35, 24824–24837.

[@weiss2013multiagent] Weiss, G. (Ed.). (2013). *Multiagent Systems* (2nd ed.). MIT Press.

[@wellman1993market] Wellman, M. P. (1993). A market-oriented programming environment and its application to distributed multicommodity flow problems. *Journal of Artificial Intelligence Research*, 1, 1–23.

[@winograd1972understanding] Winograd, T. (1972). *Understanding Natural Language*. Academic Press.

[@wooldridge1995intelligent] Wooldridge, M., & Jennings, N. R. (1995). Intelligent agents: Theory and practice. *The Knowledge Engineering Review*, 10(2), 115–152.

[@wooldridge2009introduction] Wooldridge, M. (2009). *An Introduction to MultiAgent Systems* (2nd ed.). Wiley.

[@wu2023autogen] Wu, Q., Bansal, G., Zhang, J., Wu, Y., Li, B., Zhu, E., Jiang, L., et al. (2023). *AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation*. arXiv:2308.08155.

[@xie2024osworld] Xie, T., Zhang, D., Chen, J., Li, X., Zhao, S., Cao, R., Hua, T. J., et al. (2024). *OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments*. arXiv:2404.07972.

[@yang2025neuro] Yang, X. W., et al. (2025). *Neuro-Symbolic Artificial Intelligence: Towards Improving the Reasoning Abilities of Large Language Models*. IJCAI 2025 Survey Track / arXiv:2508.13678.

[@yao2022webshop] Yao, S., Chen, H., Yang, J., & Narasimhan, K. (2022). WebShop: Towards scalable real-world web interaction with grounded language agents. *Advances in Neural Information Processing Systems*, 35, 20744–20757.

[@yao2023react] Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., & Cao, Y. (2023). ReAct: Synergizing reasoning and acting in language models. *International Conference on Learning Representations*.

[@yokoo1998distributed] Yokoo, M., Durfee, E. H., Ishida, T., & Kuwabara, K. (1998). The distributed constraint satisfaction problem: Formalization and algorithms. *IEEE Transactions on Knowledge and Data Engineering*, 10(5), 673–685.

[@yolum2002flexible] Yolum, P., & Singh, M. P. (2002). Flexible protocol specification and execution: Applying event calculus planning using commitments. *Proceedings of the First International Joint Conference on Autonomous Agents and Multiagent Systems*, 527–534.

[@zhou2023webarena] Zhou, S., Xu, F. F., Zhu, H., Zhou, X., Lo, R., Sridhar, A., Cheng, X., et al. (2023). *WebArena: A Realistic Web Environment for Building Autonomous Agents*. arXiv:2307.13854.
