### **PROJECT REPORT** 

# Adaptive Internal–External Latent Workspace 

_Full research, architecture, implementation, evaluation and development report_ 

Prepared 10 September 2026 

## **Abstract** 

This project investigates an architectural hypothesis about reasoning in artificial intelligence: useful reasoning may be better represented as the iterative transformation of a relational latent computational workspace than as a sequence of natural-language tokens. The initial proposal is a dynamic graph of continuous vector-valued latent nodes whose state, activity and relationships can change during inference. The project has since been extended with an explicit external workspace, motivated by the observation that complex problem solving often benefits from external representations such as equations, diagrams, notation, lists, plans and other structured artefacts. 

The resulting architecture is a coupled system: a persistent latent workspace and an explicit external workspace exchange information during reasoning. The system may update an existing latent state, recruit previously inactive capacity, change relational structure, externalise information, read external information back into the latent space, and continue until a learned resolution criterion is satisfied. Language is treated as an input/output interface in the core architectural hypothesis rather than as the required internal substrate of reasoning. 

The project is intentionally falsifiable. Its contribution is not a claim that vectors, graphs or latent reasoning are new individually. Continuous latent reasoning, graph neural networks, equivariance, external memory and adaptive recurrent computation all have substantial precedents [1–8]. The research question is whether a persistent, dynamically reconfigurable relational latent workspace coupled to a selective external workspace and an explicit resolution mechanism provides a measurable advantage on controlled reasoning tasks beyond simpler alternatives. 

## **Executive Summary** 

|**Dimension**|**Currentprojectposition**|
|---|---|
|Central hypothesis|Reasoning can be investigated as iterative<br>transformation of a relational latent workspace,<br>potentially coupled to an explicit external<br>workspace.|
|Primary computational object|A mutable latent graph G_t = (V_t, E_t, A_t) with<br>continuous node states,relations and activity.|
|External computational object|A structured workspace W_t containing inspectable<br>symbolic or semi-symbolic records.|
|Role of language|Input/output interface in the core hypothesis; not<br>assumed to be the reasoningsubstrate.|



|Dynamic behaviour|Vector-state evolution, adaptive activity/capacity<br>andpotentiallychangingrelationships.|
|---|---|
|Stopping|A learned resolution criterion based on task<br>suficiencyrather than rawprobabilityalone.|
|MVP strategy|Small synthetic tasks with exact solutions before<br>natural-language scale-up.|
|Primary empirical test|Compare dynamic latent-only and coupled latent–<br>external models to simpler baselines under<br>matched budgets.|
|Strongest causal test|Corrupt the external workspace after it is written<br>and measure systematic changes in subsequent<br>latent computation and outcome.|
|Research contribution if supported|Evidence for a useful structural bias: dynamic<br>relational latent reconfguration plus selective<br>externalisation can improve controlled reasoning.|



## **Contents** 

1. Project Origin and Evolution 

2. Problem Statement 

3. Research Aim, Questions and Objectives 

###### 4. Theoretical Position 

5. Conceptual Model of Understanding and Representation 

6. Why a Relational Latent Workspace? 

7. From Dynamic Latent Graph to Internal–External Workspace 

8. Final System Architecture 

9. Latent Workspace Specification 

10. External Workspace Specification 

11. Internal–External Co-Refinement 

12. Resolution, Convergence and Adaptive Halting 

13. Equivariance and Invariance 

14. Candidate Dynamic Mechanisms 

15. Probabilistic and Stochastic Components 

16. Training Objective and Learning Strategy 

17. Research Methodology 

18. Experimental Programme 

19. Baselines and Ablations 

20. Causal and Falsification Tests 

21. Evaluation Metrics and Statistical Analysis 

22. Implementation Plan and Technology Stack 

23. Repository and Data Architecture 

24. MVP Timeline and Deliverables 

25. Detailed Task Suite 

26. Worked Examples 

27. Relationship to Existing Research 

28. Novelty and Contribution 

29. Risks, Failure Modes and Mitigations 

30. Ethics, Reproducibility and Research Integrity 

31. Limitations and Open Questions 

32. Future Development Roadmap 

33. Success Criteria 

34. Expected Outputs 

35. Final Project Definition 

References 

## **1. Project Origin and Evolution** 

The project began with a conceptual question about whether physical grounding should be treated as a necessary condition for conceptual understanding. The working position is deliberately modest: direct grounding may provide useful information, but the project does not assume that every useful concept must have a single direct physical referent. Mathematics, social relationships and other abstract structures provide motivation for investigating representations that are not defined solely by sensorimotor interaction. 

A second starting point concerns language. The project distinguishes an internal conceptual state from the language used to communicate it. This motivates a view in which language can be an encoding and decoding channel while the computational substrate used during reasoning may be latent and relational. This is a hypothesis rather than a claim that natural language is unimportant. 

The third step was the recognition that continuous latent reasoning is already an active area. COCONUT feeds a hidden state back into a model as a continuous reasoning state rather than decoding every 

intermediate state into a token [1]. A recurrent depth approach similarly iterates latent computation at test time [2]. These precedents narrowed the proposed contribution: the question is no longer whether reasoning can happen without intermediate language, but whether the latent state itself should be treated as a mutable relational computational workspace. 

The architecture was then expanded from a single or small number of hidden vectors to a graph of vectorvalued nodes. This makes relationships an explicit computational object. The current state can change through node-state updates, relationship changes and activity/recruitment changes. Fixed-slot implementation is preferred initially because it approximates dynamic capacity while remaining differentiable and tractable. 

The most recent conceptual extension is the external workspace. Complex reasoning can be easier when intermediate structure is written into an explicit representation. Research on cognitive offloading finds that external representations can improve performance on memory-based tasks, while also introducing dependence on continued access to the external artefact [9,10]. In computational systems, external memory and tool use already have strong precedents, including Neural Turing Machines, Differentiable Neural Computers and more recent memory/tool-augmented language systems [11–15]. 

## **2. Problem Statement** 

The practical research problem is the lack of a clearly isolated architectural test for a persistent, adaptive, relational latent workspace that can also selectively externalise information into a structured computational space. 

Existing approaches provide important pieces: 

- Language-model chain-of-thought represents intermediate reasoning in a visible symbolic sequence. 

- Continuous latent reasoning shows that intermediate computation can occur without emitting naturallanguage tokens [1,2]. 

- Graph neural networks provide relational message passing and permutation-consistent computation [3]. 

- Continuous-time models provide mechanisms for latent trajectories and adaptive dynamics [4,5]. 

- External-memory architectures provide learned read/write access to state outside ordinary hidden activations [11,12]. 

- Memory- and tool-augmented LLM systems provide increasingly sophisticated external interaction patterns [13–15]. 

The project's unresolved question is whether these ideas can be integrated around a single, persistent computational workspace whose state is itself the object of reasoning, while preserving the ability to externalise selected information when the internal representation becomes insufficient or inefficient. 

## **3. Research Aim, Questions and Objectives** 

#### **3.1 Aim** 

To design and empirically evaluate a persistent internal–external computational workspace in which relational latent states can dynamically evolve, selectively externalise structured information, retrieve it, and stop when a task-relevant resolution criterion is satisfied. 

#### **3.2 Core research question** 

Does a dynamically reconfigurable relational latent workspace, coupled to a selective external structured workspace, provide measurable benefits for compositional reasoning compared with simpler recurrent latent, static graph and non-externalised baselines? 

#### **3.3 Objectives** 

1. Specify a tractable latent graph representation with vector states, relations and activity. 

2. Implement and evaluate a recurrent relational latent baseline before adding continuous-time dynamics. 

3. Introduce adaptive activity/recruitment and determine whether it improves efficiency or generalisation. 

4. Define and test an explicit resolution criterion for adaptive halting. 

5. Implement an inspectable structured external workspace with differentiable or parameterised read/write interfaces. 

6. Test whether external information causally influences subsequent latent computation. 

7. Learn a selective externalisation policy and compare it with never- and always-externalise controls. 

8. Evaluate the complete system on controlled reasoning tasks using matched baselines, ablations and multiple random seeds. 

9. Identify which components are genuinely necessary, which are optional, and where the architecture fails. 

## **4. Theoretical Position** 

The project treats the meaning of a candidate latent state as potentially relational rather than object-local. A vector may not have a complete, fixed interpretation in isolation; its computational role can emerge from how it interacts with other states, relations, transformations and task context. The working abstraction is: 

##### **Meaning(candidate state) ~ structure + relations + transformation behaviour + context** 

This statement is not an established theory of semantics. It is an architectural hypothesis that motivates testing structured latent computation rather than presupposing that individual embeddings are sufficient semantic objects. 

The project also distinguishes a pre-formed persistent state from an impromptu reasoning state. New input need not replace the old state. It can perturb an existing workspace and cause reorganisation. This is one motivation for carrying a graph state across related turns. 

## **5. Conceptual Model of Understanding and Representation** 

The philosophical premise is intentionally kept separate from the empirical claims. The project does not attempt to prove that a particular latent architecture 'understands' in a subjective or human sense. Instead, it operationalises a narrower engineering question: can structured latent computation produce behaviour that requires binding, composition, iterative update, alternative hypothesis management and persistence? 

|**Concept**|**Operational interpretation**|
|---|---|
|Understanding|A hypothesis about the ability to construct and<br>transform task-relevant relational structure.|
|Meaning|A property inferred from relational behaviour and<br>task use,not assumed from a single vector identity.|



|Reasoning|Iterative transformation of internal and, where<br>useful,external representations.<br>|
|---|---|
|State|The current computational confguration of the<br>latent and external workspaces.<br>|
|Resolution|A state that satisfes task-relevant constraints and<br>supports a stable correct decision/action.|
|Communication|A channel through which an input perturbs or<br>reconstructs a compatible internal state.|



## **6. Why a Relational Latent Workspace?** 

A single recurrent vector can store arbitrarily complex information in principle, but it provides no explicit architectural pressure for separable relational structure. The proposed graph introduces multiple interacting computational units and allows the system to distribute state across them. 

This does not imply that a graph is automatically more expressive than a transformer or recurrent vector model. A transformer can be interpreted graph-theoretically, and a sufficiently powerful recurrent vector can in principle encode relational information. The empirical question is whether the explicit workspace bias improves the target tasks under controlled budgets. 

- Relational binding: entities can participate in different relationships without requiring a single global vector to store everything. 

- Selective activity: irrelevant latent capacity can be suppressed. 

- State persistence: useful structure can survive across inputs. 

- Structural analysis: nodes and relationships can be inspected and intervened on. 

- Adaptive computation: reasoning depth can depend on task state. 

- Externalisation: intermediate structure can be moved into an explicit workspace when beneficial. 

## **7. From Dynamic Latent Graph to Internal–External Workspace** 

The external workspace changes the project's centre of gravity. The core computational question becomes not only how the latent graph evolves, but where useful information should reside at each stage of reasoning. 

##### **Reasoning = iterative transformation of internal and external representations** 

The system can therefore make several classes of decision at each iteration: 

- update an existing latent slot; 

- increase or reduce the activity of a latent slot; 

- modify a learned relationship; 

- write selected information into the external workspace; 

- retrieve or reinterpret external information; 

- compress external information back into the latent state; 

- continue reasoning; 

- declare the current state resolved. 

The external workspace is therefore not merely a logging device. The strongest version of the hypothesis is that it is an active computational component whose contents alter later latent updates. 

## **8. Final System Architecture** 

##### **Input → Encoder → Persistent Latent Workspace <-> External Workspace → Resolution → Invariant Readout → Decoder** 

|**Stage**|**Function**|**Key state**|
|---|---|---|
|Input|Receive task information|X|
|Encoding|Construct/perturb initial<br>representation|G_0|
|Latent computation|Relational message passing and<br>recurrence|V_t, E_t, A_t|
|Externalisation|Select useful intermediate<br>information|W_t|
|External retrieval|Inject structured external state|R_t|
|Resolution|Judge task suficiency/stability|r_t|
|Readout|<br>Order-independent fnal<br>aggregation|G*|
|Decoding|Render answer/action|Y|



Operationally, the latent workspace is the controller/interpreter of a larger computational workspace. The external workspace can be domain-specific: equations for mathematics, structured relations for planning, a board representation for games, an AST or variable state for programming, or a diagram-like structure for spatial tasks. 

## **9. Latent Workspace Specification** 

##### **G_t = (V_t, E_t, A_t)** 

V_t is a set or fixed pool of vector-valued node states. 

E_t represents relationships, either as edge weights, typed edges, relation vectors or an inferred interaction matrix. 

A_t represents participation/activity. In the first prototype this is a soft gate per slot. 

#### **9.1 Node semantics** 

Node semantics remain deliberately unresolved. Candidate interpretations include entities, concepts, relations, temporary reasoning constructs or generic learned units. The current design favours learned units whose semantics emerge from relational behaviour rather than hard-coding one node per named concept. 

#### **9.2 State update** 

##### **m_i,t = Aggregate_j [ φθ(v_i,t, v_j,t, e_ij,t) ]** 

**v_i,t+1 = Updateθ(v_i,t, m_i,t, x_i, a_i)** 

#### **9.3 Activity** 

##### **a_i,t = sigmoid(fθ(v_i,t, G_t, X))** 

Soft gating is the first implementation because it preserves differentiability and fixed tensor shapes. Functional node creation is approximated by converting an inactive slot into an active latent structure. 

## **10. External Workspace Specification** 

The external workspace should initially be structured rather than linguistic. It should be serialisable, inspectable and easy to corrupt for causal experiments. 

|**Record type**|**Example**|**Role**|
|---|---|---|
|Entity|shoe_A|Stable reference|
|Attribute|price(shoe_A)=70|Explicit value|
|Relation|money(John)=50|Task fact|
|Operation|50 + 20|Intermediate computation|
|Constraint|money>=price|Decision condition|
|Status|unresolved|State of reasoning|
|Provenance|source=fact_3|Optional traceability|



#### **10.1 External workspace invariants** 

- No direct access to the ground-truth target during normal execution. 

- Every write must be typed and loggable. 

- Every read must identify which external records influenced the latent update. 

- Workspace corruption must be possible without changing the preceding latent state. 

- The representation must support deterministic replay. 

#### **10.2 External workspace is not the same as generic external memory** 

The distinction is architectural. Generic external memory can simply store information. The proposed workspace is designed as a problem-solving medium with explicit structure, selective externalisation and bidirectional coupling to the latent reasoning state. Its most important property is causal participation in the subsequent computation. 

## **11. Internal–External Co-Refinement** 

##### **L_(t+1) = F(L_t, E_t, X)** 

##### **E_(t+1) = H(E_t, L_(t+1))** 

The two state spaces should not be considered separate pipelines. They are coupled representations of the same evolving problem. The latent workspace compresses and interprets the external state; the external workspace preserves selected structure in an explicit form that can be re-read, checked and transformed. 

#### **11.1 Why externalisation may help** 

- Intermediate structure can be preserved without forcing it to remain compressed in one latent state. 

- Explicit records can make dependencies easier to compute or verify. 

- The external state can be larger or differently structured than the latent state. 

- External state can be revisited after subsequent latent transformations. 

- The external representation can support intervention and error detection. 

#### **11.2 What would count as evidence?** 

A mere accuracy gain is weak evidence because the external module may add parameters or capacity. Stronger evidence includes selective use, external-state corruption effects, predictable changes in later latent trajectories, and gains that remain after controlling parameter count, compute and reasoning steps. 

## **12. Resolution, Convergence and Adaptive Halting** 

A high-probability latent state is not necessarily a resolved state. The model may settle into an incorrect but common state. Resolution should therefore be an explicit task-conditioned quantity [Project design]. 

##### **P(resolved | G_t, X) = Rφ(G_t, X)** 

##### **Stop when Rφ(G_t, X) >= τ** 

A complementary energy interpretation is also possible: 

##### **E(G, X) = E_constraints + E_inconsistency + E_task** 

The energy formulation is optional and should be treated as an experimental branch. The core MVP should use a learned resolution critic because this is simpler to implement and evaluate. 

#### **12.1 Stability signal** 

One practical stability statistic is the change between successive latent states, such as a normalised distance between pooled states. Stability is useful but should not be sufficient on its own; a wrong state can also be stable. 

#### **12.2 Hard safety bound** 

Every experiment must have a maximum reasoning budget. Adaptive stopping is therefore a policy within a bounded compute envelope, not an unbounded loop. 

## **13. Equivariance and Invariance** 

The architectural principle is to use equivariant computation during reasoning and invariant aggregation at the final readout. Permuting the storage order of latent slots should not change the computation's substantive result. 

Permutation equivariance is the minimal requirement for a set-like latent slot system. E(n)-equivariant mechanisms become relevant only if the latent coordinates themselves carry a formal geometric meaning [3]. 

|**Property**|**Use inproject**|
|---|---|
|Permutation equivariance|Node operations remain consistent under slot<br>reordering.|
|Geometric equivariance|Candidate mechanism only when coordinates carry<br>meaningfulgeometry.|
|Permutation invariance|Final readout should not depend on arbitrarynode|



||ordering.|
|---|---|
|Attention|Can implement learned relational weighting, but is|
||not itself the novelty.|



The project explicitly avoids claiming that equivariance 'mimics human cognition'. It is an architectural property that makes the computational representation well-defined under relabelling. 

## **14. Candidate Dynamic Mechanisms** 

|**Mechanism**|**Role**|**Status**|
|---|---|---|
|Discrete recurrent message<br>passing|First reasoning dynamics|Required starting point|
|Neural ODE|Continuous latent trajectory|Experimental branch[4]|
|Liquid Time-Constant dynamics|Adaptive time constants|Experimental branch[5]|
|Soft activity gates|Recruit/suppress latent capacity|Core MVP|
|Stochastic/reparameterised<br>gates|Uncertainty and structural<br>sampling|Later branch|
|KAN-style functions|Learned edge nonlinearities|Optional[6]|
|Fixed-point / DEQ criterion|Alternative convergence<br>mechanism|Optional [7]|



The sequencing matters. The project should not introduce continuous dynamics, stochastic gating and learned nonlinearities simultaneously. Discrete recurrent updates provide a clean reference 

implementation; complexity is added only after the previous component produces interpretable evidence. 

## **15. Probabilistic and Stochastic Components** 

The latent workspace can be partially probabilistic rather than wholly stochastic. Candidate stochastic variables include node activity, edge existence and selected latent variables. 

##### **G ~ Pθ(G | X)** 

The reparameterisation trick or relaxed categorical variables can keep stochastic decisions differentiable. The deterministic version should be established first because it gives a stable control against which stochasticity can be evaluated. 

Calibration is a first-class evaluation issue if stochastic latent states are introduced. A model should not receive credit for uncertainty merely because sampling increases robustness; the uncertainty signal must correspond to measurable correctness or decision risk. 

## **16. Training Objective and Learning Strategy** 

**L = λ_ans L_answer + λ_res L_resolution + λ_cons L_consistency + λ_dyn L_dynamics + λ_reg L_regularisation** 

Candidate terms: 

|**Loss**|**Purpose**|
|---|---|
|L_answer|Final task correctness|
|L_resolution|Resolved vs continue decision|
|L_consistency|Discourage contradictoryor unstable states|
|L_dynamics|Auxiliarysignal supportinguseful latent trajectories|
|L_regularisation|Activity sparsity, write cost, structural simplicity or<br>trajectorycomplexity|



The first system should avoid trying to learn general language understanding. Synthetic tasks with exact solutions make it possible to supervise final correctness and resolution without exposing an English chain of thought. 

#### **16.1 Curriculum** 

1. Short arithmetic compositions. 

2. Longer compositions with held-out combinations. 

3. Relational chains. 

4. Distractors and irrelevant facts. 

5. Contradictions and competing hypotheses. 

6. Tasks specifically constructed so that externalisation is helpful. 

7. Persistent multi-turn tasks. 

10. Only after controlled success: natural-language encoding/decoding. 

## **17. Research Methodology** 

The project is best characterised as design science combined with controlled experimental machine learning. The artefact is the architecture and prototype; the evaluation is an empirical test of the design hypotheses. 

|**Dimension**|**Choice**|**Rationale**|
|---|---|---|
|Research philosophy|Pragmatic / problem-oriented|Focus on whether the<br>architecture provides<br>measurable computational<br>value.|
|Approach|Deductive + iterative empirical<br>design|Derive testable predictions,<br>implement,measure,revise.|
|Method|Controlled computational<br>experiments|Exact synthetic tasks allow<br>causal intervention.|
|Strategy|Prototype + ablation +<br>comparison|Separates contribution of<br>individual components.|
|Time horizon|Finiteproject / 12-week MVP|Supports staged scope control.|
|Evidence standard|Reproducible quantitative<br>experiments plus trajectory<br>analysis|Combines performance with<br>mechanistic evidence.|



Alternative methods such as purely philosophical argument, natural-language benchmarking from the start, or human-subject cognitive experiments are not sufficient for the central architectural claim. They may become relevant later, but the MVP requires causal control that synthetic tasks provide. 

## **18. Experimental Programme** 

The experimental programme is staged so that each result can answer a distinct question. 

|**Experiment**|**Question**|**Primary comparison**|
|---|---|---|
|E1. Vector vsgraph|Does relational structure help?|Recurrent vector vs staticgraph|
|E2. Static vs dynamic|Does workspace evolution help?|Staticgraph vs dynamicgraph|
|E3. Activityablation|Does adaptive capacityhelp?|All-active vsgated|
|E4. Halting|Does resolution save compute or<br>improve correctness?|Fixed-step vs critic|
|E5. Persistence|Does retained state help?|Reset vspersistent|
|E6. Externalisation|Does explicit workspace help?|Never vs coupled|
|E7. Selectivity|Does learned write choice<br>matter?|Never vs always vs learned|
|E8. Causal intervention|Is external state computationally<br>active?|Original vs corrupted external<br>state|
|E9. Continuous dynamics|Does ODE/LTC improve<br>outcomes?|Discrete recurrent vs continuous|
|E10. Generalisation|Does structure transfer?|Train on short chains; test<br>longer/unseen compositions|



## **19. Baselines and Ablations** 

|**Model**|**Purpose**|
|---|---|
|MLP / single latent vector|Minimal structured-state control|
|Recurrent latent vector|Direct continuous-latent reasoningcontrol|
|Static GNN|Graph structure without dynamic adaptation|
|Dynamicgraph,deterministic|Core internal workspace|
|Dynamicgraph + activity|Tests adaptive capacity|
|Dynamicgraph + resolution|Tests adaptive stopping|
|Dynamicgraph + external workspace|Core coupled model|
|Always-externalise|Tests whether selectivityis necessary|
|Never-externalise|Direct externalisation control|
|Continuous-time branch|Tests dynamics mechanism|



###### Important controls: 

- Match parameter count as closely as practical. 

- Match or report compute budgets. 

- Control maximum reasoning iterations. 

- Use the same data generator and train/test split. 

- Repeat key experiments over multiple random seeds. 

## **20. Causal and Falsification Tests** 

#### **20.1 External-state intervention** 

##### **L1 → W1 → L2 → solution** 

##### **L1 → corrupted(W1) → L2′ → solution′** 

The corruption should be targeted. For example, change a relevant quantity, delete a relationship, or replace an external record with a controlled alternative. The original input and latent pre-write state remain unchanged. 

#### **20.2 Latent trajectory intervention** 

A corresponding intervention can freeze or randomise selected latent slots and test whether the model compensates through other slots or external state. 

#### **20.3 Falsification criteria** 

- The full model does not outperform simpler baselines on tasks requiring structural reconfiguration. 

- Any gain disappears after matching parameter count, compute or reasoning steps. 

- External writes improve logging but do not affect later latent trajectories. 

- The model always externalises everything or never externalises, indicating no learned policy value. 

- The graph collapses into a redundant global vector representation. 

- The resolution critic simply predicts the final answer instead of evaluating state sufficiency. 

- Continuous dynamics add cost without measurable benefit. 

## **21. Evaluation Metrics and Statistical Analysis** 

|**Metric**|**Why it matters**|
|---|---|
|Exact accuracy|Primarytask success|
|Compositionalgeneralisation|Tests structural transfer rather than memorisation|
|Reasoningiterations|Measures adaptive compute|
|External writes / reads|Measures workspace usage|
|Active-slot count|Measures capacityallocation|
|State diversity|Detects latent collapse|
|Edge entropy/ density|Detects structural collapse or explosion<br>|
|Latency/ wall-clock time|Testspractical eficiency|
|Calibration of resolution|Tests stoppingreliability|
|Corruption sensitivity|Tests causal external use|



Report mean and variance across seeds where feasible, with confidence intervals or another clearly stated uncertainty measure. Do not rely only on a single best run. For paired ablations, use the same task seeds when possible to improve sensitivity. 

## **22. Implementation Plan and Technology Stack** 

|**Layer**|**Recommended technology**|**Reason**|
|---|---|---|
|Language|Python 3.11+|Research iteration and ML|



|||ecosystem|
|---|---|---|
|Deep learning|PyTorch|Autograd and fexible custom<br>models|
|Graph|PyTorch Geometric or custom|Graph batching/messagepassing|
|Continuous dynamics|torchdifeqor equivalent|ODE experiment branch|
|Numerics|NumPy|Datageneration/analysis|
|Tracking|MLfow or Weights & Biases|Experiment management|
|Testing|pytest|Reproducibility|
|Confg|YAML + dataclasses/Hydra|Systematic ablations|
|Data|JSONL + tensors|Portable synthetic datasets|
|Visualisation|Matplotlib|Trajectorydiagnostics|
|Version control|Git + GitHub|Code and decision history|
|Environment|uv or Poetry|Reproducible dependencies|



#### **22.1 Engineering principles** 

- Keep every experiment reproducible from configuration. 

- Log all latent and external state needed for analysis. 

- Freeze earlier modules before adding new mechanisms. 

- Prefer the smallest model that can answer the research question. 

- Do not optimise away inspectability during the MVP. 

## **23. Repository and Data Architecture** 

project/ 

- ├── README.md 

- ├── pyproject.toml ├── configs/ 

- │   ── base.yaml├ 

- │   ── baselines/├ 

- │   └── experiments/ 

- ├── src/ │   ── data/├ │   ── models/├ │   │   ── encoder.py├ │   │   ── latent_workspace.py├ 

- │   │   ── message_passing.py├ │   │   ── activity.py├ │   │   ── relationships.py├ │   │   ── external_workspace.py├ │   │   ── externaliser.py├ │   │   ── reader.py├ │   │   ── resolution.py├ │   │   ── readout.py├ │   │   └── decoder.py 

- │   ── training/├ │   ── evaluation/├ │   └── utils/ 

- ├── tests/ 

- ├── scripts/ 

- ├── notebooks/ 

- ├── results/ 

- └── docs/ 

- ── architecture.md├ 

- ── experiments.md├ 

- └── decisions.md 

#### **23.1 Data schema** 

Each generated task should retain: 

- input facts / relations; 

- ground-truth answer; 

- optional proof or operation trace for supervision; 

- task family and difficulty parameters; 

- random seed; 

- train/validation/test split identifier. 

## **24. MVP Timeline and Deliverables** 

|**Phase**|**Weeks**|**Deliverable**|**Exit criterion**|
|---|---|---|---|
|1. Minimal latent<br>workspace|1–2|Vector baseline + fxed-<br>slotgraph|Stable training and<br>baseline comparison|
|2. Adaptive capacity|3|Activity gates|Measurable activity<br>behaviour|
|3. Resolution|4|Resolution critic +<br>halting|Useful adaptive<br>stopping|
|4. Persistence|5|Persistent graph state|Related-input beneft<br>test|
|5. External workspace<br>v1|6–7|Structured read/write<br>workspace|Inspectable coupling|
|6. Bidirectional coupling|8|Conceptual MVP|External intervention<br>works|
|7. Learned<br>externalisation|9–10|Selective policy|Never/always/learned<br>comparison|
|8. Evaluation|11–12|Full results package|Reproducible<br>baselines/ablations|



#### **24.1 Week-by-week** 

**Week 1:** Environment, data generator, recurrent-vector baseline. 

**Week 2:** Fixed-slot graph, message passing, permutation tests. 

**Week 3:** Activity gates, logging, sparsity controls. 

**Week 4:** Resolution critic, halting, fixed-step comparison. 

**Week 5:** Persistent state and multi-instance benchmark. 

###### **Week 6:** Structured external workspace and deterministic interfaces. 

- **Week 7:** Latent → external and external → latent integration. 

- **Week 8:** Complete loop and causal corruption experiment. 

- **Week 9:** Learned externalisation gate. 

- **Week 10:** Selective writes, write-cost and harder tasks. 

- **Week 11:** Baselines, ablations, multiple seeds. 

**Week 12:** Final analysis, documentation and dissertation-ready results. 

## **25. Detailed Task Suite** 

|**Task family**|**Example**<br>|**Primary property**|
|---|---|---|
|Arithmetic composition|50 + 20 = 70;afordability|Composition|
|Relational chains|A left of B;B left of C|Multi-hopstructure|
|Distractors|Relevant + irrelevant facts|Selective activation|
|Contradictions|Competingstatements|Hypothesis management|
|Compositionalgeneralisation|Unseen relation combinations|Structural transfer<br>|
|Externalisation-sensitive|Problems with many<br>intermediate dependencies|Workspace beneft|
|Persistence|Turn 1 state reused in Turn 2|Persistent memory|
|Corruption tests|Alter external intermediate fact|Causal usage|



Task complexity should be parameterised so that scaling curves can be measured rather than relying on a small collection of fixed examples. 

## **26. Worked Examples** 

#### **26.1 Shoe affordability** 

Input: “John has £50. Mary gives John £20. The shoe costs £70. Can John buy it?” 

1. Encode entities, quantities and relations. 

2. Activate latent slots relevant to John, money, transfer, shoe and price. 

3. Perform relational updates to combine the quantities. 

4. Optionally write money(John)=70 to the external workspace. 

5. Read the structured state back into the latent workspace. 

6. Evaluate money >= price. 

7. Check resolution and stop. 

8. Invariant readout and decode the decision. 

#### **26.2 Persistent multi-turn state** 

Turn 1: “John has £50.” Turn 2: “Mary gives him £20.” 

The intended experiment is that Turn 2 perturbs an existing state rather than requiring complete reconstruction. The benchmark can compare persistent and reset conditions while holding total information constant. 

#### **26.3 Relational chain** 

Input: “A is left of B. B is left of C. Where is A relative to C?” 

The task is useful because the correct answer requires composing two relations. The external workspace could preserve the intermediate relation or a normalised operation, allowing a direct test of whether explicit structure reduces latent burden. 

## **27. Relationship to Existing Research** 

#### **27.1 Continuous latent reasoning** 

COCONUT is the closest conceptual precedent for separating reasoning from intermediate language generation. It uses the final hidden state as a continuous reasoning representation and feeds it back as an embedding [1]. Recurrent-depth latent reasoning further demonstrates that test-time computation can be expanded through repeated latent transformations [2]. The project extends this direction by treating the latent state as a structured multi-node workspace with explicit relationships, activity and external coupling. 

#### **27.2 Graph neural networks and equivariance** 

Graph neural networks provide established relational computation; E(n)-equivariant GNNs additionally formalise symmetry-aware transformations [3]. The project should therefore not claim novelty for 'using a graph' or 'using equivariance'. 

#### **27.3 Neural ODEs and liquid dynamics** 

Neural ODEs model continuous hidden-state trajectories [4]. Liquid Time-Constant networks introduce adaptive time constants within continuous recurrent systems [5]. Both are candidate mechanisms for the latent trajectory, not the research contribution itself. 

#### **27.4 KANs** 

KANs use learnable activation functions on edges rather than fixed node activations [6]. They may be useful for future experiments on richer relational transformations but are not a required component of the MVP. 

#### **27.5 Equilibrium models** 

Deep Equilibrium models provide a formal way to represent the result of repeatedly applying a transformation as a fixed point [7]. This is relevant to the resolution/convergence question but should remain optional until the basic architecture is validated. 

#### **27.6 External memory** 

Neural Turing Machines and Differentiable Neural Computers demonstrate learned read/write interaction with external memory [11,12]. These systems establish that neural controllers can use explicit memory for 

algorithmic and structured tasks. The current project differs in making the internal–external co-refinement loop, adaptive externalisation and resolution criterion central to the reasoning hypothesis. 

#### **27.7 Cognitive offloading** 

Cognitive offloading research motivates the idea that external representations can change the computational burden of a task. The literature also warns that losing access to the external representation can harm performance [9,10]. In the project, this becomes an experimental prediction: the benefit of externalisation should be task-dependent and its causal value should be measurable. 

#### **27.8 Memory and tool-augmented language models** 

Systems such as MemGPT treat memory management as a first-class architectural concern [13], while recent surveys describe tool learning as a planning, selection, execution and response pipeline [14,15]. These are important neighbouring areas. The project's narrower focus is on the computational workspace itself and the bidirectional transformation between latent and explicit states. 

## **28. Novelty and Contribution** 

The contribution should be stated narrowly. It is not 'latent reasoning', 'graph reasoning', 'external memory' or 'equivariant neural networks' in isolation. 

|**Candidate claim**|**Assessment**|
|---|---|
|Reasoningcan happen without tokens|Establishedprecedent[1,2].|
|Graphs can encode relations|Establishedprecedent[3].|
|Continuous latent dynamics are useful|Established research direction[4,5].|
|Neural networks can use external memory|Establishedprecedent[11,12].|
|External tools/memorycan augment LLMs<br>|Established and rapidly growingarea[13–15].|
|Unifed persistent latent–external workspace with<br>selective co-refnement and explicit resolution|The proposed research contribution to test<br>experimentally.|



The strongest possible result is therefore not a philosophical proof of understanding. It is empirical evidence that the combined structural bias provides benefits that cannot be explained by parameter count, extra compute or generic memory capacity. 

## **29. Risks, Failure Modes and Mitigations** 

|**Risk**|**Failure mode**|**Mitigation**|
|---|---|---|
|Latent collapse|Slots become redundant|Monitor diversity, activity entropy<br>and ablategraph structure|
|Shortcut learning|Model memorises templates|Held-out compositions and<br>structuralgeneralisation|
|External label leakage|Workspace stores target answer|Strict schema and no target<br>access|
|External workspace ignored|Writes occur but never matter|Read ablation and corruption<br>intervention|
|External workspace overused|Always write everything|Write-cost/sparsityregularisation|
|Resolution cheating|Critic predicts answer directly|Hold-out templates and<br>trajectory-based supervision|
|Compute explosion|Too manyupdates/edges|Hard budgets and activity|



|||regularisation|
|---|---|---|
|ODE complexity|Slow without beneft|Treat continuous dynamics as<br>optional branch<br>|
|Stochastic instability|Sampling harms convergence|Deterministic model frst; gradual<br>stochasticity|
|Interpretability overclaim|Graph labels treated as literal<br>concepts|Behavioural probes rather than<br>semantic assumptions|
|Scope creep|Too manymechanisms at once|Stage and freeze modules|



## **30. Ethics, Reproducibility and Research Integrity** 

The project is primarily computational and can initially be conducted without human or sensitive personal data. This is an advantage for the MVP because it reduces privacy, consent and data-governance concerns. 

- Do not use private or sensitive user data in training or evaluation. 

- Document all generated task distributions and exclusions. 

- Report negative results and failed ablations. 

- Avoid claims of human-like understanding based only on benchmark accuracy. 

- Separate sourced claims from hypotheses and engineering choices. 

- Keep experiment configurations, seeds and code under version control. 

- Record changes to the architecture in a decision log. 

For later natural-language expansion, additional issues arise around data licensing, memorisation, bias, hallucination and the safety of autonomous tool interaction. Those are outside the controlled MVP but should be addressed before deployment-oriented work. 

## **31. Limitations and Open Questions** 

|**Openquestion**|**Currentposition**|**How to resolve**|
|---|---|---|
|What exactly is a node?|Unresolved|Behavioural probes and<br>controlled interventions|
|What exactly is an edge?|Unresolved|Compare typed vs untyped<br>relational forms|
|Does explicit topology beat<br>implicit attention?|Unknown|Matched transformer/GNN<br>controls|
|Do continuous dynamics beat<br>recurrence?|Unknown|Paired discrete vs ODE/LTC<br>experiments|
|Should state be stochastic?|Unknown|Calibration and robustness<br>experiments|
|What is the best external<br>representation?|Unknown|Compare typed records, graphs<br>and domain-specifc forms|
|When should the system<br>externalise?|Unknown<br>|Learned policy + intervention|
|What is resolution?|Operationally defned,<br>theoreticallyopen|Critic calibration and stopping<br>benchmarks|
|Doespersistence help?|Hypothesised|Reset vspersistent trials|
|Will synthetic-task gains<br>transfer?|Unknown|Later natural-language and<br>multimodal expansion|



## **32. Future Development Roadmap** 

|**Stage**|**Purpose**|
|---|---|
|Phase 0|Formalise node, edge, workspace and resolution<br>defnitions<br>|
|Phase 1|Deterministic fxed-slot relational workspace|
|Phase 2|Adaptive activity/recruitment|
|Phase 3|Learned resolution critic|
|Phase 4|Persistent state across turns<br>|
|Phase 5|External workspace + bidirectional co-refnement|
|Phase 6|Learned selective externalisation|
|Phase 7|Continuous-time dynamics|
|Phase 8|Probabilistic latent state<br>|
|Phase 9|Domain-specifc external representations|
|Phase 10|Natural-language encoder/decoder integration|
|Phase 11|Broader reasoningbenchmarks|



The ordering is strategic: the research hypothesis should be tested before the system becomes large enough that attribution of performance becomes difficult. 

## **33. Success Criteria** 

Minimum success: 

- The coupled architecture trains reliably on the controlled task suite. 

- At least three baselines are reproduced under comparable budgets. 

- The external workspace is inspectable at every reasoning iteration. 

- Adaptive stopping behaves sensibly within a hard step budget. 

- The external corruption experiment produces measurable downstream effects when relevant information is changed. 

- Ablations identify which components contribute. 

Strong success: 

- Coupled reasoning improves structural generalisation or efficiency. 

- The learned externalisation policy is selective and task-dependent. 

- The external workspace produces causal changes in later latent trajectories. 

- Benefits remain after controlling for parameters, compute and number of reasoning steps. 

- The architecture exhibits meaningful capacity recruitment rather than simple global-state collapse. 

Failure is scientifically useful if it shows that simpler recurrent vectors or static graphs explain the same results. 

## **34. Expected Outputs** 

|**Output**|**Description**|
|---|---|
|Software|Open,modular reference implementation|
|Datasetgenerators|Synthetic tasks with exact solutions|



|Experiment suite|Baselines,ablations and intervention scripts|
|---|---|
|Diagnostics|Latent activity, trajectory and external-state<br>visualisations<br>|
|Results|Accuracy, eficiency, generalisation and causal-<br>intervention tables<br>|
|Architecture report|Formal system defnition and design rationale|
|Dissertation material|Literature review, methodology, experiments,<br>results and discussion|



## **35. Final Project Definition** 

The project investigates whether reasoning can be implemented as adaptive computation over a persistent relational workspace rather than requiring language tokens as the intermediate computational substrate. Its core object is a graph-like latent workspace whose vector states, relational interactions and effective activity can evolve. The project then extends this internal workspace with an explicit external workspace that can store structured intermediate representations and feed them back into the latent computation. 

##### **Internal compression + External expansion + Iterative co-refinement + Adaptive resolution** 

The architecture is best understood as a computational workspace rather than simply a neural network layer. The latent state provides a flexible, distributed medium for implicit relational computation. The external state provides a persistent, explicit medium for selected structure that may be costly or unreliable to retain internally. The system continually decides where useful information should live and when the resulting combined state is sufficient to answer the task. 

The central empirical claim remains deliberately narrow: on problems that require composition, relational binding, iterative state construction, persistence or structured intermediate computation, a dynamically reconfigurable latent workspace with selective externalisation may outperform simpler latent-reasoning alternatives in accuracy, generalisation, robustness or computational efficiency. Whether this succeeds is an experimental question. 

## **Research Design at a Glance** 

|**Element**|**Defnition**|
|---|---|
|Independent variables|Workspace structure, dynamic activity, relationship<br>adaptation, externalisation, persistence, dynamics<br>type,stoppingrule.|
|Dependent variables|Accuracy, generalisation, reasoning steps, cost,<br>robustness,latent/external trajectoryefects.|
|Controls|<br>Vector baseline, static graph, no-external controls,<br>fxed-stepcontrols,matched budgets.|
|Interventions|External-state corruption, latent-slot ablation,<br>fxed/reordered node representations.|
|Primary evidence|<br>Comparative performance + causal trajectory<br>evidence + ablation results.|
|Reproducibility|Fixed confgs, seeds, versioned datasets, logged<br>state trajectories and checkpoints.|



## **References** 

[1] Hao, S., Sukhbaatar, S., Su, D., Li, X., Hu, Z., Weston, J., & Tian, Y. (2024). Training large language models to reason in a continuous latent space (COCONUT). arXiv:2412.06769. https://arxiv.org/abs/2412.06769 

[2] Geiping, J., McLeish, S., Jain, N., Kirchenbauer, J., Singh, S., Bartoldson, B. R., Kailkhura, B., Bhatele, A., & Goldstein, T. (2025). Scaling up test-time compute with latent reasoning: A recurrent depth approach. arXiv:2502.05171. https://arxiv.org/abs/2502.05171 

[3] Satorras, V. G., Hoogeboom, E., & Welling, M. (2021). E(n) equivariant graph neural networks. Proceedings of ICML, 139, 9323–9332. https://proceedings.mlr.press/v139/satorras21a.html 

[4] Chen, R. T. Q., Rubanova, Y., Bettencourt, J., & Duvenaud, D. (2018). Neural ordinary differential equations. arXiv:1806.07366. https://arxiv.org/abs/1806.07366 

[5] Hasani, R., Lechner, M., Amini, D., Rus, D., & Grosu, R. (2020). Liquid time-constant networks. arXiv:2006.04439. https://arxiv.org/abs/2006.04439 

[6] Liu, Z., Wang, Y., Vaidya, S., Ruehle, F., Halverson, J., Soljačić, M., Hou, T. Y., & Tegmark, M. (2024). KAN: Kolmogorov-Arnold networks. arXiv:2404.19756. https://arxiv.org/abs/2404.19756 

[7] Bai, S., Kolter, J. Z., & Koltun, V. (2019). Deep equilibrium models. arXiv:1909.01377. https://arxiv.org/abs/1909.01377 

[8] Graves, A., Wayne, G., & Danihelka, I. (2014). Neural Turing machines. arXiv:1410.5401. https://arxiv.org/abs/1410.5401 

[9] Richmond, L. L., & Taylor, R. G. (2025). The benefits and potential costs of cognitive offloading for retrospective information. Nature Reviews Psychology, 4, 312–321. https://www.nature.com/articles/s44159-025-00432-2 

[10] Burnett, K. J., & Richmond, L. L. (2026). Meta-analytic investigations of the effect of cognitive offloading on memory-based task performance and interindividual variability. Memory & Cognition, 54(1), 144–168. https://pubmed.ncbi.nlm.nih.gov/40500483/ 

[11] Graves, A., Wayne, G., Reynolds, M., et al. (2016). Hybrid computing using a neural network with dynamic external memory. Nature, 538, 471–476. https://www.nature.com/articles/nature20101 

[12] Packer, C., Wooders, S., Lin, K., Fang, V., Patil, S. G., Stoica, I., & Gonzalez, J. E. (2023). MemGPT: Towards LLMs as operating systems. arXiv:2310.08560. https://arxiv.org/abs/2310.08560 

[13] Li, Z., Song, S., Wang, H., et al. (2025). MemOS: An operating system for memory-augmented generation in large language models. arXiv:2505.22101. https://arxiv.org/abs/2505.22101 

[14] Chen, J., Wu, H., Pang, J., Wang, Y., Zhang, D., Sun, C., et al. (2025). Tool learning with language models: A comprehensive survey of methods, pipelines, and benchmarks. Machine Intelligence Research / Springer Open Access. https://link.springer.com/article/10.1007/s44336-025-00024-x 

[15] Xu, W., Huang, C., Gao, S., et al. (2025). LLM-based agents for tool learning: A survey. Data Science and Engineering, 10, 533–563. https://doi.org/10.1007/s41019-025-00296-9 

## **Project Source Basis** 

The project's own evolving design is grounded in the supplied Dynamic Vector-Graph Latent Reasoning Technical Design and the associated MVP Build Specification. Those documents provide the project's internal hypotheses, decisions, unresolved questions, prototype roadmap and implementation choices. This report expands and consolidates those materials rather than treating them as external validation. 

###### **END OF REPORT** 

