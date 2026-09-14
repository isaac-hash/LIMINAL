### **MVP BUILD SPECIFICATION** 

# Adaptive Internal–External Latent Workspace 

Technical architecture, tech stack, deliverables, milestones and evaluation 

Regenerated version — 9 September 2026 

## **1. Executive Summary** 

The MVP is a controlled research prototype for testing a persistent, adaptive latent computational workspace coupled to an explicit external workspace. It is deliberately small: the goal is to establish whether internal latent representations and explicit external representations can participate in an iterative co-refinement loop. 

##### **Input → Latent Workspace ↔ External Workspace → Resolution → Invariant Readout → Output** 

## **2. Research Questions** 

- Can a relational latent workspace perform useful multi-step reasoning? 

- Does adaptive latent activity/capacity improve over a fixed representation? 

- Does an external structured workspace causally affect later latent computation? 

- Can the system determine when a problem is sufficiently resolved? 

- Does selective externalisation outperform never- and always-externalise controls? 

## **3. Scope and Non-Goals** 

|**In scope**|**Out of scope**|
|---|---|
|Fixed latent vector slots|Large-scale LLMpretraining|
|Equivariant relational/messagepassing|Human-level intelligence claims|
|Recurrent latent updates|Open-ended agents|
|Learned activity/recruitment|Rich multimodalperception|
|Structured external scratchpad|Complex tool ecosystems|
|Bidirectional coupling|Production deployment|
|Resolution/haltingcritic|Claims about reproducinghuman cognition|



## **4. Core Architecture** 

Latent workspace: V_t contains continuous vector slots; E_t represents relationships; A_t represents effective activity. The workspace is repeatedly updated. 

External workspace: a small structured scratchpad containing entities, values, relations, operations, constraints and status. It should be inspectable, serialisable and easy to intervene on. 

Use fixed tensor shapes with soft activity gates rather than literal runtime creation of neural modules. 

## **5. Recommended Technology Stack** 

|**Layer**|**Technology**|**Purpose**|
|---|---|---|
|Language|Python 3.11+|Implementation|
|Deeplearning|PyTorch|Model and training|
|Graph|PyTorch Geometric|Messagepassing|
|Numerics|NumPy|Synthetic data/analysis|
|Tracking|MLfow or Weights & Biases|Runs/checkpoints|
|Testing|pytest|Unit/integration tests|
|Confg|YAML + dataclasses/Hydra|Reproducibility|
|Data|JSONL + PyTorch tensors|Datasets|
|Plots|Matplotlib|Analysis|
|Version control|Git + GitHub|Code/history|



Environment uv or Poetry Dependencies Compute GPU preferred; CPU baseline Training/inference 

## **6. Software Modules** 

|**Module**|**Responsibility**|
|---|---|
|encoder|Problem → initial latent state|
|latent_workspace|Stores V,E,A|
|message_passing|Relational updates|
|activity_controller|Slot activation|
|relationship_controller|Relationshipadaptation|
|external_workspace|Structured scratchpad|
|externaliser|Chooses what to write|
|reader|External state → latent signal|
|resolution|Resolved score/halting<br>|
|readout|Invariant fnal aggregation|
|decoder|Solution → answer|
|trainer|Optimisation/logging|
|evaluation|Baselines/ablations/metrics|



## **7. Mathematical Skeleton** 

G_t = (V_t, E_t, A_t) 

G_(t+1) = F(G_t, X, R_t), where X is the task encoding and R_t is information read from the external workspace. 

W_(t+1) = H(W_t, G_(t+1)). 

Soft activity gates: a_i = sigmoid(g_i). This gives the functional behaviour of recruitment while keeping tensor shapes fixed. 

Use equivariant operations during node computation and invariant aggregation for final readout. 

## **8. External Workspace v1** 

|**Record**|**Example**|**Purpose**|
|---|---|---|
|Entity|shoe_A|Persistent reference|
|Attribute|price(shoe_A)=70|Explicit value|
|Relation|money(John)=50|Structured fact|
|Operation|50 + 20|Intermediate computation|
|Constraint|money>=price<br>|Decision condition|
|Status|satisfed/unresolved|Resolution support|



Avoid free-form natural language initially. The structured form makes the workspace inspectable, corruptible and suitable for causal intervention. 

## **9. Resolution and Halting** 

Resolution should combine solution confidence, relevant constraint satisfaction and latent-state stability rather than simply choosing the highest-probability state. 

**r_t = f(confidence_t, constraints_t, stability_t)** 

Initially use supervised/teacher-forced stopping where possible plus a hard maximum iteration budget. 

## **10. Twelve-Week Timeline** 

|**Phase**|**Weeks**|**Build**<br>|**Deliverable**|
|---|---|---|---|
|Minimal latent workspace|1–2|Encoder, fxed slots,<br>equivariant message<br>passing,recurrence|Latent-only prototype|
|Adaptive capacity|3|Activity gates/recruitment|Activity analysis +<br>ablation|
|Resolution|4|Critic + adaptive halting|Variable reasoninglength|
|Persistence|5|Carry state across related<br>inputs|Persistence experiment|
|External workspace v1|6–7|Structured scratchpad +<br>read/write|Inspectable coupled<br>prototype|
|Bidirectional coupling|8|Write → read → continue|Conceptual MVP|
|Learned externalisation|9–10|Learn when/what to write|Selective externalisation|
|Evaluation|11–12|Baselines, ablations,<br>interventions|Evidence package|



## **11. Weekly Build Plan** 

#### **Week 1** 

- Repository/environment 

- Synthetic arithmetic generator 

- Single recurrent-vector baseline 

- Tests/configuration 

#### **Week 2** 

- Fixed latent slots 

- Relational message passing 

- Permutation tests 

- Vector vs graph baseline 

#### **Week 3** 

- Activity gates 

- Activity logging 

- Regularisation 

- Fixed vs adaptive ablation 

#### **Week 4** 

- Resolution critic 

- Confidence/constraint/stability signals 

- Adaptive halting 

- Fixed-step comparison 

#### **Week 5** 

- Persistent state 

- Reset/update rules 

- Related-task benchmark 

#### **Week 6** 

- Typed external state 

- Deterministic read/write 

- Serialisation/inspection 

#### **Week 7** 

- Connect latent writes 

- Connect external reads 

- Check label leakage 

#### **Week 8** 

- Full latent ↔ external loop 

- Coupled vs never-externalise 

- External corruption intervention 

#### **Week 9** 

- Learned externalisation gate 

- Candidate write operations 

- Write-cost regularisation 

#### **Week 10** 

- Selective externalisation 

- Never/always/learned comparison 

- Increase task difficulty 

#### **Week 11** 

- Core baselines 

- Ablation matrix 

- Multiple seeds 

#### **Week 12** 

- Final plots 

- Package checkpoints/configs 

- Document findings 

- Dissertation-ready tables 

## **12. MVP Task Suite** 

|**Task family**|**Capability**|**Dificulty**|
|---|---|---|
|Arithmetic composition|Multi-step quantitative reasoning|Operation count|
|Relational chains|Multi-hopreasoning|Chain length|
|Distractors|Selective relevance|Distractor ratio|
|Contradictions|Confict handling|Confict count|
|Compositionalgeneralisation|<br>Novel combinations|<br>Train/testgap|
|Externalisation-sensitive|Beneft of explicit structure|Complexity|
|Persistence|<br>Useful state across inputs|Sequence length|



## **13. First End-to-End Demonstration** 

_“John has £50. Mary gives John £20. The shoe costs £70. Can John buy it?”_ 

1. Encode entities, quantities and relations. 

2. Run several latent updates. 

3. Optionally externalise an intermediate structured fact. 

4. Read it back into the latent workspace. 

5. Continue until constraints are satisfied and the state is stable. 

6. Invariant readout → final decision. 

## **14. Baselines** 

|**Baseline**|**Purpose**|
|---|---|
|Single recurrent latent vector|Test relational workspace|
|Static latentgraph|Test dynamic adaptation|
|Dynamic latentgraph|Core internal model|
|Dynamicgraph + external workspace|Core coupled model|
|Continuous latent reasoningbaseline|Token-free latent comparison|
|Dynamic vectors only|Ablate adaptive capacity/structure|
|Dynamic activityonly|Test capacityadaptation|
|Dynamic relationships only|Test structural adaptation|
|Never externalise|Direct control|
|Always externalise|Test selectivity|
|Fixed-stepreasoning|Haltingcontrol|



## **15. Evaluation Metrics** 

- Answer accuracy/exact correctness 

- Compositional generalisation 

- Average reasoning iterations 

- External writes/reads 

- Active-slot count over time 

- Latent-state change near resolution 

- Inference cost/wall-clock time 

- Distractor robustness 

- External-state corruption robustness 

- Ablation effect sizes 

## **16. Critical Causal Experiment** 

Accuracy improvement alone is not enough. Intervene on the external state while keeping the original input and prewrite latent state unchanged. 

##### **L₁ → W₁ → L₂ → solution** 

**L₁ → corrupted(W₁) → L₂′ → solution′** 

Measure changes in the next latent state, later trajectory, reasoning length and final answer. A systematic taskrelevant effect supports the claim that the external workspace participates in computation rather than merely logging it. 

## **17. Ablations** 

|**Removed component**|**Question**|
|---|---|
|Activity gates|Does adaptive latent capacitymatter?|
|Relationshipadaptation|Does changingstructure matter?|
|External write|Can reasoningwork without externalisation?|
|External read|Is written information reused?|
|Learned externalisation|Does selective writingmatter?|
|Resolution critic|Does adaptive stoppingmatter?|
|Persistence|Does state across inputs matter?|
|Equivariant operations|Doespermutation-consistent computation matter?|



## **18. Repository Structure** 

project/ 

- ├── README.md ├── pyproject.toml ├── configs/ ├── src/ │   ── data/├ │   ── models/├ │   │   ── encoder.py├ │   │   ── latent_workspace.py├ │   │   ── message_passing.py├ │   │   ── activity.py├ │   │   ── relationships.py├ │   │   ── external_workspace.py├ │   │   ── externaliser.py├ │   │   ── resolution.py├ │   │   ── readout.py├ 

- │   │   └── decoder.py 

- │   ── training/├ 

- │   └── evaluation/ 

- ├── tests/ ├── scripts/ ├── notebooks/ 

- ├── results/ 

- └── docs/ 

## **19. Deliverables** 

|**Milestone**|**Deliverable**|**Acceptance criterion**|
|---|---|---|
|End Week 2|Latent-only prototype|Trains and solves baseline tasks|
|End Week 3|Adaptive workspace|Activity changes measurably and is<br>logged|



|End Week 4|Resolution module|Halts before maximum steps on<br>solved cases|
|---|---|---|
|End Week 5|Persistent workspace|Useful state survives related inputs|
|End Week 7|External workspace v1|Structured state can be<br>written/read/inspected|
|End Week 8|Conceptual MVP|End-to-end latent ↔ external loop<br>works|
|End Week 10|Learned externalisation|Selective writes show measurable<br>trade-ofs|
|End Week 12|Evaluation package|Baselines, ablations, interventions,<br>reproducible confgs|



## **20. Definition of Done** 

- Coupled model trains from reproducible configurations. 

- At least three meaningful baselines use comparable budgets. 

- External state is inspectable at each reasoning iteration. 

- Latent activity and write/read events are logged. 

- Adaptive stopping operates within a maximum step budget. 

- External corruption intervention is completed. 

- Ablations isolate the main architectural contributions. 

- Reported results are reproducible from committed code/configuration. 

## **21. Engineering Risks** 

|**Risk**|**Mitigation**|
|---|---|
|Shortcut learning|Held-out compositions + interventions|
|All slots stayactive|Activityregularisation + diagnostics|
|External state leaks target|Strict schema;no direct target access|
|External workspace ignored|Read ablation + corruption test|
|Unstable halting<br>|Supervised critic + hard stepcap<br>|
|Dynamicgraph too dificult|Fixed slots + softgates frst|
|Scope creep|Freeze eachphase before next|
|Hard-to-interpret results|Small synthetic tasks + inspectable state|



## **22. Research Outputs** 

- Reference implementation 

- Formal architecture and notation 

- Controlled benchmark suite 

- Baseline/ablation results 

- Latent trajectory visualisations 

- External-workspace causal intervention results 

- A defensible conclusion about the internal–external coupling hypothesis 

## **23. Recommended Build Strategy** 

**Latent vector → relational workspace → adaptive activity → resolution → external scratchpad → bidirectional loop → learned externalisation → evaluation** 

Introduce the external workspace around Week 6, not as a final add-on. This leaves enough time to test the central internal ↔ external hypothesis and change direction if the evidence is negative. 

## **24. MVP Success Criterion** 

The MVP succeeds if, under controlled conditions, a persistent adaptive latent workspace plus an explicit structured external workspace forms a useful iterative reasoning loop and the external representation has a measurable causal effect on subsequent computation. 

A strong positive result would combine improved accuracy, generalisation and/or efficiency with evidence that the model selectively externalises information and subsequently uses it in later latent updates. 

## **25. Immediate Next Actions** 

7. Create repository/environment. 

8. Implement synthetic arithmetic generator. 

9. Implement recurrent-vector baseline. 

10. Implement fixed-slot latent graph. 

11. Add permutation tests. 

12. Add activity gates/logging. 

13. Implement resolution critic. 

14. Build structured external workspace. 

15. Run external corruption intervention as soon as coupling works. 

16. Then expand learned externalisation and task diversity. 

##### **END OF MVP BUILD SPECIFICATION** 

