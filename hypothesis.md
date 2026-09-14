Yes. The **initial latent-space hypothesis** was actually simpler than where we've taken the project now.

### The original idea

The starting point was roughly:

> **Understanding/reasoning does not necessarily require manipulating language tokens. Instead, an input could be transformed into a latent conceptual representation, and reasoning could occur by allowing the components of that representation to interact and reorganise in latent space until an appropriate state is reached.**

The motivating example was the **shoe purchase** scenario:

> Person E describes a red shoe in a shopping mall, its price, and asks Person B to pay for it.

Rather than Person B internally reasoning through a sequence of English sentences, the hypothesis was that the information could produce a latent configuration containing representations corresponding to things such as:

```text
             ┌───────────┐
             │   SHOE    │
             └─────┬─────┘
                   │
          ┌────────┼────────┐
          ↓        ↓        ↓
        RED      PRICE     MALL
                   │
                   ↓
                 COST
                   │
                   ↓
                 PAY
```

But importantly, these weren't intended to be literal labelled nodes. They were **learned vector representations** whose relationships and interactions would encode the relevant conceptual structure.

---

## The crucial distinction

The original hypothesis wasn't simply:

> "Put embeddings in a graph."

It was closer to:

> **The latent representation itself becomes the computational workspace.**

So instead of:

```text
Input → tokens → attention → next token → next token → answer
```

the proposed conceptual process was:

```text
Input
  ↓
latent representation
  ↓
interacting latent components
  ↓
reorganisation / transformation
  ↓
resolved latent state
  ↓
output
```

Language could therefore act primarily as an **input/output interface**, rather than being the medium through which every intermediate reasoning operation has to be expressed.

---

# Then came the "dynamic" part

The initial static conception evolved into the idea that the latent space shouldn't merely contain a fixed set of vectors.

It should be able to **change during reasoning**.

You eventually described it in terms of something like:

$$
G_t=(V_t,E_t,A_t)
$$

where:

* \(V_t\) = latent vector nodes
* \(E_t\) = relationships between them
* \(A_t\) = their effective activity

The state could therefore evolve as:

$$
v_i(t)\rightarrow v_i(t+\Delta t)
$$

while relationships could change:

$$
e_{ij}(t)\rightarrow e_{ij}(t+\Delta t)
$$

and latent capacity could change:

$$
a_i(t)\rightarrow a_i(t+\Delta t)
$$

This led to the idea that reasoning isn't necessarily:

> **"move through a sequence of hidden states."**

It could instead be:

> **"continually reshape a computational latent space until it reaches a useful configuration."**

That's probably the most important conceptual step in the original hypothesis.

---

# And the "node creation" idea

You also explored the possibility that the system might be able to introduce new latent nodes when the existing representation wasn't sufficient.

Conceptually:

```text
Initial state

○ ○ ○ ○ ○

        ↓ problem becomes more complex

○ ○ ● ○ ● ○ ○
    ↑     ↑
 new structures
```

But we later recognised that literally creating neural parameters/nodes dynamically would make the MVP unnecessarily difficult.

So the practical implementation became:

> **fixed latent slots + learned activity/recruitment**

Meaning:

```text
inactive slot
     ↓
activated
     ↓
participates in reasoning
```

This gives us the computational behaviour of "creating a new thought structure" without requiring the neural network itself to dynamically change size.

---

# The original unresolved question

One of the most important things about the initial hypothesis is that **we did not pretend to know what a latent node represented**.

We identified this as an open research question.

A node might represent:

* an entity;
* a concept;
* a relation;
* an intermediate computation;
* a temporary reasoning construct;
* or something more abstract that doesn't map cleanly onto a human concept.

Likewise, we hadn't definitively established:

> **What constitutes a reasoning step?**

That remains important because a "reasoning step" in this architecture isn't necessarily equivalent to generating a token.

---

# Resolution was another key part

The original idea also needed an answer to:

> **When does the latent space stop changing?**

We rejected the simplistic idea that:

> "The highest-probability state is the answer."

Instead, the eventual formulation became something closer to:

$$
R_t=f(
\text{solution confidence},
\text{constraint satisfaction},
\text{state stability}
)
$$

So the system should stop when the state is:

1. sufficiently confident;
2. consistent with the relevant constraints;
3. no longer undergoing meaningful changes.

This is where the idea of **adaptive resolution** came from.

---

# The external workspace came later

This is important because it helps distinguish the **original hypothesis** from the current project.

Originally:

$$
\boxed{\text{Input}\rightarrow\text{Dynamic Latent Space}\rightarrow\text{Output}}
$$

Then the project evolved into:

$$
\boxed{
\text{Input}
\rightarrow
\text{Latent Workspace}
\leftrightarrow
\text{External Workspace}
\rightarrow
\text{Resolution}
\rightarrow
\text{Output}
}
$$

The external scratchpad idea was a **later extension**, not the original premise.

---

## So, stripped right back to its essence

I'd describe the **initial latent-space hypothesis** as:

> **A system can potentially reason by constructing and continuously reorganising a relational latent representation of a problem, rather than requiring intermediate reasoning to be expressed as language tokens. The latent representation functions as a computational workspace: its vector states and relationships change in response to the problem until a sufficiently resolved state is reached, after which that state is decoded into an output.**

And the conceptual leap that made your project distinctive was:

> **Don't just use latent space to represent the answer. Make the latent space itself the place where the computation happens.**

Everything we've added since — **dynamic nodes/activity, equivariance, continuous dynamics, persistence, external scratchpads, co-refinement and adaptive resolution** — has essentially been an attempt to make that original idea concrete and testable.
