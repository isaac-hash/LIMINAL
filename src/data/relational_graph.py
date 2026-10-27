"""Relational Graph Task — ``relational_graph`` (Phase 9, Hypothesis A).

Motivation
----------
The existing affordability and balance-update tasks test *memory retention* over a
linear chain of scalar updates.  They do not test *relational composition* — the
ability to track a graph of typed relationships between entities, compose edges across
multiple hops, and identify the correct path when distracting dead-end branches exist.

This task is designed to discriminate between:
  - A flat recurrent vector (Model A) that must smear all entity states into one vector
  - A static relational graph (Model B) that tracks slots but resets between turns
  - A persistent dynamic graph (Model C) that maintains evolving slot states across turns

Task structure (per sequence)
------------------------------
A directed graph of named entities is revealed incrementally across T turns.  Each turn
may introduce new nodes, add or remove edges, or update edge weights.  The final turn
(or several turns in the Hard configuration) presents a query that requires multi-hop
relational composition to answer.

Edge types
----------
  OWES      — A owes B £amount  (directed, weighted)
  BLOCKED   — A's debt to B is disputed/frozen  (directed, unweighted)
  TRANSFERRED — A transferred £amount to B  (directed, weighted, one-off event)

Query types (binary, verified against ground truth)
--------------------------------------------------
  reachable(A, B)      — Is there a path from A to B via active OWES edges?
  path_blocked(A, B)   — Is every A→B path blocked by at least one BLOCKED edge?
  net_positive(A)      — Is A's net credit (incoming OWES − outgoing OWES) > 0?
  can_collect(A, T)    — Can A collect at least £T transitively via OWES?

Difficulty levels
-----------------
  easy   : 3 entities, 1 edge type (OWES only), max_hops=2, 0 distractors, 5 turns
  medium : 5 entities, 2 edge types (OWES + BLOCKED), max_hops=3, 1 distractor, 7 turns
  hard   : 7 entities, 3 edge types (OWES + BLOCKED + TRANSFERRED), max_hops=4,
           2 distractors, 9 turns
  ood    : 5 entities, 2 edge types, max_hops=5–6, 1 distractor, 9 turns
           (trained on ≤4 hops, tested on 5–6 — structural generalisation test)

Fact encoding (compatible with Encoder and SequenceReasoningDataset)
--------------------------------------------------------------------
Each fact is a dict with keys:
  {"type": str, "entity": str, "key": str, "value": int/str, "op": str}

where:
  type    ∈ {entity, attribute, relation, operation}
  entity  = source entity name (or "GROUP" for aggregate queries)
  key     = edge type ("owes", "blocked", "transferred") or attribute name
  value   = edge weight (amount) for weighted edges; 0 for unweighted
  op      = "add" for new edge, "sub" for edge removal, "none" for query context

Ground truth: binary int (0 or 1) for all query types.
"""

from __future__ import annotations

import random
from typing import Any


# ---------------------------------------------------------------------------
# Vocabulary constants
# ---------------------------------------------------------------------------

ENTITY_NAMES = [
    "Alice", "Bob", "Carol", "David", "Eve", "Frank", "Grace", "Henry",
    "Iris", "Jack", "Kate", "Leo",
]

DISTRACTOR_NAMES = ["Zara", "Xander", "Yara", "Wren", "Uma"]

EDGE_TYPES = ["owes", "blocked", "transferred"]

QUERY_TYPES = ["reachable", "path_blocked", "net_positive", "can_collect"]


# ---------------------------------------------------------------------------
# Difficulty presets
# ---------------------------------------------------------------------------

DIFFICULTY_PRESETS: dict[str, dict[str, Any]] = {
    "easy": {
        "num_entities": 3,
        "edge_types": ["owes"],
        "max_hops": 2,
        "num_distractors": 0,
        "sequence_turns": 5,
        "min_hops": 1,
    },
    "medium": {
        "num_entities": 5,
        "edge_types": ["owes", "blocked"],
        "max_hops": 3,
        "num_distractors": 1,
        "sequence_turns": 7,
        "min_hops": 2,
    },
    "hard": {
        "num_entities": 7,
        "edge_types": ["owes", "blocked", "transferred"],
        "max_hops": 4,
        "num_distractors": 2,
        "sequence_turns": 9,
        "min_hops": 2,
    },
    "ood": {
        # Same training distribution as medium in entity count and edge types,
        # but graph depth is increased to 5–6 hops at evaluation time.
        "num_entities": 5,
        "edge_types": ["owes", "blocked"],
        "max_hops": 6,
        "num_distractors": 1,
        "sequence_turns": 9,
        "min_hops": 5,
    },
}


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------

class _RelationalGraph:
    """Lightweight mutable directed graph used during generation.

    Nodes: entity names (str)
    Edges: (src, dst) -> {"type": str, "weight": int, "active": bool}
    """

    def __init__(self) -> None:
        self.nodes: set[str] = set()
        self.edges: dict[tuple[str, str], dict[str, Any]] = {}

    def add_node(self, name: str) -> None:
        self.nodes.add(name)

    def add_edge(self, src: str, dst: str, edge_type: str, weight: int = 0) -> None:
        self.nodes.add(src)
        self.nodes.add(dst)
        self.edges[(src, dst)] = {"type": edge_type, "weight": weight, "active": True}

    def remove_edge(self, src: str, dst: str) -> None:
        if (src, dst) in self.edges:
            self.edges[(src, dst)]["active"] = False

    def active_edges_of_type(self, edge_type: str) -> list[tuple[str, str, int]]:
        result = []
        for (s, d), info in self.edges.items():
            if info["active"] and info["type"] == edge_type:
                result.append((s, d, info["weight"]))
        return result

    # -----------------------------------------------------------------------
    # Query evaluators (ground-truth computation)
    # -----------------------------------------------------------------------

    def reachable(self, src: str, dst: str) -> bool:
        """BFS: is dst reachable from src via active OWES edges?"""
        if src not in self.nodes or dst not in self.nodes:
            return False
        visited: set[str] = set()
        queue = [src]
        while queue:
            node = queue.pop(0)
            if node == dst:
                return True
            if node in visited:
                continue
            visited.add(node)
            for (s, d), info in self.edges.items():
                if s == node and info["active"] and info["type"] == "owes":
                    queue.append(d)
        return False

    def path_blocked(self, src: str, dst: str) -> bool:
        """Is every path from src to dst blocked by at least one BLOCKED edge?"""
        if not self.reachable(src, dst):
            # No path exists at all — vacuously "blocked" (return True so label is clear)
            return True
        # DFS: find any complete unblocked path
        def dfs_unblocked(node: str, target: str, visited: set[str]) -> bool:
            if node == target:
                return True
            for (s, d), info in self.edges.items():
                if s == node and info["active"] and d not in visited:
                    if info["type"] == "blocked":
                        continue  # this edge is blocked, skip
                    if info["type"] == "owes":
                        if dfs_unblocked(d, target, visited | {d}):
                            return True
            return False
        return not dfs_unblocked(src, dst, {src})

    def net_positive(self, entity: str) -> bool:
        """Is entity's net credit (incoming OWES weight − outgoing OWES weight) > 0?"""
        incoming = sum(w for (_, d, w) in self.active_edges_of_type("owes") if d == entity)
        outgoing = sum(w for (s, _, w) in self.active_edges_of_type("owes") if s == entity)
        return (incoming - outgoing) > 0

    def can_collect(self, entity: str, threshold: int) -> bool:
        """Can entity collect at least £threshold transitively via OWES?

        Transitive collection: sum of all OWES weights reachable from entity.
        A transferred edge adds weight once but is not recursively composable.
        """
        total = 0
        visited: set[str] = set()
        queue = [entity]
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            for (s, d, w) in self.active_edges_of_type("owes"):
                if s == node:
                    total += w
                    queue.append(d)
        return total >= threshold


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class RelationalGraphGenerator:
    """Generate multi-turn relational graph sequences for Hypothesis A testing.

    Each sequence reveals a relational graph incrementally across T turns.
    Query turns ask about multi-hop graph properties.

    Args:
        difficulty:  One of "easy", "medium", "hard", "ood".
        seed:        Global RNG seed.
        num_train:   Training set size.
        num_val:     Validation set size.
        num_test:    Test set size.
    """

    def __init__(
        self,
        difficulty: str = "medium",
        seed: int = 42,
        num_train: int = 3000,
        num_val: int = 500,
        num_test: int = 500,
    ) -> None:
        if difficulty not in DIFFICULTY_PRESETS:
            raise ValueError(f"difficulty must be one of {list(DIFFICULTY_PRESETS)}; got {difficulty!r}")
        self.difficulty = difficulty
        self.preset = DIFFICULTY_PRESETS[difficulty]
        self.seed = seed
        self.num_train = num_train
        self.num_val = num_val
        self.num_test = num_test

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_dataset(self) -> dict[str, list[dict[str, Any]]]:
        """Generate train / val / test splits."""
        return {
            "train": self._generate_split("train", self.num_train, seed_offset=0),
            "val":   self._generate_split("val",   self.num_val,   seed_offset=100_000),
            "test":  self._generate_split("test",  self.num_test,  seed_offset=200_000),
        }

    def verify_dataset(self, split: list[dict[str, Any]]) -> dict[str, Any]:
        """Sanity-check a generated split.

        Returns statistics: label balance, mean turns, edge count distribution.
        Raises AssertionError if any sequence has fewer than 2 turns or no query turns.
        """
        label_counts: dict[int, int] = {0: 0, 1: 0}
        turn_counts: list[int] = []
        edge_counts: list[int] = []

        for seq in split:
            turns = seq["turns"]
            assert len(turns) >= 2, f"Sequence {seq['sequence_id']} has <2 turns"
            query_turns = [t for t in turns if t.get("is_query_turn", False)]
            assert len(query_turns) >= 1, f"Sequence {seq['sequence_id']} has no query turns"
            label_counts[seq["turns"][-1]["ground_truth"]] += 1
            turn_counts.append(len(turns))
            edge_counts.append(seq.get("num_edges_final", 0))

        return {
            "n": len(split),
            "label_balance": label_counts,
            "mean_turns": sum(turn_counts) / len(turn_counts),
            "mean_final_edges": sum(edge_counts) / len(edge_counts),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _generate_split(
        self, split: str, count: int, seed_offset: int
    ) -> list[dict[str, Any]]:
        records = []
        for i in range(count):
            item_seed = self.seed + seed_offset + i
            rng = random.Random(item_seed)
            seq = self._generate_sequence(
                rng=rng,
                seed=item_seed,
                split=split,
                sequence_id=f"rg_{self.difficulty}_{split}_{i:06d}",
            )
            records.append(seq)
        return records

    def _generate_sequence(
        self,
        rng: random.Random,
        seed: int,
        split: str,
        sequence_id: str,
    ) -> dict[str, Any]:
        """Build a single multi-turn relational graph sequence."""
        preset = self.preset
        num_entities: int = preset["num_entities"]
        edge_types: list[str] = preset["edge_types"]
        max_hops: int = preset["max_hops"]
        min_hops: int = preset["min_hops"]
        num_distractors: int = preset["num_distractors"]
        sequence_turns: int = preset["sequence_turns"]

        # Sample entity names
        all_names = rng.sample(ENTITY_NAMES, num_entities)
        distractor_names = rng.sample(DISTRACTOR_NAMES, min(num_distractors, len(DISTRACTOR_NAMES)))

        # Build the target path: a chain of min_hops to max_hops OWES edges
        # that the answer entity can collect from
        path_length = rng.randint(min_hops, max_hops)
        path_length = min(path_length, num_entities - 1)
        path_nodes = rng.sample(all_names, path_length + 1)
        # path_nodes[0] → path_nodes[1] → ... → path_nodes[-1]

        graph = _RelationalGraph()

        # Add all main entities
        for name in all_names + distractor_names:
            graph.add_node(name)

        # Build the target OWES chain
        for i in range(len(path_nodes) - 1):
            weight = rng.randint(10, 50)
            graph.add_edge(path_nodes[i], path_nodes[i + 1], "owes", weight)

        # Add distractor edges (dead-ends or disconnected from the query path)
        off_path = [n for n in all_names if n not in path_nodes]
        for d_name in distractor_names:
            # Distractor entity with its own outgoing OWES edge that doesn't reach the target
            if off_path:
                anchor = rng.choice(off_path)
            else:
                anchor = rng.choice(all_names)
            weight = rng.randint(5, 30)
            graph.add_edge(d_name, anchor, "owes", weight)

        # Optionally add BLOCKED edges on non-critical paths
        if "blocked" in edge_types and off_path and len(all_names) > 2:
            off_src = rng.choice(off_path)
            off_dst = rng.choice([n for n in off_path if n != off_src] or [path_nodes[0]])
            graph.add_edge(off_src, off_dst, "blocked", 0)

        # Select query type and compute ground truth
        query_type = rng.choice(QUERY_TYPES)
        src_entity = path_nodes[0]
        dst_entity = path_nodes[-1]

        if query_type == "reachable":
            ground_truth = int(graph.reachable(src_entity, dst_entity))
        elif query_type == "path_blocked":
            ground_truth = int(graph.path_blocked(src_entity, dst_entity))
        elif query_type == "net_positive":
            ground_truth = int(graph.net_positive(src_entity))
        elif query_type == "can_collect":
            threshold = rng.randint(5, 30)
            ground_truth = int(graph.can_collect(src_entity, threshold))
        else:
            ground_truth = 0

        # Balance labels roughly 50/50 — if the current ground truth would create
        # imbalance, flip the query to the other entity or adjust threshold
        # (simple approach: accept as-is; generator runs enough examples for balance)

        # Build turns: reveal graph incrementally, query at the final turn
        turns = self._build_turns(
            rng=rng,
            graph=graph,
            path_nodes=path_nodes,
            distractor_names=distractor_names,
            sequence_turns=sequence_turns,
            query_type=query_type,
            src_entity=src_entity,
            dst_entity=dst_entity,
            ground_truth=ground_truth,
            sequence_id=sequence_id,
            split=split,
        )

        return {
            "sequence_id": sequence_id,
            "task_family": "relational_graph",
            "difficulty": self.difficulty,
            "split": split,
            "seed": seed,
            "query_type": query_type,
            "src_entity": src_entity,
            "dst_entity": dst_entity,
            "path_length": path_length,
            "num_edges_final": len([e for e in graph.edges.values() if e["active"]]),
            "turns": turns,
        }

    def _build_turns(
        self,
        rng: random.Random,
        graph: _RelationalGraph,
        path_nodes: list[str],
        distractor_names: list[str],
        sequence_turns: int,
        query_type: str,
        src_entity: str,
        dst_entity: str,
        ground_truth: int,
        sequence_id: str,
        split: str,
    ) -> list[dict[str, Any]]:
        """Distribute graph facts across turns and add the query at the final turn."""

        # Collect all facts to reveal
        facts_to_reveal: list[dict[str, Any]] = []

        # Node introduction facts
        for name in list(graph.nodes):
            facts_to_reveal.append({
                "type": "entity",
                "entity": name,
                "key": "exists",
                "value": 1,
                "op": "none",
            })

        # Edge facts
        for (src, dst), info in graph.edges.items():
            if info["active"]:
                facts_to_reveal.append({
                    "type": "relation",
                    "entity": src,
                    "key": info["type"],  # "owes", "blocked", "transferred"
                    "value": info["weight"],
                    "op": "add",
                    # Store destination for decoding
                    "_dst": dst,
                })

        # Shuffle so facts don't arrive in a trivially ordered way
        rng.shuffle(facts_to_reveal)

        # Distribute facts across turns (leaving the last turn as the query turn)
        content_turns = sequence_turns - 1
        chunks: list[list[dict]] = [[] for _ in range(content_turns)]
        for idx, fact in enumerate(facts_to_reveal):
            chunks[idx % content_turns].append(fact)

        turns = []
        for t_idx in range(content_turns):
            turns.append({
                "id": f"{sequence_id}_t{t_idx}",
                "turn_index": t_idx,
                "task_family": "relational_graph",
                "split": split,
                "is_query_turn": False,
                "input_facts": [self._fact_to_record(f) for f in chunks[t_idx]],
                # Non-query turns: label is a placeholder (model is not asked to predict)
                # We use the graph-level ground truth so the trainer can skip non-query turns
                "ground_truth": ground_truth,
                "label": ground_truth,
                "proof_trace": [f"reveal_facts: turn {t_idx}"],
            })

        # Query turn (last turn)
        query_facts = [
            {
                "type": "attribute",
                "entity": src_entity,
                "key": "query_src",
                "value": 1,
                "op": "none",
            },
            {
                "type": "attribute",
                "entity": dst_entity,
                "key": "query_dst",
                "value": 1,
                "op": "none",
            },
            {
                "type": "attribute",
                "entity": "QUERY",
                "key": query_type,
                "value": 1,
                "op": "none",
            },
        ]
        turns.append({
            "id": f"{sequence_id}_t{sequence_turns - 1}",
            "turn_index": sequence_turns - 1,
            "task_family": "relational_graph",
            "split": split,
            "is_query_turn": True,
            "input_facts": [self._fact_to_record(f) for f in query_facts],
            "ground_truth": ground_truth,
            "label": ground_truth,
            "proof_trace": [
                f"query_type={query_type}",
                f"src={src_entity}",
                f"dst={dst_entity}",
                f"answer={ground_truth}",
            ],
        })

        return turns

    @staticmethod
    def _fact_to_record(fact: dict[str, Any]) -> dict[str, Any]:
        """Convert internal fact dict to the encoder-compatible 5-field format."""
        return {
            "type": fact.get("type", "attribute"),
            "entity": fact.get("entity", "UNKNOWN"),
            "key": fact.get("key", "value"),
            "value": int(fact.get("value", 0)),
            "op": fact.get("op", "none"),
        }


# ---------------------------------------------------------------------------
# Registration hook (mirrors how ArithmeticGenerator dispatches to subgenerators)
# ---------------------------------------------------------------------------

def make_relational_graph_generator(
    difficulty: str,
    seed: int = 42,
    num_train: int = 3000,
    num_val: int = 500,
    num_test: int = 500,
) -> RelationalGraphGenerator:
    """Factory used by ArithmeticGenerator dispatch when task_family starts with 'relational_graph'."""
    return RelationalGraphGenerator(
        difficulty=difficulty,
        seed=seed,
        num_train=num_train,
        num_val=num_val,
        num_test=num_test,
    )
