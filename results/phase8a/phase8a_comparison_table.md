# Phase 8A — Four-Model Comparative Evaluation

> Shared evaluation set: `affordability_sequence`, 5 turns, 500 sequences, seed=999 (held-out).

| Metric | **Vector Baseline** | **Static Graph** | **Persistent Graph (Phase 4)** | **Full LIMINAL (Phase 7)** |
| :--- | :---: | :---: | :---: | :---: |
| **Overall Accuracy** | 85.20% | 86.68% | 89.72% | 38.64% |
| Turn 1 Accuracy | 88.80% | 73.00% | 99.00% | 34.80% |
| Turn 2 Accuracy | 98.20% | 100.00% | 86.60% | 37.00% |
| Turn 3 Accuracy | 90.60% | 93.40% | 86.60% | 39.20% |
| Turn 4 Accuracy | 78.40% | 85.20% | 89.40% | 39.20% |
| Turn 5 Accuracy | 70.00% | 81.80% | 87.00% | 43.00% |
| **Cross-Entropy Loss** | 2.3826 | 1.5344 | 0.9783 | 12.6041 |
| Mean Halt Steps | Fixed (4) | Fixed (4) | 0.00 | 0.00 |
| Mean Active Slots | 1/1 | 8/8 | 8/8 | 8/8 |
| Ext. Writes / Step | N/A | N/A | N/A | 0.000 |
| Trainable Parameters | 498,530 | 19,266 | 23,557 | 34,158 |
| Latency (ms/seq) | 5.20 | 4.01 | 7.16 | 13.37 |
| Persistent State | ✗ | ✗ | ✓ | ✓ |
| External Workspace | ✗ | ✗ | ✗ | ✓ |
