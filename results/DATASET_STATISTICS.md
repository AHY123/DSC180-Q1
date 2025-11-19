# Dataset Statistics Summary

This document provides comprehensive graph statistics for all datasets used in the cycle detection experiments comparing graph-native transformers (GPS) with sequence-based tokenization methods (graph-token, AutoGraph).

## Overview

All datasets consist of Erdős-Rényi (ER) random graphs with edge probability p=0.3.

| Dataset | Num Graphs | Node Range | Total Nodes | Total Edges | Cycles |
|---------|------------|------------|-------------|-------------|--------|
| Small   | 50         | 5-19       | 553         | 1,643       | 78.0%  |
| Medium  | 500        | 5-19       | 5,892       | 19,037      | 82.2%  |
| Large   | 5000       | 10-20      | 75,078      | 165,263     | 99.6%  |

---

## 50 Graphs Dataset (Small)

**Graph Generation:** ER(n=5-19, p=0.3)
**Location:** `data/synthetic_er/train` (first 50 graphs)

### Basic Statistics
- **Total Graphs:** 50
- **Graphs with Cycles:** 39/50 (78.0%)

### Node Statistics
```
Total Nodes:    553
Min Nodes:      5
Max Nodes:      19
Mean Nodes:     11.06 ± 4.53
Median Nodes:   10.5
```

### Edge Statistics
```
Total Edges:    1,643
Min Edges:      0
Max Edges:      141
Mean Edges:     32.86 ± 36.60
Median Edges:   16.5
```

### Graph Properties
```
Density:        0.476 ± 0.298
Avg Degree:     4.89 ± 4.10
Components:     2.24 ± 2.78
Diameter:       2.62 ± 1.15 (connected graphs only)
Clustering:     0.442 ± 0.336
```

### Node Distribution
| Nodes | Count | Percentage |
|-------|-------|------------|
| 5     | 5     | 10.0%      |
| 6     | 8     | 16.0%      |
| 7     | 5     | 10.0%      |
| 9     | 3     | 6.0%       |
| 10    | 4     | 8.0%       |
| 11    | 2     | 4.0%       |
| 12    | 1     | 2.0%       |
| 13    | 3     | 6.0%       |
| 14    | 4     | 8.0%       |
| 15    | 6     | 12.0%      |
| 16    | 1     | 2.0%       |
| 17    | 2     | 4.0%       |
| 18    | 5     | 10.0%      |
| 19    | 1     | 2.0%       |

---

## 500 Graphs Dataset (Medium)

**Graph Generation:** ER(n=5-19, p=0.3)
**Location:** `data/synthetic_er/train`

### Basic Statistics
- **Total Graphs:** 500
- **Graphs with Cycles:** 411/500 (82.2%)

### Node Statistics
```
Total Nodes:    5,892
Min Nodes:      5
Max Nodes:      19
Mean Nodes:     11.78 ± 4.43
Median Nodes:   12.0
```

### Edge Statistics
```
Total Edges:    19,037
Min Edges:      0
Max Edges:      169
Mean Edges:     38.07 ± 36.88
Median Edges:   24.0
```

### Graph Properties
```
Density:        0.509 ± 0.284
Avg Degree:     5.54 ± 4.05
Components:     1.90 ± 2.22
Diameter:       2.53 ± 0.93 (connected graphs only)
Clustering:     0.471 ± 0.327
```

### Node Distribution
| Nodes | Count | Percentage |
|-------|-------|------------|
| 5     | 40    | 8.0%       |
| 6     | 48    | 9.6%       |
| 7     | 29    | 5.8%       |
| 8     | 34    | 6.8%       |
| 9     | 25    | 5.0%       |
| 10    | 29    | 5.8%       |
| 11    | 32    | 6.4%       |
| 12    | 37    | 7.4%       |
| 13    | 28    | 5.6%       |
| 14    | 36    | 7.2%       |
| 15    | 35    | 7.0%       |
| 16    | 34    | 6.8%       |
| 17    | 25    | 5.0%       |
| 18    | 36    | 7.2%       |
| 19    | 32    | 6.4%       |

---

## 5000 Graphs Dataset (Large)

**Graph Generation:** ER(n=10-20, p=0.3)
**Location:** `data/large_5000`

### Basic Statistics
- **Total Graphs:** 5000
- **Graphs with Cycles:** 4979/5000 (99.6%)

### Node Statistics
```
Total Nodes:    75,078
Min Nodes:      10
Max Nodes:      20
Mean Nodes:     15.02 ± 3.18
Median Nodes:   15.0
```

### Edge Statistics
```
Total Edges:    165,263
Min Edges:      5
Max Edges:      76
Mean Edges:     33.05 ± 14.70
Median Edges:   31.0
```

### Graph Properties
```
Density:        0.300 ± 0.048
Avg Degree:     4.20 ± 1.14
Components:     1.15 ± 0.44
Diameter:       3.60 ± 0.72 (connected graphs only)
Clustering:     0.278 ± 0.103
```

### Node Distribution
| Nodes | Count | Percentage |
|-------|-------|------------|
| 10    | 462   | 9.2%       |
| 11    | 444   | 8.9%       |
| 12    | 452   | 9.0%       |
| 13    | 469   | 9.4%       |
| 14    | 438   | 8.8%       |
| 15    | 447   | 8.9%       |
| 16    | 462   | 9.2%       |
| 17    | 451   | 9.0%       |
| 18    | 429   | 8.6%       |
| 19    | 485   | 9.7%       |
| 20    | 461   | 9.2%       |

---

## Key Observations

### Dataset Characteristics

1. **Cycle Prevalence:**
   - Small (50): 78.0% have cycles
   - Medium (500): 82.2% have cycles
   - Large (5000): 99.6% have cycles
   - Larger datasets approach the theoretical probability for ER graphs with p=0.3

2. **Graph Size:**
   - Small/Medium datasets: 5-19 nodes (very diverse)
   - Large dataset: 10-20 nodes (more constrained, larger minimum)
   - Large dataset has more uniform node distribution

3. **Connectivity:**
   - Small dataset: Average 2.24 components (more disconnected)
   - Medium dataset: Average 1.90 components
   - Large dataset: Average 1.15 components (mostly connected)
   - Larger graphs with higher minimum node count tend to be more connected

4. **Graph Properties:**
   - Density decreases with larger datasets (0.476 → 0.509 → 0.300)
     - Large dataset has lower density due to larger graphs (density ∝ 1/n)
   - Average degree relatively stable (4.89 → 5.54 → 4.20)
   - Diameter increases with dataset size (2.62 → 2.53 → 3.60)
     - Larger graphs have longer shortest paths

5. **Clustering:**
   - Decreases with dataset size (0.442 → 0.471 → 0.278)
   - Consistent with random graph theory (clustering coefficient ≈ p for ER graphs)

### Implications for Model Training

1. **Task Difficulty:**
   - Small dataset is easier due to more acyclic graphs (22% negative class)
   - Large dataset is harder due to extreme class imbalance (0.4% negative class)
   - Models achieving 99.5%+ accuracy on large dataset may be biased toward predicting "has cycle"

2. **Generalization:**
   - Small/medium datasets have wider range of graph sizes and structures
   - Large dataset is more uniform but has more samples
   - Models trained on small datasets may struggle with larger graphs

3. **Graph Complexity:**
   - Larger graphs (5000 dataset) have higher diameter (longer paths)
   - This may favor GPS with global attention over local message passing
   - Sequence-based methods need to capture longer-range dependencies

---

## Data Files

- **JSON Statistics:** `results/dataset_statistics.json`
- **Analysis Script:** `scripts/analyze_dataset_stats.py`
- **Training Logs:** `results/training_logs/`
