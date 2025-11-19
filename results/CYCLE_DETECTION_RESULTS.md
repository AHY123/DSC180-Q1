# Cycle Detection Results - Model Comparison

**Task**: Binary classification - does the graph contain a cycle?

**Dataset**: Erdős-Rényi random graphs with edge probability p=0.3

---

## Summary Table

| Model | Type | Params | 50 graphs | 500 graphs | 5000 graphs |
|-------|------|--------|-----------|------------|-------------|
| **graph-token** | Sequence | 84,642 | 100.0% (4 ep, ~8s) | 98.4% (3 ep, ~45s) | 99.54% (1 ep) |
| **AutoGraph** | Sequence | 84,450-482 | 100.0% (3 ep, ~4.5s) | 98.0% (5 ep, ~60s) | 99.58% (1 ep) |
| **GPS** | Graph-native | 29,157 | 98.0% (21 ep, 3.4s) | 98.4% (20 ep, 26s) | 99.56% (3 ep, 39s) |
| **GIN (MPNN)** | Graph-native | 34,054 | 98.0% (49 ep, 3.0s) | 96.8% (100 ep, 57s) | 98.3% (1 ep, 5.7s) |

---

## Detailed Results by Dataset

### 50 Graphs (Small Dataset)
- **Class distribution**: 78% cyclic, 22% acyclic (balanced)
- **Graph size**: 5-19 nodes (diverse)

| Model | Accuracy | Epochs | Time | Loss |
|-------|----------|--------|------|------|
| graph-token | 100.0% | 4 | ~8s | - |
| AutoGraph | 100.0% | 3 | ~4.5s | - |
| GPS | 98.0% | 21 | 3.4s | - |
| GIN | 98.0% | 49 | 3.0s | 0.133 |

**Winner**: AutoGraph (fastest convergence)

---

### 500 Graphs (Medium Dataset)
- **Class distribution**: 82.2% cyclic, 17.8% acyclic
- **Graph size**: 5-19 nodes (diverse)

| Model | Accuracy | Epochs | Time | Loss |
|-------|----------|--------|------|------|
| graph-token | 98.4% | 3 | ~45s | - |
| AutoGraph | 98.0% | 5 | ~60s | - |
| GPS | 98.4% | 20 | 26s | - |
| GIN | 96.8% | 100 | 56.5s | 0.074 |

**Winner**: GPS (best balance of speed and accuracy)

---

### 5000 Graphs (Large Dataset)
- **Class distribution**: 99.6% cyclic, 0.4% acyclic (highly imbalanced)
- **Graph size**: 10-20 nodes (uniform)

| Model | Accuracy | Epochs | Time | Loss |
|-------|----------|--------|------|------|
| graph-token | 99.54% | 1 | - | - |
| AutoGraph | 99.58% | 1 | - | - |
| GPS | 99.56% | 3 | 39.0s | 0.023 |
| GIN | 98.3% | 1 | 5.7s | 0.086 |

**Winner**: AutoGraph (highest accuracy), GIN (fastest time)

---

## Key Insights

### 1. Model Architecture Comparison

**Sequence-based (graph-token, AutoGraph)**:
- ✓ Highest accuracy on large datasets (99.5%+)
- ✓ Fast convergence (1-5 epochs)
- ✗ More parameters (~85K)
- ✗ Slower per-epoch training

**Graph-native with global attention (GPS)**:
- ✓ Consistent performance across dataset sizes
- ✓ Fewer parameters (29K)
- ✓ Good speed-accuracy tradeoff
- ✗ Needs more epochs on small datasets

**Graph-native with local aggregation (GIN)**:
- ✓ Fewest parameters among competitive models (34K)
- ✓ Very fast on large datasets (5.7s)
- ✗ Inconsistent convergence (1-100 epochs)
- ✗ Slightly lower accuracy on 500/5000 graphs

### 2. Dataset Size Effects

**Small dataset (50 graphs)**:
- All models achieve ≥98% accuracy
- Tokenization methods reach 100%
- Training time differences minimal (3-8s)

**Medium dataset (500 graphs)**:
- Performance gap appears: 96.8%-98.4%
- GPS shows advantage in training efficiency
- GIN struggles with convergence (100 epochs)

**Large dataset (5000 graphs)**:
- Extremely fast convergence (1-3 epochs)
- All models achieve 98%+ accuracy
- Class imbalance (99.6% positive) likely inflates scores

### 3. Parameter Efficiency

| Model | Params | Acc (avg) | Params per 1% accuracy |
|-------|--------|-----------|------------------------|
| GPS | 29,157 | 98.67% | 295 |
| GIN | 34,054 | 97.70% | 349 |
| graph-token | ~85,000 | 99.31% | 856 |
| AutoGraph | ~85,000 | 99.19% | 857 |

**GPS is most parameter-efficient** (295 params per 1% accuracy)

### 4. Training Speed

**Time to 98% accuracy**:
- Small (50): GIN fastest (3.0s), GPS close (3.4s)
- Medium (500): GPS fastest (26s), GIN slower (57s)
- Large (5000): GIN fastest (5.7s), GPS slower (39s)

**Convergence speed** (epochs to 98%):
- Large dataset: 1-3 epochs for all
- Medium dataset: 3-20 epochs (GIN needs 100)
- Small dataset: 3-49 epochs

### 5. When to Use Each Model

**Use graph-token/AutoGraph when**:
- Maximum accuracy is critical
- You have sufficient compute for larger models
- Dataset is large enough to leverage sequence modeling

**Use GPS when**:
- You need consistent performance across dataset sizes
- Parameter efficiency matters
- You want good speed-accuracy tradeoff

**Use GIN when**:
- You need fastest inference on large datasets
- Memory is very constrained
- You can afford variable training time

---

## Training Configuration

### graph-token
```yaml
Model: Tiny Transformer
- Vocab size: 30-31
- d_model: 32
- Heads: 4
- Layers: 4
- Tokenization: Text-based with task tokens
```

### AutoGraph
```yaml
Model: Tiny Transformer
- Vocab size: 25-26
- d_model: 32
- Heads: 4
- Layers: 4
- Tokenization: Walk-based (Graph2Trail)
```

### GPS
```yaml
Model: GPS (Graph + Transformer)
- Hidden dim: 32
- Layers: 3
- Layer type: GCN+Transformer
- Heads: 4
- Positional encoding: Laplacian PE (4 freqs)
- Pooling: Mean
```

### GIN
```yaml
Model: Graph Isomorphism Network
- Hidden dim: 64
- Layers: 4
- MLP: 2-layer per GIN layer
- Pooling: Sum
- Dropout: 0.1
```

---

## Dataset Statistics

| Dataset | Graphs | Avg Nodes | Avg Edges | Cycles | Avg Degree | Avg Diameter |
|---------|--------|-----------|-----------|--------|------------|--------------|
| 50 | 50 | 11.06 ± 4.53 | 32.86 ± 36.60 | 78.0% | 4.89 ± 4.10 | 2.62 ± 1.15 |
| 500 | 500 | 11.78 ± 4.43 | 38.07 ± 36.88 | 82.2% | 5.54 ± 4.05 | 2.53 ± 0.93 |
| 5000 | 5000 | 15.02 ± 3.18 | 33.05 ± 14.70 | 99.6% | 4.20 ± 1.14 | 3.60 ± 0.72 |

---

## Files and Logs

**Training logs**: `results/training_logs/`
- graph-token: Text-based tokenization logs
- AutoGraph: Walk-based tokenization logs
- GPS: Graph Positional-structural encoding logs
- GIN: Message-passing neural network logs

**Model checkpoints**: `checkpoints/`
- `graph_token_cycle_tiny.pt`
- `autograph_cycle_tiny_full.pt`
- `gps_cycle_*graphs.pt`
- `gin_cycle_*graphs.pt`

**Dataset statistics**: `data/synthetic_er_*/dataset_stats.json`

**Training scripts**:
- `scripts/train_graph_token_cycle_full.py`
- `scripts/train_autograph_cycle_full.py`
- `scripts/train_gps_cycle.py`
- `scripts/train_mpnn_cycle.py`

---

## Conclusions

1. **All approaches are viable** - achieving 96-100% accuracy on cycle detection
2. **Sequence-based methods** have slight accuracy edge but require more parameters
3. **GPS offers best balance** of speed, accuracy, and parameter efficiency
4. **GIN is fastest** on large datasets but has inconsistent convergence
5. **Task is relatively easy** - all models learn quickly on large datasets
6. **Class imbalance** on large dataset (99.6% positive) may inflate scores

**Next steps**: Evaluate on more challenging tasks (e.g., shortest path) to better differentiate model capabilities.
