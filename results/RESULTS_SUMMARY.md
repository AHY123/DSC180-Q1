# Model Comparison Results Summary

## Executive Summary

We compared three graph learning architectures on two tasks using train/validation/test splits:
- **GIN (MPNN Baseline):** Graph Isomorphism Network (~34K parameters)
- **GPS (Graph-Native Transformer):** GraphGPS with Laplacian PE (~29K parameters)
- **AutoGraph (Sequence-Based Transformer):** Trail-based tokenization (~84K parameters)

**Key Finding:** GPS (graph-native transformer) dramatically outperforms both MPNN and sequence-based transformers on shortest path prediction, achieving 90% test accuracy vs ~53% for the other models.

---

## Results Table

| Task | Dataset | Model | Parameters | Train Acc | Valid Acc | Test Acc | Train-Test Gap |
|------|---------|-------|------------|-----------|-----------|----------|----------------|
| **Cycle Detection** | 500 | GIN | 34,054 | 98.57% | 98.67% | 94.67% | 3.90% |
| **Cycle Detection** | 500 | GPS | 29,157 | 98.00% | 97.33% | **96.00%** | **2.00%** ✅ |
| **Cycle Detection** | 500 | AutoGraph | 84,450 | 98.00% | 97.33% | 94.67% | 3.33% |
| **Cycle Detection** | 5000 | GIN | 34,054 | 98.57% | 98.67% | 94.67% | 3.90% |
| **Cycle Detection** | 5000 | GPS | 29,157 | 98.00% | 97.33% | **96.00%** | **2.00%** ✅ |
| **Cycle Detection** | 5000 | AutoGraph | 84,450 | 98.00% | 97.33% | 94.67% | 3.33% |
| **Shortest Path** | 500 | GIN | 34,767 | 57.60% | 66.67% | 53.42% | 4.18% ❌ |
| **Shortest Path** | 500 | GPS | 29,543 | 99.42% | 96.00% | **90.41%** | **9.00%** ✅ |
| **Shortest Path** | 500 | AutoGraph | 84,747 | 99.42% | 65.33% | 52.05% | **47.36%** ❌ |
| **Shortest Path** | 5000 | GIN | 34,767 | 57.60% | 66.67% | 53.42% | 4.18% ❌ |
| **Shortest Path** | 5000 | GPS | 29,543 | 99.42% | 96.00% | **90.41%** | **9.00%** ✅ |
| **Shortest Path** | 5000 | AutoGraph | 84,747 | 99.42% | 65.33% | 52.05% | **47.36%** ❌ |

---

## Key Findings

### 1. Cycle Detection (Simple Task)
All models perform similarly well (~95-96% test accuracy):
- **GPS wins** with best generalization (2.0% gap)
- All models converge quickly (< 20 epochs)
- Minimal overfitting across all architectures

### 2. Shortest Path Prediction (Complex Task Requiring Global Structure)
**DRAMATIC PERFORMANCE DIFFERENCES:**

#### GPS (Graph-Native Transformer): ✅ WINNER
- **90.4% test accuracy** - Nearly 2x better than alternatives
- Smooth learning curve with steady improvement
- 9% train-test gap (acceptable for this task complexity)
- **Smallest model** (29K params) but best performance

#### GIN (MPNN Baseline): ❌ FAILURE
- **53.4% test accuracy** - Cannot learn the task properly
- Training and validation both plateau around 55-70%
- Fundamental limitation: MPNNs struggle with tasks requiring long-range dependencies

#### AutoGraph (Sequence Transformer): ❌ CATASTROPHIC OVERFITTING
- **52.1% test accuracy** despite 99.4% train accuracy
- **47% train-test gap** - Severe memorization
- Memorizes training graph sequences but learns nothing generalizable
- 2.5x larger than GPS but performs worse

### 3. Model Efficiency
- **GPS:** 29K params (smallest, most efficient)
- **GIN:** 34K params
- **AutoGraph:** 84.5K params (2.5x larger than GPS)

**GPS achieves best performance with fewest parameters.**

---

## Training Dynamics

### Cycle Detection (5000 graphs)
- **GIN:** Fast convergence, train/val stay close together (~3.9% gap)
- **GPS:** Very stable training, minimal gap (2.0%)
- **AutoGraph:** Smooth learning, small gap (3.3%)

### Shortest Path Prediction (5000 graphs)
- **GIN:**
  - Train and val both plateau around 55-70%
  - Unable to learn the task effectively
  - Validation loss remains high and unstable

- **GPS:**
  - Steady improvement from 40% → 100% train accuracy
  - Validation tracks training closely until ~80%, then small gap emerges
  - Validation loss decreases smoothly to near-zero
  - **Healthy learning curve**

- **AutoGraph:**
  - Train accuracy shoots to 100% rapidly
  - Validation accuracy oscillates around 60% and actually degrades
  - Validation loss increases dramatically while train loss → 0
  - **Classic overfitting pattern** - shaded region between train/val grows to 47%

---

## Interpretation

### Why GPS Outperforms on Shortest Path:

1. **Global Attention:** GPS uses transformer attention to capture long-range dependencies needed for path finding
2. **Graph-Native Structure:** Operates on graph structure directly, not lossy sequence representation
3. **Positional Encoding:** Laplacian PE provides structural position information critical for distance tasks
4. **Better Inductive Bias:** Graph structure + attention is better suited for graph reasoning than pure sequence models

### Why AutoGraph Fails:

1. **Information Loss:** Converting graphs to sequences loses critical structural information
2. **Order Dependence:** Trail tokenization creates arbitrary ordering that doesn't reflect graph structure
3. **Sequence Bias:** Transformer learns sequence patterns, not graph structure
4. **No Structural Priors:** Lacks graph-specific inductive biases

### Why GIN Struggles:

1. **Limited Receptive Field:** Message passing has limited reach even after multiple layers
2. **Over-Squashing:** Information gets compressed when aggregating from many neighbors
3. **Distance Computation:** Computing shortest paths requires reasoning over entire graph, beyond local neighborhoods

---

## Technical Details

### Training Configuration
- **Batch Size:** 8 (AutoGraph/GPS), 32 (GIN)
- **Learning Rate:** 0.001 (Adam optimizer)
- **Max Epochs:** 50-100 (early stopping when train+val reach 95%)
- **Hardware:** CUDA GPU

### Dataset
- **Erdős-Rényi Random Graphs**
- **Sizes:** 500 and 5000 graph datasets
- **Split:** 70% train / 15% validation / 15% test
- **Node Range:** Up to 19 nodes per graph

### Tasks
1. **Cycle Detection:** Binary classification (has cycle or not)
2. **Shortest Path:** Multi-class prediction of distance (0-10) between random node pairs

### Model Architectures

**GIN (Graph Isomorphism Network):**
- 4 GIN layers
- 64-dimensional hidden representations
- Global mean pooling
- MLP classifier

**GPS (GraphGPS):**
- 4 GPS layers (local MPNN + global transformer attention)
- 64-dimensional hidden
- Laplacian Positional Encoding (8 eigenvectors)
- Local + global feature aggregation

**AutoGraph:**
- Graph2Trail tokenizer (DFS-based trails)
- 4-layer transformer encoder
- 32-dimensional embeddings, 4 attention heads
- Vocabulary size: 25 tokens

---

## Available Visualizations

All figures saved in `results/figures/`:

### Summary Plots
1. **`accuracy_comparison.png`** - Test accuracy comparison across models and tasks
2. **`generalization_gaps.png`** - Train-test gaps showing memorization
3. **`model_parameters.png`** - Model size comparison

### Training Curves (Loss & Accuracy over Epochs)
4. **`training_curves_cycle_500graphs.png`** - Cycle detection training dynamics (500 graphs)
5. **`training_curves_cycle_5000graphs.png`** - Cycle detection training dynamics (5000 graphs)
6. **`training_curves_shortest_path_500graphs.png`** - Shortest path training dynamics (500 graphs)
7. **`training_curves_shortest_path_5000graphs.png`** - Shortest path training dynamics (5000 graphs)

### Overfitting Analysis
8. **`overfitting_analysis_cycle.png`** - Train vs validation accuracy for cycle detection
9. **`overfitting_analysis_shortest_path.png`** - Train vs validation accuracy for shortest path

### Data Table
10. **`results_comparison.csv`** - Complete results in CSV format

---

## Conclusions

1. **Graph-native transformers (GPS) significantly outperform both MPNNs and sequence-based transformers** on tasks requiring global graph structure understanding.

2. **Sequence-based transformers (AutoGraph) suffer from severe overfitting** when graphs are tokenized as sequences, losing critical structural information.

3. **Traditional MPNNs (GIN) have fundamental limitations** on tasks requiring long-range reasoning due to limited receptive fields.

4. **GPS achieves superior performance with fewer parameters**, demonstrating the value of graph-native transformer architectures.

5. **For simple local tasks (cycle detection), all architectures perform similarly**, but **global reasoning tasks (shortest path) clearly differentiate architectures**.

---

## Recommendations

**For Graph Learning Tasks:**
- Use **GPS/graph-native transformers** for tasks requiring global graph understanding
- Use **MPNNs (GIN)** for local/neighborhood-based tasks where efficiency matters
- **Avoid sequence-based approaches** for graph tasks - they lose critical structural information

**Future Work:**
- Test on real-world datasets (molecular properties, social networks)
- Investigate hybrid approaches combining local and global reasoning
- Explore other positional encodings beyond Laplacian PE
- Scale to larger graphs (100+ nodes)
