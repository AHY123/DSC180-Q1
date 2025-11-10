Title (Proposal)
A Comparative Study of Graph-Native and Sequence-Based Transformers for Graph Representation Learning

Abstract
Classical Graph Neural Networks (GNNs), specifically Message Passing Neural Networks (MPNNs), are a standard but limited tool for graph learning, often suffering from over-smoothing and poor long-range expressiveness. Two dominant solutions have emerged: 1) "Graph-native" Transformers, like the General, Powerful, and Scalable (GPS) model, which integrate local graph structure with global attention, and 2) "Sequence-based" models, such as AutoGraph and simple index-based tokenization, which flatten graphs for use with standard, powerful Transformer backbones. This report empirically compares these two philosophies. We evaluate a baseline MPNN, the GPS Transformer, and two sequence-based models on three carefully selected tasks: synthetic cycle checking, synthetic shortest-path distance estimation, and real-world graph classification on the ZINC 12k dataset. Our analysis aims to clarify the practical trade-offs and relative power of these modern graph learning architectures.

1. Introduction
1.1 The Problem and Competing Philosophies
Graph Neural Networks (GNNs) are a successful tool in domains from molecular chemistry to social networks. The dominant paradigm, Message Passing Neural Networks (MPNNs), operates by local neighborhood aggregation. This mechanism, however, causes well-known issues like over-smoothing (where node representations become indistinguishable) and a failure to capture long-range dependencies between distant nodes.

These limitations spurred research into two competing philosophies to create more expressive models:

Graph-Native Transformers: This "fancy" approach builds principled, graph-aware architectures. Models like GPS augment local message passing with a global attention mechanism, designed to capture both local and global graph structure simultaneously.

Sequence-Based Transformers: This "simple" alternative adapts the graph for the model. By "flattening" graphs into linear sequences using methods like index-based tokenization or AutoGraph, this approach leverages the immense power and vast ecosystem of standard, high-performing Transformer backbones.

This divergence poses our core research question: Is the "fancy" graph-native design truly necessary, or can "simple" sequencing achieve comparable expressiveness? This report provides a direct empirical comparison to address this question.

1.2 Literature Review and Prior Work
The foundation of modern graph learning rests on MPNNs like the Graph Convolutional Network (GCN) (Kipf & Welling, 2017) and the Graph Attention Network (GAT) (Veličković et al., 2018). Their well-documented limitations drove innovation in the two directions we investigate.

First, Graph-Native Transformers, such as those proposed by Dwivedi & Bresson (2021), added global attention to solve the long-range problem. We use a state-of-the-art example, the GPS architecture (Rampášek et al., 2023), which creates a hybrid of local MPNN messages and global self-attention.

In parallel, the Sequence-Based approach emerged, focusing on finding an optimal "graph-to-sequence" tokenization strategy. We study two such methods: the novel flattening strategy presented in AutoGraph (Chen et al., 2025) and a more foundational index-based tokenization approach. This entire direction is bolstered by findings that standard Transformers can solve graph-algorithmic tasks (Sanford et al., 2024) and that the performance gap between GNNs and Transformers may be smaller than previously assumed (Tönshoff et al., 2023).

Our work contributes to this "fancy vs. simple" debate by directly comparing GPS (fancy) against AutoGraph and index-based tokenization (simple), using a classical MPNN as a baseline for performance.

1.3 Description of Relevant Data and Tasks
To test our central hypothesis, we selected three tasks designed to probe logical reasoning, long-range dependencies, and real-world structural pattern recognition.

Cycle Check (Synthetic): A logical reasoning task to identify non-local cyclic properties. Success here tests a model's ability to find structures that are a known weakness of many classical MPNNs.

Shortest-Path Distance (Synthetic): This is the classic long-range dependency task. It forces a model to aggregate information between distant nodes, directly comparing the global attention of GPS against the sequential attention of the flattened-graph models.

Graph Classification on ZINC 12k: A real-world molecular benchmark where success depends on identifying complex, local sub-structures (e.g., functional groups). This task tests if the principled, graph-aware architecture of GPS provides a tangible advantage over sequence models on a practical, structure-heavy problem.

This diverse suite of tasks will allow us to build a detailed picture of the models' relative strengths and weaknesses, providing clear evidence for the "fancy vs. simple" debate.