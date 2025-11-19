#!/bin/bash
# Run generalization tests for ALL transformer models (GPS, AutoGraph, graph-token) + GIN baseline

set -e

echo "======================================================================"
echo "Running Generalization Tests for ALL Transformer Models"
echo "======================================================================"

# Models to test
DATASETS=("data/synthetic_er_500" "data/synthetic_er_5000")
TASKS=("cycle" "shortest_path")

# Run GIN (MPNN baseline) with uv
echo ""
echo "======================================================================="
echo "Running GIN (MPNN Baseline)"
echo "======================================================================="
for dataset in "${DATASETS[@]}"; do
    for task in "${TASKS[@]}"; do
        echo ""
        echo "----------------------------------------------------------------------"
        echo "Running: $task on $dataset with GIN"
        echo "----------------------------------------------------------------------"

        if [ "$task" == "cycle" ]; then
            uv run python scripts/evaluate_with_splits.py \
                --base_dir "$dataset" \
                --task "$task" \
                --model_type GIN \
                --hidden_dim 64 \
                --num_layers 4 \
                --epochs 50
        else
            uv run python scripts/evaluate_with_splits.py \
                --base_dir "$dataset" \
                --task "$task" \
                --model_type GIN \
                --hidden_dim 64 \
                --num_layers 4 \
                --epochs 50 \
                --k_pairs 1 \
                --max_distance 10
        fi
    done
done

# Run GPS (Graph-Native Transformer) with gps environment
echo ""
echo "======================================================================="
echo "Running GPS (Graph-Native Transformer)"
echo "======================================================================="
for dataset in "${DATASETS[@]}"; do
    for task in "${TASKS[@]}"; do
        echo ""
        echo "----------------------------------------------------------------------"
        echo "Running: $task on $dataset with GPS"
        echo "----------------------------------------------------------------------"

        if [ "$task" == "cycle" ]; then
            mamba run -n gps python scripts/evaluate_gps_splits.py \
                --base_dir "$dataset" \
                --task "$task" \
                --config configs/gps_cycle_tiny.yaml \
                --batch_size 8 \
                --epochs 50
        else
            mamba run -n gps python scripts/evaluate_gps_splits.py \
                --base_dir "$dataset" \
                --task shortest_path \
                --config configs/gps_shortest_path_tiny.yaml \
                --batch_size 8 \
                --epochs 50 \
                --k_pairs 1 \
                --max_distance 10
        fi
    done
done

# Run AutoGraph (Sequence-Based) with autograph environment
echo ""
echo "======================================================================="
echo "Running AutoGraph (Sequence-Based Transformer)"
echo "======================================================================="
for dataset in "${DATASETS[@]}"; do
    for task in "${TASKS[@]}"; do
        echo ""
        echo "----------------------------------------------------------------------"
        echo "Running: $task on $dataset with AutoGraph"
        echo "----------------------------------------------------------------------"

        if [ "$task" == "cycle" ]; then
            mamba run -n autograph python scripts/evaluate_autograph_splits.py \
                --base_dir "$dataset" \
                --task "$task" \
                --batch_size 8 \
                --epochs 50
        else
            mamba run -n autograph python scripts/evaluate_autograph_splits.py \
                --base_dir "$dataset" \
                --task shortest_path \
                --batch_size 8 \
                --epochs 50 \
                --k_pairs 1 \
                --max_distance 10
        fi
    done
done

# Run graph-token (Sequence-Based) with autograph environment (same dependencies)
echo ""
echo "======================================================================="
echo "Running graph-token (Sequence-Based Transformer)"
echo "======================================================================="
for dataset in "${DATASETS[@]}"; do
    for task in "${TASKS[@]}"; do
        echo ""
        echo "----------------------------------------------------------------------"
        echo "Running: $task on $dataset with graph-token"
        echo "----------------------------------------------------------------------"

        if [ "$task" == "cycle" ]; then
            mamba run -n autograph python scripts/evaluate_graphtoken_splits.py \
                --base_dir "$dataset" \
                --task "$task" \
                --batch_size 8 \
                --epochs 50
        else
            mamba run -n autograph python scripts/evaluate_graphtoken_splits.py \
                --base_dir "$dataset" \
                --task shortest_path \
                --batch_size 8 \
                --epochs 50 \
                --k_pairs 1 \
                --max_distance 10
        fi
    done
done

echo ""
echo "======================================================================="
echo "All Models Complete!"
echo "======================================================================="
echo ""
echo "Generating plots..."
uv run python scripts/plot_results.py

echo ""
echo "Aggregating results..."
uv run python scripts/aggregate_results.py

echo ""
echo "======================================================================="
echo "Done! Check results/figures/ for plots"
echo "======================================================================="
