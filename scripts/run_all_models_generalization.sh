#!/bin/bash
# Run generalization tests for ALL models (GIN, GCN, GAT, GPS)

set -e

echo "======================================================================"
echo "Running Generalization Tests for ALL Models"
echo "======================================================================"

# Models to test
MPNN_MODELS=("GIN" "GCN" "GAT")
DATASETS=("data/synthetic_er_500" "data/synthetic_er_5000")
TASKS=("cycle" "shortest_path")

# Run MPNN models (GIN, GCN, GAT)
for dataset in "${DATASETS[@]}"; do
    for task in "${TASKS[@]}"; do
        for model in "${MPNN_MODELS[@]}"; do
            echo ""
            echo "----------------------------------------------------------------------"
            echo "Running: $task on $dataset with $model"
            echo "----------------------------------------------------------------------"

            if [ "$task" == "cycle" ]; then
                uv run python scripts/evaluate_with_splits.py \
                    --base_dir "$dataset" \
                    --task "$task" \
                    --model_type "$model" \
                    --hidden_dim 64 \
                    --num_layers 4 \
                    --epochs 50
            else
                uv run python scripts/evaluate_with_splits.py \
                    --base_dir "$dataset" \
                    --task "$task" \
                    --model_type "$model" \
                    --hidden_dim 64 \
                    --num_layers 4 \
                    --epochs 50 \
                    --k_pairs 1 \
                    --max_distance 10
            fi
        done
    done
done

echo ""
echo "======================================================================"
echo "All MPNN Models Complete!"
echo "======================================================================"
echo ""
echo "Now run GPS separately (needs GPS environment)"
echo ""
echo "Generating plots..."
uv run python scripts/plot_results.py

echo ""
echo "Aggregating results..."
uv run python scripts/aggregate_results.py

echo ""
echo "======================================================================"
echo "Done! Check results/figures/ for plots"
echo "======================================================================"
