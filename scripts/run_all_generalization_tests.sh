#!/bin/bash
# Run all generalization tests systematically

set -e  # Exit on error

echo "======================================================================"
echo "Running All Generalization Tests"
echo "======================================================================"

# Function to run experiment
run_exp() {
    local dataset=$1
    local task=$2
    local model=$3
    local extra_args=$4

    echo ""
    echo "----------------------------------------------------------------------"
    echo "Running: $task on $dataset with $model"
    echo "----------------------------------------------------------------------"

    uv run python scripts/evaluate_with_splits.py \
        --base_dir "$dataset" \
        --task "$task" \
        --model_type "$model" \
        --hidden_dim 64 \
        --num_layers 4 \
        --epochs 100 \
        $extra_args
}

# Cycle Detection - 500 graphs
run_exp "data/synthetic_er_500" "cycle" "GIN" ""

# Shortest Path - 500 graphs
run_exp "data/synthetic_er_500" "shortest_path" "GIN" "--k_pairs 1 --max_distance 10"

# Cycle Detection - 5000 graphs
run_exp "data/synthetic_er_5000" "cycle" "GIN" ""

# Shortest Path - 5000 graphs
run_exp "data/synthetic_er_5000" "shortest_path" "GIN" "--k_pairs 1 --max_distance 10"

echo ""
echo "======================================================================"
echo "All Experiments Complete!"
echo "======================================================================"

# Generate plots
echo ""
echo "Generating plots..."
uv run python scripts/plot_results.py

# Aggregate results
echo ""
echo "Aggregating results..."
uv run python scripts/aggregate_results.py

echo ""
echo "======================================================================"
echo "Done! Check results/figures/ for plots"
echo "======================================================================"
