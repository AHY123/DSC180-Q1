#!/bin/bash
# Run AutoGraph and graph-token only

set -e

DATASETS=("data/synthetic_er_500" "data/synthetic_er_5000")
TASKS=("cycle" "shortest_path")

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
            mamba run -n graph-token python scripts/evaluate_graphtoken_splits.py \
                --base_dir "$dataset" \
                --task "$task" \
                --batch_size 8 \
                --epochs 50
        else
            mamba run -n graph-token python scripts/evaluate_graphtoken_splits.py \
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
echo "Generating plots..."
uv run python scripts/plot_results.py

echo ""
echo "Aggregating results..."
uv run python scripts/aggregate_results.py

echo ""
echo "======================================================================="
echo "Done!"
echo "======================================================================="
