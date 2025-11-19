"""Compare graph-token vs AutoGraph tokenization approaches on cycle detection."""

import torch
import json
from pathlib import Path

def load_and_compare():
    """Load models and generate comparison statistics."""

    results = {
        "task": "cycle_detection",
        "model_architecture": "TinyTransformer (Tier 1)",
        "model_specs": {
            "d_model": 32,
            "nhead": 4,
            "num_layers": 4,
            "dim_feedforward": 128
        },
        "datasets": {
            "test_set": "50 ER graphs (n=10-20, p=0.3)",
            "full_set": "500 ER graphs (n=10-20, p=0.3)"
        },
        "tokenization_approaches": {}
    }

    # Load graph-token results (full model only - test model was overwritten)
    gt_full = torch.load('checkpoints/graph_token_cycle_tiny.pt')

    results["tokenization_approaches"]["graph-token"] = {
        "description": "Text-based graph serialization with task-specific tokens",
        "vocab_size": len(gt_full['vocab']),
        "sample_tokens": list(gt_full['vocab'].keys())[:15],
        "test_set": {
            "model_params": 68450,  # Recorded from training output
            "epochs_to_convergence": 4,
            "final_accuracy": 1.0
        },
        "full_set": {
            "model_params": sum(v.numel() for v in gt_full['model_state_dict'].values()),
            "epochs_to_convergence": 3,
            "final_accuracy": 0.984
        },
        "tokenization_stats": {
            "note": "Includes full graph structure + task question/answer",
            "format": "<bos> edges <n> nodes <q> task <p> answer <eos>",
            "avg_sequence_length": 133,
            "max_sequence_length": 533
        }
    }

    # Load AutoGraph results
    ag_test = torch.load('checkpoints/autograph_cycle_tiny.pt')
    ag_full = torch.load('checkpoints/autograph_cycle_tiny_full.pt')

    results["tokenization_approaches"]["AutoGraph"] = {
        "description": "Walk-based graph tokenization with special traversal tokens",
        "vocab_size": ag_full['tokenizer_config']['vocab_size'],
        "special_tokens": ["sos", "reset", "ladj", "radj", "eos", "pad"],
        "max_num_nodes": ag_full['tokenizer_config']['max_num_nodes'],
        "test_set": {
            "model_params": sum(v.numel() for v in ag_test['model_state_dict'].values()),
            "epochs_to_convergence": 3,
            "final_accuracy": 1.0
        },
        "full_set": {
            "model_params": sum(v.numel() for v in ag_full['model_state_dict'].values()),
            "epochs_to_convergence": 5,
            "final_accuracy": 0.980
        },
        "tokenization_stats": {
            "note": "Walk-based sequences are much shorter than graph-token",
            "format": "sos [walk tokens with adjacency markers] eos",
            "avg_sequence_length": "~30",
            "max_sequence_length": "~31"
        }
    }

    # Key observations
    results["observations"] = {
        "vocabulary": {
            "graph-token": "Larger vocab (30 tokens) with explicit task tokens",
            "AutoGraph": "Smaller vocab (25 tokens) focused on graph structure"
        },
        "sequence_length": {
            "graph-token": "Much longer sequences (max ~533 tokens observed)",
            "AutoGraph": "Compact sequences (max ~31 tokens observed)",
            "implication": "AutoGraph more memory efficient for transformers"
        },
        "convergence": {
            "test_set": "Both converge in 3-4 epochs to 100% accuracy",
            "full_set": "graph-token slightly faster (3 vs 5 epochs) but similar final accuracy"
        },
        "model_size": {
            "graph-token": "~85K params (due to 1024 positional embeddings)",
            "AutoGraph": "~84K params (shorter sequences, same architecture)",
            "note": "Difference mainly from positional embedding size"
        },
        "integration": {
            "graph-token": "Uses external/graph-token CycleCheck task (easy)",
            "AutoGraph": "Uses AutoGraph tokenizer with our task labels (moderate complexity)"
        }
    }

    results["next_steps"] = [
        "Test on more complex tasks (shortest path, subgraph matching)",
        "Implement GPS (GraphGPS) integration for comparison",
        "Evaluate on larger graphs (n > 20)",
        "Compare generalization to test sets",
        "Measure inference speed for each approach"
    ]

    return results

if __name__ == "__main__":
    results = load_and_compare()

    # Save to JSON
    output_path = Path("results/tokenization_comparison.json")
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Comparison saved to {output_path}")

    # Print summary
    print("\n" + "="*70)
    print("TOKENIZATION COMPARISON SUMMARY")
    print("="*70)
    print(f"\nTask: {results['task']}")
    print(f"Architecture: {results['model_architecture']}")
    print(f"\nDatasets:")
    print(f"  - Test: {results['datasets']['test_set']}")
    print(f"  - Full: {results['datasets']['full_set']}")

    print("\n" + "-"*70)
    print("GRAPH-TOKEN")
    print("-"*70)
    gt = results['tokenization_approaches']['graph-token']
    print(f"Vocab size: {gt['vocab_size']}")
    print(f"Test set:  {gt['test_set']['epochs_to_convergence']} epochs → {gt['test_set']['final_accuracy']*100:.1f}% acc")
    print(f"Full set:  {gt['full_set']['epochs_to_convergence']} epochs → {gt['full_set']['final_accuracy']*100:.1f}% acc")
    print(f"Parameters: {gt['full_set']['model_params']:,}")

    print("\n" + "-"*70)
    print("AUTOGRAPH")
    print("-"*70)
    ag = results['tokenization_approaches']['AutoGraph']
    print(f"Vocab size: {ag['vocab_size']}")
    print(f"Test set:  {ag['test_set']['epochs_to_convergence']} epochs → {ag['test_set']['final_accuracy']*100:.1f}% acc")
    print(f"Full set:  {ag['full_set']['epochs_to_convergence']} epochs → {ag['full_set']['final_accuracy']*100:.1f}% acc")
    print(f"Parameters: {ag['full_set']['model_params']:,}")

    print("\n" + "-"*70)
    print("KEY DIFFERENCES")
    print("-"*70)
    for category, details in results['observations'].items():
        print(f"\n{category.upper()}:")
        for approach, obs in details.items():
            print(f"  {approach}: {obs}")

    print("\n" + "="*70)
