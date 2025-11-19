"""Utilities for logging training results with timing information."""

import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import torch


class TrainingLogger:
    """Logger for training experiments with timing and result tracking."""

    def __init__(self, experiment_name: str, model_name: str, results_dir: str = "results/training_logs"):
        self.experiment_name = experiment_name
        self.model_name = model_name
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Generate unique run ID
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"{model_name}_{experiment_name}_{self.timestamp}"

        # Initialize log data
        self.log_data = {
            "run_id": self.run_id,
            "experiment_name": experiment_name,
            "model_name": model_name,
            "timestamp": self.timestamp,
            "start_time": None,
            "end_time": None,
            "total_time_seconds": None,
            "device": None,
            "model_params": None,
            "config": {},
            "training_history": [],
            "final_metrics": {},
            "dataset_info": {}
        }

    def start(self, config: Dict[str, Any], device: str, model_params: int,
              dataset_info: Optional[Dict[str, Any]] = None):
        """Start logging a training run."""
        self.log_data["start_time"] = time.time()
        self.log_data["config"] = config
        self.log_data["device"] = device
        self.log_data["model_params"] = model_params
        if dataset_info:
            self.log_data["dataset_info"] = dataset_info

        print(f"\n{'='*70}")
        print(f"Training Run: {self.run_id}")
        print(f"{'='*70}")
        print(f"Model: {self.model_name}")
        print(f"Experiment: {self.experiment_name}")
        print(f"Device: {device}")
        print(f"Parameters: {model_params:,}")
        if dataset_info:
            print(f"Dataset: {dataset_info}")
        print(f"{'='*70}\n")

    def log_epoch(self, epoch: int, metrics: Dict[str, float], epoch_time: float):
        """Log metrics for a single epoch."""
        epoch_data = {
            "epoch": epoch,
            "epoch_time": epoch_time,
            **metrics
        }
        self.log_data["training_history"].append(epoch_data)

        # Print epoch summary
        metrics_str = " | ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                                  for k, v in metrics.items()])
        print(f"Epoch {epoch:3d} | Time: {epoch_time:.2f}s | {metrics_str}")

    def finish(self, final_metrics: Dict[str, Any], save_model_path: Optional[str] = None):
        """Finish logging and save results."""
        self.log_data["end_time"] = time.time()
        self.log_data["total_time_seconds"] = self.log_data["end_time"] - self.log_data["start_time"]
        self.log_data["final_metrics"] = final_metrics
        if save_model_path:
            self.log_data["model_checkpoint_path"] = save_model_path

        # Calculate summary statistics
        if self.log_data["training_history"]:
            total_epochs = len(self.log_data["training_history"])
            avg_epoch_time = sum(e["epoch_time"] for e in self.log_data["training_history"]) / total_epochs
            self.log_data["summary"] = {
                "total_epochs": total_epochs,
                "avg_epoch_time": avg_epoch_time,
                "total_training_time": self.log_data["total_time_seconds"]
            }

        # Save to JSON
        log_file = self.results_dir / f"{self.run_id}.json"
        with open(log_file, 'w') as f:
            json.dump(self.log_data, f, indent=2)

        # Save human-readable summary
        summary_file = self.results_dir / f"{self.run_id}_summary.txt"
        self._save_summary(summary_file)

        print(f"\n{'='*70}")
        print(f"Training Complete: {self.run_id}")
        print(f"{'='*70}")
        print(f"Total Time: {self.log_data['total_time_seconds']:.2f}s ({self.log_data['total_time_seconds']/60:.2f}m)")
        print(f"Final Metrics:")
        for key, value in final_metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        print(f"\nResults saved to:")
        print(f"  - {log_file}")
        print(f"  - {summary_file}")
        if save_model_path:
            print(f"  - Model: {save_model_path}")
        print(f"{'='*70}\n")

        return log_file

    def _save_summary(self, path: Path):
        """Save human-readable summary."""
        with open(path, 'w') as f:
            f.write(f"Training Run Summary\n")
            f.write(f"{'='*70}\n\n")
            f.write(f"Run ID: {self.run_id}\n")
            f.write(f"Model: {self.model_name}\n")
            f.write(f"Experiment: {self.experiment_name}\n")
            f.write(f"Timestamp: {self.timestamp}\n\n")

            f.write(f"Configuration:\n")
            f.write(f"{'-'*70}\n")
            for key, value in self.log_data["config"].items():
                f.write(f"  {key}: {value}\n")
            f.write(f"\n")

            if self.log_data["dataset_info"]:
                f.write(f"Dataset Information:\n")
                f.write(f"{'-'*70}\n")
                for key, value in self.log_data["dataset_info"].items():
                    f.write(f"  {key}: {value}\n")
                f.write(f"\n")

            f.write(f"Model:\n")
            f.write(f"{'-'*70}\n")
            f.write(f"  Device: {self.log_data['device']}\n")
            f.write(f"  Parameters: {self.log_data['model_params']:,}\n\n")

            if self.log_data["summary"]:
                f.write(f"Training Summary:\n")
                f.write(f"{'-'*70}\n")
                f.write(f"  Total Epochs: {self.log_data['summary']['total_epochs']}\n")
                f.write(f"  Total Time: {self.log_data['summary']['total_training_time']:.2f}s ")
                f.write(f"({self.log_data['summary']['total_training_time']/60:.2f}m)\n")
                f.write(f"  Avg Epoch Time: {self.log_data['summary']['avg_epoch_time']:.2f}s\n\n")

            f.write(f"Training History:\n")
            f.write(f"{'-'*70}\n")
            for epoch_data in self.log_data["training_history"]:
                f.write(f"  Epoch {epoch_data['epoch']:3d}: ")
                metrics_str = " | ".join([f"{k}={v:.4f}" if isinstance(v, float) and k != 'epoch' and k != 'epoch_time'
                                         else "" for k, v in epoch_data.items() if k not in ['epoch', 'epoch_time']])
                f.write(f"{metrics_str} (time: {epoch_data['epoch_time']:.2f}s)\n")
            f.write(f"\n")

            f.write(f"Final Metrics:\n")
            f.write(f"{'-'*70}\n")
            for key, value in self.log_data["final_metrics"].items():
                if isinstance(value, float):
                    f.write(f"  {key}: {value:.4f}\n")
                else:
                    f.write(f"  {key}: {value}\n")


def compare_training_runs(run_ids: list[str], results_dir: str = "results/training_logs") -> Dict:
    """Compare multiple training runs."""
    results_dir = Path(results_dir)
    comparison = {
        "runs": [],
        "summary": {}
    }

    for run_id in run_ids:
        log_file = results_dir / f"{run_id}.json"
        if not log_file.exists():
            print(f"Warning: Log file not found for {run_id}")
            continue

        with open(log_file, 'r') as f:
            run_data = json.load(f)
            comparison["runs"].append(run_data)

    # Create comparison summary
    if comparison["runs"]:
        comparison["summary"] = {
            "num_runs": len(comparison["runs"]),
            "models": [r["model_name"] for r in comparison["runs"]],
            "total_times": [r["total_time_seconds"] for r in comparison["runs"]],
            "final_metrics": {r["run_id"]: r["final_metrics"] for r in comparison["runs"]}
        }

    return comparison
