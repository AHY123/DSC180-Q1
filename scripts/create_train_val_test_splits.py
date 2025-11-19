"""Create train/val/test splits for larger datasets."""

import os
import shutil
from pathlib import Path
import random
import argparse


def create_splits(source_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """Create train/val/test splits from a directory of graphs.

    Args:
        source_dir: Directory containing .graphml files
        train_ratio: Fraction for training (default 0.7)
        val_ratio: Fraction for validation (default 0.15)
        test_ratio: Fraction for testing (default 0.15)
        seed: Random seed for reproducibility
    """
    source_path = Path(source_dir)

    if not source_path.exists():
        raise ValueError(f"Source directory {source_dir} does not exist")

    # Collect all graphml files
    graphml_files = sorted(source_path.glob("*.graphml"))

    if len(graphml_files) == 0:
        raise ValueError(f"No .graphml files found in {source_dir}")

    print(f"Found {len(graphml_files)} graphs in {source_dir}")

    # Shuffle with fixed seed
    random.seed(seed)
    random.shuffle(graphml_files)

    # Calculate split sizes
    total = len(graphml_files)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    test_size = total - train_size - val_size

    print(f"Splitting into:")
    print(f"  Train: {train_size} ({train_size/total*100:.1f}%)")
    print(f"  Valid: {val_size} ({val_size/total*100:.1f}%)")
    print(f"  Test:  {test_size} ({test_size/total*100:.1f}%)")

    # Create split directories
    splits = {
        'train': graphml_files[:train_size],
        'valid': graphml_files[train_size:train_size+val_size],
        'test': graphml_files[train_size+val_size:]
    }

    for split_name, files in splits.items():
        split_dir = source_path / split_name
        split_dir.mkdir(exist_ok=True)

        # Copy files to split directory
        for src_file in files:
            dst_file = split_dir / src_file.name
            if not dst_file.exists():
                shutil.copy2(src_file, dst_file)

        print(f"Created {split_name} split with {len(files)} files")

    print(f"\nSplits created successfully in {source_dir}/{{train,valid,test}}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir', type=str, required=True,
                        help='Source directory containing .graphml files')
    parser.add_argument('--train', type=float, default=0.7,
                        help='Train ratio (default 0.7)')
    parser.add_argument('--val', type=float, default=0.15,
                        help='Validation ratio (default 0.15)')
    parser.add_argument('--test', type=float, default=0.15,
                        help='Test ratio (default 0.15)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default 42)')
    args = parser.parse_args()

    # Validate ratios
    total_ratio = args.train + args.val + args.test
    if abs(total_ratio - 1.0) > 0.001:
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")

    create_splits(
        args.source_dir,
        train_ratio=args.train,
        val_ratio=args.val,
        test_ratio=args.test,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
