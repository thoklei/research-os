# Technical Specification

This is the technical specification for the artifact detailed in research-os/artifacts/2025-10-22-large-scale-dataset-generation/spec.md

> Created: 2025-10-22
> Version: 1.0.0

## Technical Requirements

### 1. Hierarchical Directory Structure

**Objective:** Organize datasets with version control and standardized naming.

**Directory Layout:**
```
datasets/
└── YYYY-MM-DD-vN/
    ├── train.npz
    ├── val.npz
    ├── test.npz
    └── metadata.json
```

**Naming Convention:**
- Date format: `YYYY-MM-DD` (e.g., `2025-10-22`)
- Version format: `vN` where N is an integer (e.g., `v1`, `v2`)
- Full example: `datasets/2025-10-22-v1/`

**Implementation:**
- Use `os.makedirs(output_dir, exist_ok=True)` to create directory
- Default output directory: `datasets/{today}-v1`
- Allow override via `--output-dir` CLI flag
- Validate directory creation before generation starts

### 2. CLI Implementation

**Script Name:** `generate_dataset.py`

**Required Arguments:**
```python
parser = argparse.ArgumentParser(
    description='Generate large-scale atomic image datasets'
)
parser.add_argument(
    '--num-images',
    type=int,
    required=True,
    help='Total number of images to generate (e.g., 1000, 100000)'
)
parser.add_argument(
    '--output-dir',
    type=str,
    default=None,
    help='Output directory (default: datasets/YYYY-MM-DD-v1)'
)
parser.add_argument(
    '--seed',
    type=int,
    default=42,
    help='Random seed for reproducibility (default: 42)'
)
```

**Default Behavior:**
- If `--output-dir` not provided, auto-generate: `datasets/{date.today()}-v1`
- Seed defaults to 42 for reproducibility
- Validate `--num-images > 0` before proceeding

**Example Usage:**
```bash
python generate_dataset.py --num-images 100000
python generate_dataset.py --num-images 50000 --output-dir datasets/2025-10-22-v2 --seed 123
```

### 3. Memory Estimation and User Confirmation

**Memory Formula:**
```python
def estimate_memory(num_images: int, grid_size: int = 16) -> dict:
    """
    Calculate memory requirements for dataset generation.

    Args:
        num_images: Total number of images to generate
        grid_size: Grid dimension (default 16x16)

    Returns:
        dict with memory estimates in bytes and human-readable format
    """
    bytes_per_image = grid_size * grid_size  # uint8: 1 byte per pixel
    total_bytes = num_images * bytes_per_image

    # Convert to human-readable
    if total_bytes < 1024:
        readable = f"{total_bytes} bytes"
    elif total_bytes < 1024**2:
        readable = f"{total_bytes / 1024:.2f} KB"
    elif total_bytes < 1024**3:
        readable = f"{total_bytes / (1024**2):.2f} MB"
    else:
        readable = f"{total_bytes / (1024**3):.2f} GB"

    return {
        'bytes': total_bytes,
        'readable': readable,
        'num_images': num_images
    }
```

**User Confirmation Flow:**
```python
# Display memory estimate
mem_info = estimate_memory(args.num_images)
print(f"\nDataset Generation Plan:")
print(f"  Images: {mem_info['num_images']:,}")
print(f"  Memory: {mem_info['readable']}")
print(f"  Output: {output_dir}")

# Prompt for confirmation
response = input("\nProceed with generation? (y/n): ")
if response.lower() not in ['y', 'yes']:
    print("Generation cancelled.")
    sys.exit(0)
```

**Display Before Generation:**
- Total images to generate
- Estimated memory usage (human-readable)
- Output directory path
- Wait for explicit user confirmation (y/n)

### 4. Data Type Conversion (uint8)

**Objective:** Reduce memory footprint by 87.5% (int64 → uint8).

**Technical Details:**
- Current grid values: 0-255 (already in uint8 range)
- Default NumPy dtype: int64 (8 bytes per value)
- Target dtype: uint8 (1 byte per value)
- Savings: 256 bytes vs 2048 bytes per 16x16 grid

**Implementation in `visualization.py`:**

Modify `save_corpus()` function:
```python
def save_corpus(splits: dict[str, tuple], output_dir: str, dtype=np.uint8):
    """
    Save train/val/test splits to .npz files with efficient dtype.

    Args:
        splits: Dict with 'train', 'val', 'test' keys
        output_dir: Directory to save .npz files
        dtype: NumPy dtype for arrays (default: np.uint8)
    """
    os.makedirs(output_dir, exist_ok=True)

    for split_name, (images, labels) in splits.items():
        # Convert to uint8 for memory efficiency
        images_uint8 = images.astype(dtype)
        labels_uint8 = labels.astype(dtype)

        filepath = os.path.join(output_dir, f"{split_name}.npz")
        np.savez_compressed(
            filepath,
            images=images_uint8,
            labels=labels_uint8
        )
        print(f"Saved {split_name}: {filepath}")
```

**Backward Compatibility:**
- Add `dtype` parameter with default `np.uint8`
- Existing code continues to work without changes
- Explicit dtype conversion before saving

### 5. Progress Tracking

**Library:** `tqdm` (already in tech stack)

**Implementation:**
```python
from tqdm import tqdm

def generate_corpus(num_samples: int, show_progress: bool = True) -> tuple:
    """
    Generate atomic image corpus with optional progress bar.

    Args:
        num_samples: Number of images to generate
        show_progress: Display tqdm progress bar (default: True)

    Returns:
        Tuple of (images, labels) as numpy arrays
    """
    images = []
    labels = []

    # Create progress bar
    iterator = range(num_samples)
    if show_progress:
        iterator = tqdm(iterator, desc="Generating images", unit="img")

    for _ in iterator:
        grid, label = generate_atomic_image()
        images.append(grid)
        labels.append(label)

    return np.array(images), np.array(labels)
```

**Progress Bar Format:**
- Description: "Generating images"
- Unit: "img"
- Show iteration count, rate, and ETA
- Example: `Generating images: 45000/100000 [01:23<01:05, 543.2img/s]`

### 6. Metadata JSON Structure

**Filename:** `metadata.json`

**Required Fields:**
```json
{
  "version": "1.0.0",
  "timestamp": "2025-10-22T14:32:15.123456",
  "generation_params": {
    "num_images": 100000,
    "seed": 42,
    "grid_size": 16,
    "splits": {
      "train": 0.8,
      "val": 0.1,
      "test": 0.1
    }
  },
  "dataset_info": {
    "train_count": 80000,
    "val_count": 10000,
    "test_count": 10000,
    "total_count": 100000
  },
  "memory_info": {
    "bytes_per_image": 256,
    "total_bytes": 25600000,
    "readable": "25.60 MB"
  },
  "file_format": {
    "dtype": "uint8",
    "compression": "npz"
  }
}
```

**Implementation:**
```python
import json
from datetime import datetime

def save_metadata(output_dir: str, num_images: int, seed: int, splits_info: dict):
    """Save dataset metadata to JSON file."""
    metadata = {
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "generation_params": {
            "num_images": num_images,
            "seed": seed,
            "grid_size": 16,
            "splits": {
                "train": 0.8,
                "val": 0.1,
                "test": 0.1
            }
        },
        "dataset_info": splits_info,
        "memory_info": estimate_memory(num_images),
        "file_format": {
            "dtype": "uint8",
            "compression": "npz"
        }
    }

    filepath = os.path.join(output_dir, "metadata.json")
    with open(filepath, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved metadata: {filepath}")
```

### 7. Visual Validation Tool

**Objective:** Quick quality check of generated dataset.

**Functionality:**
- Load 100 random samples from train split
- Display in 10x10 grid
- Show label distribution histogram
- Save validation report as PNG

**Implementation:**
```python
def validate_dataset(dataset_dir: str, num_samples: int = 100):
    """
    Visual validation of generated dataset.

    Args:
        dataset_dir: Path to dataset directory (contains train.npz)
        num_samples: Number of random samples to inspect (default: 100)
    """
    # Load train split
    train_data = np.load(os.path.join(dataset_dir, "train.npz"))
    images = train_data['images']
    labels = train_data['labels']

    # Random sample
    num_samples = min(num_samples, len(images))
    indices = np.random.choice(len(images), num_samples, replace=False)
    sample_images = images[indices]
    sample_labels = labels[indices]

    # Create visualization
    fig, axes = plt.subplots(11, 10, figsize=(20, 22))

    # Plot 100 images (10x10 grid)
    for i in range(100):
        if i < num_samples:
            row, col = divmod(i, 10)
            ax = axes[row, col]
            ax.imshow(sample_images[i], cmap='gray', vmin=0, vmax=255)
            ax.set_title(f"L:{sample_labels[i]}", fontsize=8)
            ax.axis('off')

    # Label distribution histogram (bottom row)
    ax_hist = plt.subplot(11, 1, 11)
    ax_hist.hist(labels, bins=256, color='blue', alpha=0.7)
    ax_hist.set_xlabel('Label Value')
    ax_hist.set_ylabel('Frequency')
    ax_hist.set_title('Label Distribution (Full Dataset)')

    plt.tight_layout()

    # Save validation report
    output_path = os.path.join(dataset_dir, "validation_report.png")
    plt.savefig(output_path, dpi=150)
    print(f"Validation report saved: {output_path}")
    plt.show()
```

**Usage:**
```python
# In generate_dataset.py, after generation completes
validate_dataset(args.output_dir)
```

## Code Modifications

### 1. Changes to `pipeline.py`

**Function: `generate_corpus`**
- Add `show_progress` parameter (default: `True`)
- Integrate tqdm progress bar around generation loop
- Maintain existing return signature: `tuple[np.ndarray, np.ndarray]`

**Function: `split_corpus`**
- No changes required
- Already returns correct split format
- Compatible with uint8 conversion downstream

**New Function: `estimate_memory`**
```python
def estimate_memory(num_images: int, grid_size: int = 16) -> dict:
    """Calculate memory requirements for dataset generation."""
    bytes_per_image = grid_size * grid_size
    total_bytes = num_images * bytes_per_image

    if total_bytes < 1024:
        readable = f"{total_bytes} bytes"
    elif total_bytes < 1024**2:
        readable = f"{total_bytes / 1024:.2f} KB"
    elif total_bytes < 1024**3:
        readable = f"{total_bytes / (1024**2):.2f} MB"
    else:
        readable = f"{total_bytes / (1024**3):.2f} GB"

    return {
        'bytes': total_bytes,
        'readable': readable,
        'num_images': num_images,
        'bytes_per_image': bytes_per_image
    }
```

### 2. Changes to `visualization.py`

**Function: `save_corpus`**
- Add `dtype` parameter (default: `np.uint8`)
- Explicit dtype conversion before `np.savez_compressed`
- Add split size logging

**Updated Signature:**
```python
def save_corpus(splits: dict[str, tuple], output_dir: str, dtype=np.uint8):
    """Save train/val/test splits with specified dtype."""
    os.makedirs(output_dir, exist_ok=True)

    for split_name, (images, labels) in splits.items():
        images_typed = images.astype(dtype)
        labels_typed = labels.astype(dtype)

        filepath = os.path.join(output_dir, f"{split_name}.npz")
        np.savez_compressed(filepath, images=images_typed, labels=labels_typed)

        print(f"Saved {split_name}: {len(images_typed)} samples → {filepath}")
```

**New Function: `save_metadata`**
```python
def save_metadata(output_dir: str, num_images: int, seed: int, splits_info: dict):
    """Save dataset metadata to JSON file."""
    metadata = {
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "generation_params": {
            "num_images": num_images,
            "seed": seed,
            "grid_size": 16,
            "splits": {"train": 0.8, "val": 0.1, "test": 0.1}
        },
        "dataset_info": splits_info,
        "memory_info": estimate_memory(num_images),
        "file_format": {"dtype": "uint8", "compression": "npz"}
    }

    with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
```

**New Function: `validate_dataset`**
- See implementation in Technical Requirements section 7
- Independent utility function
- Can be called from CLI or used standalone

### 3. New CLI Script: `generate_dataset.py`

**Location:** Project root or `src/` directory

**Structure:**
```python
#!/usr/bin/env python3
"""
Large-scale atomic image dataset generator.

Usage:
    python generate_dataset.py --num-images 100000
    python generate_dataset.py --num-images 50000 --output-dir datasets/custom --seed 123
"""

import argparse
import sys
from datetime import date
from pipeline import generate_corpus, split_corpus, estimate_memory
from visualization import save_corpus, save_metadata, validate_dataset

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Generate large-scale atomic image datasets'
    )
    parser.add_argument('--num-images', type=int, required=True,
                        help='Total number of images to generate')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: datasets/YYYY-MM-DD-v1)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    args = parser.parse_args()

    # Validate inputs
    if args.num_images <= 0:
        print("Error: --num-images must be positive")
        sys.exit(1)

    # Set output directory
    output_dir = args.output_dir or f"datasets/{date.today()}-v1"

    # Display memory estimate and confirm
    mem_info = estimate_memory(args.num_images)
    print(f"\nDataset Generation Plan:")
    print(f"  Images: {mem_info['num_images']:,}")
    print(f"  Memory: {mem_info['readable']}")
    print(f"  Output: {output_dir}")

    response = input("\nProceed with generation? (y/n): ")
    if response.lower() not in ['y', 'yes']:
        print("Generation cancelled.")
        sys.exit(0)

    # Set random seed
    np.random.seed(args.seed)

    # Generate corpus
    print(f"\nGenerating {args.num_images:,} images...")
    images, labels = generate_corpus(args.num_images, show_progress=True)

    # Split corpus
    print("\nSplitting dataset...")
    splits = split_corpus(images, labels)

    # Calculate split sizes
    splits_info = {
        'train_count': len(splits['train'][0]),
        'val_count': len(splits['val'][0]),
        'test_count': len(splits['test'][0]),
        'total_count': args.num_images
    }

    # Save dataset
    print(f"\nSaving dataset to {output_dir}...")
    save_corpus(splits, output_dir, dtype=np.uint8)
    save_metadata(output_dir, args.num_images, args.seed, splits_info)

    # Validate dataset
    print("\nValidating dataset...")
    validate_dataset(output_dir)

    print(f"\nDataset generation complete!")
    print(f"Location: {output_dir}")

if __name__ == '__main__':
    main()
```

### 4. Backward Compatibility

**Existing Code Compatibility:**
- `generate_corpus()`: New `show_progress` parameter defaults to `True`, existing calls work unchanged
- `save_corpus()`: New `dtype` parameter defaults to `np.uint8`, existing calls auto-convert to uint8
- `load_corpus()`: No changes needed, works with uint8 .npz files
- All existing functions maintain original signatures with new optional parameters

**Migration Path:**
- Existing scripts continue to work without modification
- New features available via optional parameters
- No breaking changes to public API

**Testing Backward Compatibility:**
```python
# Old code still works
images, labels = generate_corpus(1000)  # Progress bar shown by default
splits = split_corpus(images, labels)
save_corpus(splits, "output")  # Automatically saves as uint8

# New code with explicit control
images, labels = generate_corpus(100000, show_progress=True)
save_corpus(splits, "output", dtype=np.uint8)
```

## External Dependencies

No new external dependencies required. All required libraries are already in the project tech stack:

- `numpy`: Array operations, dtype conversion, .npz file format
- `matplotlib`: Visualization and validation reporting
- `tqdm`: Progress bar display
- Standard library: `argparse`, `json`, `datetime`, `os`, `sys`
