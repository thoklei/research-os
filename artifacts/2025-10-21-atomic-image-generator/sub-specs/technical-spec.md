# Technical Specification

This is the technical specification for the artifact detailed in research-os/artifacts/2025-10-21-atomic-image-generator/spec.md

> Created: 2025-10-21
> Version: 1.0.0

## Technical Requirements

### Core Parameters

```python
GRID_SIZE = 16  # Fixed 16x16 grid
CORPUS_SIZE = 10  # Default corpus size (configurable)
MIN_PIXELS_PER_OBJECT = 2
MAX_PIXELS_PER_OBJECT = 15
MIN_OBJECTS_PER_IMAGE = 1
MAX_OBJECTS_PER_IMAGE = 4
COLOR_PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]  # ARC color palette
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1
```

### Object Type Distribution

```python
OBJECT_TYPES = {
    'blob': 0.40,      # 40% random blobs
    'rectangle': 0.20,  # 20% rectangles
    'line': 0.20,       # 20% lines
    'pattern': 0.20     # 20% patterns
}
```

### Grid Generation Algorithm

**High-level flow:**
1. Initialize empty 16x16 grid (all zeros)
2. Determine number of objects (random 1-4)
3. For each object:
   - Select object type based on distribution
   - Generate candidate object
   - Validate size constraint (2-15 pixels)
   - If valid, attempt placement with collision detection
   - If invalid size, regenerate object (max 10 attempts)
   - If placement fails after 50 attempts, skip object
4. Return completed grid

**Data Structure:**
```python
grid = np.zeros((16, 16), dtype=np.uint8)
occupied = np.zeros((16, 16), dtype=bool)  # Collision mask
```

### Object Generators

#### 1. Blob Generator (Connectivity-Biased Growth)

**Algorithm:**
```
Input: target_size (random 2-15), color
Output: Set of (row, col) coordinates

1. Select random seed point (r, c)
2. Initialize pixels = {(r, c)}
3. While len(pixels) < target_size:
   a. Build frontier of unoccupied neighbors adjacent to current pixels
   b. If frontier empty, break (constrained by occupied space)
   c. Select random frontier cell with bias toward connected growth
   d. Add cell to pixels
4. Return pixels
```

**Connectivity Bias:**
- Weight frontier cells by number of existing neighbors
- Cells with 2+ neighbors are 3x more likely to be selected
- Creates cohesive blob shapes rather than scattered pixels

**Implementation Notes:**
- Use 4-connectivity (up, down, left, right)
- Track frontier as set for O(1) lookups
- Validate each addition doesn't exceed grid bounds

#### 2. Rectangle Generator

**Algorithm:**
```
Input: None
Output: Set of (row, col) coordinates

1. Determine target size (random 2-15 pixels)
2. Calculate possible dimensions:
   - For size S, find all (h, w) where h * w == S or h * w ≈ S
   - Prefer exact matches, allow ±1 pixel for better variety
3. Select random (height, width) from valid dimensions
4. Select random top-left corner position
5. Fill rectangle coordinates
6. Return pixels
```

**Dimension Selection:**
- For size 6: could be 2x3, 3x2, or 1x6, 6x1
- For size 7: use 2x3 (6 pixels) or 2x4 (8 pixels), adjust to 7
- Bias toward more square-like rectangles (aspect ratio < 3:1)

#### 3. Line Generator

**Algorithm:**
```
Input: None
Output: Set of (row, col) coordinates

1. Determine target length (random 2-15 pixels)
2. Select orientation: horizontal, vertical, or diagonal (45°, -45°)
3. Select random starting point
4. Draw line in selected direction for target length
5. Return pixels
```

**Orientation Distribution:**
- Horizontal: 30%
- Vertical: 30%
- Diagonal (45°): 20%
- Diagonal (-45°): 20%

**Implementation:**
- Use Bresenham's line algorithm for diagonals
- Ensure line stays within grid bounds
- Truncate if hits boundary before reaching target length

#### 4. Pattern Generator

**Algorithm:**
```
Input: None
Output: Set of (row, col) coordinates

1. Select pattern type: checkerboard, L-shape, T-shape, plus-sign, zigzag
2. Generate base pattern coordinates
3. Scale to meet size constraint (2-15 pixels)
4. Return pixels
```

**Pattern Templates:**

**Checkerboard (2x2 to 4x4):**
```
1 0    1 0 1
0 1    0 1 0
       1 0 1
```

**L-shape:**
```
1 0
1 0
1 1
```

**T-shape:**
```
1 1 1
0 1 0
```

**Plus-sign:**
```
0 1 0
1 1 1
0 1 0
```

**Zigzag:**
```
1 0
1 1
0 1
```

**Scaling Strategy:**
- Generate pattern at base size
- If too small, extend pattern (repeat or grow)
- If too large, truncate
- Ensure final size is 2-15 pixels

### Collision Detection and Placement

**Algorithm:**
```
Input: object_pixels (set of relative coordinates), grid, occupied_mask
Output: Boolean success, updated grid and mask

1. For max_attempts (50):
   a. Select random anchor point (r_anchor, c_anchor)
   b. Translate object coordinates to absolute positions
   c. Check bounds: all pixels within [0, 16)
   d. Check collisions: no pixel overlaps occupied_mask
   e. If valid:
      - Mark pixels in occupied_mask
      - Set pixels in grid to object color
      - Return True
2. Return False (placement failed)
```

**Collision Check:**
```python
def check_collision(pixels, occupied):
    for (r, c) in pixels:
        if r < 0 or r >= 16 or c < 0 or c >= 16:
            return True  # Out of bounds
        if occupied[r, c]:
            return True  # Collision
    return False
```

### Color Sampling Strategy

**Uniform Random Sampling:**
```python
color = np.random.choice(COLOR_PALETTE)  # Uniform distribution [1-9]
```

**Per-Object Color:**
- Each object gets one color
- All pixels in an object share the same color
- Different objects may have same or different colors (random)

**Background:**
- Grid background is 0 (black in ARC palette)
- Colors 1-9 represent the standard ARC colors

### Size Constraint Enforcement

**Hard Constraint: 2-15 Pixels**

**Validation:**
```python
def validate_object_size(pixels):
    size = len(pixels)
    return MIN_PIXELS_PER_OBJECT <= size <= MAX_PIXELS_PER_OBJECT
```

**Regeneration Strategy:**
1. Generate object candidate
2. Check size: `validate_object_size(pixels)`
3. If invalid:
   - Regenerate with new random parameters
   - Max 10 regeneration attempts per object
4. If all attempts fail, log warning and skip object
5. Continue with remaining objects

**Why Regeneration vs. Adjustment:**
- Preserves object type integrity (don't distort blobs/rectangles)
- Cleaner algorithm (no complex resizing logic)
- 10 attempts sufficient for random generation success rate

### Output Format

**Data Structure:**
```python
# Single image
image = np.ndarray(shape=(16, 16), dtype=np.uint8)
# Values range: 0-9 (0=background, 1-9=colors)

# Full corpus
corpus = {
    'train': np.ndarray(shape=(N_train, 16, 16), dtype=np.uint8),
    'val': np.ndarray(shape=(N_val, 16, 16), dtype=np.uint8),
    'test': np.ndarray(shape=(N_test, 16, 16), dtype=np.uint8)
}
```

**File Format:**
```python
# Compressed numpy archive
np.savez_compressed(
    'atomic_corpus.npz',
    train=train_images,
    val=val_images,
    test=test_images,
    metadata={
        'total_size': CORPUS_SIZE,
        'grid_size': GRID_SIZE,
        'train_split': TRAIN_SPLIT,
        'val_split': VAL_SPLIT,
        'test_split': TEST_SPLIT,
        'generation_date': timestamp,
        'color_palette': COLOR_PALETTE
    }
)
```

**Split Calculation:**
```python
n_train = int(CORPUS_SIZE * 0.8)
n_val = int(CORPUS_SIZE * 0.1)
n_test = CORPUS_SIZE - n_train - n_val  # Remaining to handle rounding
```

### Configurable Parameters

**Command-line Interface:**
```python
python generate_corpus.py --corpus_size 10 --output atomic_corpus.npz
```

**Parameters:**
- `--corpus_size`: Number of images to generate (default: 10)
- `--output`: Output file path (default: atomic_corpus.npz)
- `--seed`: Random seed for reproducibility (optional)
- `--train_split`: Training set proportion (default: 0.8)
- `--val_split`: Validation set proportion (default: 0.1)

**Configuration File Support:**
```python
# config.json
{
    "corpus_size": 10,
    "grid_size": 16,
    "min_objects": 1,
    "max_objects": 4,
    "min_pixels": 2,
    "max_pixels": 15,
    "object_distribution": {
        "blob": 0.40,
        "rectangle": 0.20,
        "line": 0.20,
        "pattern": 0.20
    }
}
```

## Visualization Requirements

### Matplotlib Integration

**Color Mapping:**
```python
# ARC color palette (approximation for matplotlib)
ARC_COLORS = {
    0: '#000000',  # Black (background)
    1: '#0074D9',  # Blue
    2: '#FF4136',  # Red
    3: '#2ECC40',  # Green
    4: '#FFDC00',  # Yellow
    5: '#AAAAAA',  # Gray
    6: '#F012BE',  # Magenta
    7: '#FF851B',  # Orange
    8: '#7FDBFF',  # Cyan
    9: '#870C25',  # Maroon
}
```

### Preview Functionality

**Single Image Display:**
```python
def visualize_grid(grid, title="Generated Grid", save_path=None):
    """
    Display single 16x16 grid with ARC colors

    Args:
        grid: np.ndarray (16, 16) with values 0-9
        title: Plot title
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    # Create RGB image from color palette
    rgb_image = np.zeros((16, 16, 3))
    for i in range(16):
        for j in range(16):
            color_hex = ARC_COLORS[grid[i, j]]
            rgb_image[i, j] = hex_to_rgb(color_hex)

    ax.imshow(rgb_image, interpolation='nearest')
    ax.set_title(title)
    ax.grid(True, which='both', color='white', linewidth=0.5, alpha=0.3)
    ax.set_xticks(np.arange(-0.5, 16, 1))
    ax.set_yticks(np.arange(-0.5, 16, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
```

**Multi-Image Gallery:**
```python
def visualize_corpus_sample(images, n_samples=9, save_path=None):
    """
    Display grid of sample images from corpus

    Args:
        images: np.ndarray (N, 16, 16)
        n_samples: Number of samples to display (default 9 for 3x3 grid)
        save_path: Optional path to save figure
    """
    n_rows = int(np.sqrt(n_samples))
    n_cols = int(np.ceil(n_samples / n_rows))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 12))
    axes = axes.flatten()

    sample_indices = np.random.choice(len(images), n_samples, replace=False)

    for idx, ax in enumerate(axes):
        if idx < n_samples:
            grid = images[sample_indices[idx]]
            rgb_image = grid_to_rgb(grid)  # Convert using ARC_COLORS
            ax.imshow(rgb_image, interpolation='nearest')
            ax.set_title(f'Sample {sample_indices[idx]}', fontsize=10)
            ax.grid(True, which='both', color='white', linewidth=0.3, alpha=0.3)
        ax.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
```

### Quality Verification Visual Checks

**Object Count Distribution:**
```python
def plot_object_count_distribution(corpus_images):
    """
    Histogram showing distribution of object counts per image
    """
    object_counts = []
    for img in corpus_images:
        # Count connected components (objects)
        count = count_objects(img)
        object_counts.append(count)

    plt.figure(figsize=(8, 5))
    plt.hist(object_counts, bins=range(1, 6), edgecolor='black', alpha=0.7)
    plt.xlabel('Number of Objects per Image')
    plt.ylabel('Frequency')
    plt.title('Object Count Distribution')
    plt.xticks(range(1, 5))
    plt.grid(axis='y', alpha=0.3)
    plt.show()
```

**Object Size Distribution:**
```python
def plot_object_size_distribution(corpus_images):
    """
    Histogram showing distribution of object sizes
    """
    object_sizes = []
    for img in corpus_images:
        sizes = extract_object_sizes(img)  # Extract each object's pixel count
        object_sizes.extend(sizes)

    plt.figure(figsize=(8, 5))
    plt.hist(object_sizes, bins=range(2, 17), edgecolor='black', alpha=0.7)
    plt.xlabel('Object Size (pixels)')
    plt.ylabel('Frequency')
    plt.title('Object Size Distribution')
    plt.axvline(x=2, color='r', linestyle='--', label='Min Size')
    plt.axvline(x=15, color='r', linestyle='--', label='Max Size')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.show()
```

**Object Type Distribution:**
```python
def plot_object_type_distribution(corpus_metadata):
    """
    Bar chart showing actual vs expected object type distribution

    Args:
        corpus_metadata: Dictionary with object type counts during generation
    """
    types = ['Blob', 'Rectangle', 'Line', 'Pattern']
    expected = [0.40, 0.20, 0.20, 0.20]
    actual = corpus_metadata['type_distribution']

    x = np.arange(len(types))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, expected, width, label='Expected', alpha=0.7)
    ax.bar(x + width/2, actual, width, label='Actual', alpha=0.7)

    ax.set_ylabel('Proportion')
    ax.set_title('Object Type Distribution')
    ax.set_xticks(x)
    ax.set_xticklabels(types)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.show()
```

**Color Usage Distribution:**
```python
def plot_color_distribution(corpus_images):
    """
    Bar chart showing frequency of each color in corpus
    """
    color_counts = np.zeros(10)  # 0-9
    for img in corpus_images:
        unique, counts = np.unique(img, return_counts=True)
        for color, count in zip(unique, counts):
            color_counts[color] += count

    # Exclude background (0) from visualization
    colors = range(1, 10)
    counts = color_counts[1:]

    plt.figure(figsize=(10, 5))
    bars = plt.bar(colors, counts, color=[ARC_COLORS[i] for i in colors],
                   edgecolor='black', alpha=0.8)
    plt.xlabel('Color ID')
    plt.ylabel('Total Pixel Count')
    plt.title('Color Usage Distribution Across Corpus')
    plt.xticks(colors)
    plt.grid(axis='y', alpha=0.3)
    plt.show()
```

### Interactive Exploration

**Optional: Interactive Grid Viewer:**
```python
def interactive_viewer(corpus_images):
    """
    Simple keyboard-based viewer to browse corpus

    Keys:
    - Right arrow: Next image
    - Left arrow: Previous image
    - Q: Quit
    """
    import matplotlib.pyplot as plt

    idx = [0]  # Mutable counter

    def on_key(event):
        if event.key == 'right':
            idx[0] = (idx[0] + 1) % len(corpus_images)
        elif event.key == 'left':
            idx[0] = (idx[0] - 1) % len(corpus_images)
        elif event.key == 'q':
            plt.close()
            return

        update_plot()

    def update_plot():
        ax.clear()
        grid = corpus_images[idx[0]]
        rgb_image = grid_to_rgb(grid)
        ax.imshow(rgb_image, interpolation='nearest')
        ax.set_title(f'Image {idx[0] + 1}/{len(corpus_images)}')
        ax.grid(True, which='both', color='white', linewidth=0.5, alpha=0.3)
        ax.axis('off')
        fig.canvas.draw()

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.canvas.mpl_connect('key_press_event', on_key)
    update_plot()
    plt.show()
```

## External Dependencies

### Required Python Packages

**NumPy (version >= 1.20.0)**
- **Purpose:** Core array operations and grid manipulation
- **Usage:**
  - Grid storage (`np.ndarray`)
  - Random number generation (`np.random`)
  - Array operations (indexing, masking, boolean operations)
  - File I/O (`np.savez_compressed`, `np.load`)
- **Justification:** Essential for efficient numerical operations and data storage

**Matplotlib (version >= 3.7.0)**
- **Purpose:** Visualization and quality verification
- **Usage:**
  - Grid visualization with ARC color palette
  - Distribution plots (histograms, bar charts)
  - Multi-image gallery displays
  - Export visualizations to PNG/PDF
- **Justification:** Standard Python visualization library, required for preview functionality and quality checks

**SciPy (version >= 1.10.0)** - Optional
- **Purpose:** Connected component analysis (optional utility)
- **Usage:**
  - `scipy.ndimage.label()` for counting objects in generated grids
  - Useful for validation and statistics
- **Justification:** Provides robust connected component labeling for quality verification. Can be replaced with custom implementation if dependency minimization is desired.

### Optional Dependencies

**PyTorch (version >= 2.0.0)** - Future Compatibility
- **Purpose:** DataLoader integration (not required for generation)
- **Usage:**
  - Custom `Dataset` class wrapper for generated corpus
  - Batching and shuffling for model training
- **Justification:** Mentioned in roadmap for future ML training compatibility. Not needed for initial generation phase.

**Pillow (version >= 9.0.0)** - Optional
- **Purpose:** Alternative image export format
- **Usage:**
  - Export grids as PNG images directly
  - Higher quality image exports
- **Justification:** Useful if users want standalone image files instead of matplotlib figures. Optional enhancement.

### Minimal Installation

```bash
# Core requirements
pip install numpy>=1.20.0 matplotlib>=3.7.0

# Optional for validation
pip install scipy>=1.10.0

# Future compatibility (not required for v1.0)
pip install torch>=2.0.0
```

### Dependency Rationale

**Why not PIL/Pillow for core generation?**
- NumPy arrays are sufficient for grid representation
- Matplotlib provides visualization without additional image library
- Keeps dependencies minimal

**Why include SciPy as optional?**
- Connected component labeling is non-trivial to implement correctly
- SciPy is widely available in scientific Python environments
- Can be replaced with custom BFS/DFS if needed

**Why Matplotlib >= 3.7.0?**
- Improved color handling and interpolation options
- Better grid display functionality
- Modern API for subplots and figure management

### Development Dependencies

```bash
# For testing and development only
pip install pytest>=7.0.0  # Unit testing
pip install black>=23.0.0  # Code formatting
pip install mypy>=1.0.0    # Type checking
```

**Not required for production corpus generation, only for development workflow.**
