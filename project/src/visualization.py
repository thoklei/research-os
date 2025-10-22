"""
Visualization and Output Serialization - Task 5

This module implements:
- Matplotlib grid visualization with ARC color scheme
- Gallery visualization for multiple grids
- .npz compression and serialization
- Corpus save/load with train/val/test splits
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from typing import List, Union, Optional, Dict
from atomic_generator import Grid


# ARC color scheme (10 colors: 0-9)
# 0 = black (background)
# 1-9 = object colors
ARC_COLORMAP = [
    '#000000',  # 0: Black (background)
    '#0074D9',  # 1: Blue
    '#FF4136',  # 2: Red
    '#2ECC40',  # 3: Green
    '#FFDC00',  # 4: Yellow
    '#AAAAAA',  # 5: Gray
    '#F012BE',  # 6: Magenta
    '#FF851B',  # 7: Orange
    '#7FDBFF',  # 8: Cyan
    '#870C25',  # 9: Maroon
]


def visualize_grid(grid: Union[Grid, np.ndarray],
                   title: str = "ARC Grid",
                   show: bool = True,
                   figsize: tuple = (6, 6)):
    """
    Visualize a single grid using matplotlib with ARC color scheme.

    Args:
        grid: Grid instance or numpy array (16x16)
        title: Title for the plot
        show: Whether to display the plot
        figsize: Figure size (width, height)

    Returns:
        matplotlib Figure instance
    """
    # Extract data array
    if isinstance(grid, Grid):
        data = grid.data
    else:
        data = grid

    # Create colormap
    cmap = ListedColormap(ARC_COLORMAP)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Display grid
    im = ax.imshow(data, cmap=cmap, vmin=0, vmax=9, interpolation='nearest')

    # Add gridlines
    ax.set_xticks(np.arange(-0.5, 16, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 16, 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)

    # Remove major ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Set title
    ax.set_title(title, fontsize=12, fontweight='bold')

    # Tight layout
    plt.tight_layout()

    if show:
        plt.show()

    return fig


def visualize_gallery(grids: Union[List[Grid], np.ndarray],
                     rows: int = 2,
                     cols: int = 3,
                     titles: Optional[List[str]] = None,
                     show: bool = True,
                     figsize: tuple = (12, 8)):
    """
    Visualize multiple grids in a gallery layout.

    Args:
        grids: List of Grid instances or numpy array (N, 16, 16)
        rows: Number of rows in gallery
        cols: Number of columns in gallery
        titles: Optional list of titles for each grid
        show: Whether to display the plot
        figsize: Figure size (width, height)

    Returns:
        matplotlib Figure instance
    """
    # Convert to list of arrays
    if isinstance(grids, np.ndarray):
        grid_arrays = [grids[i] for i in range(min(len(grids), rows * cols))]
    else:
        grid_arrays = [g.data if isinstance(g, Grid) else g for g in grids[:rows * cols]]

    # Create colormap
    cmap = ListedColormap(ARC_COLORMAP)

    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    # Handle single subplot case
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1 or cols == 1:
        axes = axes.reshape(rows, cols)

    # Plot each grid
    idx = 0
    for i in range(rows):
        for j in range(cols):
            ax = axes[i, j]

            if idx < len(grid_arrays):
                # Display grid
                ax.imshow(grid_arrays[idx], cmap=cmap, vmin=0, vmax=9, interpolation='nearest')

                # Add gridlines
                ax.set_xticks(np.arange(-0.5, 16, 1), minor=True)
                ax.set_yticks(np.arange(-0.5, 16, 1), minor=True)
                ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.2)

                # Remove ticks
                ax.set_xticks([])
                ax.set_yticks([])

                # Set title
                if titles and idx < len(titles):
                    ax.set_title(titles[idx], fontsize=10)
                else:
                    ax.set_title(f"Grid {idx + 1}", fontsize=10)

                idx += 1
            else:
                # Hide empty subplots
                ax.axis('off')

    plt.tight_layout()

    if show:
        plt.show()

    return fig


def save_corpus(corpus: List[Grid],
                filepath: str,
                train: Optional[List[Grid]] = None,
                val: Optional[List[Grid]] = None,
                test: Optional[List[Grid]] = None):
    """
    Save corpus to compressed .npz file.

    Args:
        corpus: List of Grid instances
        filepath: Path to save .npz file
        train: Optional training set split
        val: Optional validation set split
        test: Optional test set split
    """
    # Convert grids to numpy array
    images = np.array([g.data if isinstance(g, Grid) else g for g in corpus])

    # Prepare data dictionary
    data = {'images': images}

    # Add splits if provided
    if train is not None:
        train_array = np.array([g.data if isinstance(g, Grid) else g for g in train])
        data['train'] = train_array

    if val is not None:
        val_array = np.array([g.data if isinstance(g, Grid) else g for g in val])
        data['val'] = val_array

    if test is not None:
        test_array = np.array([g.data if isinstance(g, Grid) else g for g in test])
        data['test'] = test_array

    # Save compressed
    np.savez_compressed(filepath, **data)


def load_corpus(filepath: str) -> Dict[str, np.ndarray]:
    """
    Load corpus from .npz file.

    Args:
        filepath: Path to .npz file

    Returns:
        Dictionary containing 'images' and optionally 'train', 'val', 'test' arrays
    """
    data = np.load(filepath)

    # Convert to regular dict
    result = {}
    for key in data.files:
        result[key] = data[key]

    return result
