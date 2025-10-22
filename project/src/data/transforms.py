"""
Data Augmentation Transforms for ARC Grids - Task 2.2

Implements geometric transformations for 16×16 ARC grids:
- Random rotations (0°, 90°, 180°, 270°)
- Random horizontal flip
- Random vertical flip

All transforms preserve integer labels [0-9].
"""

import torch
import numpy as np
from typing import Callable, List, Optional
import random


class RandomRotation:
    """
    Randomly rotate grid by 0°, 90°, 180°, or 270°.

    Args:
        angles (list): List of rotation angles in degrees (default: [0, 90, 180, 270])
        p (float): Probability of applying rotation (default: 1.0)
    """

    def __init__(self, angles: List[int] = [0, 90, 180, 270], p: float = 1.0):
        self.angles = angles
        self.p = p

        # Map angles to number of 90° rotations
        self.k_map = {
            0: 0,
            90: 1,
            180: 2,
            270: 3,
        }

    def __call__(self, grid: torch.Tensor) -> torch.Tensor:
        """
        Apply random rotation to grid.

        Args:
            grid (torch.Tensor): Input grid with shape (16, 16)

        Returns:
            torch.Tensor: Rotated grid with shape (16, 16)
        """
        if random.random() > self.p:
            return grid

        # Select random angle
        angle = random.choice(self.angles)
        k = self.k_map[angle]

        if k == 0:
            return grid

        # Rotate using torch.rot90
        # k=1: 90° counter-clockwise
        # k=2: 180°
        # k=3: 270° counter-clockwise (= 90° clockwise)
        return torch.rot90(grid, k=k, dims=(0, 1))

    def __repr__(self):
        return f"RandomRotation(angles={self.angles}, p={self.p})"


class RandomHorizontalFlip:
    """
    Randomly flip grid horizontally.

    Args:
        p (float): Probability of applying flip (default: 0.5)
    """

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, grid: torch.Tensor) -> torch.Tensor:
        """
        Apply random horizontal flip to grid.

        Args:
            grid (torch.Tensor): Input grid with shape (16, 16)

        Returns:
            torch.Tensor: Flipped grid with shape (16, 16)
        """
        if random.random() < self.p:
            return torch.flip(grid, dims=(1,))  # Flip along width
        return grid

    def __repr__(self):
        return f"RandomHorizontalFlip(p={self.p})"


class RandomVerticalFlip:
    """
    Randomly flip grid vertically.

    Args:
        p (float): Probability of applying flip (default: 0.5)
    """

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, grid: torch.Tensor) -> torch.Tensor:
        """
        Apply random vertical flip to grid.

        Args:
            grid (torch.Tensor): Input grid with shape (16, 16)

        Returns:
            torch.Tensor: Flipped grid with shape (16, 16)
        """
        if random.random() < self.p:
            return torch.flip(grid, dims=(0,))  # Flip along height
        return grid

    def __repr__(self):
        return f"RandomVerticalFlip(p={self.p})"


class Compose:
    """
    Compose multiple transforms together.

    Args:
        transforms (list): List of transform callables
    """

    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, grid: torch.Tensor) -> torch.Tensor:
        """
        Apply all transforms in sequence.

        Args:
            grid (torch.Tensor): Input grid with shape (16, 16)

        Returns:
            torch.Tensor: Transformed grid with shape (16, 16)
        """
        for t in self.transforms:
            grid = t(grid)
        return grid

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string


class ToTensor:
    """
    Convert numpy array to torch tensor.

    Note: This is primarily for compatibility. ARCDataset already returns tensors.
    """

    def __call__(self, grid: np.ndarray) -> torch.Tensor:
        """
        Convert numpy array to tensor.

        Args:
            grid (np.ndarray): Input grid with shape (16, 16)

        Returns:
            torch.Tensor: Grid tensor with shape (16, 16)
        """
        if isinstance(grid, torch.Tensor):
            return grid
        return torch.from_numpy(grid).long()

    def __repr__(self):
        return "ToTensor()"


class Identity:
    """
    Identity transform (no-op). Useful for validation/test sets.
    """

    def __call__(self, grid: torch.Tensor) -> torch.Tensor:
        """Return grid unchanged."""
        return grid

    def __repr__(self):
        return "Identity()"


def get_train_transforms() -> Compose:
    """
    Get standard training transforms with data augmentation.

    Returns:
        Compose: Composed transform with rotations and flips
    """
    return Compose([
        RandomRotation(angles=[0, 90, 180, 270], p=1.0),
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
    ])


def get_val_transforms() -> Identity:
    """
    Get validation transforms (identity - no augmentation).

    Returns:
        Identity: No-op transform
    """
    return Identity()


def get_test_transforms() -> Identity:
    """
    Get test transforms (identity - no augmentation).

    Returns:
        Identity: No-op transform
    """
    return Identity()


# Convenience function for creating custom augmentation pipelines
def create_augmentation_pipeline(
    rotation: bool = True,
    rotation_angles: List[int] = [0, 90, 180, 270],
    horizontal_flip: bool = True,
    horizontal_flip_p: float = 0.5,
    vertical_flip: bool = True,
    vertical_flip_p: float = 0.5,
) -> Compose:
    """
    Create custom augmentation pipeline.

    Args:
        rotation: Whether to include rotation
        rotation_angles: List of rotation angles
        horizontal_flip: Whether to include horizontal flip
        horizontal_flip_p: Probability of horizontal flip
        vertical_flip: Whether to include vertical flip
        vertical_flip_p: Probability of vertical flip

    Returns:
        Compose: Composed transform pipeline
    """
    transforms = []

    if rotation:
        transforms.append(RandomRotation(angles=rotation_angles, p=1.0))

    if horizontal_flip:
        transforms.append(RandomHorizontalFlip(p=horizontal_flip_p))

    if vertical_flip:
        transforms.append(RandomVerticalFlip(p=vertical_flip_p))

    if not transforms:
        transforms.append(Identity())

    return Compose(transforms)
