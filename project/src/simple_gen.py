import numpy as np
import matplotlib.pyplot as plt

def generate_arc_image(grid_size=(10, 10), num_objects=None, 
                        color_palette=range(1, 10)):
    """
    Generate atomic ARC-like images with object-based structure.
    
    Args:
        grid_size: (height, width) of the grid
        num_objects: Number of objects to generate (None = random 1-4)
        color_palette: Available colors (0 reserved for background)
    """

    # Initialize with background (color 0)
    grid = np.zeros(grid_size, dtype=int)

    # Decide number of objects
    if num_objects is None:
        num_objects = np.random.randint(1, 5)

    objects_placed = 0
    max_attempts = 100

    for attempt in range(max_attempts):
        if objects_placed >= num_objects:
            break

        # 1. Choose object type with probabilities
        object_type = np.random.choice([
            'blob',      # Random growth (40%)
            'rectangle', # Filled rectangle (20%)
            'line',      # Straight line (20%)
            'pattern'    # Small repeated pattern (20%)
        ], p=[0.4, 0.2, 0.2, 0.2])

        # 2. Generate object
        obj = generate_object(object_type, grid_size)

        # 3. Choose random placement
        if can_place_object(grid, obj):
            color = np.random.choice(color_palette)
            place_object(grid, obj, color)
            objects_placed += 1

    return grid


def generate_object(object_type, max_size=(10, 10)):
    """Generate a single object as a set of relative coordinates."""

    if object_type == 'blob':
        return generate_blob(max_size)
    elif object_type == 'rectangle':
        return generate_rectangle(max_size)
    elif object_type == 'line':
        return generate_line(max_size)
    elif object_type == 'pattern':
        return generate_pattern(max_size)


def generate_blob(max_size, min_pixels=2, max_pixels=15):
    """
    Grow a connected blob using random walk with connectivity bias.
    """
    # Start with seed pixel
    pixels = {(0, 0)}
    target_size = np.random.randint(min_pixels, max_pixels + 1)

    # 8-neighborhood
    neighbors = [(-1,-1), (-1,0), (-1,1),
                (0,-1),          (0,1),
                (1,-1),  (1,0),  (1,1)]

    # Grow blob
    for _ in range(target_size - 1):
        # Get all possible growth points
        candidates = []
        for pixel in pixels:
            for dy, dx in neighbors:
                new_pixel = (pixel[0] + dy, pixel[1] + dx)
                if new_pixel not in pixels:
                    # Bias towards pixels with more neighbors (creates cohesion)
                    neighbor_count = sum(
                        1 for ndy, ndx in neighbors
                        if (new_pixel[0] + ndy, new_pixel[1] + ndx) in pixels
                    )
                    # Add multiple times based on connectivity
                    candidates.extend([new_pixel] * (neighbor_count + 1))

        if not candidates:
            break

        # Choose weighted by connectivity
        new_pixel = candidates[np.random.randint(len(candidates))]
        pixels.add(new_pixel)

        # Check size constraints
        # if get_bounding_box_size(pixels) > max_size:
        #     pixels.remove(new_pixel)
        #     break

    return pixels #normalize_object(pixels)


def generate_rectangle(max_size):
    """Generate a filled rectangle."""
    h = np.random.randint(1, min(6, max_size[0]))
    w = np.random.randint(1, min(6, max_size[1]))

    pixels = {(y, x) for y in range(h) for x in range(w)}
    return pixels


def generate_line(max_size):
    """Generate a straight line (horizontal, vertical, or diagonal)."""
    direction = np.random.choice(['horizontal', 'vertical', 'diagonal'])
    length = np.random.randint(2, min(8, max(max_size)))

    if direction == 'horizontal':
        pixels = {(0, x) for x in range(length)}
    elif direction == 'vertical':
        pixels = {(y, 0) for y in range(length)}
    else:  # diagonal
        pixels = {(i, i) for i in range(length)}

    return pixels


def generate_pattern(max_size):
    """Generate a small repeated pattern (2x2 or 3x3)."""
    # Define some basic patterns
    patterns = [
        {(0,0), (0,1), (1,0)},  # L-shape
        {(0,0), (0,1), (1,1)},  # Corner
        {(0,0), (1,1)},         # Diagonal
        {(0,1), (1,0), (1,1), (1,2)},  # T-shape
    ]

    return np.random.choice(patterns)


def can_place_object(grid, obj, min_spacing=1):
    """Check if object can be placed with minimum spacing."""
    h, w = grid.shape

    # Try random positions
    for _ in range(20):
        y_offset = np.random.randint(0, h - get_height(obj) + 1)
        x_offset = np.random.randint(0, w - get_width(obj) + 1)

        # Check if all pixels are free (with spacing)
        collision = False
        for dy, dx in obj:
            for spacing_dy in range(-min_spacing, min_spacing + 1):
                for spacing_dx in range(-min_spacing, min_spacing + 1):
                    check_y = y_offset + dy + spacing_dy
                    check_x = x_offset + dx + spacing_dx

                    if (0 <= check_y < h and 0 <= check_x < w and
                        grid[check_y, check_x] != 0):
                        collision = True
                        break
                if collision:
                    break
            if collision:
                break

        if not collision:
            return (y_offset, x_offset)

    return None


def place_object(grid, obj, color, position):
    """Place object on grid at position with given color."""
    y_offset, x_offset = position
    for dy, dx in obj:
        grid[y_offset + dy, x_offset + dx] = color


if __name__ == '__main__':
    img = generate_arc_image(grid_size=(10, 10), num_objects=None, 
                        color_palette=range(1, 10))