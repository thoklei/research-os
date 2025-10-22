"""
Visualization script to display generated atomic images
"""

import numpy as np
from pipeline import generate_corpus
from visualization import visualize_gallery

def main():
    print("Generating 12 atomic images...")

    # Set seed for reproducibility
    np.random.seed(42)

    # Generate corpus
    corpus = generate_corpus(corpus_size=12)

    print(f"Generated {len(corpus)} images")
    print("Displaying gallery...")

    # Create gallery with 3 rows and 4 columns
    fig = visualize_gallery(
        corpus,
        rows=3,
        cols=4,
        titles=[f"Image {i+1}" for i in range(12)],
        show=True,
        figsize=(16, 12)
    )

if __name__ == "__main__":
    main()
