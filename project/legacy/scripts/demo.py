"""
Demo script for Atomic Image Generator

This script demonstrates the complete pipeline:
1. Generate a corpus of 10 atomic images
2. Split into train/val/test sets
3. Visualize sample images
4. Save to compressed .npz file
5. Load and verify
"""

import numpy as np
from pipeline import generate_corpus, split_corpus
from visualization import visualize_gallery, save_corpus, load_corpus

def main():
    print("=" * 60)
    print("Atomic Image Generator - Demo")
    print("=" * 60)

    # Set seed for reproducibility
    np.random.seed(42)

    # 1. Generate corpus
    print("\n1. Generating corpus of 10 images...")
    corpus = generate_corpus(corpus_size=10)
    print(f"   ✓ Generated {len(corpus)} images")

    # 2. Split corpus
    print("\n2. Splitting corpus (80/10/10)...")
    train, val, test = split_corpus(corpus)
    print(f"   ✓ Train: {len(train)} images")
    print(f"   ✓ Val:   {len(val)} images")
    print(f"   ✓ Test:  {len(test)} images")

    # 3. Visualize sample images
    print("\n3. Visualizing first 6 images...")
    fig = visualize_gallery(corpus[:6], rows=2, cols=3, show=False)
    print("   ✓ Gallery created (not displayed in script mode)")

    # 4. Save corpus
    print("\n4. Saving corpus to 'atomic_corpus.npz'...")
    save_corpus(corpus, 'atomic_corpus.npz', train=train, val=val, test=test)
    print("   ✓ Corpus saved with compression")

    # 5. Load and verify
    print("\n5. Loading and verifying corpus...")
    loaded = load_corpus('atomic_corpus.npz')
    print(f"   ✓ Loaded corpus with keys: {list(loaded.keys())}")
    print(f"   ✓ Images shape: {loaded['images'].shape}")
    print(f"   ✓ Train shape:  {loaded['train'].shape}")
    print(f"   ✓ Val shape:    {loaded['val'].shape}")
    print(f"   ✓ Test shape:   {loaded['test'].shape}")

    # Verify data integrity
    print("\n6. Verifying data integrity...")
    for i in range(len(corpus)):
        assert np.array_equal(loaded['images'][i], corpus[i].data)
    print("   ✓ All images match original corpus")

    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)

    # Print sample statistics
    print("\nSample Statistics:")
    print(f"- Grid size: 16x16")
    print(f"- Color palette: 0-9 (0=background, 1-9=objects)")
    print(f"- Objects per image: 1-4")
    print(f"- Object types: Blobs (40%), Rectangles (20%), Lines (20%), Patterns (20%)")
    print(f"- Object size constraint: 2-15 pixels")

    # Analyze first image
    first_image = corpus[0].data
    unique_colors = set(first_image.flatten()) - {0}
    num_objects = len(unique_colors)
    num_pixels = np.sum(first_image != 0)

    print(f"\nFirst Image Analysis:")
    print(f"- Number of objects (unique colors): {num_objects}")
    print(f"- Total colored pixels: {num_pixels}")
    print(f"- Colors used: {sorted(unique_colors)}")

if __name__ == "__main__":
    main()
