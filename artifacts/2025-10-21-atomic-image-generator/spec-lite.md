# Atomic Image Generator - Lite Summary

Build a procedural generator that creates 16x16 ARC-like training grids containing 1-4 non-overlapping geometric objects (blobs, rectangles, lines, and patterns) using a 9-color palette. The system generates configurable batches of synthetic training data for the encoder, starting with 10 images and scaling as needed.

## Key Points
- Generates 16x16 grids with 1-4 geometric objects per image
- Four object types: blobs (irregular shapes), rectangles, lines (horizontal/vertical), and patterns (repeating motifs)
- 9-color palette with black background and non-overlapping object placement
- Configurable batch generation starting at 10 images, scalable for training data needs
