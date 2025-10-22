# Spec Requirements Document

> Artifact: Atomic Image Generator
> Created: 2025-10-21
> Status: Planning

## Overview

Build a procedural generator that creates individual ARC-like grids containing 1-4 distinct geometric objects to serve as training data for an encoder model. The generator will produce 16x16 grids with random blobs, filled rectangles, straight lines, and small patterns drawn from a 9-color palette, enforcing object size constraints and non-overlapping placement.

## User Stories

**Story 1: Generate Training Corpus**
- As a machine learning researcher, I want to generate a configurable corpus of synthetic ARC-like images (starting with 10, scalable to thousands) so that I can train an encoder to learn meaningful representations of atomic geometric primitives.
- Problem Solved: Eliminates manual dataset creation and provides unlimited scalable training data with controlled characteristics.

**Story 2: Visualize Generated Images**
- As a developer, I want to visualize batches of generated images using matplotlib so that I can verify the quality, diversity, and correctness of the procedural generation before using the data for training.
- Problem Solved: Enables rapid quality assurance and debugging of the generation algorithm without needing to inspect raw numpy arrays.

**Story 3: Control Object Characteristics**
- As a researcher, I want each generated image to contain 1-4 non-overlapping objects where each object is between 2-15 pixels in size so that the training data matches the atomic complexity level found in real ARC tasks.
- Problem Solved: Ensures generated data has appropriate complexity - not too simple (single pixels) and not too complex (large composite scenes).

## Artifact Scope

1. **Procedural Grid Generation**: Generate 16x16 grids with 1-4 randomly placed objects, where each object is drawn from four types (random blobs 40%, filled rectangles 20%, straight lines 20%, small patterns 20%) and sized between 2-15 pixels.

2. **Color Palette System**: Implement 9-color palette {1-9} with uniform random sampling for object colors, ensuring each object is a single cohesive color against background (color 0).

3. **Non-Overlapping Placement Engine**: Enforce hard constraint that objects do not overlap by tracking occupied cells and rejecting placements that would cause collisions, with retry logic or object count reduction if placement fails.

4. **Configurable Corpus Generation**: Expose corpus size as a configurable parameter (default 10 images for initial testing) with ability to scale to thousands of images for production training runs.

5. **Visualization Tooling**: Provide matplotlib-based visualization functions to display individual images or grids of multiple generated samples for quality inspection and debugging.

## Out of Scope

- Object spacing requirements (objects can be adjacent, just not overlapping)
- Advanced object types beyond the four specified primitives (blobs, rectangles, lines, patterns)
- Grid sizes other than 16x16 (keeping implementation simple for initial version)
- Statistical testing of color distribution (uniform sampling assumed to be sufficient)
- Object relationship constraints (symmetry, alignment, grouping)
- Compression or storage optimization for the generated corpus
- Training loop integration or encoder architecture (this artifact only generates data)

## Expected Deliverable

1. **Python Script Execution**: Run a Python script that generates a configurable number of 16x16 numpy arrays (default 10) and saves them to disk, completing in under 10 seconds for the initial corpus size.

2. **Visual Quality Verification**: Execute a visualization script that displays a 4x4 or 5x5 grid of generated images in matplotlib, where visual inspection confirms: (a) 1-4 distinct objects per image, (b) objects are cohesive connected components, (c) no overlapping objects, (d) images resemble atomic ARC task elements.

3. **Constraint Validation**: Run a validation function that programmatically checks a generated corpus and confirms 100% compliance with constraints: all objects are 2-15 pixels, all grids are 16x16, all colors are in range {0-9}, and no object pixels overlap.

## Artifact Documentation

- Tasks: @research-os/artifacts/2025-10-21-atomic-image-generator/tasks.md
- Technical Specification: @research-os/artifacts/2025-10-21-atomic-image-generator/sub-specs/technical-spec.md
