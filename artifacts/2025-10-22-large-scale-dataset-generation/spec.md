# Spec Requirements Document

> Artifact: Large-Scale Dataset Generation System
> Created: 2025-10-22
> Status: Planning

## Overview

This artifact scales the existing demo-scale dataset generation pipeline (currently working at 10 images) to production-scale datasets supporting 1,000 to 100,000 images. The system will provide memory-efficient storage using uint8 datatypes, hierarchical directory organization with versioning, and validation tools to ensure data quality before full-scale generation.

## User Stories

**As a researcher, I want to:**

1. Generate an initial validation dataset of N=1,000 images to verify the pipeline works correctly at moderate scale before committing computational resources to larger generation runs.

2. Receive a memory estimate and confirm resource allocation before generation begins, so I can ensure my system has sufficient capacity and avoid failed runs due to resource exhaustion.

3. Monitor generation progress in real-time via a progress bar, so I can estimate completion time and identify any performance bottlenecks in the pipeline.

4. Visually inspect a random sample of 100 generated instances, so I can quickly validate data quality and correctness before proceeding to larger-scale generation.

5. Scale confidently to N=100,000 images after validation, using a robust CLI interface with reproducible random seeds and organized output directories.

## Artifact Scope

This artifact includes the following features:

1. **Hierarchical Versioned Storage**: Implement Option C directory structure with timestamped versions, uint8 NPZ storage for 16x16 images (16x space savings over float64), and single metadata JSON file per version containing generation parameters and timestamps.

2. **CLI Interface**: Create command-line tool with `--num-images`, `--output-dir`, and `--seed` arguments that wraps existing pipeline.py generate_corpus() function with minimal code changes.

3. **Memory Management**: Calculate and display memory estimates before generation begins, requiring user confirmation to proceed, preventing out-of-memory failures during long-running generation tasks.

4. **Progress Monitoring**: Integrate tqdm progress bar into generation loop, displaying current image count, elapsed time, and estimated time remaining for user awareness during long runs.

5. **Validation Tools**: Implement visual inspection utility that randomly samples 100 instances from generated dataset and creates inspection visualization grid for quick quality assessment.

## Out of Scope

The following features are explicitly NOT included in this artifact:

1. **Checkpointing/Resume Capability**: No mid-generation state saving or ability to resume interrupted runs. Keep implementation simple; re-run generation if interrupted.

2. **Weights & Biases Integration**: No experiment tracking, logging, or cloud integration. Local-only implementation focused on core generation functionality.

3. **Distributed Generation**: Single-machine, single-process execution only. No multi-GPU, multi-node, or parallel generation strategies.

4. **Advanced Compression**: Use only uint8 datatype conversion. No additional compression algorithms (gzip, lz4, etc.) or optimization techniques.

5. **Interactive UI**: Command-line interface only. No web dashboard, GUI, or interactive visualization tools beyond static validation outputs.

## Expected Deliverable

The artifact is complete when the following outcomes are testable:

1. **Successful 1K Generation**: Run `python generate_cli.py --num-images 1000 --output-dir ./data --seed 42` and produce hierarchically organized dataset with metadata.json containing timestamp, version, seed, and generation parameters.

2. **Memory Estimation Accuracy**: System displays memory estimate (e.g., "Estimated memory: 2.5 GB for 1000 images") and requires user confirmation before proceeding. Estimate accuracy within 20% of actual peak memory usage.

3. **uint8 Space Savings**: Verify 16x storage reduction by comparing NPZ file sizes between original float64 implementation and new uint8 implementation for identical datasets.

4. **Progress Visibility**: During generation, tqdm displays real-time progress bar showing: `Generating: 547/1000 [01:23<01:08, 6.6img/s]` with accurate time estimates.

5. **Quality Validation**: Run validation tool on 1K dataset, generating inspection grid of 100 random samples. Visual review confirms correct rendering, appropriate variety, and absence of artifacts or errors.

6. **100K Scale Test**: After 1K validation passes, successfully generate N=100,000 dataset without memory errors or crashes, completing in reasonable time (under 6 hours on standard hardware).

## Artifact Documentation

- Tasks: @research-os/artifacts/2025-10-22-large-scale-dataset-generation/tasks.md
- Technical Specification: @research-os/artifacts/2025-10-22-large-scale-dataset-generation/sub-specs/technical-spec.md
