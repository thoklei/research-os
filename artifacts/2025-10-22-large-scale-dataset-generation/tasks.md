# Artifact Tasks

These are the tasks to be completed for the artifact detailed in research-os/artifacts/2025-10-22-large-scale-dataset-generation/spec.md

> Created: 2025-10-22
> Status: Ready for Implementation

## Tasks

- [x] 1. Adapt pipeline.py for uint8 dtype and memory estimation
  - [x] 1.1 Write tests for uint8 conversion and memory estimation functions
  - [x] 1.2 Add estimate_memory() function (calculate bytes: num_images × 16 × 16 × dtype_size)
  - [x] 1.3 Add dtype parameter to generate_corpus() (default uint8, backward compatible)
  - [x] 1.4 Add show_progress parameter with tqdm integration
  - [x] 1.5 Convert grid.data to specified dtype in generate_corpus()
  - [x] 1.6 Verify backward compatibility with existing demo.py workflow
  - [x] 1.7 Verify all tests pass (23/23 passing)

- [x] 2. Enhance visualization.py with metadata support and validation
  - [x] 2.1 Write tests for metadata generation and validation functions
  - [x] 2.2 Add create_metadata() function to generate JSON with timestamp, version, parameters, and dataset stats
  - [x] 2.3 Implement save_metadata() to write JSON file alongside dataset
  - [x] 2.4 Add validate_dataset() function to check shape (N,16,16), values (0-9), dtype
  - [x] 2.5 Verify uint8 handling in save_corpus and load_corpus
  - [x] 2.6 Verify all tests pass (22/22 passing)

- [x] 3. Implement hierarchical directory structure and CLI script
  - [x] 3.1 Write tests for directory creation and CLI argument parsing
  - [x] 3.2 Create generate_dataset.py CLI script with argparse (--num-images, --output-dir, --seed)
  - [x] 3.3 Implement hierarchical directory structure creation (datasets/YYYY-MM-DD-vN/)
  - [x] 3.4 Add progress tracking with tqdm for batch generation
  - [x] 3.5 Integrate memory estimation with user confirmation before generation
  - [x] 3.6 Add error handling for invalid parameters and directory creation
  - [x] 3.7 Implement train/val/test splitting and metadata generation
  - [x] 3.8 Verify all tests pass (21/21 passing)

- [x] 4. Create visual validation tool for quality assurance
  - [x] 4.1 Write tests for sample selection and grid visualization
  - [x] 4.2 Create validate_visual.py script to load and display 100 random samples
  - [x] 4.3 Implement 10x10 grid layout with matplotlib
  - [x] 4.4 Add metadata overlay (version, timestamp, sample indices)
  - [x] 4.5 Include visual statistics (mean, std, min, max pixel values)
  - [x] 4.6 Save validation grid as PNG for documentation
  - [x] 4.7 Verify all tests pass (20/20 passing)

- [x] 5. End-to-end testing and validation at scale
  - [x] 5.1 Write integration tests for complete pipeline (1K and 100K scales) (20 tests passing)
  - [x] 5.2 Generate 1K dataset and verify size (~256 KB), metadata, and visual quality
  - [x] 5.3 Run visual validation on 1K dataset
  - [x] 5.4 Generate 100K dataset and verify size (~2.8 MB compressed), metadata, and visual quality
  - [x] 5.5 Run visual validation on 100K dataset
  - [x] 5.6 Verify memory usage stays within expected bounds during generation
  - [x] 5.7 Test CLI with various parameter combinations (different seeds, output dirs)
  - [x] 5.8 Verify all integration tests pass (20/20 passing)
