# Technical Stack Documentation

## Overview

This document provides comprehensive technical specifications for implementing the controllable visual reasoning task generator. The system generates ARC-like tasks through compositional latent transformations with mathematically interpretable difficulty parameters.

---

## 1. Development Frameworks

### 1.1 Core Machine Learning Framework

**Primary Framework: PyTorch 2.0+**
- **Rationale**: Industry-standard for research, excellent autograd support for custom encoders, strong ecosystem for slot attention implementations
- **Key Libraries**:
  - `torch.nn` for neural encoder architecture
  - `torch.optim` for encoder training (Adam optimizer with learning rate 3e-4)
  - `torchvision.transforms` for data augmentation during encoder training
  - `torch.utils.data.DataLoader` for efficient batch processing

**Alternative Framework (if needed): JAX + Flax**
- **Use Case**: If differentiable parametric rendering required
- **Advantage**: Better support for functional programming and custom gradient definitions
- **Trade-off**: Smaller ecosystem, steeper learning curve

### 1.2 Encoder / Decoder
- currently an open question how we are going to go from image to latent

### 1.3 Visualization and Analysis Tools

**Grid Visualization**:
- **matplotlib 3.7+**: Primary plotting library for grid rendering
  - Custom colormap for 10 ARC colors (black=0, blue=1, red=2, green=3, yellow=4, grey=5, pink=6, orange=7, teal=8, brown=9)
  - Grid display with `imshow()` using nearest-neighbor interpolation
  - Figure size: 5x5 inches per grid for publication quality

**Interactive Visualization**:
- **plotly 5.0+**: Interactive task browser for exploring generated dataset
- **streamlit 1.25+**: Web interface for human study task presentation
  - Real-time difficulty estimation display
  - Input interface for participant responses
  - Timer functionality for solve time measurement

**Latent Space Analysis**:
- **scikit-learn 1.3+**:
  - `sklearn.manifold.TSNE` for 2D latent space visualization
  - `sklearn.decomposition.PCA` for dimensionality analysis
  - `sklearn.metrics` for clustering and disentanglement metrics
- **seaborn 0.12+**: Statistical visualization for correlation heatmaps, distribution plots

### 1.4 Experiment Tracking and Reproducibility

**Weights & Biases (wandb) 0.15+**:
- Track encoder training curves (reconstruction loss, accuracy)
- Log generated task samples during development
- Compare ablation study results
- Store hyperparameters and model checkpoints
- **Configuration**:
  - Project name: `arc-controllable-generation`
  - Logging frequency: Every 10 training steps
  - Artifact storage: Encoder checkpoints every 10 epochs

**Alternative: MLflow 2.5+**:
- Self-hosted option if cloud logging not preferred
- Similar tracking capabilities with local backend

**Version Control**:
- **git-lfs**: Track large binary files (encoder weights, generated task datasets)
- **DVC (Data Version Control) 3.0+**: Track dataset versions and experiment outputs
  - Remote storage: AWS S3 or Google Cloud Storage
  - Track: Generated tasks, human study data, model predictions

### 1.5 Human Study Infrastructure

**Prolific or Amazon Mechanical Turk**:
- Participant recruitment platform
- Target: 50-150 participants across phases
- Compensation: $15/hour (approximately $2.50 per 10-minute task session)
- Quality filters: English proficiency, 95%+ approval rate, age 18+

**Study Platform: Custom Streamlit Application**:
- **Features**:
  - Task presentation with 2-5 input-output examples displayed
  - Drawing interface for participant responses (16x16 grid selector)
  - Built-in timer for solve time measurement
  - Likert scale (1-7) for difficulty rating
  - Session management and data logging
- **Deployment**: Hosted on Heroku or AWS EC2
- **Database**: PostgreSQL for storing participant responses, timestamps, ratings

**Alternative Platform: jsPsych 7.0+**:
- JavaScript-based experiment framework
- Better for precise timing measurements
- Deployable to any web hosting service

---

## 2. Datasets

### 2.1 Training Data for Encoder

**Synthetic Grid Dataset**:
- **Size**: 10,000 grids (8,000 training, 1,000 validation, 1,000 test)
- **Generation Method**: Hand-crafted synthetic grids with simple objects
  - Shapes: Circles, squares, triangles, lines, crosses (5 types)
  - Colors: All 10 ARC colors uniformly sampled
  - Objects per grid: 1-6 (uniform distribution)
  - Position: Random (x, y) coordinates with no overlap
  - Size: 1-5 pixels (radius or half-width)
- **Format**:
  - NumPy arrays: `(N, 16, 16)` with integer values 0-9
  - Metadata JSON: Object positions, shapes, colors for each grid
- **Storage**: ~40 MB compressed (gzip), ~200 MB uncompressed
- **Purpose**: Train slot attention encoder to achieve ≥90% reconstruction accuracy

**Data Augmentation** (during encoder training):
- Random horizontal/vertical flips (50% probability)
- Random 90° rotations (25% probability each direction)
- Color permutation (randomly shuffle color assignments)
- Note: No continuous transformations (rotation, scaling) since outputs are discrete

### 2.2 Generated Task Benchmark

**Primary Dataset: Controllable ARC-like Tasks**:
- **Size**: 3,000 tasks total
  - Training: 2,000 tasks
  - Validation: 500 tasks
  - Test: 500 tasks (solutions held-out for evaluation)
- **Stratification**:
  - Composition depth `n`: 1, 2, 3, 4, 5 (5 levels)
  - Object count `k`: 1, 2, 3, 4, 5, 6 (6 levels)
  - 100 tasks per (n, k) combination for training
  - 16-17 tasks per combination for validation/test
- **Task Format** (matches ARC-AGI):
  - 2-5 input-output example pairs per task
  - Input grids: 16x16, 10 colors (integer values 0-9)
  - Output grids: 16x16, 10 colors (result of applying transformation F)
  - Test input: Single grid requiring participant/model to predict output
- **Metadata per Task**:
  - Difficulty parameters: `(n, k)`
  - Latent representations: Input and output slots (k × d arrays)
  - Composition sequence: List of sub-functions applied `[f_1, f_2, ..., f_n]`
  - Function parameters: Arguments for each sub-function (e.g., `rotate(90)`, `translate(2, -1)`)
  - Ground truth output grid
- **Storage**:
  - Grids: ~150 MB (gzip compressed NumPy)
  - Metadata: ~50 MB (JSON)
  - Total: ~200 MB compressed
- **Format**: HDF5 file with hierarchical structure:
  ```
  tasks.h5
  ├── train/
  │   ├── task_0000/
  │   │   ├── examples (N_examples × 2 × 16 × 16)
  │   │   ├── test_input (16 × 16)
  │   │   ├── test_output (16 × 16)
  │   │   └── metadata (JSON string)
  │   ├── task_0001/
  │   └── ...
  ├── validation/
  └── test/
  ```

### 2.3 Reference Dataset

**Original ARC-AGI Dataset**:
- **Source**: [https://github.com/fchollet/ARC-AGI](https://github.com/fchollet/ARC-AGI)
- **Usage**:
  - Diversity comparison baseline (400 public tasks)
  - Visual resemblance validation
  - Structural analysis reference
- **Not Used For**: Training (our approach generates tasks from scratch)
- **Subset**: 400 public tasks (800 hidden for official leaderboard, not needed)
- **Storage**: ~5 MB (JSON format)

**ConceptARC Dataset** (optional):
- **Source**: Simplified ARC variant
- **Usage**: Additional comparison point for structural diversity metrics
- **Size**: ~100 tasks

### 2.4 Human Study Data

**Collected Data**:
- **Experiment 2.1 (Composition Depth)**:
  - 50 participants × 50 tasks = 2,500 response records
  - Fields: participant_id, task_id, response_grid, solve_time, accuracy, difficulty_rating (1-7)
- **Experiment 2.2 (Object Count)**:
  - 50 participants × 60 tasks = 3,000 response records
  - Same fields as above
- **Experiment 5.1 (Large-Scale Validation)**:
  - 150 participants × 200 tasks = 30,000 response records (sampled, not all combinations)
- **Total Storage**: ~100 MB (CSV + grid arrays)
- **Privacy**: Anonymized participant IDs, no personal information collected
- **Backup**: Encrypted backup on institutional secure storage

### 2.5 AI Solver Predictions

**Baseline CNN Predictions**:
- 500 test tasks × model predictions = 500 grid arrays
- Storage: ~2 MB per model
- Metadata: Prediction confidence, inference time per task

**External Solver Predictions** (if reproduced):
- ViTARC predictions: 500 grids
- Neural CA predictions: 500 grids
- Storage: ~2 MB per solver

---

## 3. Evaluation Metrics

### 3.1 Task Difficulty Metrics

**Composition Depth `n`**:
- **Definition**: Number of sequential sub-function compositions in transformation F
- **Formula**: For `F = f_n ∘ f_{n-1} ∘ ... ∘ f_1`, depth = n
- **Range**: {1, 2, 3, 4, 5} (discrete levels)
- **Hypothesis**: Higher n → higher human/AI difficulty

**Object Count `k`**:
- **Definition**: Number of distinct objects (slots) in input latent representation
- **Range**: {1, 2, 3, 4, 5, 6} (discrete levels)
- **Hypothesis**: Higher k → higher difficulty due to visual complexity

**Combined Difficulty Score**:
- **Formula**: `D = α·n + β·k` where α, β are learned weights
- **Calibration**: Fit via linear regression on human solve rates
- **Expected**: α > β (composition depth dominates object count)
- **Range**: Continuous, normalized to [0, 1] for standardization

### 3.2 Human Performance Metrics

**Solve Accuracy**:
- **Definition**: Proportion of tasks where participant's output grid exactly matches ground truth
- **Formula**: `Accuracy = (Exact matches) / (Total tasks attempted)`
- **Range**: [0, 1] (higher is better)
- **Expected Results**:
  - n=1: 94% accuracy
  - n=3: ~70% accuracy
  - n=5: 41% accuracy

**Solve Time**:
- **Definition**: Time from task presentation to response submission (seconds)
- **Measurement**: JavaScript `performance.now()` for precise timing
- **Exclusion Criteria**: Times >300s (5 minutes) treated as timeouts
- **Expected Results**:
  - n=1: 12.3s mean (SD ~5s)
  - n=3: ~40s mean (SD ~20s)
  - n=5: 67.8s mean (SD ~35s)
- **Analysis**: Median and IQR reported (robust to outliers)

**Difficulty Rating**:
- **Definition**: Subjective difficulty rating by participant
- **Scale**: 1 (very easy) to 7 (very hard), Likert scale
- **Collection**: After each task attempt
- **Expected Correlation**: Spearman ρ ≥ 0.75 with composition depth

**Difficulty Correlation (Spearman ρ)**:
- **Definition**: Spearman rank correlation between difficulty parameters (n, k) and human metrics (solve time, accuracy, ratings)
- **Formula**: Standard Spearman correlation coefficient
- **Expected**: ρ ≥ 0.70 for composition depth, ρ ≥ 0.50 for object count
- **Significance**: p < 0.01 threshold (Bonferroni corrected for multiple comparisons)

**Inter-Cohort Reliability**:
- **Definition**: Consistency of solve rates across independent participant cohorts
- **Formula**: Pearson correlation between cohort-averaged solve rates
- **Expected**: r ≥ 0.90 (high reliability)
- **Purpose**: Validate that difficulty patterns are not cohort-specific

### 3.3 AI Solver Performance Metrics

**Exact Match Accuracy**:
- **Definition**: Proportion of test tasks where AI solver output exactly matches ground truth grid
- **Formula**: `Accuracy = (Exact matches) / (Total test tasks)`
- **Range**: [0, 1]
- **Expected Results**:
  - ViTARC: 87% (n=1) → 23% (n=5)
  - Product-of-Experts: 72% (n=1) → 18% (n=5)
  - Baseline CNN: Lower overall, similar degradation pattern

**Difficulty Sensitivity**:
- **Definition**: Rate of accuracy degradation as difficulty increases
- **Formula**: Slope of linear regression `Accuracy ~ β₀ + β₁·n`
- **Interpretation**: More negative β₁ indicates stronger difficulty sensitivity
- **Expected**: AI solvers more sensitive than humans (steeper negative slope)

**Variance Explained (R²)**:
- **Definition**: Proportion of variance in solver accuracy explained by difficulty parameters
- **Formula**: R² from regression `Accuracy ~ β₀ + β₁·n + β₂·k + ε`
- **Expected Results** (from mission):
  - Composition depth: 31% variance explained
  - Object count: 18% variance explained
- **Interpretation**: Validates that difficulty parameters meaningfully predict failure modes

**Inference Time**:
- **Definition**: Time for AI solver to generate output for single task (seconds)
- **Measurement**: Wall-clock time, averaged over 5 runs
- **Purpose**: Assess computational efficiency, identify slow solvers
- **Expected**: <1s per task for CNN baseline, variable for SOTA solvers

### 3.4 Encoder Quality Metrics

**Reconstruction Accuracy**:
- **Definition**: Proportion of pixels correctly reconstructed by encoder φ
- **Formula**: `Acc = (Correct pixels) / (16 × 16 × Total grids)`
- **Range**: [0, 1] (higher is better)
- **Expected**: ≥96.4% overall, ≥90% on simple tasks (n ≤ 3, k ≤ 4)
- **Failure Analysis**: Accuracy stratified by composition depth and object count

**Object-Level Accuracy**:
- **Definition**: Proportion of objects correctly reconstructed (position, shape, color all match)
- **Formula**: `Obj_Acc = (Correct objects) / (Total objects across grids)`
- **More Stringent**: Single pixel error fails entire object
- **Expected**: ≥85% on simple tasks

**Slot Disentanglement (Mutual Information)**:
- **Definition**: Degree to which individual slot dimensions encode interpretable features
- **Formula**: `MI(z_i, property_j) = ∑ P(z_i, property_j) log [P(z_i, property_j) / (P(z_i)P(property_j))]`
  - `z_i`: Slot dimension i
  - `property_j`: Ground-truth object property (e.g., x-position, color)
- **Computation**: Discretize continuous dimensions into bins, compute MI for each (z_i, property) pair
- **Expected**: High MI (>2.0 bits) between slot dimensions and object properties
- **Purity Metric**: For each property, identify highest-MI dimension and compute purity = MI(best_z, property) / Entropy(property)
- **Target**: ≥70% purity for position, shape, color

**Failure Mode Classification**:
- **Categories**:
  1. **Color Errors**: Correct shape/position, wrong color
  2. **Position Errors**: Correct shape/color, wrong position (>2 pixels off)
  3. **Shape Errors**: Correct position/color, wrong shape
  4. **Missing Objects**: Slot not rendered or invisible
  5. **Spurious Objects**: Extra objects not in ground truth
- **Quantification**: Proportion of failures in each category
- **Purpose**: Identify systematic weaknesses in encoder design

### 3.5 Diversity Metrics

**Function Composition Diversity**:
- **Definition**: Number of unique composition sequences across generated tasks
- **Computation**: Count distinct ordered sequences `[f_1, f_2, ..., f_n]` up to depth n=4
- **Expected**: >100,000 unique patterns (combinatorial explosion: 15^4 = 50,625 sequences, more with function parameters)
- **Metric**: Coverage = (Unique sequences observed) / (Total tasks)

**Structural Diversity (Graph Edit Distance)**:
- **Definition**: Average pairwise graph edit distance between task structures
- **Task Graph Representation**:
  - Nodes: Objects (slots) in initial latent state
  - Edges: Transformations between objects (e.g., "connect", "union")
  - Node attributes: Shape, color
  - Edge attributes: Transformation type
- **Graph Edit Distance (GED)**: Minimum number of edit operations (node/edge insertion, deletion, substitution) to transform one graph into another
- **Computation**: Use approximate GED algorithm (exact computation NP-hard)
  - Library: `networkx.algorithms.similarity.graph_edit_distance()`
  - Sample: Compute GED for 1,000 random task pairs, report mean
- **Baseline Comparison**: Compute same metric for 400 public ARC tasks
- **Expected**: 2.3x higher GED for our generated tasks vs. ARC
- **Formula**: `Diversity_Score = Mean_GED(Generated) / Mean_GED(ARC_Public)`

**Visual Diversity (Perceptual Hash Distance)**:
- **Definition**: Average pairwise Hamming distance between perceptual hashes of task grids
- **Computation**:
  1. Compute perceptual hash (pHash) for each input grid using `imagehash` library
  2. pHash: 8×8 DCT-based hash, robust to minor variations
  3. Hamming distance: Number of differing bits between two hashes
- **Formula**: `Visual_Diversity = Mean(Hamming_Distance(pHash_i, pHash_j))` for all pairs (i, j)
- **Range**: [0, 64] (64-bit hash)
- **Purpose**: Ensure visual appearance varies sufficiently across tasks

**Color Entropy**:
- **Definition**: Shannon entropy of color distribution across dataset
- **Formula**: `H(Color) = -∑_{c=0}^9 P(color=c) log_2 P(color=c)`
  - P(color=c): Proportion of all pixels with color c
- **Range**: [0, log_2(10)] = [0, 3.32] bits
- **Expected**: >2.5 bits (indicates most colors used frequently)
- **Purpose**: Verify no color bias in generation

**Shape Distribution (Chi-Squared Test)**:
- **Definition**: Statistical test of uniform shape distribution
- **Null Hypothesis**: All 5 shape types (circle, square, triangle, line, cross) appear equally often
- **Formula**: `χ² = ∑ (Observed - Expected)² / Expected`
- **Degrees of Freedom**: 4 (5 shapes - 1)
- **Significance**: Reject null if p < 0.05 (indicates non-uniform distribution)
- **Expected**: Fail to reject null (approximately uniform distribution)

### 3.6 Curriculum Learning Metrics

**Final Test Accuracy Improvement**:
- **Definition**: Difference in test accuracy between easy-to-hard curriculum and random curriculum
- **Formula**: `Δ_Acc = Acc_Easy2Hard - Acc_Random`
- **Expected**: ≥5% improvement (e.g., 68% vs 63%)
- **Significance**: t-test comparing final epoch accuracies across 5 random seeds

**Generalization to Hard Tasks**:
- **Definition**: Accuracy on hardest task subset (n=5, k=6) after curriculum training
- **Comparison**: Easy-to-hard vs. random curriculum
- **Expected**: Curriculum shows better generalization to difficult tasks
- **Metric**: Accuracy on hard subset, stratified by training curriculum

**Learning Efficiency (AUC of Learning Curve)**:
- **Definition**: Area under the curve (AUC) of validation accuracy vs. training epoch
- **Formula**: `AUC = ∫_{epoch=0}^{100} Validation_Accuracy(epoch) d(epoch)` (trapezoidal integration)
- **Interpretation**: Higher AUC indicates faster learning
- **Expected**: Easy-to-hard curriculum achieves higher AUC than random/hard-to-easy

---

## 4. Computational Requirements

### 4.1 Hardware Requirements

**Encoder Training**:
- **GPU**: NVIDIA GPU with ≥16 GB VRAM (e.g., V100, A100, RTX 4090)
  - Rationale: Slot attention with batch size 64 requires ~12 GB
  - Fallback: Reduce batch size to 32 for GPUs with 8-12 GB (RTX 3080, RTX 4080)
- **CPU**: 8+ cores for parallel data loading
- **RAM**: 32 GB system memory
- **Storage**: 500 GB SSD for datasets, checkpoints, experiment logs
- **Estimated Training Time**:
  - 10K synthetic grids, 100 epochs, batch size 64
  - V100: ~4-6 hours
  - RTX 4090: ~3-4 hours
  - CPU-only (not recommended): ~72 hours

**Task Generation Pipeline**:
- **GPU**: Not required (generation runs on CPU)
- **CPU**: 4+ cores, benefit from parallelization
- **RAM**: 16 GB sufficient
- **Storage**: 200 GB for 3,000 task benchmark + metadata
- **Estimated Time**:
  - 3,000 tasks with rendering: ~2-3 hours (single-threaded)
  - Parallelized (8 cores): ~20-30 minutes

**AI Solver Evaluation**:
- **GPU**: Depends on solver architecture
  - CNN Baseline: 8 GB VRAM sufficient
  - ViTARC (if reproduced): 16-24 GB VRAM
- **Estimated Inference Time**:
  - CNN Baseline: 500 tasks in ~10 minutes (V100)
  - SOTA Solvers: Variable, potentially 1-2 hours

**Human Study Infrastructure**:
- **Server**: AWS EC2 t3.medium or equivalent (2 vCPU, 4 GB RAM)
- **Database**: PostgreSQL (db.t3.micro RDS instance, 20 GB storage)
- **Cost**: ~$50-100/month during active study periods

### 4.2 Software Environment

**Python Version**: 3.9-3.11 (3.10 recommended)

**Core Dependencies** (`requirements.txt`):
```
torch==2.0.1
torchvision==0.15.2
numpy==1.24.3
scipy==1.11.1
matplotlib==3.7.2
seaborn==0.12.2
plotly==5.15.0
scikit-learn==1.3.0
pandas==2.0.3
h5py==3.9.0
Pillow==10.0.0
imagehash==4.3.1
networkx==3.1
tqdm==4.65.0
wandb==0.15.5
streamlit==1.25.0
psycopg2-binary==2.9.6
pyyaml==6.0.1
```

**Development Tools**:
```
jupyter==1.0.0
ipython==8.14.0
black==23.7.0  # Code formatter
flake8==6.0.0  # Linter
pytest==7.4.0  # Testing
```

**Optional (for JAX fallback)**:
```
jax[cuda11_cudnn82]==0.4.13  # If parametric rendering needed
flax==0.7.0
optax==0.1.7
```

**Docker Container** (reproducibility):
```dockerfile
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
RUN apt-get update && apt-get install -y git
COPY requirements.txt /workspace/
RUN pip install -r /workspace/requirements.txt
WORKDIR /workspace
```

### 4.3 Cloud Computing Options

**Option 1: University/Institutional Cluster**:
- **Advantages**: Free, high-performance GPUs (A100s), large storage
- **Disadvantages**: Job queues, limited availability
- **Recommended For**: Encoder training, large-scale experiments

**Option 2: Google Colab Pro** ($10/month):
- **GPU**: V100 or T4 (occasionally A100)
- **Advantages**: Cheap, easy setup, Jupyter interface
- **Disadvantages**: Session timeouts, 12-hour limits, unreliable GPU allocation
- **Recommended For**: Prototyping, small experiments, triage phase

**Option 3: AWS EC2**:
- **Instance**: `p3.2xlarge` (V100, $3.06/hour) or `g5.xlarge` (A10G, $1.006/hour)
- **Advantages**: Full control, reliable, scalable
- **Disadvantages**: Cost ($50-200 for full project)
- **Recommended For**: Final experiments, reproducibility runs

**Option 4: Lambda Labs** (GPU Cloud):
- **Instance**: A100 at ~$1.10/hour (cheaper than AWS)
- **Advantages**: GPU-focused, simple pricing
- **Disadvantages**: Smaller ecosystem than AWS
- **Recommended For**: Extended training runs

**Cost Estimate (Full Project)**:
- Encoder training: 6 hours × $3/hour = $18
- Solver evaluation: 2 hours × $1/hour = $2
- Ablation studies: 10 hours × $1/hour = $10
- Buffer for failures: $20
- **Total GPU Cost**: ~$50-70

---

## 5. Deployment and Release

### 5.1 Code Repository

**GitHub Repository Structure**:
```
arc-controllable-generation/
├── README.md                    # Installation, usage, citation
├── requirements.txt             # Python dependencies
├── setup.py                     # Package installation
├── LICENSE                      # MIT or Apache 2.0
├── .gitignore
├── .gitattributes               # Git-LFS configuration
├── src/
│   ├── __init__.py
│   ├── encoder/                 # Slot attention implementation
│   │   ├── slot_attention.py
│   │   ├── encoder.py
│   │   └── decoder.py
│   ├── functions/               # Sub-function library
│   │   ├── geometric.py         # translate, rotate, scale, reflect
│   │   ├── topological.py       # connect, group, separate
│   │   ├── set_operations.py    # union, intersection, difference
│   │   └── patterns.py          # repeat, symmetrize, tile
│   ├── generator/               # Task generation pipeline
│   │   ├── sampler.py           # Latent sampling
│   │   ├── composer.py          # Function composition
│   │   └── renderer.py          # Grid rendering
│   ├── evaluation/              # Metrics and analysis
│   │   ├── diversity.py
│   │   ├── difficulty.py
│   │   └── solvers.py
│   └── utils/
│       ├── visualization.py
│       └── data.py
├── scripts/
│   ├── train_encoder.py         # Train slot attention encoder
│   ├── generate_dataset.py      # Generate 3K task benchmark
│   ├── evaluate_solver.py       # Run AI solver evaluation
│   └── analyze_results.py       # Compute metrics, create plots
├── configs/
│   ├── encoder_config.yaml      # Hyperparameters for encoder
│   └── generation_config.yaml   # Task generation parameters
├── experiments/                 # Experiment logs (not tracked)
│   ├── 0.1-triage/
│   ├── 1.1-encoder/
│   └── ...
├── notebooks/
│   ├── 01_encoder_training.ipynb
│   ├── 02_task_generation.ipynb
│   ├── 03_human_study_analysis.ipynb
│   └── 04_ai_evaluation.ipynb
└── tests/
    ├── test_encoder.py
    ├── test_functions.py
    └── test_generator.py
```

**Git-LFS Configuration** (`.gitattributes`):
```
*.h5 filter=lfs diff=lfs merge=lfs -text
*.npy filter=lfs diff=lfs merge=lfs -text
*.pth filter=lfs diff=lfs merge=lfs -text
```

### 5.2 Dataset Release

**Hosted on Hugging Face Datasets**:
- **URL**: `huggingface.co/datasets/[username]/arc-controllable-tasks`
- **Advantages**:
  - Free hosting, version control
  - Direct integration with `datasets` library
  - DOI assignment for citability
  - Community access and feedback

**Dataset Card** (required by Hugging Face):
```markdown
# ARC Controllable Tasks Dataset

## Dataset Description
3,000 ARC-like visual reasoning tasks with controllable difficulty parameters.

## Dataset Structure
- Train: 2,000 tasks
- Validation: 500 tasks
- Test: 500 tasks (solutions provided)

## Features
- `examples`: List of input-output grid pairs (2-5 per task)
- `test_input`: Test input grid (16×16)
- `test_output`: Ground truth output grid (16×16)
- `composition_depth`: Difficulty parameter n ∈ {1,2,3,4,5}
- `object_count`: Difficulty parameter k ∈ {1,2,3,4,5,6}
- `function_sequence`: List of sub-functions applied

## Citation
[BibTeX entry]

## License
CC BY 4.0 (Creative Commons Attribution)
```

**Loading Interface**:
```python
from datasets import load_dataset
dataset = load_dataset("username/arc-controllable-tasks")
train_data = dataset['train']
```

### 5.3 Pre-trained Models

**Encoder Checkpoint**:
- **Format**: PyTorch `.pth` file (state dict)
- **Size**: ~50 MB (depends on architecture)
- **Hosted On**:
  - Hugging Face Model Hub: `huggingface.co/[username]/arc-encoder`
  - Zenodo (long-term archival with DOI)
- **Metadata**:
  - Hyperparameters (slot count, dimension, training epochs)
  - Reconstruction accuracy achieved
  - Training dataset description

**Baseline CNN Solver**:
- **Format**: PyTorch `.pth` file
- **Size**: ~200 MB (ResNet-style architecture)
- **Hosted On**: Same as encoder
- **Metadata**:
  - Architecture details
  - Training accuracy on generated tasks
  - Test accuracy stratified by difficulty

**Loading Interface**:
```python
from transformers import AutoModel  # If using HF Hub
encoder = AutoModel.from_pretrained("username/arc-encoder")

# Or direct PyTorch:
import torch
encoder = SlotAttentionEncoder(num_slots=6, slot_dim=8)
encoder.load_state_dict(torch.load("arc_encoder.pth"))
```

### 5.4 Interactive Demo

**Streamlit Web App**:
- **URL**: `arc-controllable-demo.streamlit.app` (Streamlit Community Cloud, free hosting)
- **Features**:
  1. **Task Browser**: Explore generated tasks, filter by difficulty
  2. **Custom Task Generation**:
     - Sliders to control composition depth (n) and object count (k)
     - Real-time task generation and rendering
     - Display predicted human solve rate
  3. **Difficulty Visualizations**:
     - Heatmaps of solve rates across (n, k) space
     - Learning curves from curriculum experiments
  4. **Try It Yourself**: Interactive grid editor for users to attempt tasks
- **Deployment**:
  - GitHub repo linked to Streamlit Cloud (automatic deployment)
  - Runs on lightweight CPU instance (no GPU needed for inference)

**Alternative: Observable Notebook**:
- JavaScript-based interactive notebook
- Advantages: More flexible visualizations, D3.js integration
- URL: `observablehq.com/@[username]/arc-controllable-tasks`

### 5.5 Paper Supplementary Materials

**Hosted on Paper Website** (GitHub Pages):
- **URL**: `[username].github.io/arc-controllable-generation/`
- **Contents**:
  - High-resolution figures (SVG format)
  - Interactive visualizations (Plotly embeds)
  - Video demonstrations of task generation
  - Appendix with extended results tables
  - Links to dataset, code, models
  - BibTeX citation

**Supplementary PDF** (submitted with paper):
- Extended ablation results
- Complete sub-function library specifications
- Failure mode examples (encoder errors)
- Additional human study demographics and consent forms

### 5.6 Reproducibility Checklist

**Provided Artifacts**:
- [ ] Complete source code with installation instructions
- [ ] Pre-trained encoder weights
- [ ] Generated task benchmark (3,000 tasks)
- [ ] Human study raw data (anonymized)
- [ ] AI solver predictions
- [ ] Statistical analysis scripts (Jupyter notebooks)
- [ ] Configuration files with all hyperparameters
- [ ] Docker container for environment replication
- [ ] Random seeds documented for all stochastic processes

**Documentation**:
- [ ] README with quick start guide
- [ ] API documentation (docstrings in code)
- [ ] Tutorial notebooks for common use cases
- [ ] Troubleshooting guide for common errors
- [ ] Expected runtimes for each script

**Compliance**:
- [ ] Code passes linting (flake8) and formatting (black)
- [ ] Unit tests for critical functions (pytest)
- [ ] License file (MIT or Apache 2.0)
- [ ] Code of conduct for contributors
- [ ] Contribution guidelines

---

## 6. Dependencies and External Tools

### 6.1 Critical External Dependencies

**Slot Attention Reference Implementation**:
- **Source**: [google-research/slot-attention](https://github.com/google-research/slot-attention)
- **License**: Apache 2.0
- **Usage**: Adapt architecture for discrete grid outputs
- **Note**: Original implementation for continuous images (Clevr dataset); requires modifications for 16×16 discrete grids

**Alternative Slot Attention Implementations**:
- **PyTorch Lightning**: [slot-attention-lightning](https://github.com/lucidrains/slot-attention) (community implementation)
- **JAX/Flax**: Official Google implementation if using JAX

### 6.2 ARC-AGI Solver Baselines

**ViTARC** (arXiv:2410.06405):
- **Expected Availability**: Code likely not public (recent preprint)
- **Fallback**: Implement vision transformer baseline from scratch
  - Architecture: ViT-Base (12 layers, 768 hidden dim, 12 attention heads)
  - Modification: Add object-based positional encoding (as described in paper)

**Neural Cellular Automata** (Xu & Miikkulainen, 2025):
- **Expected Availability**: Check author websites for code release
- **Fallback**: Use simpler CA baseline from [Neural CA tutorial](https://distill.pub/2020/growing-ca/)

**Product of Experts** (Franzen et al., 2025):
- **Expected Availability**: Likely proprietary (71.6% is competition-level performance)
- **Fallback**: Not critical; proceed with CNN and ViT baselines

### 6.3 Pre-trained Models (External)

**No Pre-trained Models Required**:
- This research trains encoders from scratch on synthetic data
- No reliance on ImageNet, CLIP, or other pre-trained vision models
- Advantage: Full control and interpretability

**Optional for Ablation**:
- **ResNet-18 Pre-trained on ImageNet**: Could test as alternative encoder backbone
  - Source: `torchvision.models.resnet18(pretrained=True)`
  - Hypothesis: Pre-training may hurt performance on abstract grids (distribution mismatch)

### 6.4 Human Study Tools

**Survey Platforms** (if not using custom Streamlit):
- **Qualtrics**: Institutional license often available
  - Advantages: Reliable, feature-rich, IRB-approved
  - Disadvantages: Limited customization for grid drawing interface
- **jsPsych**: Open-source JavaScript experiment framework
  - Advantages: Precise timing, flexible, free
  - Disadvantages: Requires web hosting, more setup

**Participant Recruitment**:
- **Prolific** (`prolific.co`): Recommended for research quality
  - Costs: ~$10-12/hour effective rate (including platform fees)
  - Advantages: High-quality participants, research-focused, GDPR compliant
- **Amazon Mechanical Turk**: Alternative, lower cost
  - Costs: ~$8-10/hour effective rate
  - Advantages: Large participant pool
  - Disadvantages: More quality control needed

### 6.5 Visualization and Analysis Libraries

**Network Analysis** (for graph edit distance):
- **NetworkX**: `pip install networkx`
- **Python-igraph**: Alternative, faster for large graphs
  - Installation: `pip install python-igraph`

**Statistical Analysis**:
- **Scipy**: Standard library for correlations, t-tests, ANOVA
- **Statsmodels**: Advanced regression models, if needed
  - Installation: `pip install statsmodels`
- **Pingouin**: User-friendly stats library with effect sizes
  - Installation: `pip install pingouin`

**Image Hashing** (for visual diversity):
- **ImageHash**: `pip install imagehash`
- Based on: Perceptual hashing (pHash) algorithm

### 6.6 Version Control and Experiment Tracking

**Git-LFS** (Large File Storage):
- **Installation**: [git-lfs.github.com](https://git-lfs.github.com/)
- **Usage**: Track `.h5`, `.pth`, `.npy` files
- **Free Tier**: GitHub allows 1 GB storage, 1 GB/month bandwidth
- **Paid Tier**: $5/month for 50 GB storage (if needed)

**DVC (Data Version Control)**:
- **Installation**: `pip install dvc[s3]` (if using AWS S3 remote)
- **Remote Storage Options**:
  - AWS S3: ~$0.023/GB/month
  - Google Cloud Storage: ~$0.020/GB/month
  - Local network storage: Free (if institutional server available)

**Weights & Biases** (Experiment Tracking):
- **Free Tier**: 100 GB storage, unlimited experiments
- **Paid Tier**: $50/month/user for teams (not needed for solo research)

---

## 7. Technical Standards and Best Practices

### 7.1 Code Quality Standards

**PEP 8 Compliance**:
- Use `black` for automatic formatting (line length: 100 characters)
- Use `flake8` for linting (ignore E203, W503 for black compatibility)
- Type hints for all function signatures (Python 3.9+ syntax)

**Documentation Standards**:
- Docstrings for all public functions (Google style)
- Example:
  ```python
  def translate(slots: np.ndarray, dx: float, dy: float) -> np.ndarray:
      """Translate all slots by (dx, dy) in latent space.

      Args:
          slots: Array of shape (k, d) representing k objects with d dimensions.
          dx: Translation in x-direction (latent units).
          dy: Translation in y-direction (latent units).

      Returns:
          Translated slots array of shape (k, d).

      Example:
          >>> slots = np.array([[0.5, 0.5, 1], [0.2, 0.8, 2]])
          >>> translate(slots, dx=0.1, dy=-0.1)
          array([[0.6, 0.4, 1], [0.3, 0.7, 2]])
      """
  ```

**Testing Standards**:
- Unit tests for all sub-functions (≥90% code coverage goal)
- Integration tests for end-to-end task generation
- Regression tests for encoder (ensure accuracy doesn't degrade)
- Use `pytest` with fixtures for common setup

### 7.2 Data Management Standards

**File Naming Conventions**:
- Tasks: `task_{split}_{index:06d}.h5` (e.g., `task_train_000042.h5`)
- Encoder checkpoints: `encoder_epoch_{epoch:03d}_acc_{acc:.4f}.pth`
- Experiment logs: `{date}_{experiment_id}_{description}.log`

**Metadata Standards** (JSON schema for tasks):
```json
{
  "task_id": "string",
  "split": "train|val|test",
  "difficulty": {
    "composition_depth": "int (1-5)",
    "object_count": "int (1-6)",
    "combined_score": "float (0-1)"
  },
  "composition": [
    {"function": "translate", "params": {"dx": 0.1, "dy": -0.2}},
    {"function": "rotate", "params": {"angle": 90}}
  ],
  "latent_input": "array shape (k, d)",
  "latent_output": "array shape (k, d)",
  "generated_timestamp": "ISO 8601 datetime",
  "encoder_version": "string (commit hash or version number)"
}
```

### 7.3 Reproducibility Standards

**Random Seed Management**:
- Set seeds for all stochastic processes:
  ```python
  import random, numpy as np, torch
  def set_seed(seed: int = 42):
      random.seed(seed)
      np.random.seed(seed)
      torch.manual_seed(seed)
      torch.cuda.manual_seed_all(seed)
      torch.backends.cudnn.deterministic = True
  ```
- Document seeds used for each experiment phase

**Hyperparameter Logging**:
- All hyperparameters in YAML config files (no hard-coded values)
- Log configs to experiment tracker (wandb) for every run
- Include config hash in checkpoint filenames for traceability

**Environment Pinning**:
- Use `pip freeze > requirements_frozen.txt` for exact package versions
- Provide Docker container with frozen environment
- Document Python version, CUDA version, cuDNN version

---

## 8. Open Research Questions

### 8.1 Encoder Architecture

**Question**: Should we use learned (neural) or hand-crafted (parametric) encoder?
- **Trade-offs**:
  - Neural: More flexible, potentially better representations, but harder to interpret
  - Parametric: Perfect reconstruction, fully interpretable, but less flexible
- **Decision Point**: Experiment 0.1 (triage) and 1.1 will determine
- **Current Plan**: Start with parametric for rapid prototyping, pursue neural if time permits

### 8.2 Sub-function Granularity

**Question**: Should sub-functions have continuous or discrete parameters?
- **Examples**:
  - Continuous: `rotate(angle)` where angle ∈ [0°, 360°]
  - Discrete: `rotate(angle)` where angle ∈ {0°, 90°, 180°, 270°}
- **Trade-offs**:
  - Continuous: More expressive, harder to ensure discrete grid outputs
  - Discrete: Simpler, more ARC-like, easier to implement
- **Current Plan**: Start with discrete, evaluate if continuous needed

### 8.3 Difficulty Model Form

**Question**: Linear vs. non-linear difficulty model?
- **Linear**: `D = α·n + β·k`
- **Non-linear**: `D = α·n + β·k + γ·(n×k)` (interaction term)
- **Trade-offs**: Non-linear more expressive but harder to interpret
- **Decision Point**: Experiment 2.3 (calibration) will test both

### 8.4 Human Study Design

**Question**: Within-subjects vs. between-subjects for difficulty validation?
- **Within-subjects**: Each participant sees multiple difficulty levels (current plan)
  - Advantage: More statistical power, controls for individual differences
  - Disadvantage: Potential learning effects
- **Between-subjects**: Different participants see different difficulties
  - Advantage: No learning effects
  - Disadvantage: Requires larger sample (150-300 participants)
- **Current Plan**: Within-subjects with counterbalanced task order

---

## Summary

This technical stack provides a comprehensive foundation for implementing controllable ARC-like task generation. Key decisions include:

1. **PyTorch** for neural encoder development with **Slot Attention** architecture
2. **Hand-crafted sub-function library** (15 functions) operating in latent space (d < 10)
3. **HDF5 format** for 3,000-task benchmark with full metadata
4. **Streamlit** for human studies, **Prolific** for recruitment (N=50-150)
5. **Multiple difficulty metrics**: Composition depth, object count, combined model
6. **Evaluation across human and AI solvers** with focus on difficulty correlation
7. **Hugging Face** for dataset release, **GitHub** for code, **Wandb** for experiment tracking

The stack is designed for reproducibility, with Docker containers, frozen dependencies, and comprehensive documentation. Estimated compute costs are ~$50-70 (GPU time) and ~$500-1000 (human study compensation).
