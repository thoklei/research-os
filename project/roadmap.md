# Research Experiment Roadmap

## Overview
This roadmap outlines the experimental plan for validating controllable difficulty generation in ARC-like visual reasoning tasks through compositional latent transformations. The experiments are organized by dependencies, with each phase building on validated results from previous phases. The critical innovation is demonstrating that composition depth and object count parameters enable precise control over task difficulty.

## Phase 0: Minimum Triage Experiment (Days 1-3)
**CRITICAL: This experiment determines go/no-go for the entire research project**

### Experiment 0.1: Core Hypothesis Validation - Hand-Crafted Encoder
- **Objective**: Validate that compositional transformations in latent space can produce meaningful difficulty variations with minimal investment
- **Duration**: 2-3 days maximum
- **Approach**:
  - Implement minimal hand-crafted encoder φ: Z → Grid (16x16, 10 colors)
    - Use simple geometric shapes (circle, square, triangle) positioned at latent coordinates
    - Map each slot (x, y, shape_type, color) directly to grid rendering
    - Avoid complex learned representations
  - Implement 3 basic sub-functions in latent space:
    - `translate(dx, dy)`: Move object position
    - `rotate_90()`: Rotate object 90 degrees
    - `scale(factor)`: Change object size
  - Generate 50 minimal tasks with varying composition depth (n=1, n=2, n=3)
  - Test with 5-10 human participants (informal pilot)
- **Required Resources**:
  - Basic Python with NumPy for grid rendering
  - No GPU required
  - 5-10 volunteer participants
- **Baseline Comparison**:
  - Naive baseline: Random grid transformations (no compositional structure)
  - Measure: Can humans distinguish n=1 vs n=3 tasks by difficulty?
  - Threshold: At least 60% of participants should rate n=3 tasks as harder than n=1
- **Success Criteria**:
  - [ ] Hand-crafted encoder produces recognizable ARC-like grids
  - [ ] Compositional transformations produce visually distinct outputs
  - [ ] Mean human solve time increases from n=1 to n=3 (even if not statistically significant)
  - [ ] No fundamental rendering bugs or edge case failures
  - [ ] Tasks visually resemble ARC examples (confirmed by manual inspection)
- **Decision Gate**:
  - **GO**: If human solve time shows upward trend with composition depth (p < 0.3 acceptable for pilot) AND grids look ARC-like
  - **PIVOT**: If mechanism works but rendering quality is poor → explore alternative shape libraries or grid sizes
  - **NO-GO**: If composition depth shows no relationship to perceived difficulty OR fundamental technical blockers discovered

**Rationale**: This triage experiment avoids the critical encoder design challenge by using hand-crafted rendering. If compositional structure fails to produce difficulty variations even with perfect rendering, the core hypothesis is invalid. If successful, we have evidence to justify investing in sophisticated encoder development.

---

## Phase 1: Foundation & Encoder Development (Weeks 1-3)

### Experiment 1.1: Encoder Architecture Search
- **Depends on**: Experiment 0.1 success (GO decision)
- **Objective**: Develop robust encoder φ: Z → Grid that maps slot-based latent representations to discrete ARC-like grids
- **Duration**: 5-7 days
- **Critical Challenge**: No existing autoencoders for ARC grids in literature (gap identified in related work analysis)
- **Approach 1 - Neural Encoder (Primary)**:
  - Train slot-based autoencoder on synthetic grid data
  - Architecture: Adapt Slot Attention (Locatello et al.) for discrete outputs
    - Encoder: CNN → Slot Attention (k slots, d<10 dimensions each)
    - Decoder: Slot representations → Spatial broadcast → CNN → 16x16x10 grid (softmax over colors)
  - Training data: 10,000 synthetic grids with 1-6 simple objects (circles, squares, triangles)
  - Loss: Cross-entropy reconstruction loss on discrete grid outputs
- **Approach 2 - Parametric Rendering (Fallback)**:
  - Design differentiable rendering function mapping slots to shapes
  - Each slot (x, y, shape_type, size, color, orientation) → rendered shape
  - Composite shapes additively with occlusion rules
  - Parameters manually designed, not learned
- **Implementation Milestones**:
  - [ ] Generate synthetic training dataset of 10K grids
  - [ ] Implement slot attention architecture for discrete grids
  - [ ] Train autoencoder for 50-100 epochs
  - [ ] Evaluate reconstruction accuracy on held-out grids
  - [ ] If neural approach fails, implement parametric rendering
- **Expected Results**:
  - Neural encoder: ≥90% reconstruction accuracy on simple grids
  - Parametric encoder: Perfect reconstruction by design
- **Success Criteria**:
  - [ ] Encoder achieves ≥90% reconstruction accuracy OR parametric rendering works correctly
  - [ ] Slots capture interpretable object properties (position, shape, color)
  - [ ] No systematic failure modes on basic shapes
  - [ ] Rendering completes in <100ms per grid for real-time generation
- **Fallback**: If neural encoder fails (<80% accuracy), proceed with parametric rendering and note as limitation

### Experiment 1.2: Sub-function Library Implementation
- **Depends on**: Experiment 1.1 completion (either encoder approach)
- **Objective**: Implement comprehensive library of interpretable sub-functions operating in latent space
- **Duration**: 4-5 days
- **Sub-function Categories**:
  - **Geometric (5 functions)**:
    - `translate(dx, dy)`: Move object position in latent space
    - `rotate(angle)`: Rotate object (discrete angles: 0°, 90°, 180°, 270°)
    - `scale(factor)`: Change object size
    - `reflect_x()`, `reflect_y()`: Mirror transformations
  - **Topological (3 functions)**:
    - `connect(slot_i, slot_j)`: Create line/connection between objects
    - `group(slots)`: Treat multiple slots as single unit
    - `separate(slot)`: Split composite object into components
  - **Set Operations (4 functions)**:
    - `union(slot_i, slot_j)`: Merge objects
    - `intersection(slot_i, slot_j)`: Keep overlapping region
    - `difference(slot_i, slot_j)`: Subtract one object from another
    - `duplicate(slot)`: Create copy of object
  - **Pattern Operations (3 functions)**:
    - `repeat(slot, n, direction)`: Create n copies in direction
    - `symmetrize(slots, axis)`: Make pattern symmetric
    - `tile(slot, grid_pattern)`: Tile object in pattern
- **Implementation Requirements**:
  - Each function operates on latent slot representations (not pixel space)
  - Functions must be composable: F = f_n ∘ f_{n-1} ∘ ... ∘ f_1
  - Handle edge cases: objects moving outside boundaries, empty slots, invalid operations
  - All functions deterministic given parameters
- **Validation**:
  - Unit test each function with 10+ test cases
  - Verify composition: f_2(f_1(z)) produces expected latent state
  - Render composed transformations and visually inspect outputs
- **Success Criteria**:
  - [ ] All 15 sub-functions implemented and tested
  - [ ] Functions compose correctly (no composition errors)
  - [ ] Edge cases handled gracefully (objects clipped at boundaries)
  - [ ] Rendered outputs visually match expected transformations

### Experiment 1.3: Data Generation Pipeline
- **Depends on**: Experiments 1.1 AND 1.2 completion
- **Objective**: Build end-to-end pipeline for generating ARC-like tasks with controllable difficulty
- **Duration**: 3-4 days
- **Pipeline Components**:
  1. **Input Latent Sampling**: Sample k objects (1-6) with random positions, shapes, colors
  2. **Transformation Composition**: Sample composition depth n (1-5), randomly select n sub-functions
  3. **Task Instance Generation**: Apply F to input latent → get output latent
  4. **Multi-example Tasks**: Generate 2-5 input-output pairs by local sampling in latent space (nearby initial states)
  5. **Rendering**: Map input/output latents to grids via encoder φ
- **Difficulty Control**:
  - Composition depth n: 1, 2, 3, 4, 5 (discrete levels)
  - Object count k: 1, 2, 3, 4, 5, 6 (discrete levels)
  - Combined difficulty score: weighted sum α·n + β·k (to be calibrated)
- **Generation Protocol**:
  - For each difficulty level (n, k), generate 100 task instances
  - Total: 5 depth levels × 6 object counts × 100 instances = 3,000 tasks
  - Split: 2,000 training, 500 validation, 500 test
- **Implementation Milestones**:
  - [ ] Sampling functions for latent configurations
  - [ ] Composition engine for applying function sequences
  - [ ] Multi-example task bundling
  - [ ] Batch rendering pipeline
  - [ ] Generate initial benchmark of 3,000 tasks
- **Success Criteria**:
  - [ ] Pipeline generates 3,000 diverse tasks without errors
  - [ ] Tasks match ARC format (2-5 examples, 16x16 grids, 10 colors)
  - [ ] Visual inspection confirms ARC-like appearance
  - [ ] Generated tasks span full difficulty range (n=1 to n=5, k=1 to k=6)

---

## Phase 2: Difficulty Validation with Human Studies (Weeks 4-6)

### Experiment 2.1: Pilot Human Study - Composition Depth
- **Depends on**: Generated task benchmark from 1.3
- **Objective**: Validate that composition depth n correlates with human-perceived difficulty
- **Duration**: 5-7 days (including recruitment, data collection, analysis)
- **Experimental Design**:
  - Participants: 50 humans recruited via Prolific or MTurk
  - Tasks: 50 generated tasks (10 per depth level: n=1,2,3,4,5) with fixed object count (k=3)
  - Procedure: Participants solve tasks and rate difficulty on 1-7 Likert scale
  - Metrics: Solve accuracy, solve time, difficulty rating
- **Expected Results** (from mission):
  - n=1: 94% solve rate, 12.3s mean time
  - n=3: ~70% solve rate, ~40s mean time
  - n=5: 41% solve rate, 67.8s mean time
  - Spearman correlation ρ ≥ 0.70 between depth and difficulty rating
- **Analysis Plan**:
  - Spearman correlation between n and solve time, accuracy, ratings
  - One-way ANOVA testing effect of composition depth on solve time
  - Post-hoc pairwise comparisons (Bonferroni corrected)
  - Visualization: Solve rate and time vs composition depth
- **Success Criteria**:
  - [ ] Spearman ρ ≥ 0.60 between composition depth and solve time (p < 0.01)
  - [ ] Monotonic decrease in solve rate from n=1 to n=5
  - [ ] Effect size (Cohen's d) ≥ 0.8 comparing n=1 vs n=5
  - [ ] Participant feedback confirms tasks resemble ARC

### Experiment 2.2: Human Study - Object Count
- **Depends on**: Experiment 2.1 completion
- **Objective**: Validate that object count k correlates with human-perceived difficulty
- **Duration**: 5-7 days
- **Experimental Design**:
  - Participants: Same 50 humans from 2.1 (within-subjects design)
  - Tasks: 60 generated tasks (10 per object count: k=1,2,3,4,5,6) with fixed depth (n=2)
  - Procedure: Same as 2.1
  - Metrics: Solve accuracy, solve time, difficulty rating
- **Expected Results** (from mission):
  - k=2: 89% solve rate
  - k=4: ~60% solve rate
  - k=6: 38% solve rate
  - Spearman correlation ρ = -0.71 between object count and solve rate
- **Analysis Plan**:
  - Same statistical tests as 2.1
  - Compare effect sizes: composition depth vs object count
  - Test interaction: Does depth effect vary by object count?
- **Success Criteria**:
  - [ ] Spearman ρ ≥ 0.50 between object count and solve time (p < 0.01)
  - [ ] Solve rate decreases from k=1 to k=6
  - [ ] Composition depth shows larger effect than object count (validates mission claim)

### Experiment 2.3: Difficulty Model Calibration
- **Depends on**: Experiments 2.1 AND 2.2 completion
- **Objective**: Fit predictive model for task difficulty based on composition depth and object count
- **Duration**: 2-3 days
- **Modeling Approach**:
  - Linear regression: `solve_rate = β₀ + β₁·n + β₂·k + β₃·(n×k) + ε`
  - Logistic regression: `P(solve) = logit⁻¹(β₀ + β₁·n + β₂·k)`
  - Compare models via R² and cross-validated MSE
- **Calibration Goal**:
  - Given target solve rate (e.g., 70%), predict optimal (n, k) parameters
  - From mission: n=2.4, k=3-4 should yield 68-73% solve rate
- **Analysis**:
  - Fit models on 80% of human data, validate on 20%
  - Test predictions: Generate 200 tasks at n=2.4, k=3.5 (interpolated), validate with 20 new participants
- **Success Criteria**:
  - [ ] Model achieves R² ≥ 0.60 predicting solve rate
  - [ ] Calibration test validates 68-73% solve rate for recommended parameters
  - [ ] Composition depth contributes more variance than object count (β₁ > β₂)

---

## Phase 3: AI Solver Evaluation (Weeks 7-9)

### Experiment 3.1: Baseline Solver Reproduction
- **Depends on**: Generated benchmark from 1.3 (can run in parallel with Phase 2)
- **Objective**: Reproduce or adapt state-of-the-art ARC solvers to evaluate on generated tasks
- **Duration**: 7-10 days
- **Solvers to Evaluate**:
  1. **ViTARC** (arXiv:2410.06405):
     - Claims nearly 100% on half of 400 public ARC tasks
     - Uses pixel-level input + object-based positional encoding
     - Official implementation: [search for repo if available]
  2. **Product of Experts (Franzen et al., 2025)**:
     - Achieves 71.6% on public ARC-AGI
     - May not have public implementation → use simpler alternative
  3. **Neural Cellular Automata (Xu & Miikkulainen, 2025)**:
     - Alternative approach for grid transformations
  4. **Baseline: Simple CNN**:
     - Train supervised CNN on our generated training set (2,000 tasks)
     - Architecture: ResNet-style encoder → transformer → grid decoder
- **Implementation Strategy**:
  - Prioritize ViTARC if code available (most relevant)
  - If no implementations available, train our own CNN baseline
  - Adapt solvers to our task format (may require minor modifications)
- **Evaluation Protocol**:
  - Test on 500-task test set (from 1.3)
  - Metrics: Exact match accuracy (output grid matches ground truth)
  - Stratify results by composition depth (n=1-5) and object count (k=1-6)
- **Expected Results** (from mission):
  - ViTARC: 87% (n=1) → 23% (n=5)
  - Product-of-Experts: 72% (n=1) → 18% (n=5)
  - Baseline CNN: Lower overall but similar degradation pattern
- **Success Criteria**:
  - [ ] At least one SOTA solver successfully evaluated
  - [ ] Baseline CNN trained and evaluated
  - [ ] Results stratified by difficulty parameters
- **Fallback**: If no SOTA implementations available, proceed with CNN baseline only and note as limitation

### Experiment 3.2: AI Difficulty Correlation Analysis
- **Depends on**: Experiment 3.1 completion
- **Objective**: Validate that AI solver performance degrades with composition depth and object count
- **Duration**: 3-4 days
- **Analysis Plan**:
  - Correlation analysis: Spearman ρ between (n, k) and solver accuracy
  - Regression: `accuracy = β₀ + β₁·n + β₂·k + ε`
  - Compare AI sensitivity vs human sensitivity to difficulty parameters
  - Variance explained: Does composition depth explain more variance than object count?
- **Expected Results** (from mission):
  - Composition depth explains 31% variance in solver failure
  - Object count explains 18% variance in solver failure
  - Strong negative correlation between depth and accuracy
- **Visualizations**:
  - Accuracy heatmap: n (x-axis) vs k (y-axis)
  - Degradation curves: Accuracy vs composition depth for each solver
  - Human-AI comparison: Overlayed solve rates
- **Success Criteria**:
  - [ ] Spearman ρ ≤ -0.60 between composition depth and solver accuracy (p < 0.01)
  - [ ] Composition depth explains ≥25% variance (validates mission claim)
  - [ ] AI solvers degrade faster than humans (larger slope in regression)

### Experiment 3.3: Ablation Study - Sub-function Contributions
- **Depends on**: Experiment 3.2 completion
- **Objective**: Identify which sub-function categories drive difficulty
- **Duration**: 3-4 days
- **Ablation Conditions**:
  1. **Geometric only**: Compositions use only geometric transformations (translate, rotate, scale)
  2. **Topological only**: Use only topological operations (connect, group, separate)
  3. **Set operations only**: Use only set operations (union, intersection, difference)
  4. **Pattern operations only**: Use only pattern operations (repeat, symmetrize, tile)
  5. **Full library**: All sub-functions available (baseline)
- **Protocol**:
  - Generate 500 tasks per condition (same difficulty distribution: n=1-5, k=2-4)
  - Evaluate with human participants (20 per condition, within-subjects)
  - Evaluate with best-performing AI solver from 3.1
- **Analysis**:
  - Compare solve rates across ablation conditions
  - Identify which function categories are hardest for humans vs AI
  - Test hypothesis: Topological and set operations harder than geometric
- **Success Criteria**:
  - [ ] Geometric-only tasks significantly easier than full library (p < 0.05)
  - [ ] Identify at least one function category that disproportionately challenges AI
  - [ ] Results inform which sub-functions contribute most to controllable difficulty

---

## Phase 4: Diversity and Quality Analysis (Weeks 10-11)

### Experiment 4.1: Structural Diversity Evaluation
- **Depends on**: Generated benchmark from 1.3
- **Objective**: Quantify diversity of generated tasks and compare to public ARC dataset
- **Duration**: 3-4 days
- **Diversity Metrics**:
  1. **Function Composition Diversity**:
     - Count unique composition sequences across 3,000 generated tasks
     - Expected: >100,000 unique patterns up to n=4 (from mission)
  2. **Visual Diversity**:
     - Perceptual hash distance between task grids
     - Compare distribution to 400 public ARC tasks
  3. **Structural Diversity (Graph Edit Distance)**:
     - Represent each task as graph (objects = nodes, transformations = edges)
     - Compute average graph edit distance between task pairs
     - Expected: 2.3x higher than public ARC (from mission)
  4. **Color and Shape Diversity**:
     - Entropy of color distributions
     - Distribution of shape types across dataset
- **Comparison Baseline**:
  - Download 400 public ARC tasks
  - Compute same diversity metrics
  - Statistical comparison: t-tests and effect sizes
- **Analysis**:
  - Diversity scores by difficulty level (does complexity reduce diversity?)
  - Coverage: Do all 15 sub-functions appear sufficiently?
  - Uniqueness: What percentage of tasks are truly unique?
- **Success Criteria**:
  - [ ] Generated tasks achieve ≥2x structural diversity vs public ARC
  - [ ] All 15 sub-functions appear in ≥5% of tasks
  - [ ] <5% duplicate tasks (near-identical compositions)
  - [ ] Diversity maintained across difficulty levels

### Experiment 4.2: Encoder Quality Analysis
- **Depends on**: Experiment 1.1 (encoder architecture)
- **Objective**: Evaluate reconstruction accuracy and failure modes of encoder φ
- **Duration**: 2-3 days
- **Evaluation Protocol**:
  - Test set: 1,000 held-out latent configurations
  - Metrics:
    - Pixel-wise accuracy (exact match on discrete grids)
    - Object-level accuracy (correct object count, positions, colors)
    - Slot disentanglement: Do slot dimensions encode interpretable features?
- **Expected Results** (from mission):
  - Overall reconstruction accuracy: 96.4%
  - Failure concentrated on high-composition-depth tasks (n ≥ 6)
- **Failure Mode Analysis**:
  - Categorize reconstruction errors:
    - Color errors (wrong color assignment)
    - Position errors (object displaced)
    - Shape errors (object shape incorrect)
    - Missing objects (slot not rendered)
  - Identify systematic patterns: Which sub-functions cause errors?
  - Analyze: Are failures correlated with task difficulty?
- **Disentanglement Analysis**:
  - Vary single latent dimension, observe grid changes
  - Compute mutual information between slots and object properties
  - Qualitative visualization: Latent traversals
- **Success Criteria**:
  - [ ] Reconstruction accuracy ≥90% on simple tasks (n ≤ 3, k ≤ 4)
  - [ ] Interpretable failure modes identified and documented
  - [ ] Slot dimensions show disentanglement (≥70% purity for position, shape, color)

### Experiment 4.3: Curriculum Learning Validation
- **Depends on**: Difficulty model from 2.3
- **Objective**: Demonstrate that controllable difficulty enables effective curriculum learning
- **Duration**: 4-5 days
- **Experimental Design**:
  - Train CNN baseline (from 3.1) with three training curricula:
    1. **Random**: Sample tasks uniformly from all difficulty levels
    2. **Easy-to-Hard**: Start with n=1,k=1, gradually increase to n=5,k=6 over epochs
    3. **Hard-to-Easy**: Reverse curriculum
  - Training: 2,000 generated tasks, 100 epochs, standard augmentation
  - Evaluation: Same 500-task test set across all curricula
- **Hypothesis**:
  - Easy-to-hard curriculum should outperform random and hard-to-easy
  - Validates that difficulty control enables curriculum design
- **Metrics**:
  - Final test accuracy
  - Learning curves (accuracy vs epoch)
  - Generalization: Accuracy on hardest tasks (n=5, k=6)
- **Success Criteria**:
  - [ ] Easy-to-hard curriculum achieves ≥5% higher test accuracy than random
  - [ ] Learning curves show clear differences between curricula
  - [ ] Demonstrates practical utility of controllable difficulty

---

## Phase 5: Final Validation & Reproducibility (Weeks 12-13)

### Experiment 5.1: Large-Scale Human Validation
- **Depends on**: All previous experiments
- **Objective**: Validate key findings with larger participant pool and final benchmark
- **Duration**: 5-7 days
- **Experimental Design**:
  - Participants: 150 humans (3 cohorts of 50)
  - Tasks: 200 carefully selected tasks spanning difficulty levels
  - Protocol: Same as 2.1/2.2 but with refined task selection
- **Validation Goals**:
  - Replicate Spearman ρ = 0.78 between composition depth and difficulty
  - Validate solve rate progressions: 92-95% (n=1) → 38-44% (n=5)
  - Confirm consistent patterns across three independent cohorts
- **Expected Results** (from mission):
  - Spearman ρ = 0.78, p < 0.001
  - Consistent solve rates across cohorts
  - Difficulty model validated on new participants
- **Analysis**:
  - Inter-cohort reliability (correlation between cohort solve rates)
  - Aggregate statistics with confidence intervals
  - Finalize difficulty model with full dataset
- **Success Criteria**:
  - [ ] Spearman ρ ≥ 0.75 (p < 0.001) between depth and difficulty rating
  - [ ] Cohort solve rates within ±5% of each other (high reliability)
  - [ ] All mission claims validated with statistical significance

### Experiment 5.2: Benchmark Dataset Finalization
- **Depends on**: Experiment 5.1
- **Objective**: Create final public benchmark with metadata and documentation
- **Duration**: 3-4 days
- **Dataset Components**:
  - **Training Set**: 2,000 tasks with difficulty labels (n, k)
  - **Validation Set**: 500 tasks
  - **Test Set**: 500 tasks (with held-out solutions)
  - **Metadata**: JSON files with latent representations, composition sequences, difficulty scores
  - **Documentation**: README with task format, evaluation protocol, baseline results
- **Quality Assurance**:
  - Visual inspection of all test set tasks (flag any rendering errors)
  - Verify no train-test leakage (check for duplicate compositions)
  - Validate difficulty labels match calibrated model predictions
- **Baseline Results Package**:
  - Human performance statistics (solve rates, times, ratings)
  - AI solver performance (from 3.1/3.2)
  - Difficulty model coefficients and predictions
- **Success Criteria**:
  - [ ] 3,000 high-quality tasks with complete metadata
  - [ ] <1% rendering errors or edge cases
  - [ ] Comprehensive documentation and baseline results
  - [ ] Dataset ready for public release

### Experiment 5.3: Reproducibility Package
- **Depends on**: All previous experiments
- **Objective**: Ensure complete reproducibility of results
- **Duration**: 3-4 days
- **Package Contents**:
  1. **Code Repository**:
     - Encoder architecture (neural or parametric)
     - Sub-function library implementation
     - Task generation pipeline
     - Evaluation scripts
     - Statistical analysis notebooks
  2. **Pretrained Models**:
     - Encoder weights (if neural approach used)
     - Baseline CNN solver weights
  3. **Data**:
     - Generated benchmark dataset (3,000 tasks)
     - Human study raw data (anonymized)
     - AI solver predictions
  4. **Documentation**:
     - Installation instructions
     - Usage examples
     - Hyperparameter settings
     - Expected runtimes
- **Reproducibility Tests**:
  - Fresh clone and install on clean environment
  - Re-run encoder training (verify convergence to similar accuracy)
  - Re-generate 100 tasks (verify similarity to benchmark)
  - Re-run statistical analyses (verify identical results)
- **Success Criteria**:
  - [ ] Fresh install reproduces encoder within ±2% reconstruction accuracy
  - [ ] Task generation produces visually similar outputs
  - [ ] Statistical analyses reproduce exact p-values and correlations
  - [ ] Clear README enables reproduction by external researchers

---

## Risk Mitigation & Contingency Plans

### High-Risk Elements

#### Risk 1: Encoder Development Failure
- **Description**: Neural encoder fails to achieve ≥90% reconstruction accuracy on ARC-like grids
- **Probability**: Medium (no existing work for guidance)
- **Impact**: Critical (encoder is core component)
- **Mitigation**:
  - Start with parametric rendering fallback in parallel with neural approach
  - If neural encoder underperforms, use parametric rendering
  - Document as limitation: "Hand-crafted encoder pending learned approach"
- **Fallback**:
  - Proceed with parametric rendering
  - Frame contribution as "controllable difficulty generation" independent of encoder method
  - Note as future work: "Learned encoder for discrete grids remains open challenge"

#### Risk 2: Weak Difficulty Correlations
- **Description**: Composition depth and object count show weak or no correlation with human difficulty (ρ < 0.50)
- **Probability**: Low-Medium (triage experiment reduces risk)
- **Impact**: High (invalidates core hypothesis)
- **Mitigation**:
  - Triage experiment (Phase 0) provides early warning
  - If correlations weak, explore alternative parameters:
    - Specific sub-function types (e.g., topological harder than geometric)
    - Latent distance traveled by transformations
    - Visual complexity metrics (edges, color changes)
- **Fallback**:
  - Pivot to "diversity analysis" and "curriculum learning" contributions
  - Re-frame as "exploration of difficulty factors" rather than "controllable difficulty"
  - Emphasize compositional generation framework as main contribution

#### Risk 3: Human Study Recruitment Challenges
- **Description**: Difficulty recruiting 50-150 participants or high dropout rates
- **Probability**: Medium
- **Impact**: Medium (delays timeline, reduces statistical power)
- **Mitigation**:
  - Use Prolific or MTurk with fair compensation ($15/hour)
  - Keep tasks short (10-15 minutes per participant)
  - Pilot with smaller sample (20 participants) first
- **Fallback**:
  - Reduce sample size to 30 participants (still adequate for correlations)
  - Emphasize AI evaluation (which doesn't require recruitment)
  - Note limited generalizability but still report significant findings

#### Risk 4: SOTA Solver Reproduction Issues
- **Description**: No public implementations of ViTARC or Product-of-Experts available
- **Probability**: High (many papers lack code releases)
- **Impact**: Medium (limits AI evaluation comparisons)
- **Mitigation**:
  - Prioritize simpler baselines (CNN, transformer) that we can implement
  - Use Neural Cellular Automata if code available
  - Focus on comparing our CNN baseline across difficulty levels
- **Fallback**:
  - Train multiple baseline architectures (CNN, ResNet, Transformer)
  - Compare baselines to each other on our benchmark
  - Emphasize difficulty correlation analysis over absolute performance

#### Risk 5: Generated Tasks Don't Resemble ARC
- **Description**: Visual appearance significantly different from original ARC tasks
- **Probability**: Low-Medium
- **Impact**: Medium (reduces ecological validity)
- **Mitigation**:
  - Use exact ARC specifications (16x16 grids, 10-color palette)
  - Include qualitative comparison figures in paper
  - Conduct Turing test: Can humans distinguish our tasks from real ARC?
- **Fallback**:
  - Frame as "ARC-inspired" rather than "ARC-like"
  - Emphasize controllable generation over exact resemblance
  - Argue that structural properties matter more than surface similarity

### Timeline Buffer

**Pessimistic Timeline** (assumes 1-2 risks materialize):
- **Weeks 0-1**: Phase 0 (Triage) + Risk assessment
- **Weeks 1-4**: Phase 1 (Encoder development with fallback)
- **Weeks 4-7**: Phase 2 (Human studies with smaller sample)
- **Weeks 7-10**: Phase 3 (AI evaluation with baseline only)
- **Weeks 10-12**: Phase 4 (Diversity analysis)
- **Weeks 12-14**: Phase 5 (Final validation)
- **Total**: 14 weeks (~3.5 months)

**Optimistic Timeline** (no major risks):
- **Weeks 0-1**: Phase 0 (Triage)
- **Weeks 1-3**: Phase 1 (Encoder + sub-functions)
- **Weeks 4-6**: Phase 2 (Human studies)
- **Weeks 7-9**: Phase 3 (AI evaluation)
- **Weeks 10-11**: Phase 4 (Analysis)
- **Weeks 12-13**: Phase 5 (Final validation)
- **Total**: 13 weeks (~3 months)

**Recommended Buffer**: Plan for 14-16 weeks total, with weeks 14-16 reserved for:
- Additional experiments based on reviewer feedback
- Writing and polishing paper
- Preparing supplementary materials
- Creating demo visualizations

---

## Dependencies Summary

```
Experiment 0.1 (Triage: Hand-Crafted Encoder Test)
    ↓ (GO decision)
    ├─→ Experiment 1.1 (Encoder Architecture Search)
    │       ↓
    │   Experiment 1.2 (Sub-function Library)
    │       ↓
    │   Experiment 1.3 (Data Generation Pipeline)
    │       ↓
    │       ├─→ Experiment 2.1 (Human Study: Depth)
    │       │       ↓
    │       │   Experiment 2.2 (Human Study: Object Count)
    │       │       ↓
    │       │   Experiment 2.3 (Difficulty Model Calibration)
    │       │
    │       ├─→ Experiment 3.1 (Baseline Solver Reproduction)
    │       │       ↓
    │       │   Experiment 3.2 (AI Difficulty Correlation)
    │       │       ↓
    │       │   Experiment 3.3 (Ablation Study: Sub-functions)
    │       │
    │       └─→ Experiment 4.1 (Structural Diversity)
    │
    ├─→ Experiment 4.2 (Encoder Quality Analysis)
    │
    └─→ Experiment 4.3 (Curriculum Learning Validation)
            ↓
        Experiment 5.1 (Large-Scale Human Validation)
            ↓
        Experiment 5.2 (Benchmark Dataset Finalization)
            ↓
        Experiment 5.3 (Reproducibility Package)
```

**Critical Path**: 0.1 → 1.1 → 1.2 → 1.3 → 2.1 → 2.2 → 2.3 → 5.1 → 5.2 → 5.3

**Parallel Tracks**:
- Human evaluation (Phase 2) and AI evaluation (Phase 3) can partially overlap after 1.3
- Diversity analysis (4.1) can run immediately after 1.3
- Encoder quality analysis (4.2) can run after 1.1

---

## Success Metrics

Overall project success requires meeting these criteria:

### Phase 0 (Triage) - GO Decision
- [ ] Hand-crafted encoder produces recognizable ARC-like grids
- [ ] Composition depth shows upward trend with human difficulty (p < 0.3)
- [ ] No fundamental technical blockers discovered

### Phase 1 (Foundation) - Technical Viability
- [ ] Encoder achieves ≥90% reconstruction accuracy (or parametric fallback works)
- [ ] All 15 sub-functions implemented and composable
- [ ] Pipeline generates 3,000 diverse tasks matching ARC format

### Phase 2 (Human Validation) - Core Hypothesis
- [ ] Spearman ρ ≥ 0.70 between composition depth and difficulty (p < 0.01)
- [ ] Monotonic solve rate decrease across difficulty levels
- [ ] Difficulty model achieves R² ≥ 0.60 predicting human performance

### Phase 3 (AI Validation) - Generalizability
- [ ] At least one AI solver evaluated successfully
- [ ] Composition depth explains ≥25% variance in solver failure
- [ ] AI shows steeper degradation than humans (validates reasoning difficulty)

### Phase 4 (Quality Analysis) - Dataset Contribution
- [ ] Generated tasks achieve ≥2x structural diversity vs public ARC
- [ ] Encoder maintains ≥90% reconstruction on simple tasks
- [ ] Curriculum learning demonstrates practical utility

### Phase 5 (Final Validation) - Publication Readiness
- [ ] Large-scale human study (N=150) replicates key findings (ρ ≥ 0.75)
- [ ] Complete benchmark dataset with metadata and documentation
- [ ] Reproducibility package enables external validation

### Minimum Publishable Unit (if timeline compressed)
If forced to truncate experiments, the minimum viable paper requires:
- [ ] Triage experiment + full encoder development (Phase 0-1)
- [ ] Single human study validating composition depth effect (Experiment 2.1)
- [ ] Diversity analysis comparing to public ARC (Experiment 4.1)
- [ ] Reproducible task generation pipeline

This would support publication at a workshop or short paper venue, with full validation deferred to journal extension.

---

## Experiment Tracking

For each experiment, maintain:
1. **Experiment Log**: Date, duration, deviations from plan
2. **Results Summary**: Key metrics, statistical tests, visualizations
3. **Code Commit**: Tagged version of code used
4. **Data Archive**: Stored outputs, predictions, raw data
5. **Lessons Learned**: What worked, what didn't, adjustments made

Centralize in research repository:
```
research-os/
├── experiments/
│   ├── 0.1-triage/
│   │   ├── log.md
│   │   ├── results.json
│   │   ├── tasks/
│   │   └── analysis.ipynb
│   ├── 1.1-encoder/
│   ├── 2.1-human-depth/
│   └── ...
└── results/
    ├── figures/
    ├── tables/
    └── final-results.json
```

This structure enables systematic tracking and supports reproducibility requirements.
