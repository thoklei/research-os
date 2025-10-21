# Related Work

## Core Papers on ARC-AGI

### Neural Cellular Automata for ARC-AGI
- **Authors**: Xu & Miikkulainen
- **Year**: 2025
- **Venue**: arXiv preprint
- **Summary**: Applies NCAs with iterative transformation rules for grid-based tasks, showing promise for abstract grid reasoning. Uses local update rules to transform grids through multiple iterations.
- **Key Results**: Demonstrates emergent behavior for solving certain ARC patterns through learned update rules
- **Relation to Project**: Alternative approach - we use latent space formulation rather than cellular automata

### Product of Experts with LLMs for ARC
- **Authors**: Franzen et al.
- **Year**: 2025
- **Venue**: arXiv preprint
- **Summary**: Achieves state-of-the-art 71.6% accuracy on public ARC-AGI using task-specific augmentation and depth-first search. Combines multiple expert models.
- **Key Results**: 71.6% on public ARC test set (previous best: ~21%)
- **Relation to Project**: Shows current SOTA performance but lacks interpretable structure - our work focuses on controllable generation

### Object-Centric Decision Transformer for ARC
- **Authors**: Park et al.
- **Year**: 2023
- **Venue**: Conference paper
- **Summary**: Uses object-centric representations with clustering for ARC tasks. Identifies objects through clustering and applies transformations.
- **Key Results**: Improved performance on subset of ARC tasks requiring object manipulation
- **Relation to Project**: Closest to our object-centric approach but doesn't control generation - we add controllable difficulty

### Solving ARC with Neural Embeddings and Vector Arithmetic
- **Authors**: Not specified in abstract
- **Year**: 2023
- **Venue**: arXiv:2311.08083
- **Summary**: Uses VAE to transform ARC items into low-dimensional latent vectors, then applies simple vector arithmetic to discover patterns.
- **Key Results**: Only 2% on official ARC, 8.8% on ConceptARC. Works well only on simpler items with fewer colors.
- **Relation to Project**: **CRITICAL** - Only existing VAE approach for ARC, but very limited success. Highlights the challenge of designing encoder φ

### ViTARC: Vision Transformers for ARC
- **Authors**: Not specified
- **Year**: 2024
- **Venue**: arXiv:2410.06405
- **Summary**: Addresses representational deficiency of ViT architecture for ARC through pixel-level input and object-based positional encoding.
- **Key Results**: Nearly 100% solve rate on more than half of 400 public ARC tasks
- **Relation to Project**: Different approach using transformers - we focus on interpretable latent generation

### LatFormer: Infusing Lattice Symmetry Priors
- **Authors**: Not specified
- **Year**: 2023
- **Venue**: arXiv:2306.03175
- **Summary**: Incorporates lattice symmetry priors in attention masks for improved sample efficiency on grid-based tasks.
- **Key Results**: 2 orders of magnitude fewer data required than standard attention
- **Relation to Project**: Shows importance of structural priors for grid tasks - we incorporate structure through latent design

### NSA: Neuro-symbolic ARC Challenge
- **Authors**: Not specified
- **Year**: 2025
- **Venue**: arXiv:2501.04424
- **Summary**: Combines transformer for proposal generation with combinatorial search, pre-trains with synthetic data.
- **Key Results**: Improved performance through neuro-symbolic combination
- **Relation to Project**: Uses synthetic data generation but not with controllable difficulty parameters

## Object-Centric and Compositional Learning

### Slot-Based Representations

#### SlotAdapt
- **Authors**: Akan & Yemez
- **Year**: Recent
- **Venue**: Conference/Journal
- **Summary**: Combines slot attention with pretrained diffusion models for compositional generation without text-centric bias
- **Key Results**: Improved compositional generation quality
- **Relation to Project**: Relevant slot-based architecture we can adapt for discrete grids

#### Slot-MLLM
- **Authors**: Chi et al.
- **Year**: Recent
- **Venue**: Conference/Journal
- **Summary**: Object-centric visual tokenizer using Slot Attention encoding local visual details while maintaining high-level semantics
- **Key Results**: Better object-level understanding in multimodal models
- **Relation to Project**: Shows slot attention can capture both local and global features

#### Disentangled Slot Attention
- **Authors**: Chen et al.
- **Year**: Recent
- **Venue**: Conference/Journal
- **Summary**: Separates scene-dependent attributes (scale, position) from scene-independent representations (appearance, shape)
- **Key Results**: Improved disentanglement of object properties
- **Relation to Project**: Informs our design of slot dimensions with semantic meaning

#### Slot Mixture Module
- **Authors**: Kirilenko et al.
- **Year**: Recent
- **Venue**: Conference/Journal
- **Summary**: Represents slots as Gaussian Mixture Model cluster centers for more expressive slot representations
- **Key Results**: More flexible slot representations
- **Relation to Project**: Alternative slot representation approach to consider

#### Decomposer Networks
- **Authors**: Mohsen Joneidi
- **Year**: 2024
- **Venue**: arXiv:2510.09825
- **Summary**: Semantic autoencoder that factorizes input into multiple interpretable components through parallel branches with residual updates. Positioned relative to MONet, IODINE, Slot Attention.
- **Key Results**: Parsimonious, semantically meaningful representations through explicit component competition
- **Relation to Project**: Shows how object-centric methods could adapt to structured domains - relevant for our encoder design

### Compositional Generation

#### GANformer2
- **Authors**: Hudson & Zitnick
- **Year**: 2021
- **Venue**: NeurIPS
- **Summary**: Moves from flat latent spaces to object-oriented structure with compositional generation capabilities
- **Key Results**: Better compositional image generation
- **Relation to Project**: Shows benefits of structured latent spaces but not for reasoning tasks

#### SlotDiffusion
- **Authors**: Not specified
- **Year**: Recent
- **Venue**: Conference/Journal
- **Summary**: Combines slot-based representations with diffusion models for object-centric generation with state-of-the-art video prediction
- **Key Results**: SOTA video prediction performance
- **Relation to Project**: Demonstrates slots work well with generative models

#### Additive Decoders
- **Authors**: Not specified
- **Year**: Recent
- **Venue**: Conference/Journal
- **Summary**: Framework decomposing images as sums of object-specific images, enabling novel factor combinations
- **Key Results**: Zero-shot compositional generation
- **Relation to Project**: Relevant rendering approach - each slot contributes additively to final image

#### CoLa: Compositional Latent Components
- **Authors**: Shi et al.
- **Year**: 2025
- **Venue**: Conference/Journal
- **Summary**: Learns compositional latent components without predefined decomposition, achieving zero-shot generalization
- **Key Results**: Strong zero-shot performance on Chinese character generation
- **Relation to Project**: Shows compositional structure enables generalization in different domain

## Compositional Function Learning

### Algorithmic Primitives
- **Authors**: Lippl et al.
- **Year**: Recent
- **Venue**: Conference/Journal
- **Summary**: Function vector methods derive primitive vectors as reusable compositional building blocks
- **Key Results**: Learned reusable function primitives
- **Relation to Project**: Relevant for our sub-function library design

### Latent Zoning Network
- **Authors**: Lin et al.
- **Year**: Recent
- **Venue**: Conference/Journal
- **Summary**: Tasks expressed as compositions of encoders and decoders with shared Gaussian latent space
- **Key Results**: Improved compositional task learning
- **Relation to Project**: Similar compositional approach but we focus on interpretable transformations

### Layer Specialization in Transformers
- **Authors**: Liu
- **Year**: 2025
- **Venue**: Conference/Journal
- **Summary**: Demonstrates transformers develop modular, interpretable mechanisms for compositional reasoning through layer specialization
- **Key Results**: Emergent modular structure in transformers
- **Relation to Project**: Shows importance of modularity for compositional reasoning

## Controllable Difficulty Generation

### CAPTCHA-X
- **Authors**: Not specified
- **Year**: Recent
- **Venue**: Conference/Journal
- **Summary**: Defines five reasoning-oriented metrics for spatial reasoning task evaluation with categorical difficulty levels
- **Key Results**: Structured difficulty metrics for visual reasoning
- **Relation to Project**: Example of difficulty parametrization, though categorical rather than continuous

### DiscoveryWorld
- **Authors**: Not specified
- **Year**: Recent
- **Venue**: Conference/Journal
- **Summary**: Structures 120 tasks with three explicit difficulty levels and parametric variations
- **Key Results**: Systematic difficulty progression in task design
- **Relation to Project**: Shows value of explicit difficulty levels - we extend to continuous parameters

### JRDB-Reasoning
- **Authors**: Jahangard et al.
- **Year**: Recent
- **Venue**: Conference/Journal
- **Summary**: Formalizes reasoning complexity with structured levels and adaptive query engine
- **Key Results**: Formalized reasoning complexity metrics
- **Relation to Project**: Framework for thinking about reasoning difficulty

### MORSE-500
- **Authors**: Cai et al.
- **Year**: Recent
- **Venue**: Conference/Journal
- **Summary**: Script-driven generation with fine-grained control over visual complexity and temporal dynamics
- **Key Results**: Fine-grained control over generated task complexity
- **Relation to Project**: Similar goal of controllable generation but for video domain

### Game-RL
- **Authors**: Tong et al.
- **Year**: Recent
- **Venue**: Conference/Journal
- **Summary**: Provides controllable difficulty gradation across 30 games using Code2Logic approach
- **Key Results**: Systematic difficulty control across diverse games
- **Relation to Project**: Shows value of controllable difficulty for training

## Interpretable Representations

### EnCoBo: Energy-based Concept Bottlenecks
- **Authors**: Kim et al.
- **Year**: 2025
- **Venue**: Conference/Journal
- **Summary**: Energy-based concept bottlenecks enable concept composition and negation with interpretability
- **Key Results**: Interpretable compositional operations
- **Relation to Project**: Relevant for interpretable latent spaces and compositional operations

### Explicitly Disentangled Representations
- **Authors**: Majellaro et al.
- **Year**: Recent
- **Venue**: Conference/Journal
- **Summary**: Separates shape and texture components into non-overlapping latent space dimensions
- **Key Results**: Clean separation of visual attributes
- **Relation to Project**: Informs our design of semantic latent dimensions

### Complex-valued Autoencoders
- **Authors**: Not specified
- **Year**: Recent
- **Venue**: Conference/Journal
- **Summary**: Uses complex numbers where magnitudes express feature presence and phases encode other properties
- **Key Results**: More expressive latent representations
- **Relation to Project**: Alternative representation approach for richer latent spaces

## Datasets and Benchmarks

### ARC-AGI Dataset
- **Source**: Chollet et al.
- **Size**: ~1000 tasks with 2-5 input-output examples each
- **Standard Metrics**: Accuracy on held-out test inputs
- **Usage in Literature**: Standard benchmark for abstract reasoning
- **Our Usage**: Inspiration for grid structure, color palette, task format

### ConceptARC
- **Source**: Simplified version of ARC
- **Size**: Subset with more structured patterns
- **Standard Metrics**: Same as ARC but easier baseline
- **Usage in Literature**: Testing ground for ARC approaches
- **Our Usage**: Reference for simplification strategies

## Key Gaps in Literature

### Major Gap: Autoencoders for ARC Grids
- **Finding**: No robust autoencoders trained specifically on ARC-like discrete grids
- **Evidence**: Only one VAE attempt (arXiv:2311.08083) with very limited success (2% accuracy)
- **Implication**: The encoder φ: Z → Images is a critical technical challenge requiring novel design

### Gap: Controllable Difficulty Parameters
- **Finding**: No existing work on generating ARC-like tasks with mathematically controllable difficulty
- **Evidence**: Existing benchmarks use categorical levels rather than continuous parameters
- **Implication**: Our main contribution - interpretable difficulty control

### Gap: Compositional Transformations in Latent Space
- **Finding**: No work applying compositional functions in latent space for grid-based reasoning tasks
- **Evidence**: Existing work operates in pixel space or uses black-box transformations
- **Implication**: Novel approach combining latent representations with compositional structure

## Methodological References

### Slot Attention
- **Introduced By**: Locatello et al.
- **Common Implementation**: Available in various repos
- **Our Adaptation**: Apply to discrete grids with fixed number of slots for objects

### Variational Autoencoders (VAEs)
- **Introduced By**: Kingma & Welling
- **Common Implementation**: Standard in deep learning frameworks
- **Our Adaptation**: Need custom architecture for discrete grid outputs

### Compositional Function Spaces
- **Introduced By**: Various works in program synthesis
- **Common Implementation**: Domain-specific languages
- **Our Adaptation**: Library of interpretable sub-functions operating in latent space