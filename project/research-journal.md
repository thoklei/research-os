# Research Planning Journal

## Initial Research Idea

**Core Problem:**
ARC-AGI-v1 is a difficult challenge for current visual reasoning models. It consists of ~1k tasks, each with 2-5 "training" input-output pairs and a final "test" input to be completed. Both inputs and outputs are image-like pixel grids.

The ARC-AGI-v1 dataset tests "reasoning", but the definition of "reasoning" remains opaque. The generating process of tasks and their correct latent representation is unclear, making models' performance difficult to assess.

**Proposed Solution:**
Build a simplified version of ARC-AGI-v1 that resembles the original challenge but with precise control over image generation. Create a structured generator for ARC-like tasks with:

1. A generating map F from $\mathcal{Z} \subset \mathbb{R}^d$ to itself
2. A map $\varphi \colon \mathcal{Z} \longrightarrow \textnormal{Images}$
3. Given subset $\mathcal{I}$ of $\mathcal{Z}$: input image = $\varphi(\mathcal{I})$, output image = $(F \circ \varphi)(\mathcal{I})$

The choice of sets and maps should consist of few interpretable parameters, with latent space dimensionality smaller than image space. Parameters randomly sampled from prior distribution for new task instances.

Note: $\varphi$ decomposes into two operations:
- Map latent point $z \in \mathcal{Z}$ to Image space
- Map Image to discretized grid for rendering (fixed at 16x16 for input and output)

**Example Task:**
Input: Two points (one blue, one red) on black background
Latent: $(x,y)$ coordinate of red point + distance $d$ between points
Output: Concentric green squares, smallest at $(x,y)$ of red point, gap size controlled by $d$

**Goals:**
1. Generate fully understood, structured dataset resembling ARC tasks
2. Image encoding in structured, interpretable latent space
3. Full control of generating map → notion of task-difficulty
4. Image encoding in structured, interpretable latent space
5. Images with clearly defined "objects" (object-centric learning sense)

**Proposed Approach:**
Find set of simple sub-functions with few parameters that compose to build many functions F. Mapping $\varphi$ composed of continuous mapping to $\mathbb{R}^2$ followed by discretization for discrete ARC-like images.

**Target Venue:**
Any top-tier journal

---

## Iteration 1: Initial Exploration

### Related Work Search 1
- Search queries: ["ARC-AGI benchmark visual reasoning", "synthetic task generation object-centric", "interpretable latent space compositional"]
- Key findings:
  - **Neural Cellular Automata for ARC-AGI (Xu & Miikkulainen, 2025)**: Applies NCAs with iterative transformation rules for grid-based tasks, showing promise for abstract grid tasks. Directly tackles ARC but doesn't address interpretability or controlled generation.
  - **Product of Experts with LLMs (Franzen et al., 2025)**: Achieves 71.6% on public ARC-AGI using task-specific augmentation and DFS. Shows strong performance but lacks interpretable latent structure.
  - **Object-Centric Decision Transformer (Park et al., 2023)**: Uses object-centric representations with clustering for ARC tasks, highlighting need for better data and training methods. Closest to object-centric approach but doesn't control generation.
  - **GANformer2 (Hudson & Zitnick, 2021)**: Moves from flat latent spaces to object-oriented structure with compositional generation. Relevant for structured latent spaces but not designed for reasoning tasks.
  - **CoLa (Shi et al., 2025)**: Learns compositional latent components without predefined decomposition, achieving zero-shot generalization. Shows promise for compositional structure but in different domain (Chinese characters).
  - **EnCoBo (Kim et al., 2025)**: Energy-based concept bottlenecks enable concept composition and negation with interpretability. Relevant for interpretable latent spaces and compositional operations.

### Questions Asked

**Q1:** I found papers on Neural Cellular Automata (Xu & Miikkulainen, 2025) that tackle ARC-AGI directly. Are you building on this cellular automata approach or taking a different approach?
**A1:** Different from cellular automata approach - pursuing latent space formulation

**Q2:** The standard validation for ARC-AGI is comparing against the original test set. Will you validate by having models solve your generated tasks, or a different validation approach?
**A2:** Focus on generating the dataset for now, without worrying about evaluation yet

**Q3:** I'm assuming you're targeting a top venue like NeurIPS, ICLR, or ICML. Is that correct, or are you aiming for a different venue?
**A3:** NeurIPS

**Q4:** Recent works achieve 71.6% on ARC-AGI (Franzen et al., 2025). What level of task complexity are you planning - simpler than ARC-AGI for better analysis, or matching its difficulty?
**A4:** Not focusing on comparing, focus on creating the dataset

**Q5:** For the compositional sub-functions, I assume you'll design a library of primitives (rotation, reflection, grouping). Will these be hand-designed or learned from data?
**A5:** Design a library of sub-functions (not learning from data)

**Q6:** The typical baseline for synthetic dataset generation is to compare against existing generators. Will you compare against any existing ARC task generators or procedural methods?
**A6:** Out of current scope

**Q7:** I notice existing work doesn't address controllable difficulty through interpretable parameters. Is this gap (control over difficulty of generated tasks) your main differentiator?
**A7:** YES - controllable difficulty through interpretable parameters and control over difficulty of generated tasks

**Q8:** Are there any specific aspects you want to exclude from the initial scope (e.g., multi-modal inputs, temporal reasoning, or tasks requiring counting)?
**A8:** Not worrying about this for now, focus on constructing actionable task generation

**Existing Research Code:**
**A9:** Starting from scratch

### Insights Gained
- Pursuing latent space formulation, NOT cellular automata approach
- Target venue: NeurIPS
- Main contribution: controllable difficulty through interpretable parameters
- Will design (not learn) library of compositional sub-functions
- Focus on dataset generation first, evaluation later
- Starting implementation from scratch
- Key differentiator: control over difficulty of generated tasks through interpretable parameters

---

## Iteration 2: Focused Investigation

### Related Work Search 2
- Search queries: ["compositional task generation", "difficulty metrics visual reasoning", "object-centric latent generation", "ARC program synthesis"]
- Key findings:

**Compositional Generation Approaches:**
- **Layer Specialization in Transformers (Liu, 2025)**: Demonstrates transformers develop modular, interpretable mechanisms for compositional reasoning through layer specialization
- **SlotDiffusion**: Combines slot-based representations with diffusion for object-centric generation with state-of-the-art video prediction
- **Additive Decoders**: Framework decomposing images as sums of object-specific images, enabling novel factor combinations

**Difficulty Metrics in Visual Reasoning:**
- **CAPTCHA-X**: Defines five reasoning-oriented metrics for spatial reasoning task evaluation with categorical difficulty levels
- **DiscoveryWorld**: Structures 120 tasks with three explicit difficulty levels and parametric variations
- Most benchmarks use categorical levels rather than continuous mathematical metrics

**ARC-Specific Insights:**
- Humans achieve 64-76% accuracy while AI reaches only 40-60%
- Key primitives: object detection, spatial transformations (rotation, reflection, translation), grid operations, compositional rules
- Difficulty factors: input-output diversity constraints, visual complexity, compositional reasoning demands
- Program synthesis approaches use neurosymbolic methods and domain-specific languages

**Object-Centric Representations:**
- Slot attention mechanisms for interpretable object decomposition
- Disentangled latent dimensions separating shape and texture
- Energy-based models with permutation-invariant attention
- Complex-valued autoencoders where magnitudes express feature presence

### Technical Questions and Detailed Responses

**Q1: Latent Space Structure**
**Answer:** $d < 10$ (low-dimensional). Dimensions should have some semantics. Relates to "object-centric" because images are composed of objects.

**Q2: Sub-function Library Design**
**Answer:** Library can consist of all categories: geometric, topological, set operations, pattern operations. Sub-functions operate in latent space.

**Q3: Difficulty Parametrization**
**Answer:** Composition depth as starting point (NOTE: caveat that more compositions might actually lead to easier functions). Number of involved objects is a decent heuristic for now.

**Q4: Mapping Function φ**
**Answer:** Objects in latent space should map to structured patterns. Will ensure resemblance to ARC tasks by using similarly sized grids and same 10 color options.

**Q5: Composition Mechanism**
**Answer:** Sequential application (composition in mathematical sense): $F = f_n \circ f_{n-1} \circ ... \circ f_1$. Think analogously to how latent dimensions of a VAE map to attributes of the image.

**Q6: Task Instance Generation**
**Answer:** Both difficulty-dependent priors and hierarchical sampling make sense. Not worried about diversity for now.

**Q7: Object-Centric Aspects**
**Answer:** Use slot-based representations for now.

**Q8: Resemblance to ARC**
**Answer:** PRESERVE: grid size, color palette (10 colors), few-shot structure (2-5 examples). NOT PRESERVE: transformation types (ill-defined for ARC; want to design these explicitly).

### Technical Specifications Crystallized
- **Latent Space**: Low-dimensional ($d < 10$) with semantic dimensions
- **Object Representation**: Slot-based, each slot has $d$ dimensions
- **Sub-function Library**: Comprehensive (geometric, topological, set, pattern operations) operating in latent space
- **Composition**: Sequential function composition $F = f_n \circ ... \circ f_1$
- **Difficulty Metrics**: Composition depth and number of objects as heuristics
- **Visual Rendering**: Structured patterns on grids matching ARC (size, 10 colors)
- **Task Structure**: 2-5 input-output examples per task

---

## Iteration 3: Deep Technical Investigation

### Related Work Search 3 - Targeted Technical Search

**Slot-Based Object Representations:**
- **SlotAdapt (Akan & Yemez)**: Combines slot attention with pretrained diffusion models for compositional generation without text-centric bias
- **Slot-MLLM (Chi et al.)**: Object-centric visual tokenizer using Slot Attention encoding local visual details while maintaining high-level semantics
- **Disentangled Slot Attention (Chen et al.)**: Separates scene-dependent attributes (scale, position) from scene-independent representations (appearance, shape)
- **Explicitly Disentangled Representations (Majellaro et al.)**: Separates shape and texture components into non-overlapping latent space dimensions
- **Slot Mixture Module (Kirilenko et al.)**: Represents slots as Gaussian Mixture Model cluster centers for expressive slot representations
- **GLASS (Singh et al.)**: Combines slot attention with diffusion models using semantic and instance guidance modules for complex scenes

**Compositional Function Learning:**
- **Algorithmic Primitives (Lippl et al.)**: Function vector methods derive primitive vectors as reusable compositional building blocks
- **Latent Zoning Network (Lin et al.)**: Tasks expressed as compositions of encoders and decoders with shared Gaussian latent space
- **Neuro-Symbolic AI (Oikonomou et al.)**: Constructs analytical solutions through composition of foundational solution components
- **PARL (Liu et al.)**: Discovers symbolic and abstract action space from language-annotated demonstrations
- **Compositional Neural Textures (Tu et al.)**: Models textures as discrete composition of Gaussian textons with spatial support

**Controllable Difficulty Generation:**
- **JRDB-Reasoning (Jahangard et al.)**: Formalizes reasoning complexity with structured levels and adaptive query engine
- **MORSE-500 (Cai et al.)**: Script-driven generation with fine-grained control over visual complexity and temporal dynamics
- **CrossWordBench (Leng et al.)**: Adjustable puzzle complexity through prefill ratio control
- **Game-RL (Tong et al.)**: Provides controllable difficulty gradation across 30 games using Code2Logic approach
- **FineCops-Ref (Liu et al.)**: Multi-level fine-grained reasoning with controllable varying levels of difficulty

### Final Technical Clarification Questions and User Responses

**Q1: Slot-Latent Interaction:** Each slot has $d$ dimensions. Will slots share the same semantic interpretation of dimensions (e.g., dimension 1 = x-position for all slots), or will each slot have its own interpretation?
**A1:** Either approach is fine (shared semantic interpretation or slot-specific)

**Q2: Concrete Sub-function Examples:** Could you provide 2-3 specific examples of sub-functions in each category?
**A2 - CRITICAL CLARIFICATION FROM USER:**
- For map $F$ (operates in latent space): simple operations like rotate(angle), translate(dx, dy), etc. are all great
- **CRITICAL RESEARCH QUESTION**: What is the harder part - coming up with appropriate encoder $\varphi$ which maps from latent space to image space
  - **Is there one in literature?**
  - **Have people trained an autoencoder on ARC-like grids that we could build on?**

**Q3: Handling Edge Cases:** How should the system handle objects that would overlap, go outside boundaries, or empty slots?
**A3:** Avoid edge cases

**Q4: Rendering Pipeline:** For $\varphi: \mathcal{Z} \rightarrow \text{Images}$, will each slot render to a single colored pixel, a shape, or a learnable pattern?
**A4:** Every slot will render to a shape

**Q5: Task Consistency:** How will you ensure that the same F produces consistent transformations across the 2-5 input-output examples within a task?
**A5:** Should happen by design, as long as you sample the input latents close to each other

---

## Iteration 4: Critical Literature Gap Investigation

### Related Work Search 4 - Autoencoders for ARC

**CRITICAL FINDING: Limited Work on ARC-Specific Autoencoders**

After extensive search across arXiv and other sources, I found:

**Existing VAE Approach for ARC:**
- **"Solving ARC with Neural Embeddings and Vector Arithmetic" (arXiv:2311.08083)**:
  - Uses VAE to transform ARC items into low-dimensional latent vectors
  - Applies simple vector arithmetic to discover patterns
  - Achieved only 2% on official ARC, 8.8% on ConceptARC
  - Works well only on simpler items with fewer colors and uniform shapes
  - Requires high VAE reconstruction accuracy for success
  - **LIMITATION**: Paper abstract doesn't specify encoder architecture or latent dimensionality

**Vision Transformer Approaches (Not Autoencoders):**
- **ViTARC (arXiv:2410.06405)**: Uses pixel-level input and object-based positional encoding, but not an autoencoder
- **LatFormer (arXiv:2306.03175)**: Incorporates lattice symmetry priors but not focused on learned representations

**Object-Centric Models (General, Not ARC-Specific):**
- **Decomposer Networks (arXiv:2510.09825)**: Semantic autoencoder factorizing inputs into interpretable components, positioned relative to MONet, IODINE, Slot Attention
- No papers found specifically applying MONet, IODINE, or Slot Attention to ARC grids

### Key Research Gap Identified

**MAJOR GAP**: No existing work on training robust autoencoders specifically for ARC-like discrete grids with interpretable latent spaces. The single VAE attempt (2311.08083) achieved very limited success and doesn't provide architectural details.

This gap represents either:
1. **A significant contribution opportunity** - developing the first effective autoencoder for ARC grids
2. **A critical design challenge** - the encoder $\varphi$ from latent to image space needs to be carefully designed or learned

### Refined Understanding After All Iterations

**Core Innovation Points:**
1. **Controllable difficulty generation** through interpretable parameters (unique contribution)
2. **Structured latent space** with slot-based object representations
3. **Compositional function library** operating in latent space
4. **Critical Challenge**: Designing or learning appropriate encoder $\varphi: \mathcal{Z} \rightarrow \text{Images}$
   - No existing robust autoencoder for ARC grids in literature
   - May need to develop custom architecture or use hand-designed rendering

**Technical Decisions Finalized:**
- Slot-based representation with $d < 10$ dimensions per slot
- Each slot renders to a shape (not just pixels)
- Sequential function composition for transformations
- Avoid edge cases in initial design
- Task consistency through local sampling in latent space

---

## Final Research Vision

### Research Statement
Develop a controllable generator for ARC-like visual reasoning tasks with interpretable latent representations, enabling precise control over task difficulty through compositional transformations in a structured latent space.

### Key Differentiators
1. Unlike existing ARC approaches, this research provides full control over task generation through interpretable parameters
2. The approach introduces compositional functions operating in low-dimensional latent space rather than pixel space
3. Expected to enable systematic analysis of what makes visual reasoning tasks difficult

### Methodology Overview
- **Core Approach**: Slot-based latent representations with compositional transformations
- **Baseline**: Building on slot attention concepts, but applied to discrete grid generation
- **Novel Components**:
  - Library of interpretable sub-functions in latent space
  - Controllable difficulty through composition depth and object count
  - Custom encoder $\varphi$ for mapping latents to ARC-like grids (major technical challenge)
- **Evaluation Plan**: Generate dataset with varying difficulty levels, analyze structural properties

### Target Venue
NeurIPS - Appropriate for novel dataset generation methodology with theoretical grounding in compositional learning

### Expected Contributions
1. First controllable generator for ARC-like tasks with interpretable difficulty parameters
2. Novel compositional framework operating in structured latent space for visual reasoning
3. Systematic analysis of difficulty factors in abstract visual reasoning tasks
4. Potential foundation for training more interpretable models on structured reasoning

### Critical Open Challenge
**Encoder Design**: The mapping $\varphi: \mathcal{Z} \rightarrow \text{Images}$ remains the hardest technical challenge, with no existing robust autoencoders for ARC grids in literature. This may require either:
- Developing a custom neural encoder architecture
- Hand-designing a parametric rendering function
- Training a new autoencoder on synthetic grid data first