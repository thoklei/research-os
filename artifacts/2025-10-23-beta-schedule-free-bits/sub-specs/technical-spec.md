# Technical Specification

This is the technical specification for the artifact detailed in research-os/artifacts/2025-10-23-beta-schedule-free-bits/spec.md

> Created: 2025-10-23
> Version: 1.0.0

## Technical Requirements

### 1. Beta Schedule Implementations

**Location:** `project/src/training/utils.py`

Extend the existing `get_beta_schedule()` function to support three new schedule types:

#### 1.1 Ultra-Conservative Schedule (`ultra_conservative`)

```python
def get_beta_schedule(epoch: int, schedule_type: str = "linear_warmup",
                      max_beta: float = 0.5, warmup_epochs: int = 10,
                      ramp_epochs: int = 50) -> float:
    """
    Extended beta schedule with ultra-conservative option.

    Args:
        epoch: Current epoch (1-indexed)
        schedule_type: Type of schedule
        max_beta: Maximum beta value to reach
        warmup_epochs: Epochs to stay at beta=0
        ramp_epochs: Epochs to ramp from 0 to max_beta
    """
    if schedule_type == "ultra_conservative":
        if epoch <= warmup_epochs:
            return 0.0
        elif epoch <= warmup_epochs + ramp_epochs:
            progress = (epoch - warmup_epochs) / ramp_epochs
            return progress * max_beta
        else:
            return max_beta
    # ... existing schedules
```

**Default parameters for ultra_conservative:**
- `warmup_epochs = 20` (double the current 10)
- `ramp_epochs = 60` (20% longer than current 50)
- `max_beta = 0.1` (5x lower than current 0.5)

#### 1.2 Cyclical Schedule (`cyclical`)

```python
if schedule_type == "cyclical":
    # Cycle length in epochs
    cycle_length = 20  # Configurable via parameter
    cycle_position = epoch % cycle_length

    if cycle_position < cycle_length / 2:
        # Ascending: 0 -> max_beta
        return (cycle_position / (cycle_length / 2)) * max_beta
    else:
        # Descending: max_beta -> 0
        return ((cycle_length - cycle_position) / (cycle_length / 2)) * max_beta
```

**Default parameters for cyclical:**
- `cycle_length = 20` epochs per cycle
- `max_beta = 0.1` (conservative max)

#### 1.3 BetaScheduler Class Updates

Update the `BetaScheduler` class to accept and pass through the new parameters:

```python
class BetaScheduler:
    def __init__(self, schedule_type: str = "linear_warmup",
                 max_beta: float = 0.5,
                 warmup_epochs: int = 10,
                 ramp_epochs: int = 50,
                 cycle_length: int = 20):
        self.schedule_type = schedule_type
        self.max_beta = max_beta
        self.warmup_epochs = warmup_epochs
        self.ramp_epochs = ramp_epochs
        self.cycle_length = cycle_length
        self.current_epoch = 0
        self.beta_history: List[float] = []

    def step(self, epoch: int) -> float:
        self.current_epoch = epoch
        beta = get_beta_schedule(
            epoch,
            self.schedule_type,
            max_beta=self.max_beta,
            warmup_epochs=self.warmup_epochs,
            ramp_epochs=self.ramp_epochs,
            cycle_length=self.cycle_length
        )
        self.beta_history.append(beta)
        return beta
```

### 2. Free Bits Implementation

**Location:** `project/src/models/beta_vae.py`

Modify the `loss_function()` method to add free bits mechanism:

```python
def loss_function(self, recon_logits, x, mu, logvar, beta=1.0, free_bits=0.0):
    """
    Compute β-VAE loss with optional free bits.

    Args:
        recon_logits: Reconstruction logits (batch, num_colors, H, W)
        x: Ground truth grid (batch, H, W)
        mu: Latent mean (batch, latent_dim)
        logvar: Latent log variance (batch, latent_dim)
        beta: β parameter for KL weighting
        free_bits: Minimum KL per dimension (nats). If 0, no clamping.

    Returns:
        dict: Dictionary containing loss components and per-dim KL
    """
    # Reconstruction loss (existing focal loss or cross-entropy)
    if self.use_focal_loss:
        recon_loss = self.recon_loss_fn(recon_logits, x.long())
    else:
        # ... existing fallback

    # KL divergence per dimension
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # (batch, latent_dim)

    # Apply free bits if specified
    if free_bits > 0:
        # Clamp per dimension, then average over batch
        kl_per_dim_clamped = torch.clamp(kl_per_dim, min=free_bits)
        kl_loss = kl_per_dim_clamped.sum(dim=1).mean()  # Sum over dims, mean over batch

        # Store unclamped KL/dim for monitoring
        kl_per_dim_actual = kl_per_dim.mean(dim=0)  # (latent_dim,)
    else:
        # Standard VAE: no clamping
        kl_loss = kl_per_dim.sum(dim=1).mean()
        kl_per_dim_actual = kl_per_dim.mean(dim=0)

    # Total loss
    total_loss = recon_loss + beta * kl_loss

    return {
        'loss': total_loss,
        'recon_loss': recon_loss,
        'kl_loss': kl_loss,
        'kl_per_dim': kl_per_dim_actual  # For monitoring (latent_dim,)
    }
```

**Key Implementation Details:**
- Free bits applied per dimension, not globally
- Clamping uses `torch.clamp(kl_per_dim, min=free_bits)`
- Return unclamped `kl_per_dim` for monitoring (to see actual collapse)
- Backward pass goes through clamped values (prevents gradient flow when below threshold)

### 3. Configuration Updates

**Location:** `project/src/training/config.py`

Add new configuration parameters to `TrainingConfig`:

```python
@dataclass
class TrainingConfig:
    # Existing parameters...

    # β-annealing schedule (UPDATED)
    beta_schedule: str = "linear_warmup"  # Options: linear_warmup, constant, aggressive, ultra_conservative, cyclical
    beta_max: float = 0.5  # Maximum beta value
    beta_warmup_epochs: int = 10  # Epochs at beta=0
    beta_ramp_epochs: int = 50  # Epochs to ramp to max_beta
    beta_cycle_length: int = 20  # For cyclical schedule

    # Free bits (NEW)
    use_free_bits: bool = False  # Enable free bits mechanism
    free_bits_lambda: float = 0.0  # Free bits per dimension (nats)

    # ... rest of config
```

**Recommended Default Configs:**

Add helper functions:

```python
def get_ultra_conservative_config(run_id: Optional[str] = None) -> TrainingConfig:
    """Get ultra-conservative beta schedule config."""
    return TrainingConfig(
        run_id=run_id,
        beta_schedule="ultra_conservative",
        beta_max=0.1,
        beta_warmup_epochs=20,
        beta_ramp_epochs=60,
        use_free_bits=True,
        free_bits_lambda=0.3,
        max_epochs=100,  # Longer training for slow schedule
    )

def get_cyclical_config(run_id: Optional[str] = None) -> TrainingConfig:
    """Get cyclical beta schedule config."""
    return TrainingConfig(
        run_id=run_id,
        beta_schedule="cyclical",
        beta_max=0.1,
        beta_cycle_length=20,
        use_free_bits=True,
        free_bits_lambda=0.3,
    )
```

### 4. Trainer Updates

**Location:** `project/src/training/trainer.py`

#### 4.1 Pass Free Bits to Loss Function

Update `train_epoch()` method:

```python
def train_epoch(self, epoch: int) -> Dict[str, float]:
    """Train for one epoch."""
    self.model.train()
    self.train_metrics.reset()

    beta = self.beta_scheduler.step(epoch)

    for batch_idx, batch in enumerate(pbar):
        # ... existing setup

        # Forward pass
        recon_logits, mu, logvar = self.model(batch)

        # Compute loss WITH free bits
        loss_dict = self.model.loss_function(
            recon_logits, batch, mu, logvar,
            beta=beta,
            free_bits=self.config.free_bits_lambda if self.config.use_free_bits else 0.0
        )

        # ... rest of training loop
```

#### 4.2 Enhanced Logging

Add per-dimension KL logging:

```python
# In train_epoch(), after computing loss_dict
if self.global_step % self.config.log_every_n_steps == 0:
    # Existing logging...

    # Add per-dimension KL logging
    if 'kl_per_dim' in loss_dict and loss_dict['kl_per_dim'].numel() > 1:
        kl_per_dim_values = loss_dict['kl_per_dim']  # (latent_dim,)

        if self.use_wandb:
            for dim_idx, kl_val in enumerate(kl_per_dim_values):
                log_dict[f'train/kl_per_dim/dim_{dim_idx}'] = kl_val.item()

            # Also log min, max, mean KL per dim
            log_dict['train/kl_per_dim/min'] = kl_per_dim_values.min().item()
            log_dict['train/kl_per_dim/max'] = kl_per_dim_values.max().item()
            log_dict['train/kl_per_dim/mean'] = kl_per_dim_values.mean().item()
```

### 5. Command-Line Interface

**Location:** `project/src/train_vae.py`

Add command-line arguments for new features:

```python
parser.add_argument(
    '--beta-schedule',
    type=str,
    default=None,
    choices=['linear_warmup', 'constant', 'aggressive', 'ultra_conservative', 'cyclical'],
    help='Beta annealing schedule type'
)
parser.add_argument(
    '--beta-max',
    type=float,
    default=None,
    help='Maximum beta value'
)
parser.add_argument(
    '--free-bits',
    type=float,
    default=None,
    help='Free bits per dimension (enables free bits if > 0)'
)

# In main(), apply overrides
if args.beta_schedule is not None:
    config.beta_schedule = args.beta_schedule

if args.beta_max is not None:
    config.beta_max = args.beta_max

if args.free_bits is not None:
    config.free_bits_lambda = args.free_bits
    config.use_free_bits = True
```

### 6. Testing and Validation

**Success Criteria:**

1. **KL/dim monitoring works**: Can observe per-dimension KL values in W&B or logs
2. **Free bits prevents collapse**: When free_bits=0.3, no dimension's KL drops below 0.3
3. **Ultra-conservative schedule trains**: Model with beta_max=0.1 maintains non-black accuracy > 30%
4. **Cyclical schedule cycles**: Beta values oscillate according to cycle_length parameter
5. **Backward compatibility**: Existing configs (linear_warmup, free_bits=0) produce identical results

**Test Commands:**

```bash
# Test ultra-conservative schedule with free bits
python src/train_vae.py --beta-schedule ultra_conservative --beta-max 0.1 --free-bits 0.3

# Test cyclical schedule
python src/train_vae.py --beta-schedule cyclical --beta-max 0.1 --free-bits 0.3

# Test free bits alone with standard schedule
python src/train_vae.py --free-bits 0.5

# Baseline (no changes)
python src/train_vae.py
```

## Approach

### Implementation Strategy

The implementation follows a modular approach where each feature can be independently tested:

1. **Beta schedules** are pure functions that can be validated with unit tests
2. **Free bits** is a self-contained modification to the loss function
3. **Configuration** uses dataclass with backward compatibility
4. **Logging** is additive and doesn't break existing metrics

### Key Design Decisions

**Free Bits Per-Dimension vs Global:**
- Chose per-dimension approach (following Kingma et al. 2016)
- Prevents selective collapse while allowing informed dimensions to use more bits
- Each dimension gets a minimum allocation, preventing gradient starvation

**Cyclical Schedule Design:**
- Triangle wave pattern (linear up, linear down)
- Simpler than cosine annealing, easier to reason about
- Full cycle = 20 epochs by default (can sweep in experiments)

**Configuration Architecture:**
- Keep all parameters in TrainingConfig for reproducibility
- Provide helper functions for common patterns (ultra_conservative, cyclical)
- CLI args override config values (for quick experiments)

## Performance Considerations

1. **Free bits overhead**: Minimal - just one `torch.clamp()` operation per batch
2. **Memory**: No additional memory overhead
3. **Logging overhead**: Per-dimension KL logging adds ~10 scalar logs per step (for d=10), negligible impact

## Implementation Order

1. Add configuration parameters to `TrainingConfig`
2. Implement ultra_conservative and cyclical schedules in `utils.py`
3. Add free bits mechanism to `beta_vae.py loss_function()`
4. Update `BetaScheduler` class to pass through new parameters
5. Modify `trainer.py` to use free bits and log per-dimension KL
6. Add command-line arguments to `train_vae.py`
7. Test each component with ablation studies

## External Dependencies

No new external dependencies required. All implementations use existing PyTorch operations and Python standard library features.
