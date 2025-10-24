# Spec Summary (Lite)

Implement conservative beta-annealing schedules and free bits mechanism to prevent posterior collapse in beta-VAE training with severe class imbalance (93% black pixels). Current training works at beta=0 but collapses to all-black predictions when beta increases, despite focal loss and class weighting. Solution adds ultra-conservative schedule (longer warmup, lower max beta), per-dimension KL clamping via free bits, and cyclical annealing options to maintain diverse reconstructions while enabling regularization.
