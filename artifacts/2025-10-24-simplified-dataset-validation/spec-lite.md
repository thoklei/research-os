# Spec Summary (Lite)

Validate β-VAE model capacity by training on a simplified 100k dataset containing only parameterized shapes from shape_generators.py (Lines, Rectangles, Checkerboards, L-shapes, T-shapes, Plus, Zigzag), excluding blob objects to reduce variability. Train with beta effectively disabled (max_beta ≈ 0) using linear schedule to isolate reconstruction capacity evaluation without KL regularization. Success criteria: >95% pixel accuracy across all 10 color classes without collapse to black pixels (avoiding the 93% trivial solution).
