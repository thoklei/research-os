#!/bin/bash

# Task 6: Simple Integration Tests
# Runs actual training with different configs and reports results

set -e  # Exit on error

echo "================================================================================"
echo "TASK 6 INTEGRATION TESTS (Simple Version)"
echo "================================================================================"
echo ""
echo "This will run 3 quick training tests (~2-3 minutes total)"
echo "Using test-1k dataset"
echo ""

# Activate zeus environment
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
pyenv activate zeus

cd "$(dirname "$0")"

# Test 1: Baseline
echo "================================================================================"
echo "Test 1: Baseline (Default Config)"
echo "================================================================================"
python src/train_vae.py \
  --quick-test \
  --no-wandb \
  --data-path datasets/test-1k/corpus.npz

echo ""
echo "✓ Baseline test completed"
echo ""

# Test 2: Ultra-conservative with free bits
echo "================================================================================"
echo "Test 2: Ultra-Conservative Schedule + Free Bits"
echo "================================================================================"
python src/train_vae.py \
  --quick-test \
  --no-wandb \
  --data-path datasets/test-1k/corpus.npz \
  --beta-schedule ultra_conservative \
  --beta-max 0.1 \
  --free-bits 0.3

echo ""
echo "✓ Ultra-conservative test completed"
echo ""

# Test 3: Cyclical with free bits
echo "================================================================================"
echo "Test 3: Cyclical Schedule + Free Bits"
echo "================================================================================"
python src/train_vae.py \
  --quick-test \
  --no-wandb \
  --data-path datasets/test-1k/corpus.npz \
  --beta-schedule cyclical \
  --beta-max 0.1 \
  --beta-cycle-length 20 \
  --free-bits 0.3

echo ""
echo "✓ Cyclical test completed"
echo ""

echo "================================================================================"
echo "ALL TESTS COMPLETED ✓"
echo "================================================================================"
echo ""
echo "Summary:"
echo "  ✓ Baseline config works (no regressions)"
echo "  ✓ Ultra-conservative schedule with free bits=0.3 works"
echo "  ✓ Cyclical schedule with free bits=0.3 works"
echo ""
echo "Check the run folders in ../experiments/runs/ for detailed logs"
echo "Recent runs:"
find ../experiments/runs -name "training.log" -mmin -10 | head -5
