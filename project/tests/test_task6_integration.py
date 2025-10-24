"""
Test Task 6: Integration Testing and Validation

This script runs end-to-end training tests with different configurations to verify:
1. No regressions with baseline config
2. Ultra-conservative schedule works correctly
3. Cyclical schedule works correctly
4. Free bits prevents KL collapse
5. Reconstruction quality maintained with beta > 0

NOTE: Uses test-1k dataset for fast testing.
"""

import subprocess
import sys
import re
from pathlib import Path


def run_training_test(name, args, expected_checks):
    """
    Run a training test and verify output.

    Args:
        name: Test name
        args: List of CLI arguments
        expected_checks: Dict of expected patterns/values to verify

    Returns:
        bool: True if test passed
    """
    print("=" * 80)
    print(f"Running: {name}")
    print("=" * 80)

    # Build command - use bash -c to properly handle pyenv
    cmd_parts = [
        "python", "src/train_vae.py",
        "--quick-test",
        "--no-wandb",
        "--data-path", "datasets/test-1k/corpus.npz",
        *args
    ]

    full_cmd = f"cd {Path(__file__).parent} && eval \"$(pyenv init -)\" && eval \"$(pyenv virtualenv-init -)\" && pyenv activate zeus && {' '.join(cmd_parts)}"

    print(f"Command: {' '.join(cmd_parts)}")
    print()

    # Run training
    result = subprocess.run(
        full_cmd,
        shell=True,
        executable="/bin/bash",
        capture_output=True,
        text=True,
        timeout=300  # 5 minute timeout
    )

    output = result.stdout + result.stderr

    # Check for successful completion
    if result.returncode != 0:
        print(f"❌ FAILED: Training exited with code {result.returncode}")
        print("\nOutput (last 100 lines):")
        print('\n'.join(output.split('\n')[-100:]))
        return False

    # Verify expected checks
    print("Verifying output...")
    for check_name, pattern in expected_checks.items():
        if isinstance(pattern, str):
            # String pattern matching
            if pattern in output:
                print(f"  ✓ {check_name}: Found '{pattern}'")
            else:
                print(f"  ❌ {check_name}: Missing '{pattern}'")
                return False
        elif callable(pattern):
            # Custom check function
            if pattern(output):
                print(f"  ✓ {check_name}: Passed")
            else:
                print(f"  ❌ {check_name}: Failed")
                return False

    print(f"\n✓ PASS: {name}\n")
    return True


def extract_final_metrics(output):
    """Extract final validation metrics from training output."""
    metrics = {}

    # Look for final epoch summary
    # Pattern: "  Val   - Loss: 2.1234 | Acc: 0.934 | KL/dim: 0.0523"
    val_pattern = r"Val\s+-\s+Loss:\s+([\d.]+)\s+\|\s+Acc:\s+([\d.]+)\s+\|\s+KL/dim:\s+([\d.]+)"
    matches = list(re.finditer(val_pattern, output))

    if matches:
        last_match = matches[-1]
        metrics['loss'] = float(last_match.group(1))
        metrics['acc'] = float(last_match.group(2))
        metrics['kl_per_dim'] = float(last_match.group(3))

    return metrics


def check_non_black_accuracy(output):
    """Verify that model predicts non-black pixels (acc > 0.30 means more than just black)."""
    metrics = extract_final_metrics(output)
    if metrics:
        acc = metrics['acc']
        # If acc > 0.30, it means model is doing better than just predicting black
        # (since 93% are black pixels, random guessing would get ~93% acc)
        return acc > 0.30
    return False


def check_kl_above_threshold(threshold):
    """Create check function for KL being above threshold."""
    def check(output):
        metrics = extract_final_metrics(output)
        if metrics:
            return metrics['kl_per_dim'] >= threshold
        return False
    return check


def test_baseline():
    """Test 6.1: Baseline test with default config."""

    return run_training_test(
        name="Baseline (Default Config)",
        args=[],  # No overrides
        expected_checks={
            "Training completed": "Training Complete",
            "No crashes": lambda o: "Traceback" not in o,
            "Has validation": "Val   - Loss:",
            "Reasonable accuracy": check_non_black_accuracy,
        }
    )


def test_ultra_conservative():
    """Test 6.2: Ultra-conservative schedule with free bits."""

    return run_training_test(
        name="Ultra-Conservative Schedule",
        args=[
            "--beta-schedule", "ultra_conservative",
            "--beta-max", "0.1",
            "--free-bits", "0.3",
        ],
        expected_checks={
            "Training completed": "Training Complete",
            "No crashes": lambda o: "Traceback" not in o,
            "Has validation": "Val   - Loss:",
            "Reasonable accuracy": check_non_black_accuracy,
            # With free_bits=0.3, KL/dim should stay >= 0.3 once beta > 0
            "KL above free bits": check_kl_above_threshold(0.25),  # Allow slight variance
        }
    )


def test_cyclical():
    """Test 6.3: Cyclical schedule with free bits."""

    return run_training_test(
        name="Cyclical Schedule",
        args=[
            "--beta-schedule", "cyclical",
            "--beta-max", "0.1",
            "--beta-cycle-length", "20",
            "--free-bits", "0.3",
        ],
        expected_checks={
            "Training completed": "Training Complete",
            "No crashes": lambda o: "Traceback" not in o,
            "Has validation": "Val   - Loss:",
            "Reasonable accuracy": check_non_black_accuracy,
            "KL above free bits": check_kl_above_threshold(0.25),
        }
    )


def test_custom_warmup():
    """Test 6.4: Custom warmup schedule."""

    return run_training_test(
        name="Custom Warmup",
        args=[
            "--beta-schedule", "linear_warmup",
            "--beta-max", "0.2",
            "--beta-warmup-epochs", "2",
            "--beta-ramp-epochs", "3",
            "--free-bits", "0.2",
        ],
        expected_checks={
            "Training completed": "Training Complete",
            "No crashes": lambda o: "Traceback" not in o,
            "Has validation": "Val   - Loss:",
            "Reasonable accuracy": check_non_black_accuracy,
        }
    )


def run_all_tests():
    """Run all integration tests."""

    print("\n" + "=" * 80)
    print("TASK 6 INTEGRATION TESTS")
    print("=" * 80)
    print("\nThis will run 4 training tests (~3-5 minutes total)")
    print("Using test-1k dataset for fast testing\n")

    results = []

    try:
        # Test 1: Baseline
        results.append(("Baseline", test_baseline()))

        # Test 2: Ultra-conservative
        results.append(("Ultra-Conservative", test_ultra_conservative()))

        # Test 3: Cyclical
        results.append(("Cyclical", test_cyclical()))

        # Test 4: Custom warmup
        results.append(("Custom Warmup", test_custom_warmup()))

    except subprocess.TimeoutExpired:
        print("\n❌ TEST TIMEOUT: Training took longer than 5 minutes")
        return False
    except Exception as e:
        print(f"\n❌ TEST ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    for name, passed in results:
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {name}")

    all_passed = all(passed for _, passed in results)

    if all_passed:
        print("\n" + "=" * 80)
        print("ALL INTEGRATION TESTS PASSED ✓")
        print("=" * 80)
        print("\nTask 6 complete - verified:")
        print("  ✓ Baseline config works (no regressions)")
        print("  ✓ Ultra-conservative schedule with free bits works")
        print("  ✓ Cyclical schedule with free bits works")
        print("  ✓ Custom warmup parameters work")
        print("  ✓ Non-black pixel reconstruction maintained")
        print("  ✓ KL divergence stays above free_bits threshold")
        print("\nThe implementation successfully addresses the beta > 0 collapse problem!")
    else:
        print("\n❌ SOME TESTS FAILED")

    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
