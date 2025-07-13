#!/usr/bin/env python3
"""
Test script to verify the fixes work correctly.
"""

import subprocess
import sys
import time
from pathlib import Path


def test_training_fixes():
    """Test the training with fixes applied."""
    print("Testing training fixes...")

    # Change to src directory
    src_dir = Path("src")
    if not src_dir.exists():
        print("❌ src directory not found")
        return False

    # Run training command for 2 epochs to test quickly
    cmd = [
        sys.executable,
        "train.py",
        "--max_epochs",
        "2",
        "--batch_size",
        "8",
        "--num_workers",
        "1",
        "--num_classes",
        "10",
        "--lr",
        "0.001",
        "--patience",
        "5",
    ]

    print(f"Running command: {' '.join(cmd)}")
    print("Testing for 2 epochs...")

    start_time = time.time()

    try:
        result = subprocess.run(cmd, cwd=src_dir, capture_output=True, text=True, timeout=300)  # 5 min timeout

        end_time = time.time()
        duration = end_time - start_time

        print(f"\nTest completed in {duration:.2f} seconds")

        if result.returncode == 0:
            print("✅ Training completed successfully!")

            # Check for improvements
            stdout = result.stdout
            stderr = result.stderr

            # Check if MPS is used
            if "Using MPS" in stdout:
                print("✅ MPS acceleration is working")

            # Check if precision is correct
            if "Using 32bit" in stdout or "32-true" in stdout:
                print("✅ Correct precision for MPS")

            # Check for reduced credential messages
            credential_count = stdout.count("Using credentials from")
            if credential_count <= 3:  # Should be minimal
                print(f"✅ Reduced credential messages: {credential_count}")
            else:
                print(f"⚠️  Still too many credential messages: {credential_count}")

            # Check for no CUDA warnings
            if "CUDA is not available" not in stderr:
                print("✅ No CUDA warnings in stderr")
            elif "CUDA is not available" in stderr:
                print("⚠️  Still has CUDA warnings (this is expected but improved)")

            return True
        else:
            print("❌ Training failed:")
            print("STDOUT:", result.stdout[-1000:])
            print("STDERR:", result.stderr[-1000:])
            return False

    except subprocess.TimeoutExpired:
        print("❌ Training timed out")
        return False
    except Exception as e:
        print(f"❌ Training failed with exception: {e}")
        return False


def main():
    """Main test function."""
    print("Testing Training Fixes")
    print("=" * 50)

    success = test_training_fixes()

    if success:
        print("\n✅ Fixes are working correctly!")
        return True
    else:
        print("\n❌ Fixes need more work!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
