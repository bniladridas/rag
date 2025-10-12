#!/usr/bin/env python3
"""
Test runner script for the RAG Transformer project
"""
import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and return success"""
    print(f"Running {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} passed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed:")
        print(e.stdout)
        print(e.stderr)
        return False

def main():
    """Main test function"""
    os.chdir(os.path.join(os.path.dirname(__file__), '..'))  # Go to project root

    success = True

    # Run pytest
    success &= run_command("python3 -m pytest tests/ -v", "Unit and E2E tests")

    # Run with coverage if available
    try:
        import pytest_cov
        success &= run_command("python3 -m pytest tests/ --cov=src --cov-report=term", "Tests with coverage")
    except ImportError:
        print("pytest-cov not installed, skipping coverage...")

    if success:
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())