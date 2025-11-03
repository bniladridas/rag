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
        result = subprocess.run(
            command, shell=True, check=True, capture_output=True, text=True
        )
        print(f"PASS: {description} passed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"FAIL: {description} failed:")
        print(e.stdout)
        print(e.stderr)
        return False


def main():
    """Main test function"""
    os.chdir(os.path.join(os.path.dirname(__file__), ".."))  # Go to project root

    success = True

    # Run pytest
    if not run_command(f"{sys.executable} -m pytest tests/ -v", "Unit and E2E tests"):
        success = False

    # Run with coverage if installed
    try:
        import pytest_cov  # noqa: F401

        if not run_command(
            f"{sys.executable} -m pytest tests/ --cov=src --cov-report=term",
            "Tests with coverage",
        ):
            success = False
    except ImportError:
        print("pytest-cov not installed, skipping coverage...")

    if success:
        print("\nAll tests passed!")
        return 0
    else:
        print("\nSome tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
