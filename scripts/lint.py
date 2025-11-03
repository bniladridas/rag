#!/usr/bin/env python3
"""
Linting script for the RAG Transformer project
"""
import subprocess
import sys
import os


def run_command(command, description):
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
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    success = True
    success &= run_command("python3 -m py_compile src/rag/*.py", "Python syntax check")

    try:
        import flake8

        success &= run_command(
            "python3 -m flake8 src/rag/ --max-line-length=100 --ignore=E203,W503",
            "Flake8 linting",
        )
    except ImportError:
        print("Flake8 not installed, skipping...")

    try:
        import mypy

        success &= run_command(
            "python3 -m mypy src/rag/ --ignore-missing-imports", "MyPy type checking"
        )
    except ImportError:
        print("MyPy not installed, skipping...")

    if success:
        print("\nAll linting checks passed!")
        return 0
    else:
        print("\nSome linting checks failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
