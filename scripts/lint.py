#!/usr/bin/env python3
"""
Linting script for the RAG Transformer project
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
    """Main linting function"""
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Go to project root

    success = True

    # Check Python syntax
    if not run_command(f"{sys.executable} -m py_compile src/*.py", "Python syntax check"):
        success = False

    # Run flake8 if installed
    try:
        import flake8  # noqa: F401
        if not run_command(f"{sys.executable} -m flake8 src/ --max-line-length=100 --ignore=E203,W503", "Flake8 linting"):
            success = False
    except ImportError:
        print("Flake8 not installed, skipping...")

    # Run mypy if installed
    try:
        import mypy  # noqa: F401
        if not run_command(f"{sys.executable} -m mypy src/ --ignore-missing-imports", "MyPy type checking"):
            success = False
    except ImportError:
        print("MyPy not installed, skipping...")

    if success:
        print("\n✓ All linting checks passed!")
        return 0
    else:
        print("\n✗ Some linting checks failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
