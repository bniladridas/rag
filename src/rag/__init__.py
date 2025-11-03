import logging
import os
import subprocess
from importlib.metadata import PackageNotFoundError, version
from typing import Optional

# Import main at the top to avoid E402, but use a different name to avoid circular imports
from .__main__ import main as _main  # noqa: F401

# Set up logging
logger = logging.getLogger(__name__)

# Re-export main as part of the public API
main = _main


def get_version_from_git() -> Optional[str]:
    """Get version from git tag or return None if not available.

    Returns:
        Optional[str]: Version string (without 'v' prefix) or None if not available.
    """
    try:
        # Check if we're in a git repository
        if not os.path.exists(".git") and not os.environ.get("GIT_DIR"):
            return None

        # Get the most recent tag that is reachable from the current commit
        tag = (
            subprocess.check_output(
                ["git", "describe", "--tags", "--abbrev=0"],
                stderr=subprocess.PIPE,
                timeout=5,  # Prevent hanging if git is slow
            )
            .decode("utf-8")
            .strip()
        )

        if not tag:
            return None

        # Remove 'v' prefix if present (e.g., v1.0.0 -> 1.0.0)
        return tag[1:] if tag.startswith("v") else tag

    except subprocess.TimeoutExpired:
        logger.warning("Git version check timed out")
    except subprocess.CalledProcessError as e:
        if e.returncode != 128:  # 128 means no tags found, which is expected
            logger.warning(f"Git command failed: {e.stderr.decode('utf-8').strip()}")
    except Exception as e:
        logger.warning(f"Error getting version from git: {str(e)}")

    return None


# Allow version to be overridden by environment variable
__version__ = os.environ.get("RAG_VERSION")

if not __version__:
    # First try to get version from git tag (for development and releases)
    __version__ = get_version_from_git()

    # If not in a git repo or no tags, fall back to package metadata
    if not __version__:
        try:
            __version__ = version("rag")
        except PackageNotFoundError:
            # Fallback for development installs without package metadata
            __version__ = "0.0.0-dev"
            logger.debug("Using development version (0.0.0-dev)")

__all__ = ["__version__", "main"]
