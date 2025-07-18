#!/usr/bin/env python3
"""
Script to update the conda environment.yml file with current dependencies.
This script exports the current conda environment and updates the environment.yml file
with cross-platform compatibility.
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(command, check=True):
    """Run a shell command and return the result."""
    try:
        result = subprocess.run(
            command, shell=True, check=check, capture_output=True, text=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error running command '{command}': {e}")
        if check:
            sys.exit(1)
        return None


def update_environment_yml():
    """Update the environment.yml file with current conda environment."""

    # Get the project root directory
    project_root = Path(__file__).parent.parent
    env_file = project_root / "environment.yml"

    print("Updating conda environment.yml file with cross-platform compatibility...")

    # Check if we're in the correct conda environment
    current_env = run_command("conda info --envs | grep '*' | awk '{print $1}'")
    if not current_env:
        print("Error: Could not determine current conda environment")
        sys.exit(1)

    print(f"Current conda environment: {current_env}")

    # Get conda packages (only package names, no versions or build strings)
    print("Exporting conda packages...")
    conda_packages = run_command(
        "conda list --export | grep -v '^#' | grep -v '^@' | cut -d'=' -f1 | grep -v '^$'"
    )

    if not conda_packages:
        print("Error: Failed to export conda packages")
        sys.exit(1)

    # Get pip packages
    print("Exporting pip packages...")
    pip_packages = run_command("pip list --format=freeze | grep -v '^#'")

    # Create the new environment.yml content
    content = [
        "name: llm-guided-curriculum-rl",
        "channels:",
        "  - conda-forge",
        "  - defaults",
        "dependencies:",
    ]

    # Add conda packages (only names, no versions for cross-platform compatibility)
    for package in conda_packages.split("\n"):
        if package.strip():
            content.append(f"  - {package.strip()}")

    # Add pip packages if any
    if pip_packages:
        content.append("  - pip:")
        for package in pip_packages.split("\n"):
            if package.strip():
                content.append(f"    - {package.strip()}")

    # Write the updated environment.yml
    with open(env_file, "w") as f:
        f.write("\n".join(content))

    print(f"✅ Successfully updated {env_file} with cross-platform compatibility")
    print("You can now commit and push the changes to trigger the GitHub Action.")


if __name__ == "__main__":
    update_environment_yml()
