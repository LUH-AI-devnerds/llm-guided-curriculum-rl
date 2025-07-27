#!/usr/bin/env python3

import subprocess
import sys
import os
from pathlib import Path


def run_command(command, check=True):
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
    project_root = Path(__file__).parent.parent
    env_file = project_root / "environment.yml"

    print("Updating conda environment.yml file with cross-platform compatibility...")

    current_env = run_command("conda info --envs | grep '*' | awk '{print $1}'")
    if not current_env:
        print("Error: Could not determine current conda environment")
        sys.exit(1)

    print(f"Current conda environment: {current_env}")

    print("Exporting conda packages...")
    conda_packages = run_command(
        "conda list --export | grep -v '^#' | grep -v '^@' | cut -d'=' -f1 | grep -v '^$'"
    )

    if not conda_packages:
        print("Error: Failed to export conda packages")
        sys.exit(1)

    print("Exporting pip packages...")
    pip_packages = run_command("pip list --format=freeze | grep -v '^#'")

    content = [
        "name: llm-guided-curriculum-rl",
        "channels:",
        "  - conda-forge",
        "  - defaults",
        "dependencies:",
    ]

    for package in conda_packages.split("\n"):
        if package.strip():
            content.append(f"  - {package.strip()}")

    if pip_packages:
        content.append("  - pip:")
        for package in pip_packages.split("\n"):
            if package.strip():
                content.append(f"    - {package.strip()}")

    with open(env_file, "w") as f:
        f.write("\n".join(content))

    print(f"✅ Successfully updated {env_file} with cross-platform compatibility")
    print("You can now commit and push the changes to trigger the GitHub Action.")


if __name__ == "__main__":
    update_environment_yml()
