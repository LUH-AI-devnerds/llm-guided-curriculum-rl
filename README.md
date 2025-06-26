# LLM-Guided Curriculum Reinforcement Learning

This repository contains code for LLM-guided curriculum reinforcement learning experiments.

## Environment Setup

This project uses conda for environment management. The `environment.yml` file contains all the necessary dependencies and is designed to be **cross-platform compatible** (works on Windows, macOS, and Linux).

### Automatic Environment Updates

The repository includes automation to keep the conda environment file up-to-date:

#### GitHub Actions Workflow
- **Location**: `.github/workflows/update-environment.yml`
- **Trigger**: Automatically runs on every push to `main` or `master` branch
- **Function**: Updates `environment.yml` with current dependencies and commits the changes
- **Cross-Platform**: Generates environment files compatible with Windows, macOS, and Linux

#### Local Update Scripts
You can also manually update the environment file using the provided scripts:

```bash
# Using the shell script wrapper (recommended)
./scripts/update_env.sh

# Or directly with Python
python scripts/update_environment.py
```

### Manual Environment Setup
```bash
# Create the conda environment
conda env create -f environment.yml

# Activate the environment
conda activate llm-guided-curriculum-rl
```

## Cross-Platform Compatibility

The environment automation ensures that:
- ✅ **No platform-specific build strings** (like `h99b78c6_7`, `hbd8a1cb_0`)
- ✅ **No absolute paths** (like `/opt/anaconda3/envs/...`)
- ✅ **Minimal version pinning** for maximum compatibility
- ✅ **Works on Windows, macOS, and Linux**

## Project Structure
```
llm-guided-curriculum-rl/
├── .github/workflows/    # GitHub Actions workflows
├── scripts/              # Utility scripts
├── docs/                 # Documentation
├── environment.yml       # Cross-platform conda environment specification
└── README.md            # This file
```

## Contributing

When you add new dependencies to your local environment, the automation will automatically update the `environment.yml` file on the next push to GitHub. This ensures that all collaborators have access to the same environment setup across different operating systems.