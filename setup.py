#!/usr/bin/env python3

import os
import sys
from setuptools import setup, find_packages


def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return "LLM-Guided Curriculum Learning for Reinforcement Learning"


def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
    if os.path.exists(requirements_path):
        with open(requirements_path, "r", encoding="utf-8") as f:
            return [
                line.strip() for line in f if line.strip() and not line.startswith("#")
            ]
    return []


PACKAGE_NAME = "llm-guided-curriculum-rl"
PACKAGE_VERSION = "1.0.0"
PACKAGE_DESCRIPTION = "LLM-Guided Curriculum Learning for Reinforcement Learning"
PACKAGE_LONG_DESCRIPTION = read_readme()
PACKAGE_AUTHOR = "Amirreza Alasti"
PACKAGE_AUTHOR_EMAIL = "amirreza.alasti@example.com"
PACKAGE_URL = "https://github.com/amirrezaalasti/llm-guided-curriculum-rl"
PACKAGE_LICENSE = "MIT"
PACKAGE_CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "Intended Audience :: Education",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Scientific/Engineering :: Information Analysis",
    "Topic :: Games/Entertainment :: Simulation",
    "Topic :: Education",
    "Framework :: Jupyter",
]

PACKAGE_KEYWORDS = [
    "reinforcement-learning",
    "curriculum-learning",
    "large-language-models",
    "llm",
    "blackjack",
    "multi-agent",
    "deep-learning",
    "neural-networks",
    "q-learning",
    "dqn",
    "artificial-intelligence",
    "machine-learning",
    "education",
    "simulation",
    "gaming",
]

PROJECT_URLS = {
    "Bug Reports": "https://github.com/amirrezaalasti/llm-guided-curriculum-rl/issues",
    "Source": "https://github.com/amirrezaalasti/llm-guided-curriculum-rl",
    "Documentation": "https://github.com/amirrezaalasti/llm-guided-curriculum-rl/blob/main/README.md",
    "Changelog": "https://github.com/amirrezaalasti/llm-guided-curriculum-rl/blob/main/CHANGELOG.md",
}

PYTHON_REQUIRES = ">=3.8"

INSTALL_REQUIRES = read_requirements()

EXTRAS_REQUIRE = {
    "dev": [
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
        "black>=23.0.0",
        "flake8>=6.0.0",
        "mypy>=1.0.0",
        "pre-commit>=3.0.0",
        "jupyter>=1.0.0",
        "ipykernel>=6.0.0",
    ],
    "docs": [
        "sphinx>=6.0.0",
        "sphinx-rtd-theme>=1.0.0",
        "myst-parser>=1.0.0",
    ],
    "analysis": [
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "pandas>=1.5.0",
        "plotly>=5.0.0",
        "scikit-learn>=1.0.0",
    ],
    "full": [
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
        "black>=23.0.0",
        "flake8>=6.0.0",
        "mypy>=1.0.0",
        "pre-commit>=3.0.0",
        "jupyter>=1.0.0",
        "ipykernel>=6.0.0",
        "sphinx>=6.0.0",
        "sphinx-rtd-theme>=1.0.0",
        "myst-parser>=1.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "pandas>=1.5.0",
        "plotly>=5.0.0",
        "scikit-learn>=1.0.0",
    ],
}

PACKAGE_DATA = {
    "": ["*.md", "*.txt", "*.yml", "*.yaml", "*.json"],
}

ENTRY_POINTS = {
    "console_scripts": [
        "blackjack-rl=scripts.RLAgent:main",
        "blackjack-gui=blackjack_gui:main",
        "analyze-logs=scripts.analyze_logs:main",
        "curriculum-train=scripts.curriculum_multi_agent_rl:main",
    ],
}

PACKAGES = find_packages(include=["scripts", "scripts.*"])

INCLUDE_PACKAGE_DATA = True

ZIP_SAFE = False


def main():
    setup(
        name=PACKAGE_NAME,
        version=PACKAGE_VERSION,
        description=PACKAGE_DESCRIPTION,
        long_description=PACKAGE_LONG_DESCRIPTION,
        long_description_content_type="text/markdown",
        author=PACKAGE_AUTHOR,
        author_email=PACKAGE_AUTHOR_EMAIL,
        url=PACKAGE_URL,
        project_urls=PROJECT_URLS,
        license=PACKAGE_LICENSE,
        classifiers=PACKAGE_CLASSIFIERS,
        keywords=PACKAGE_KEYWORDS,
        python_requires=PYTHON_REQUIRES,
        packages=PACKAGES,
        include_package_data=INCLUDE_PACKAGE_DATA,
        package_data=PACKAGE_DATA,
        install_requires=INSTALL_REQUIRES,
        extras_require=EXTRAS_REQUIRE,
        entry_points=ENTRY_POINTS,
        zip_safe=ZIP_SAFE,
        # Additional metadata
        platforms=["any"],
        requires_python=PYTHON_REQUIRES,
        # Development and testing
        test_suite="tests",
        # Package discovery
        package_dir={"": "."},
    )


if __name__ == "__main__":
    main()
