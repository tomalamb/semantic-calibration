"""Setup script for open-uncertainty package."""

from setuptools import setup, find_packages

# Read the requirements from requirements.txt
# Skip lines that start with # or -e (editable installs)
with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [
        line.strip() for line in f 
        if line.strip() 
        and not line.startswith("#") 
        and not line.startswith("-e")
    ]

setup(
    name="open-uncertainty",
    version="0.1.0",
    author="Tom Lamb",
    description="Calibration and uncertainty quantification for Large Language Models",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/tomalamb/open-uncertainty",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "head-temp-training=src.head_temp_training:main",
            "scalar-temp-training=src.scalar_temp_training:main",
        ],
    },
)
