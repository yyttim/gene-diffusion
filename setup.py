"""Setup script for GeneDiffusion"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="gene-diffusion",
    version="0.1.0",
    author="GeneDiffusion Contributors",
    author_email="your.email@institution.edu",
    description="Masked Diffusion Model for Mammalian Genomic Sequences",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yyttim/gene-diffusion",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": ["pytest>=7.0.0", "black>=22.0.0", "flake8>=4.0.0"],
    },
    entry_points={
        "console_scripts": [
            "gene-diffusion-train=src.train:main",
            "gene-diffusion-demo=demo:main",
        ],
    },
) 