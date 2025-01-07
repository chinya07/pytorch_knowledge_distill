"""Setup configuration for pytorch-knowledge-distill package."""

from setuptools import setup, find_packages

setup(
    name="pytorch_knowledge_distill",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A simple PyTorch implementation of Knowledge Distillation",
    url="https://github.com/yourusername/pytorch-knowledge-distill",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.7.0",
        "torchvision>=0.8.0",
    ],
) 