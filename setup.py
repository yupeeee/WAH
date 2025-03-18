from pathlib import Path

from setuptools import find_packages, setup

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="wah",
    version="1.13.18",
    description="a library so simple you will learn Within An Hour",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Juyeop Kim",
    author_email="juyeopkim@yonsei.ac.kr",
    url="https://github.com/yupeeee/WAH",
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    entry_points={
        "console_scripts": [
            "wah=wah.cli:main",
        ],
    },
    install_requires=[
        "setuptools>=61.0",
        "accelerate",
        "bitsandbytes",
        "diffusers",
        "ftfy",
        "lightning",
        "matplotlib",
        "numpy",
        "pandas",
        "PyYAML",
        "pillow",
        "regex",
        "requests",
        "sentencepiece",
        "tensorboard",
        "timm",
        "torch",
        "torchmetrics",
        "torchvision",
        "tqdm",
        "transformers",
    ],
)
