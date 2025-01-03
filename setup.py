from pathlib import Path
from setuptools import setup, find_packages

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="wah",
    version="1.11.2",
    description="a library so simple you will learn Within An Hour",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Juyeop Kim",
    author_email="juyeopkim@yonsei.ac.kr",
    url="https://github.com/yupeeee/WAH",
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "setuptools>=61.0",
        "lightning",
        "matplotlib",
        "numpy",
        "pandas",
        "pyperclip",
        "PyYAML",
        "selenium",
        "tensorboard",
        "timm",
        "torch",
        "torchaudio",
        "torchmetrics",
        "torchvision",
        "webdriver_manager",
    ],
)
