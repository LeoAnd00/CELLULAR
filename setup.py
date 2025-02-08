from pathlib import Path
from setuptools import setup, find_packages

setup(
    name="CELLULAR",
    version="0.0.1",
    author="Leo Andrekson, Rocío Mercado",
    author_email="leo.andrekson@gmail.com, rocom@chalmers.se",
    description="A package for generating an embedding space from scRNA-Seq. This space can be used for cell type annotation, novel cell type detection, cell type representations, and visualization.",
    long_description="Read README.md",
    long_description_content_type="text/markdown",
    url="https://github.com/LeoAnd00/CELLULAR",
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.10.5",
    packages=find_packages(),
    install_requires=[
        l.strip() for l in Path('requirements.txt').read_text('utf-8').splitlines()
        if l.strip() and not l.startswith("--")  # Ignore empty lines and options like --extra-index-url
    ]
)