from setuptools import setup, find_packages

setup(
    name="sproxel-analyser",
    description="A package for analyzing Sproxel data.",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "matplotlib>=3.5.0",
        "numpy-stl>=2.16.0",
    ],
    entry_points={
        "console_scripts": [
            "analyse=sproxel_analyser.analyse:main",
        ],
    },
)
