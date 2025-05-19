# Sproxel Analyser

A Python package for analyzing and visualizing 3D voxel models from Sproxel CSV files.

## Installation

```bash
pip install -e .
```

## Usage

You can run the analysis tool on a CSV file:

```bash
analyse scene-sproxel.csv
```

Or import the package in your own Python code:

```python
from sproxel_analyser.analyse import load_voxel_data, visualize_voxel_model

# Load data from a CSV file
voxel_array = load_voxel_data('scene-sproxel.csv')

# Visualize the 3D model
visualize_voxel_model(voxel_array)
```

## Requirements

- Python 3.7+
- NumPy
- Pandas
- Matplotlib
