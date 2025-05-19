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

Export an STL file with 3D numbers marking each face:

```bash
analyse scene-sproxel.csv --output-stl output.stl --number 3
```

Or import the package in your own Python code:

```python
from sproxel_analyser.analyse import load_voxel_data, visualize_voxel_model
from sproxel_analyser.stl_exporter import export_visible_faces_to_stl

# Load data from a CSV file
voxel_array = load_voxel_data('scene-sproxel.csv')

# Visualize the 3D model
visualize_voxel_model(voxel_array)

# Export to STL with 3D number shapes (default is 3)
export_visible_faces_to_stl(voxel_array, 'output.stl', number=5)
```

## Requirements

- Python 3.7+
- NumPy
- Pandas
- Matplotlib
- numpy-stl (for STL export)
