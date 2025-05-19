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
analyse scene-sproxel.csv --output-stl output.stl
```

The STL output contains two numbers on each face:
- **Top number**: Tile category (0-6), sized 30% smaller to fit on the face
  - **0**: Standard tile (no modifications needed)
  - **1**: Edge tile with one 45° cut
  - **2**: Edge tile with two 45° cuts
  - **3**: Edge tile with three 45° cuts
  - **4**: Corner tile with one corner
  - **5**: Corner tile with two corners
  - **6**: Complex tile (multiple modifications)
  
- **Bottom number**: Color category (00-31), sized 50% smaller with two digits
  - Each voxel's color is categorized into one of 32 possible colors
  - Always displayed as a two-digit number (e.g., "01" instead of just "1")
  - Colors are automatically detected and categorized from the source data

You can also specify a default number (0-9) to use if categorization fails:

```bash
analyse scene-sproxel.csv --output-stl output.stl --number 3
```

Or import the package in your own Python code:

```python
from sproxel_analyser import load_voxel_data, visualize_voxel_model, export_visible_faces_to_stl

# Load data from a CSV file
voxel_array = load_voxel_data('scene-sproxel.csv')

# Visualize the 3D model
visualize_voxel_model(voxel_array)

# Export to STL with dual numbers for tile category and color
export_visible_faces_to_stl(voxel_array, 'output.stl')
```

## Requirements

- Python 3.7+
- NumPy
- Pandas
- Matplotlib
- numpy-stl (for STL export)
