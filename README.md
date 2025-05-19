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

Export an STL file with indicators on each face:

```bash
analyse scene-sproxel.csv --output-stl output.stl
```

There are two display modes available for the STL:

### Numbers Mode (default)

```bash
analyse scene-sproxel.csv --output-stl output.stl --display-mode numbers
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

### Cuboids Mode

```bash
analyse scene-sproxel.csv --output-stl output.stl --display-mode cuboids
```

In cuboids mode, each face displays a 2×2 grid of cuboids:
- Each cuboid is 4mm wide with a constant 3mm thickness
- 2mm gap between cuboids
- Cuboids are oriented with their thinner face parallel to the tile face
- No tile plane is created, only the 4 cuboids are present
- Cuboids are placed 3mm inside the tile (inset from the face)
- Top row: Represents tile category (left and right cuboids)
- Bottom row: Represents color category (left and right cuboids)

You can also specify a default number (0-9) to use if categorization fails:

```bash
analyse scene-sproxel.csv --output-stl output.stl --number 3
```

## Python API

```python
from sproxel_analyser import load_voxel_data, visualize_voxel_model, export_visible_faces_to_stl

# Load data from a CSV file
voxel_array = load_voxel_data('scene-sproxel.csv')

# Visualize the 3D model
visualize_voxel_model(voxel_array)

# Export to STL with numbers or cuboids
export_visible_faces_to_stl(voxel_array, 'output.stl', display_mode='numbers')
# or
export_visible_faces_to_stl(voxel_array, 'output.stl', display_mode='cuboids')
```

## Requirements

- Python 3.7+
- NumPy
- Pandas
- Matplotlib
- numpy-stl (for STL export)
