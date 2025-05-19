"""
Tools for analyzing and visualizing 3D voxel models in the Sproxel format.
"""

__version__ = '0.1.0'

# Expose key functionality at the package level
from sproxel_analyser.analyse import (
    load_voxel_data,
    count_visible_faces,
    visualize_voxel_model,
    export_visible_faces_to_stl
)
from sproxel_analyser.stl_exporter import export_visible_faces_to_stl
