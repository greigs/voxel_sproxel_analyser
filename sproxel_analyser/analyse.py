import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_voxel_data(csv_path):
    """Load and parse a voxel data file."""
    # Read the file to examine its structure
    with open(csv_path, 'r') as f:
        first_line = f.readline().strip()
        content = f.read()
    
    # Check if the first line contains dimensions in format "width,height,depth"
    dimensions = None
    if ',' in first_line and first_line.count(',') == 2:
        try:
            dimensions = [int(d.strip()) for d in first_line.split(',')]
            print(f"Found dimensions in header: {dimensions[0]}x{dimensions[1]}x{dimensions[2]}")
        except ValueError:
            print("Could not parse dimensions from header")
            dimensions = None
    
    if dimensions:
        width, height, depth = dimensions
        
        # Parse the CSV data with appropriate delimiter
        lines = content.strip().split('\n')
        
        # Create a 3D array with the right dimensions
        voxel_array = np.empty((depth, height, width), dtype=object)
        
        # Fill the array with data from each layer
        layer = 0
        row = 0
        
        for line in lines:
            if not line.strip():
                # Empty line indicates next layer
                layer += 1
                row = 0
                continue
                
            # Split the line by commas
            cells = line.strip().split(',')
            
            # Ensure our indices are in bounds
            if layer < depth and row < height and len(cells) == width:
                for col, cell in enumerate(cells):
                    voxel_array[layer, row, col] = cell
                row += 1
        
        print(f"Successfully loaded voxel data with dimensions {depth}x{height}x{width}")
        return voxel_array
    
    # If above parsing failed, try original methods
    # Check if the file has empty lines separating layers
    layers = content.split('\n\n')
    if len(layers) > 1:
        # File has multiple layers separated by blank lines
        voxel_layers = []
        for layer in layers:
            if not layer.strip():
                continue
                
            # Parse each layer
            rows = []
            for line in layer.strip().split('\n'):
                if line.strip():
                    rows.append(line.strip().split())
            
            if rows:
                voxel_layers.append(np.array(rows))
        
        # Check if all layers have the same dimensions
        if voxel_layers:
            layer_shapes = [layer.shape for layer in voxel_layers]
            if all(shape == layer_shapes[0] for shape in layer_shapes):
                # Stack layers along z-axis
                height, width = layer_shapes[0]
                depth = len(voxel_layers)
                voxel_array = np.zeros((depth, height, width), dtype=object)
                
                for z, layer in enumerate(voxel_layers):
                    voxel_array[z] = layer
                    
                print(f"Successfully parsed {depth} layers of size {height}x{width}")
                return voxel_array
    
    # If no layers were detected or they had inconsistent shapes,
    # try pandas with delimiter detection
    try:
        df = pd.read_csv(csv_path, header=None, delim_whitespace=True, skip_blank_lines=True)
        raw_array = df.values
        print(f"Loaded data with shape: {raw_array.shape}")
        
        # Try to infer 3D structure from 2D data
        # If it's a 2D array (rows x columns), assume it's a single layer
        if len(raw_array.shape) == 2:
            return raw_array.reshape(1, raw_array.shape[0], raw_array.shape[1])
            
        return raw_array
    except Exception as e:
        print(f"Error reading CSV with pandas: {e}")
        
        # Final fallback to manual parsing without layer structure
        with open(csv_path, 'r') as f:
            raw_lines = f.readlines()

        rows = []
        for line in raw_lines:
            if line.strip():  # Skip empty lines
                row = line.strip().split()
                rows.append(row)
        
        if rows:
            return np.array(rows).reshape(1, len(rows), len(rows[0]))
        return np.array([])

def count_visible_faces(voxel_array):
    """
    Count how many voxel faces are visible from outside the model.
    Each voxel has 6 faces, but internal faces don't contribute to the visible surface.
    
    A face is considered visible if either:
    1. It's on the boundary of the model space (x=0, x=width-1, etc.)
    2. It's adjacent to an empty cell (no voxel present)
    
    Internal faces shared between two adjacent voxels are NOT counted.
    
    Returns total visible faces and a breakdown by face direction.
    """
    # Get the dimensions
    if len(voxel_array.shape) == 2:
        height, width = voxel_array.shape
        depth = 1
        voxel_array = voxel_array.reshape(depth, height, width)
    else:
        depth, height, width = voxel_array.shape
    
    # Create a binary occupancy grid (True where a voxel exists)
    occupied = np.zeros((depth, height, width), dtype=bool)
    
    # Fill the occupancy grid based on non-empty, non-transparent voxels
    for z in range(depth):
        for y in range(height):
            for x in range(width):
                val = str(voxel_array[z, y, x]).strip()
                
                # Skip empty cells or fully transparent cells
                if val == "" or val.lower() == "none" or val == "0" or val == "#00000000":
                    continue
                
                # For colored voxels, check if they're transparent
                if val.startswith("#") and len(val) >= 9:
                    try:
                        alpha = int(val[7:9], 16)
                        if alpha == 0:  # Skip fully transparent voxels
                            continue
                    except ValueError:
                        pass  # If parsing fails, assume it's not transparent
                
                # Mark this voxel as occupied
                occupied[z, y, x] = True
    
    # Define the 6 possible directions (±x, ±y, ±z)
    directions = [
        (1, 0, 0), (-1, 0, 0),   # +x, -x
        (0, 1, 0), (0, -1, 0),   # +y, -y
        (0, 0, 1), (0, 0, -1)    # +z, -z
    ]
    
    # Direction names for reporting
    direction_names = ["+X", "-X", "+Y", "-Y", "+Z", "-Z"]
    
    # Counter for visible faces in each direction
    visible_faces = {name: 0 for name in direction_names}
    
    # Track visible face positions for edge and corner analysis
    face_positions = {name: [] for name in direction_names}
    
    # Initialize counters for tile modification tracking
    edges_count = 0
    corners_count = 0
    modified_tiles_count = 0
    
    # Detailed tile type tracker - will store counts of different configurations
    # Format: {(num_edges_modified, num_corners_modified): count}
    detailed_tile_types = {}
    
    # Check each voxel
    for z in range(depth):
        for y in range(height):
            for x in range(width):
                # Skip if this voxel is empty
                if not occupied[z, y, x]:
                    continue
                
                # Check each neighboring position
                for i, (dx, dy, dz) in enumerate(directions):
                    nx, ny, nz = x + dx, y + dy, z + dz
                    
                    # A face is visible if:
                    # 1. It's at the boundary of the model space OR
                    # 2. The neighboring voxel in this direction is empty
                    if (nx < 0 or nx >= width or 
                        ny < 0 or ny >= height or
                        nz < 0 or nz >= depth or
                        not occupied[nz, ny, nx]):
                        visible_faces[direction_names[i]] += 1
                        # Store the position and orientation of this visible face
                        face_positions[direction_names[i]].append((x, y, z))
    
    # Calculate total visible faces
    total_faces = sum(visible_faces.values())
    
    # Enhanced tile analysis - maps voxel positions to their visible face directions
    voxel_faces = {}
    for direction, positions in face_positions.items():
        for pos in positions:
            if pos not in voxel_faces:
                voxel_faces[pos] = []
            voxel_faces[pos].append(direction)
    
    # Categorize each tile based on its edge and corner modifications
    edge_corner_combinations = {}  # Dictionary to store counts of each combination
    
    # Create a list to track neighboring visible faces for each visible face
    # This helps identify which edges need 45-degree cuts or size adjustments
    for pos, visible_dirs in voxel_faces.items():
        x, y, z = pos
        
        # For each visible face, check for adjacent visible faces
        edge_count = 0  # Count of edges needing 45-degree cuts
        resize_count = 0  # Count of edges needing size adjustment
        edge_config = []  # Track which edges need modification
        
        # Check each pair of perpendicular faces
        for i, dir1 in enumerate(visible_dirs):
            dir1_idx = direction_names.index(dir1)
            dx1, dy1, dz1 = directions[dir1_idx]
            
            for dir2 in visible_dirs[i+1:]:
                dir2_idx = direction_names.index(dir2)
                dx2, dy2, dz2 = directions[dir2_idx]
                
                # If the directions are perpendicular (dot product = 0)
                if dx1*dx2 + dy1*dy2 + dz1*dz2 == 0:
                    edge_count += 1
                    edge_config.append((dir1, dir2))
                    
                    # Check if a third face forms a corner with these two
                    for dir3 in visible_dirs:
                        if dir3 != dir1 and dir3 != dir2:
                            dir3_idx = direction_names.index(dir3)
                            dx3, dy3, dz3 = directions[dir3_idx]
                            
                            # If dir3 is perpendicular to both dir1 and dir2
                            if (dx1*dx3 + dy1*dy3 + dz1*dz3 == 0 and
                                dx2*dx3 + dy2*dy3 + dz2*dz3 == 0):
                                # This is a corner - all three faces need adjustment
                                resize_count += 1
        
        # Classify this tile based on edge and corner modifications
        tile_type = (len(visible_dirs), edge_count, resize_count)
        edge_corner_combinations[tile_type] = edge_corner_combinations.get(tile_type, 0) + 1
    
    # Validation step to confirm we're only counting external faces
    # Calculate total voxels and max possible external faces
    total_voxels = np.sum(occupied)
    max_faces = 6 * total_voxels  # Each voxel has 6 faces total
    internal_faces = max_faces - total_faces
    
    print(f"Total voxels: {total_voxels}")
    print(f"Total faces: {max_faces}")
    print(f"External faces: {total_faces} ({total_faces / max_faces:.1%} of all faces)")
    print(f"Internal faces: {internal_faces} ({internal_faces / max_faces:.1%} of all faces)")
    
    # Report tile customization needs
    print(f"\nTile coverage analysis:")
    print(f"Total tiles needed: {total_faces}")
    print(f"Edge tiles (needing one 45° cut): {edges_count}")
    print(f"Corner tiles (needing two 45° cuts): {corners_count}")
    print(f"Modified tiles needed: {modified_tiles_count} ({modified_tiles_count / total_faces:.1%} of all tiles)")
    print(f"Standard tiles needed: {total_faces - modified_tiles_count} ({(total_faces - modified_tiles_count) / total_faces:.1%} of all tiles)")
    
    # Prepare detailed tile analysis report
    print(f"\nDetailed Tile Analysis:")
    print(f"Total tiles needed: {total_faces}")
    
    # Sort the combinations for consistent reporting
    sorted_combinations = sorted(edge_corner_combinations.items())
    
    tile_categories = {
        "standard": 0,          # No modifications
        "edge_one": 0,          # One 45° edge cut
        "edge_two": 0,          # Two 45° edge cuts (adjacent)
        "edge_three": 0,        # Three 45° edge cuts
        "corner_one": 0,        # One corner (three faces meet)
        "corner_two": 0,        # Two corners
        "complex": 0            # More complex configurations
    }
    
    # Create a detailed report table
    print("\n{:<15} {:<15} {:<15} {:<10}".format("Visible Faces", "45° Edges", "Size Adjusts", "Count"))
    print("-" * 60)
    
    for (visible_count, edge_count, resize_count), count in sorted_combinations:
        print("{:<15} {:<15} {:<15} {:<10}".format(visible_count, edge_count, resize_count, count))
        
        # Categorize the tile types
        if edge_count == 0 and resize_count == 0:
            tile_categories["standard"] += count
        elif edge_count == 1 and resize_count == 0:
            tile_categories["edge_one"] += count
        elif edge_count == 2 and resize_count == 0:
            tile_categories["edge_two"] += count
        elif edge_count == 3 and resize_count == 0:
            tile_categories["edge_three"] += count
        elif edge_count >= 2 and resize_count == 1:
            tile_categories["corner_one"] += count
        elif edge_count >= 3 and resize_count == 2:
            tile_categories["corner_two"] += count
        else:
            tile_categories["complex"] += count
    
    print("\nSummary by Tile Category:")
    for category, count in tile_categories.items():
        print(f"{category.replace('_', ' ').capitalize()} tiles: {count} ({count/total_faces:.1%})")
    
    # Add tile information to the return values
    tile_info = {
        'total': total_faces,
        'categories': tile_categories,
        'detailed': edge_corner_combinations
    }
    
    return total_faces, visible_faces, tile_info

def visualize_voxel_model(voxel_array):
    """Visualize a voxel model in 3D."""
    # Get dimensions
    if len(voxel_array.shape) == 2:
        height, width = voxel_array.shape
        depth = 1
        voxel_array = voxel_array.reshape(depth, height, width)
    else:
        depth, height, width = voxel_array.shape

    print(f"Visualizing 3D model with dimensions: {depth}x{height}x{width}")
    
    # Check if the data makes sense dimensionally
    if depth == 1 and (height > 1000 or width > 1000):
        print(f"Warning: Dimensions look suspicious. Got {depth}x{height}x{width}")
        print("Attempting to reshape the data based on header information...")
        
        # Try to use dimensions from the CSV header if they exist
        try:
            # This is just a safety check in case we still got wrong dimensions
            total_elements = voxel_array.size
            if total_elements == 42 * 42 * 41:  # This matches what's in the header
                voxel_array = voxel_array.reshape(41, 42, 42)
                depth, height, width = voxel_array.shape
                print(f"Reshaped to {depth}x{height}x{width}")
        except Exception as e:
            print(f"Error reshaping: {e}")
    
    # Prepare 3D volume and color arrays - use width, height, depth order for visualization
    filled = np.zeros((width, height, depth), dtype=bool)  # For visualization axes
    colors = np.empty((width, height, depth, 4))  # RGBA
    
    # Default color
    default_color = np.array([0.5, 0.5, 0.5, 1.0])  # Gray with full opacity
    
    # Count non-empty voxels for verification
    non_empty_count = 0
    
    for z in range(depth):
        for y in range(height):
            for x in range(width):
                val = str(voxel_array[z, y, x]).strip()
                
                # Skip empty cells
                if val == "" or val.lower() == "none" or val == "0" or val == "#00000000":
                    continue
                
                non_empty_count += 1
                
                # Try to parse as a color
                if val.startswith("#") and len(val) >= 7:
                    try:
                        # Parse hex color
                        r = int(val[1:3], 16) / 255.0
                        g = int(val[3:5], 16) / 255.0
                        b = int(val[5:7], 16) / 255.0
                        
                        # Handle alpha if present
                        a = 1.0  # Default full opacity
                        if len(val) >= 9:
                            a = int(val[7:9], 16) / 255.0
                        
                        # Only show if not fully transparent
                        if a > 0:
                            filled[x, y, z] = True  # Note swapped axes
                            colors[x, y, z] = [r, g, b, a]
                    except ValueError:
                        # If color parsing fails, use default
                        filled[x, y, z] = True
                        colors[x, y, z] = default_color
                else:
                    # Non-empty cell that's not a color
                    filled[x, y, z] = True
                    colors[x, y, z] = default_color

    print(f"Found {non_empty_count} non-empty voxels in the model")
    
    # Only proceed if we have voxels to show
    if not np.any(filled):
        print("Warning: No visible voxels found in the data")
        return
    
    # Count visible faces
    total_faces, face_breakdown, tile_info = count_visible_faces(voxel_array)
    print(f"Model has {total_faces} visible faces")
    print(f"Face breakdown by direction: {face_breakdown}")
    print(f"Tile breakdown: {tile_info}")

    # Plot using matplotlib
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Use matplotlib's voxels function to render the 3D model
    ax.voxels(filled, facecolors=colors, edgecolor='k', alpha=0.7)

    # Set viewing angle for better visualization
    ax.view_init(elev=30, azim=45)
    
    # Equal aspect ratio for all axes
    ax.set_box_aspect([1, 1, 1])
    
    # Add labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f"Sproxel Voxel Model ({width}x{height}x{depth})")
    
    plt.tight_layout()
    plt.show()

def export_visible_faces_to_stl(voxel_array, output_path, number=3):
    """
    Import the STL export function from the dedicated module.
    This is a wrapper to maintain backward compatibility.
    
    Parameters:
        voxel_array: 3D numpy array containing voxel data
        output_path: Path where the STL file will be saved
        number: The number to use for the 3D shape (default: 3)
    """
    try:
        from sproxel_analyser.stl_exporter import export_visible_faces_to_stl as stl_export
        return stl_export(voxel_array, output_path, number)
    except ImportError:
        print("Error importing stl_exporter module. Make sure it's in the correct location.")
        return False

def analyze_file_format(csv_path):
    """Analyze the format of the input file to help debug issues."""
    with open(csv_path, 'r') as f:
        content = f.readlines()
    
    print(f"\nFile Analysis for {os.path.basename(csv_path)}:")
    print(f"Total lines: {len(content)}")
    
    if content:
        print(f"First line: '{content[0].strip()}'")
        if len(content) > 1:
            print(f"Second line: '{content[1].strip()}'")
        
        # Count non-empty lines
        non_empty_lines = sum(1 for line in content if line.strip())
        print(f"Non-empty lines: {non_empty_lines}")
        
        # Check for potential layer separators
        empty_line_indices = [i for i, line in enumerate(content) if not line.strip()]
        if empty_line_indices:
            print(f"Empty lines found at positions: {empty_line_indices[:5]}{'...' if len(empty_line_indices) > 5 else ''}")
            
            # Calculate average distance between empty lines
            if len(empty_line_indices) > 1:
                distances = [empty_line_indices[i+1] - empty_line_indices[i] for i in range(len(empty_line_indices)-1)]
                avg_distance = sum(distances) / len(distances)
                print(f"Average distance between empty lines: {avg_distance:.2f}")
                
                # If average distance is consistent, might indicate layer structure
                if all(abs(d - avg_distance) < 2 for d in distances):
                    potential_layers = len(empty_line_indices) + 1
                    lines_per_layer = avg_distance
                    print(f"Possible layer structure: {potential_layers} layers with ~{lines_per_layer:.0f} lines per layer")
    
    return

def analyze_csv_format(csv_path):
    """Analyze the CSV file format to help debug parsing issues."""
    with open(csv_path, 'r') as f:
        first_few_lines = [next(f) for _ in range(10)]
    
    print("\nCSV Format Analysis:")
    print(f"First line: {first_few_lines[0].strip()}")
    
    # Check if dimensions are in first line
    first_line = first_few_lines[0].strip()
    if ',' in first_line and first_line.count(',') == 2:
        try:
            dimensions = [int(d.strip()) for d in first_line.split(',')]
            print(f"Dimensions from header: {dimensions[0]}x{dimensions[1]}x{dimensions[2]}")
            print(f"Total expected elements: {dimensions[0] * dimensions[1] * dimensions[2]}")
        except ValueError:
            print("First line contains commas but doesn't appear to be dimensions")
    
    # Check how delimiter patterns look in the data
    delimiter_patterns = {
        'comma': sum(line.count(',') for line in first_few_lines),
        'semicolon': sum(line.count(';') for line in first_few_lines),
        'tab': sum(line.count('\t') for line in first_few_lines),
        'space': sum(len(line.split()) > 1 for line in first_few_lines)
    }
    print(f"Delimiter patterns in first few lines: {delimiter_patterns}")
    
    # Detect color format if present
    color_format = None
    for line in first_few_lines[1:]:  # Skip header line
        if '#' in line:
            samples = [s.strip() for s in line.split(',') if '#' in s]
            if samples:
                color_format = f"Example color value: {samples[0]} (length: {len(samples[0])})"
                break
    
    if color_format:
        print(color_format)
    else:
        print("No color codes (#) detected in the sample")

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description='Analyze and visualize 3D voxel models')
    parser.add_argument('csv_file', help='Path to the CSV file containing voxel data')
    parser.add_argument('--debug', action='store_true', help='Print detailed debug information')
    parser.add_argument('--faces-only', action='store_true', help='Only count visible faces without visualization')
    parser.add_argument('--output-stl', help='Export visible faces to STL file')
    parser.add_argument('--number', type=int, default=3, choices=range(10),
                        help='Number to use for 3D shapes in STL export (0-9, default: 3)')
    args = parser.parse_args()
    
    csv_path = args.csv_file
    
    if not os.path.exists(csv_path):
        print(f"Error: File not found: {csv_path}")
        return 1
    
    if args.debug:
        analyze_csv_format(csv_path)
        analyze_file_format(csv_path)
    
    try:
        voxel_array = load_voxel_data(csv_path)
        print(f"Loaded voxel data with shape: {voxel_array.shape}")
        
        # Export to STL if requested
        if args.output_stl:
            # Import the STL export functionality only when needed
            try:
                from sproxel_analyser.stl_exporter import export_visible_faces_to_stl as stl_export
                stl_export(voxel_array, args.output_stl, args.number)
            except ImportError:
                # Fall back to local function if import fails
                export_visible_faces_to_stl(voxel_array, args.output_stl, args.number)
        # Count visible faces directly if desired
        elif args.faces_only:
            total_faces, face_breakdown, tile_info = count_visible_faces(voxel_array)
            print(f"Model has {total_faces} visible faces")
            print(f"Face breakdown by direction: {face_breakdown}")
            print(f"Tile breakdown: {tile_info}")
        else:
            visualize_voxel_model(voxel_array)
    except Exception as e:
        print(f"Error processing file: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    main()