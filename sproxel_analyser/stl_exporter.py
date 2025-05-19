import numpy as np
from math import sin, cos, pi

def export_visible_faces_to_stl(voxel_array, output_path, number=3):
    """
    Export only the visible faces of a voxel model to an STL file.
    Each visible face is represented by a 1mm thick 3D number.
    Each voxel is modeled at 2cm width and length.
    The numbers are oriented with their flat surfaces parallel to the voxel faces.
    
    Parameters:
        voxel_array: 3D numpy array containing voxel data
        output_path: Path where the STL file will be saved
        number: The number to use for the 3D shape (default: 3)
    """
    try:
        # Try to import numpy-stl package
        import numpy as np
        from stl import mesh
    except ImportError:
        print("Error: The numpy-stl package is required for STL export.")
        print("Please install it using: pip install numpy-stl")
        return False
    
    print(f"Preparing STL export to {output_path} using 3D number '{number}'...")
    
    # Get the dimensions
    if len(voxel_array.shape) == 2:
        height, width = voxel_array.shape
        depth = 1
        voxel_array = voxel_array.reshape(depth, height, width)
    else:
        depth, height, width = voxel_array.shape
    
    # Set voxel size to 2cm (20mm)
    voxel_size = 20.0  # in mm
    
    print(f"Model dimensions: {width}x{height}x{depth} voxels")
    print(f"Physical dimensions: {width*voxel_size/10:.1f}cm x {height*voxel_size/10:.1f}cm x {depth*voxel_size/10:.1f}cm")
    
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
    
    # Define number parameters
    number_height = 1.0  # 1mm thick (depth/thickness of the number)
    number_size = 10.0   # 10mm size (height of the number)
    
    # Function to define vertices and faces for different numbers
    def get_number_shape(num):
        """Get vertices and faces for 3D representation of a number"""
        # Define the number as a series of points on a 2D grid
        # Each segment is a list of points forming a line
        if num == 0:
            segments = [
                [(0, 0), (0, 2), (2, 2), (2, 0), (0, 0)]  # Outer rectangle
            ]
        elif num == 1:
            segments = [
                [(1, 0), (1, 2)]  # Vertical line
            ]
        elif num == 2:
            segments = [
                [(0, 2), (2, 2), (2, 1), (0, 1), (0, 0), (2, 0)]
            ]
        elif num == 3:
            segments = [
                [(0, 2), (2, 2), (2, 0), (0, 0)],
                [(0, 1), (2, 1)]
            ]
        elif num == 4:
            segments = [
                [(0, 2), (0, 1), (2, 1)],
                [(2, 2), (2, 0)]
            ]
        elif num == 5:
            segments = [
                [(2, 2), (0, 2), (0, 1), (2, 1), (2, 0), (0, 0)]
            ]
        elif num == 6:
            segments = [
                [(2, 2), (0, 2), (0, 0), (2, 0), (2, 1), (0, 1)]
            ]
        elif num == 7:
            segments = [
                [(0, 2), (2, 2), (2, 0)]
            ]
        elif num == 8:
            segments = [
                [(0, 0), (0, 2), (2, 2), (2, 0), (0, 0)],
                [(0, 1), (2, 1)]
            ]
        elif num == 9:
            segments = [
                [(0, 0), (2, 0), (2, 2), (0, 2), (0, 1), (2, 1)]
            ]
        else:
            # Default to 3 if unsupported number
            segments = [
                [(0, 2), (2, 2), (2, 0), (0, 0)],
                [(0, 1), (2, 1)]
            ]
        
        # Convert to 3D mesh with specified thickness
        vertices = []
        faces = []
        
        # Segment thickness (width of the number's strokes)
        segment_thickness = number_size * 0.2  # 20% of number size
        
        # Process each segment
        for segment in segments:
            # For each segment, create a rectangular prism along the path
            for i in range(len(segment) - 1):
                # Get current and next point
                x1, y1 = segment[i]
                x2, y2 = segment[i + 1]
                
                # Scale to desired size
                x1 = (x1 / 2.0 - 0.5) * number_size
                y1 = (y1 / 2.0 - 0.5) * number_size
                x2 = (x2 / 2.0 - 0.5) * number_size
                y2 = (y2 / 2.0 - 0.5) * number_size
                
                # Calculate segment direction and length
                dx = x2 - x1
                dy = y2 - y1
                segment_length = np.sqrt(dx**2 + dy**2)
                
                # Skip if segment is too short
                if segment_length < 0.001:
                    continue
                
                # Normalize direction
                dx /= segment_length
                dy /= segment_length
                
                # Calculate perpendicular vector for segment width
                nx = -dy
                ny = dx
                
                # Define vertices of rectangular prism
                # Front face
                v1 = [x1 + nx * segment_thickness/2, y1 + ny * segment_thickness/2, number_height/2]
                v2 = [x1 - nx * segment_thickness/2, y1 - ny * segment_thickness/2, number_height/2]
                v3 = [x2 - nx * segment_thickness/2, y2 - ny * segment_thickness/2, number_height/2]
                v4 = [x2 + nx * segment_thickness/2, y2 + ny * segment_thickness/2, number_height/2]
                
                # Back face
                v5 = [x1 + nx * segment_thickness/2, y1 + ny * segment_thickness/2, -number_height/2]
                v6 = [x1 - nx * segment_thickness/2, y1 - ny * segment_thickness/2, -number_height/2]
                v7 = [x2 - nx * segment_thickness/2, y2 - ny * segment_thickness/2, -number_height/2]
                v8 = [x2 + nx * segment_thickness/2, y2 + ny * segment_thickness/2, -number_height/2]
                
                # Add vertices
                base_idx = len(vertices)
                vertices.extend([v1, v2, v3, v4, v5, v6, v7, v8])
                
                # Add faces (triangles)
                # Front face
                faces.append([base_idx, base_idx + 1, base_idx + 2])
                faces.append([base_idx, base_idx + 2, base_idx + 3])
                
                # Back face
                faces.append([base_idx + 4, base_idx + 6, base_idx + 5])
                faces.append([base_idx + 4, base_idx + 7, base_idx + 6])
                
                # Side faces
                faces.append([base_idx, base_idx + 4, base_idx + 5])
                faces.append([base_idx, base_idx + 5, base_idx + 1])
                
                faces.append([base_idx + 1, base_idx + 5, base_idx + 6])
                faces.append([base_idx + 1, base_idx + 6, base_idx + 2])
                
                faces.append([base_idx + 2, base_idx + 6, base_idx + 7])
                faces.append([base_idx + 2, base_idx + 7, base_idx + 3])
                
                faces.append([base_idx + 3, base_idx + 7, base_idx + 4])
                faces.append([base_idx + 3, base_idx + 4, base_idx + 0])
        
        return vertices, faces
    
    # Get the vertices and faces for the selected number
    number_vertices, number_faces = get_number_shape(number)
    
    # Function to transform and place a 3D number at a specific position and orientation
    def place_number(center_x, center_y, center_z, direction_vector):
        """Transform and place a 3D number at the specified position with the given orientation"""
        # Normalize direction vector
        dx, dy, dz = direction_vector
        length = np.sqrt(dx**2 + dy**2 + dz**2)
        if length == 0:
            dx, dy, dz = 0, 0, 1  # Default to z-axis
        else:
            dx, dy, dz = dx/length, dy/length, dz/length
            
        # Determine perpendicular vectors for number orientation
        # Find two vectors perpendicular to the direction vector
        if abs(dx) < abs(dy) and abs(dx) < abs(dz):
            perp1 = np.array([0, -dz, dy])
        elif abs(dy) < abs(dz):
            perp1 = np.array([-dz, 0, dx])
        else:
            perp1 = np.array([-dy, dx, 0])
            
        # Normalize perp1
        perp1 = perp1 / np.sqrt(np.sum(perp1**2))
        # Create perp2 using cross product
        perp2 = np.cross([dx, dy, dz], perp1)
        
        # Transform vertices
        transformed_vertices = []
        for vertex in number_vertices:
            # Original vertex coordinates
            vx, vy, vz = vertex
            
            # Apply orientation transformation
            transformed = np.zeros(3)
            transformed += vx * perp1
            transformed += vy * perp2
            transformed += vz * np.array([dx, dy, dz])
            
            # Translate to the center position
            transformed[0] += center_x
            transformed[1] += center_y
            transformed[2] += center_z
            
            transformed_vertices.append(transformed.tolist())
        
        # Return transformed vertices and original faces
        return transformed_vertices, number_faces
    
    # Collect all vertices and faces for the entire model
    all_vertices = []
    all_faces = []
    number_count = 0
    
    # For each voxel, check all 6 faces to see if they're visible
    for z in range(depth):
        for y in range(height):
            for x in range(width):
                # Skip if this voxel is empty
                if not occupied[z, y, x]:
                    continue
                
                # Position of the center of the voxel using the fixed voxel size
                center_x = (x + 0.5) * voxel_size
                center_y = (y + 0.5) * voxel_size
                center_z = (z + 0.5) * voxel_size
                
                # Check all 6 directions to see if this face is visible
                for i, (dx, dy, dz) in enumerate(directions):
                    nx, ny, nz = x + dx, y + dy, z + dz
                    
                    # A face is visible if:
                    # 1. It's at the boundary of the model space OR
                    # 2. The neighboring voxel in this direction is empty
                    if (nx < 0 or nx >= width or 
                        ny < 0 or ny >= height or
                        nz < 0 or nz >= depth or
                        not occupied[nz, ny, nx]):
                        
                        # Create a 3D number at the position of this face
                        face_center_x = center_x + dx * voxel_size * 0.5
                        face_center_y = center_y + dy * voxel_size * 0.5
                        face_center_z = center_z + dz * voxel_size * 0.5
                        
                        num_verts, num_faces = place_number(
                            face_center_x, face_center_y, face_center_z, 
                            (dx, dy, dz)
                        )
                        
                        # Adjust face indices for the global list
                        vertex_base = len(all_vertices)
                        for face in num_faces:
                            all_faces.append([
                                vertex_base + face[0], 
                                vertex_base + face[1], 
                                vertex_base + face[2]
                            ])
                        
                        # Add number vertices to the global list
                        all_vertices.extend(num_verts)
                        number_count += 1
    
    # Create the mesh
    if not all_vertices:
        print("No visible faces found.")
        return False
    
    print(f"Generated mesh with {len(all_vertices)} vertices, {len(all_faces)} triangles, and {number_count} 3D numbers")
    
    # Convert to numpy arrays
    vertices = np.array(all_vertices)
    faces = np.array(all_faces)
    
    # Create the mesh
    model_mesh = mesh.Mesh(np.zeros(len(all_faces), dtype=mesh.Mesh.dtype))
    
    # Set the vertices for each face
    for i, f in enumerate(all_faces):
        for j in range(3):
            model_mesh.vectors[i][j] = vertices[f[j]]
    
    # Save the mesh to STL file
    try:
        model_mesh.save(output_path)
        print(f"Successfully saved STL file to {output_path}")
        print(f"Model uses {voxel_size/10:.1f}cm voxels with {number_count} 3D numbers of value '{number}'")
        return True
    except Exception as e:
        print(f"Error saving STL file: {e}")
        return False
