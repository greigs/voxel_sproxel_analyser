import numpy as np
from math import sin, cos, pi

def export_visible_faces_to_stl(voxel_array, output_path):
    """
    Export only the visible faces of a voxel model to an STL file.
    Each visible face is represented as a 1mm thick, 1cm diameter cylinder.
    Each voxel is modeled at 2cm width and length.
    The cylinders are oriented with their flat surfaces parallel to the voxel faces.
    
    Parameters:
        voxel_array: 3D numpy array containing voxel data
        output_path: Path where the STL file will be saved
    """
    try:
        # Try to import numpy-stl package
        import numpy as np
        from stl import mesh
        # For cylinder generation
        from math import sin, cos, pi
    except ImportError:
        print("Error: The numpy-stl package is required for STL export.")
        print("Please install it using: pip install numpy-stl")
        return False
    
    print(f"Preparing STL export to {output_path}...")
    
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
    
    # Define cylinder parameters
    cylinder_radius = 5.0  # 5mm = 1cm diameter
    cylinder_height = 1.0  # 1mm thick
    cylinder_segments = 12  # Number of segments for the cylinder
    
    # Function to create a cylinder mesh at a specified position and orientation
    def create_cylinder(center_x, center_y, center_z, direction_vector):
        # Vertices for a cylinder with specified dimensions
        vertices = []
        faces = []
        
        # Normalize direction vector
        dx, dy, dz = direction_vector
        length = (dx**2 + dy**2 + dz**2)**0.5
        if length == 0:
            dx, dy, dz = 0, 0, 1  # Default to z-axis
        else:
            dx, dy, dz = dx/length, dy/length, dz/length
            
        # Determine perpendicular vectors for cylinder orientation
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
        
        # Create cylinder vertices for both ends - making sure flat sides are parallel to voxel face
        for i in range(cylinder_segments):
            angle = 2 * pi * i / cylinder_segments
            # Coordinate on unit circle
            circle_x = cos(angle)
            circle_y = sin(angle)
            
            # First end vertices (scaled and positioned)
            # Positioned at face center minus half the cylinder height in the normal direction
            p1 = center_x - (dx * cylinder_height/2)
            p2 = center_y - (dy * cylinder_height/2)
            p3 = center_z - (dz * cylinder_height/2)
            
            # Then add the circle point scaled by the radius
            p1 += circle_x * perp1[0] * cylinder_radius
            p2 += circle_x * perp1[1] * cylinder_radius
            p3 += circle_x * perp1[2] * cylinder_radius
            
            p1 += circle_y * perp2[0] * cylinder_radius
            p2 += circle_y * perp2[1] * cylinder_radius
            p3 += circle_y * perp2[2] * cylinder_radius
            
            vertices.append([p1, p2, p3])
            
            # Second end vertices (at face center plus half the cylinder height)
            p1 = center_x + (dx * cylinder_height/2)
            p2 = center_y + (dy * cylinder_height/2)
            p3 = center_z + (dz * cylinder_height/2)
            
            # Then add the circle point
            p1 += circle_x * perp1[0] * cylinder_radius
            p2 += circle_x * perp1[1] * cylinder_radius
            p3 += circle_x * perp1[2] * cylinder_radius
            
            p1 += circle_y * perp2[0] * cylinder_radius
            p2 += circle_y * perp2[1] * cylinder_radius
            p3 += circle_y * perp2[2] * cylinder_radius
            
            vertices.append([p1, p2, p3])
        
        # Create faces for the cylinder
        vertex_count = len(vertices)
        base = len(vertices) - 2*cylinder_segments
        
        # Side faces
        for i in range(cylinder_segments):
            # Use modulo to wrap around to the first vertex
            next_i = (i + 1) % cylinder_segments
            
            # First triangle (connecting vertices of both ends)
            faces.append([
                base + 2*i,
                base + 2*i+1,
                base + 2*next_i
            ])
            
            # Second triangle
            faces.append([
                base + 2*i+1,
                base + 2*next_i+1,
                base + 2*next_i
            ])
        
        # End cap 1
        for i in range(1, cylinder_segments-1):
            faces.append([
                base,
                base + 2*i,
                base + 2*(i+1)
            ])
        
        # End cap 2
        for i in range(1, cylinder_segments-1):
            faces.append([
                base + 1,
                base + 2*(i+1)+1,
                base + 2*i+1
            ])
            
        return vertices, faces
    
    # Collect all vertices and faces for the entire model
    all_vertices = []
    all_faces = []
    cylinder_count = 0
    
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
                        
                        # Create a cylinder at the position of this face
                        face_center_x = center_x + dx * voxel_size * 0.5
                        face_center_y = center_y + dy * voxel_size * 0.5
                        face_center_z = center_z + dz * voxel_size * 0.5
                        
                        cylinder_verts, cylinder_faces = create_cylinder(
                            face_center_x, face_center_y, face_center_z, 
                            (dx, dy, dz)
                        )
                        
                        # Adjust face indices for the global list
                        vertex_base = len(all_vertices)
                        for face in cylinder_faces:
                            all_faces.append([vertex_base + face[0], 
                                             vertex_base + face[1], 
                                             vertex_base + face[2]])
                        
                        # Add cylinder vertices to the global list
                        all_vertices.extend(cylinder_verts)
                        cylinder_count += 1
    
    # Create the mesh
    if not all_vertices:
        print("No visible faces found.")
        return False
    
    print(f"Generated mesh with {len(all_vertices)} vertices, {len(all_faces)} triangles, and {cylinder_count} cylinders")
    
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
        print(f"Model uses {voxel_size/10:.1f}cm voxels with {cylinder_count} cylinders (1cm diameter, 1mm thick)")
        return True
    except Exception as e:
        print(f"Error saving STL file: {e}")
        return False
