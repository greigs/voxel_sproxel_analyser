import numpy as np
from math import sin, cos, pi

def export_visible_faces_to_stl(voxel_array, output_path, number=3, display_mode='numbers'):
    """
    Export only the visible faces of a voxel model to an STL file.
    Each visible face can display either numbers or cuboids:
    
    Numbers mode:
    - Top number: The tile category (0-6), sized 30% smaller
    - Bottom number: The color category (00-31), sized 50% smaller with two digits
    
    Cuboids mode:
    - 2x2 grid of cuboids, each 4mm wide with 3mm thickness
    - 2mm gap between cuboids
    - Cuboids are oriented so the thinner face is parallel with the tile face
    - No tile face plane is created, only the cuboids are present
    
    Tile categories:
    0: Standard tile (no modifications)
    1: Edge tile with one 45° cut
    2: Edge tile with two 45° cuts
    3: Edge tile with three 45° cuts
    4: Corner tile with one corner
    5: Corner tile with two corners
    6: Complex tile (multiple modifications)
    
    Each voxel is modeled at 2cm width and length.
    
    Parameters:
        voxel_array: 3D numpy array containing voxel data
        output_path: Path where the STL file will be saved
        number: Default number to use if categorization fails (default: 3)
        display_mode: 'numbers' or 'cuboids' (default: 'numbers')
    """
    try:
        # Try to import numpy-stl package
        import numpy as np
        from stl import mesh
    except ImportError:
        print("Error: The numpy-stl package is required for STL export.")
        print("Please install it using: pip install numpy-stl")
        return False
    
    display_mode = display_mode.lower()
    if display_mode not in ['numbers', 'cuboids']:
        print(f"Warning: Unknown display mode '{display_mode}'. Using 'numbers' mode.")
        display_mode = 'numbers'
    
    print(f"Preparing STL export to {output_path} using {display_mode} mode...")
    
    # Get the dimensions
    if len(voxel_array.shape) == 2:
        height, width = voxel_array.shape
        depth = 1
        voxel_array = voxel_array.reshape(depth, height, width)
    else:
        depth, height, width = voxel_array.shape
    
    # Set voxel size to 1.25cm (12.5mm)
    voxel_size = 12.5  # in mm
    
    print(f"Model dimensions: {width}x{height}x{depth} voxels")
    print(f"Physical dimensions: {width*voxel_size/10:.1f}cm x {height*voxel_size/10:.1f}cm x {depth*voxel_size/10:.1f}cm")
    
    # Create a binary occupancy grid (True where a voxel exists)
    occupied = np.zeros((depth, height, width), dtype=bool)
    
    # Also track color values for each voxel
    voxel_colors = {}
    
    # Collect all unique colors
    unique_colors = set()
    
    # Fill the occupancy grid and collect color information
    for z in range(depth):
        for y in range(height):
            for x in range(width):
                val = str(voxel_array[z, y, x]).strip()
                
                # Skip empty cells or fully transparent cells
                if val == "" or val.lower() == "none" or val == "0" or val == "#00000000":
                    continue
                
                # For colored voxels, check if they're transparent and collect color info
                color_val = None
                if val.startswith("#") and len(val) >= 7:
                    try:
                        alpha = 255  # Default full opacity
                        if len(val) >= 9:
                            alpha = int(val[7:9], 16)
                            if alpha == 0:  # Skip fully transparent voxels
                                continue
                        
                        # Extract color without alpha for categorization
                        color_val = val[:7].upper()  # Just the RGB part
                        unique_colors.add(color_val)
                        
                    except ValueError:
                        pass  # If parsing fails, assume it's not transparent
                
                # Mark this voxel as occupied
                occupied[z, y, x] = True
                
                # Store the color value if we have one
                if color_val:
                    voxel_colors[(z, y, x)] = color_val
    
    # Create a mapping from unique colors to indices (0-31)
    print(f"Found {len(unique_colors)} unique colors")
    color_to_index = {}
    for i, color in enumerate(sorted(unique_colors)):
        if i < 32:  # Limit to 32 colors
            color_to_index[color] = i
        else:
            print(f"Warning: More than 32 unique colors found. Limiting to first 32.")
            break
    
    # Create color category map for reporting
    color_categories = {i: [] for i in range(min(32, len(unique_colors)))}
    for color, index in color_to_index.items():
        color_categories[index].append(color)
            
    # Define the 6 possible directions (±x, ±y, ±z)
    directions = [
        (1, 0, 0), (-1, 0, 0),   # +x, -x
        (0, 1, 0), (0, -1, 0),   # +y, -y
        (0, 0, 1), (0, 0, -1)    # +z, -z
    ]
    
    # Direction names for reporting
    direction_names = ["+X", "-X", "+Y", "-Y", "+Z", "-Z"]
    
    # Define display parameters
    if display_mode == 'numbers':
        number_height = 1.0  # 1mm thick (depth/thickness of the number)
        category_number_size = 7.0 * 0.7   # 7mm size reduced by 30% for top number
        color_number_size = 7.0 * 0.5      # 7mm size reduced by 50% for bottom number
    else:  # cuboids mode
        cuboid_width = 2.0       # 2.0mm width for each cuboid
        cuboid_thickness = 1.3   # 1.3mm thickness for each cuboid
        cuboid_gap = 1.0         # 1mm gap between cuboids
    
    # First, analyze the model to categorize faces
    # Track visible face positions for edge and corner analysis
    face_positions = {name: [] for name in direction_names}
    
    # Check each voxel for visible faces
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
                        # Store the position and orientation of this visible face
                        face_positions[direction_names[i]].append((x, y, z))
    
    # Enhanced tile analysis - maps voxel positions to their visible face directions
    voxel_faces = {}
    for direction, positions in face_positions.items():
        for pos in positions:
            if pos not in voxel_faces:
                voxel_faces[pos] = []
            voxel_faces[pos].append(direction)
    
    # Map to store the category number for each face
    face_categories = {}
    
    # Categorize each face based on its edge and corner modifications
    for pos, visible_dirs in voxel_faces.items():
        x, y, z = pos
        
        # For each visible face, check for adjacent visible faces
        # Check each visible face individually to categorize it
        for dir_idx, direction in enumerate(visible_dirs):
            # Get the direction vector for this face
            dir_vector_idx = direction_names.index(direction)
            dx, dy, dz = directions[dir_vector_idx]
            
            # Count edges and corners involving this face
            edge_count = 0
            corner_count = 0
            
            # This face forms edges with perpendicular faces
            for other_dir in visible_dirs:
                if other_dir == direction:
                    continue
                    
                other_dir_idx = direction_names.index(other_dir)
                odx, ody, odz = directions[other_dir_idx]
                
                # If the directions are perpendicular (dot product = 0)
                if dx*odx + dy*ody + dz*odz == 0:
                    edge_count += 1
                    
                    # Check if there's a third face forming a corner with these two
                    for third_dir in visible_dirs:
                        if third_dir == direction or third_dir == other_dir:
                            continue
                        
                        third_dir_idx = direction_names.index(third_dir)
                        tdx, tdy, tdz = directions[third_dir_idx]
                        
                        # If the third direction is perpendicular to both current directions
                        if (dx*tdx + dy*tdy + dz*tdz == 0 and
                            odx*tdx + ody*tdy + odz*tdz == 0):
                            corner_count += 1
            
            # Categorize the face (subtract 1 from corner_count as each corner is counted twice)
            if edge_count == 0:
                # Standard tile
                category = 0
            elif edge_count == 1:
                # One edge tile
                category = 1
            elif edge_count == 2 and corner_count == 0:
                # Two edge tile
                category = 2
            elif edge_count == 3 and corner_count <= 1:
                # Three edge tile
                category = 3
            elif corner_count >= 2 and corner_count <= 3:
                # One corner tile
                category = 4
            elif corner_count >= 4:
                # Two corner tile
                category = 5
            else:
                # Complex tile
                category = 6
            
            # Store the category for this face
            face_key = (x, y, z, direction)
            face_categories[face_key] = category
    
    if display_mode == 'numbers':
        # ------- NUMBERS MODE -------
        # Function to define vertices and faces for different numbers
        def get_number_shape(num, is_digit=True, is_color=False, is_tens_digit=False):
            """
            Get vertices and faces for 3D representation of a number or letter
            
            Parameters:
                num: The number (0-9) or letter
                is_digit: True if num is a digit (0-9), False if it's a letter
                is_color: True if this is a color number (uses smaller size)
                is_tens_digit: True if this is the tens digit of a two-digit number
            """
            # Choose appropriate size based on whether it's a tile category or color number
            if is_color:
                size = color_number_size
            else:
                size = category_number_size
                
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
                # Default to 0 if unsupported
                segments = [
                    [(0, 0), (0, 2), (2, 2), (2, 0), (0, 0)]
                ]
            
            # Convert to 3D mesh with specified thickness
            vertices = []
            faces = []
            
            # Segment thickness (width of the number's strokes)
            segment_thickness = size * 0.2  # 20% of number size
            
            # Process each segment
            for segment in segments:
                # For each segment, create a rectangular prism along the path
                for i in range(len(segment) - 1):
                    # Get current and next point
                    x1, y1 = segment[i]
                    x2, y2 = segment[i + 1]
                    
                    # Scale to desired size
                    x1 = (x1 / 2.0 - 0.5) * size
                    y1 = (y1 / 2.0 - 0.5) * size
                    x2 = (x2 / 2.0 - 0.5) * size
                    y2 = (y2 / 2.0 - 0.5) * size
                    
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
        
        # Create a cache of number shapes to avoid regenerating them
        number_cache = {}
        
        # Cache digits 0-9 for tile categories (top number)
        for i in range(10):
            number_cache[f"category_{i}"] = get_number_shape(i, is_color=False)
        
        # Cache digits 0-9 for color digits (bottom number)
        for i in range(10):
            number_cache[f"color_units_{i}"] = get_number_shape(i, is_color=True)
            number_cache[f"color_tens_{i}"] = get_number_shape(i, is_color=True)
        
        # Function to transform and place a 3D number at a specific position and orientation
        def place_number(center_x, center_y, center_z, direction_vector, number, 
                        is_category=True, digit_position=None):
            """
            Transform and place a 3D number at the specified position with the given orientation.
            
            Parameters:
                center_x, center_y, center_z: The center position of the face
                direction_vector: The normal vector of the face
                number: The number value to display
                is_category: True for category number, False for color number
                digit_position: None for category, 'tens' or 'units' for color digits
            """
            # Determine y offset (vertical positioning)
            if is_category:
                y_offset = category_number_size * 0.7  # Top position for category
                x_offset = 0
                cache_key = f"category_{number}"
            else:
                y_offset = -color_number_size * 0.9  # Bottom position for color
                
                # Determine x offset (horizontal positioning) for color digits
                if digit_position == 'tens':
                    x_offset = -color_number_size * 0.75
                    cache_key = f"color_tens_{number // 10}"
                else:  # units digit
                    if number >= 10:  # Two digits
                        x_offset = color_number_size * 0.75
                    else:  # Single digit
                        x_offset = 0
                    cache_key = f"color_units_{number % 10}"
            
            # Get the cached number shape
            if cache_key not in number_cache:
                # Fall back to default if not found
                cache_key = "category_0"
            
            number_vertices, number_faces = number_cache[cache_key]
            
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
                transformed += (vx + x_offset) * perp1  # Apply horizontal offset
                transformed += (vy + y_offset) * perp2  # Apply vertical offset
                transformed += vz * np.array([dx, dy, dz])
                
                # Translate to the center position
                transformed[0] += center_x
                transformed[1] += center_y
                transformed[2] += center_z
                
                transformed_vertices.append(transformed.tolist())
            
            # Return transformed vertices and original faces
            return transformed_vertices, number_faces
    
    else:  # CUBOIDS MODE
        # Function to create a single cuboid with specified dimensions
        def create_cuboid(center_x, center_y, center_z, width, height, depth):
            """Create a cuboid centered at the given position with specified dimensions"""
            # Half dimensions for vertex positions
            half_width = width / 2
            half_height = height / 2
            half_depth = depth / 2
            
            # Define the 8 vertices of the cuboid
            vertices = [
                # Front face (z+)
                [center_x - half_width, center_y - half_height, center_z + half_depth],  # v0
                [center_x + half_width, center_y - half_height, center_z + half_depth],  # v1
                [center_x + half_width, center_y + half_height, center_z + half_depth],  # v2
                [center_x - half_width, center_y + half_height, center_z + half_depth],  # v3
                
                # Back face (z-)
                [center_x - half_width, center_y - half_height, center_z - half_depth],  # v4
                [center_x + half_width, center_y - half_height, center_z - half_depth],  # v5
                [center_x + half_width, center_y + half_height, center_z - half_depth],  # v6
                [center_x - half_width, center_y + half_height, center_z - half_depth],  # v7
            ]
            
            # Define the 12 triangles (6 faces, 2 triangles each)
            faces = [
                # Front face
                [0, 1, 2], [0, 2, 3],
                # Back face
                [4, 6, 5], [4, 7, 6],
                # Left face
                [0, 3, 7], [0, 7, 4],
                # Right face
                [1, 5, 6], [1, 6, 2],
                # Top face
                [3, 2, 6], [3, 6, 7],
                # Bottom face
                [0, 4, 5], [0, 5, 1],
            ]
            
            return vertices, faces
        
        # Function to place a cuboid at a specific position and orientation on a voxel face
        def place_cuboid(center_x, center_y, center_z, direction_vector, position, category=0, color=0):
            """
            Create and place a cuboid at a specified position on a voxel face
            
            Parameters:
                center_x, center_y, center_z: The center position of the voxel face
                direction_vector: The normal vector of the face
                position: Position index (0-3) determining quadrant in the 2x2 grid
                category: Category value (0-6) - not used for thickness anymore
                color: Color value (0-31) - not used for thickness anymore
            """
            # Normalize direction vector
            dx, dy, dz = direction_vector
            length = np.sqrt(dx**2 + dy**2 + dz**2)
            if length == 0:
                dx, dy, dz = 0, 0, 1  # Default to z-axis
            else:
                dx, dy, dz = dx/length, dy/length, dz/length
                
            # Find perpendicular vectors for the face's coordinate system
            if abs(dx) < abs(dy) and abs(dx) < abs(dz):
                perp1 = np.array([0, -dz, dy])
            elif abs(dy) < abs(dz):
                perp1 = np.array([-dz, 0, dx])
            else:
                perp1 = np.array([-dy, dx, 0])
                
            # Normalize perp1
            perp1_length = np.sqrt(np.sum(perp1**2))
            perp1 = perp1 / perp1_length
            
            # Create perp2 using cross product (perpendicular to both direction and perp1)
            perp2 = np.cross([dx, dy, dz], perp1)
            
            # Calculate position offsets for the 2x2 grid layout
            grid_offset = cuboid_width + cuboid_gap
            
            # Determine position in the 2x2 grid
            if position == 0:  # Top-left
                x_offset = -grid_offset / 2
                y_offset = grid_offset / 2
            elif position == 1:  # Top-right
                x_offset = grid_offset / 2
                y_offset = grid_offset / 2
            elif position == 2:  # Bottom-left
                x_offset = -grid_offset / 2
                y_offset = -grid_offset / 2
            else:  # Bottom-right (position == 3)
                x_offset = grid_offset / 2
                y_offset = -grid_offset / 2

            # Use constant thickness for all cuboids (3mm)
            thickness = cuboid_thickness
            
            # Calculate the cuboid position in the face's plane
            planar_offset = x_offset * perp1 + y_offset * perp2
            
            # Position the cuboid center 3mm inside the face (inset)
            inset = 3.0  # 3mm inset from the face
            inset_vector = -inset * np.array([dx, dy, dz])  # Negative direction to move inside
            
            cuboid_center = np.array([
                center_x + planar_offset[0] + inset_vector[0],
                center_y + planar_offset[1] + inset_vector[1], 
                center_z + planar_offset[2] + inset_vector[2]
            ])
            
            # Create the cuboid with its thinner dimension along the face normal
            # The cuboid's local coordinate system:
            # Z axis = direction vector (normal to face)
            # X axis = perp1
            # Y axis = perp2
            
            # Create vertices for a cuboid where the Z dimension is the thickness
            # and X and Y dimensions are the cuboid width
            half_width = cuboid_width / 2
            half_thickness = thickness / 2
            
            # Define the 8 vertices in local space
            local_vertices = [
                # Front face (Z+)
                [-half_width, -half_width, half_thickness],   # 0: front bottom left
                [half_width, -half_width, half_thickness],    # 1: front bottom right
                [half_width, half_width, half_thickness],     # 2: front top right
                [-half_width, half_width, half_thickness],    # 3: front top left
                
                # Back face (Z-)
                [-half_width, -half_width, -half_thickness],  # 4: back bottom left
                [half_width, -half_width, -half_thickness],   # 5: back bottom right
                [half_width, half_width, -half_thickness],    # 6: back top right
                [-half_width, half_width, -half_thickness],   # 7: back top left
            ]
            
            # Transform vertices to global space
            transformed_vertices = []
            for vertex in local_vertices:
                local_x, local_y, local_z = vertex
                
                # Transform using the local-to-global coordinate system
                global_position = (
                    cuboid_center + 
                    local_x * perp1 + 
                    local_y * perp2 + 
                    local_z * np.array([dx, dy, dz])
                )
                
                transformed_vertices.append(global_position.tolist())
            
            # Define the faces (triangles) of the cuboid
            faces = [
                # Front face (Z+)
                [0, 1, 2], [0, 2, 3],
                # Back face (Z-)
                [4, 6, 5], [4, 7, 6],
                # Left face (X-)
                [0, 3, 7], [0, 7, 4],
                # Right face (X+)
                [1, 5, 6], [1, 6, 2],
                # Top face (Y+)
                [3, 2, 6], [3, 6, 7],
                # Bottom face (Y-)
                [0, 4, 5], [0, 5, 1],
            ]
            
            return transformed_vertices, faces
        
        # Function to add a 2x2 grid of cuboids on a voxel face
        def add_cuboid_grid(center_x, center_y, center_z, direction_vector, category, color_idx):
            """Add a 2x2 grid of cuboids to represent category and color values"""
            all_vertices = []
            all_faces = []
            
            # Add 4 cuboids in a grid pattern
            for position in range(4):
                # Create and place the cuboid
                vertices, faces = place_cuboid(
                    center_x, center_y, center_z, 
                    direction_vector, position,
                    category, color_idx
                )
                
                # Adjust face indices for global vertex list
                vertex_base = len(all_vertices)
                for face in faces:
                    all_faces.append([
                        vertex_base + face[0], 
                        vertex_base + face[1], 
                        vertex_base + face[2]
                    ])
                
                # Add vertices to the global list
                all_vertices.extend(vertices)
            
            return all_vertices, all_faces
    
    # Collect all vertices and faces for the entire model
    all_vertices = []
    all_faces = []
    
    # Counters for statistics
    number_counts = {i: 0 for i in range(7)}  # For tile categories
    color_counts = {i: 0 for i in range(min(32, len(color_to_index)))}  # For color categories
    
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
                
                # Get the color index for this voxel
                color_key = (z, y, x)
                color_val = voxel_colors.get(color_key, "#FFFFFF")  # Default to white if no color
                color_idx = color_to_index.get(color_val, 0)        # Default to 0 if not in map
                
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
                        
                        # Calculate the center of this face
                        face_center_x = center_x + dx * voxel_size * 0.5
                        face_center_y = center_y + dy * voxel_size * 0.5
                        face_center_z = center_z + dz * voxel_size * 0.5
                        
                        # Get the category for this face
                        face_key = (x, y, z, direction_names[i])
                        category = face_categories.get(face_key, number)  # Use default if not found
                        
                        # Count these categories
                        number_counts[category] = number_counts.get(category, 0) + 1
                        color_counts[color_idx] = color_counts.get(color_idx, 0) + 1
                        
                        if display_mode == 'numbers':
                            # Add the top number (tile category)
                            cat_verts, cat_faces = place_number(
                                face_center_x, face_center_y, face_center_z, 
                                (dx, dy, dz), category, is_category=True
                            )
                            vertex_base = len(all_vertices)
                            for face in cat_faces:
                                all_faces.append([
                                    vertex_base + face[0], 
                                    vertex_base + face[1], 
                                    vertex_base + face[2]
                                ])
                            all_vertices.extend(cat_verts)
                            
                            # Add two-digit color number at the bottom
                            # First, add tens digit if > 9
                            if color_idx > 9:
                                tens_verts, tens_faces = place_number(
                                    face_center_x, face_center_y, face_center_z, 
                                    (dx, dy, dz), color_idx, 
                                    is_category=False, digit_position='tens'
                                )
                                # Add tens digit to mesh
                                vertex_base = len(all_vertices)
                                for face in tens_faces:
                                    all_faces.append([
                                        vertex_base + face[0], 
                                        vertex_base + face[1], 
                                        vertex_base + face[2]
                                    ])
                                all_vertices.extend(tens_verts)
                            

                            # Add units digit
                            units_verts, units_faces = place_number(
                                face_center_x, face_center_y, face_center_z, 
                                (dx, dy, dz), color_idx,
                                is_category=False, digit_position='units'
                            )
                            # Add units digit to mesh
                            vertex_base = len(all_vertices)
                            for face in units_faces:
                                all_faces.append([
                                    vertex_base + face[0], 
                                    vertex_base + face[1], 
                                    vertex_base + face[2]
                                ])
                            all_vertices.extend(units_verts)
                        else:  # cuboids mode
                            # Add a 2x2 grid of cuboids representing category and color
                            grid_verts, grid_faces = add_cuboid_grid(
                                face_center_x, face_center_y, face_center_z,
                                (dx, dy, dz), category, color_idx
                            )
                            
                            # Add the grid to the mesh
                            vertex_base = len(all_vertices)
                            for face in grid_faces:
                                all_faces.append([
                                    vertex_base + face[0], 
                                    vertex_base + face[1], 
                                    vertex_base + face[2]
                                ])
                            all_vertices.extend(grid_verts)
    
    # Create the mesh
    if not all_vertices:
        print("No visible faces found.")
        return False
    
    print(f"Generated mesh with {len(all_vertices)} vertices and {len(all_faces)} triangles")
    print("Numbers by tile category:")
    category_names = [
        "Standard tiles (0)", 
        "Edge tiles with one cut (1)", 
        "Edge tiles with two cuts (2)",
        "Edge tiles with three cuts (3)",
        "Corner tiles with one corner (4)",
        "Corner tiles with two corners (5)",
        "Complex tiles (6)"
    ]
    
    total_numbers = sum(number_counts.values())
    for i, name in enumerate(category_names):
        if i < len(number_counts):
            count = number_counts.get(i, 0)
            percentage = (count / total_numbers) * 100 if total_numbers > 0 else 0
            print(f"  {name}: {count} ({percentage:.1f}%)")
    
    print("\nColors categorized (showing first 10):")
    for i, colors in sorted(color_categories.items())[:10]:
        count = color_counts.get(i, 0)
        percentage = (count / total_numbers) * 100 if total_numbers > 0 else 0
        print(f"  Category {i:02d}: {count} faces ({percentage:.1f}%) - {colors[0]}" + 
              (f" and {len(colors)-1} more" if len(colors) > 1 else ""))
    
    if len(color_categories) > 10:
        print(f"  ... and {len(color_categories) - 10} more color categories")
    
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
        print(f"Model uses {voxel_size/10:.1f}cm voxels with {display_mode} representation")
        return True
    except Exception as e:
        print(f"Error saving STL file: {e}")
        return False
