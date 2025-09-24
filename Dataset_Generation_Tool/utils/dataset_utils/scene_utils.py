# import modules
import bpy
import numpy as np
import csv
import bpy_extras
import mathutils
from mathutils import Vector
import math
from math import sin,cos,tan
import random

from utils.blender_utils.bpy_utils import add_background_plane,add_light_source,add_camera,add_beizer_curve,add_nurbs_path
# from utils.blender_utils.bpy_utils import *


def create_trajectory_scene(trajectory_type:str):
    """
    Creates the required scene
    Args:
        trajectory_type: str (line,circle)
    """
    
    # add background plane
    background_plane = add_background_plane(plane_name='Background_plane',plane_size=10.0)
    light_source = add_light_source(light_name='Sun',light_type='SUN',shadow=True)
    main_camera = add_camera(camera_name='Camera',location=(-1.2058,2.8936,2.2521),rotation=(53.924,0.49219,-158.44))
    robot_camera = add_camera(camera_name='Robot_Camera',location=(0,0,0),rotation=(0,0,0))

    # add curves for following the path
    light_track = add_beizer_curve(name='light_track', location=(0, 0, 2.42459), scale=(1, 1, 1))
    light_source.constraints.new(type='FOLLOW_PATH')
    light_source.constraints['Follow Path'].target = light_track
    
    if trajectory_type=='LINE':
        camera_track = add_nurbs_path(path_name='camera_track',location=(0.5, 0, 1),rotation=(10,120,10),scale=(1,1,1))
    elif trajectory_type == "CURVE":
        camera_track = add_beizer_curve(name='camera_track',location=(0,0,2),scale=(1,1,1))
    
    # Camera constraints
    robot_camera.constraints.new(type='FOLLOW_PATH')
    robot_camera.constraints['Follow Path'].target = camera_track
    robot_camera.constraints.new(type="TRACK_TO")

    return background_plane, light_source, robot_camera,main_camera,camera_track, light_track


def calculate_optimal_radius(obj_names, camera, padding=0.5):
    """
    Calculates an optimal radius based on object sizes and camera visibility.
    
    Keyword arguments:
        obj_names -- list of object names
        camera_name -- name of the camera
        padding -- multiplier to add extra spacing between objects
    """
    objects = [bpy.context.scene.objects.get(name) for name in obj_names if bpy.context.scene.objects.get(name)]
    
    if not objects:
        return 1.0 
    
    # Estimate object size
    avg_diameter = sum((max(obj.dimensions.x, obj.dimensions.y) for obj in objects)) / len(objects)

    # Minimum radius to prevent overlap
    min_radius = (avg_diameter / 2) * len(objects) / math.pi * padding

    if camera:
        cam_distance = camera.location.length
        max_radius = cam_distance * 0.5
        return min(min_radius, max_radius)
    
    return min_radius

# def place_objects_in_circular_arrangement(obj_names,camera):
#     """
#     Places objects in a circular arrangement dynamically calculating radius.
    
#     Keyword arguments:
#         obj_names -- list of object names
#         camera_name -- name of the camera
#     """
#     radius = calculate_optimal_radius(obj_names, camera)
#     angle_increment = 2 * math.pi / len(obj_names)
    
#     # Place objects randomly within the circular layout
#     for index, obj_name in enumerate(obj_names):
#         obj = bpy.context.scene.objects.get(obj_name)
#         if obj:
#             angle = index * angle_increment
#             x = radius * math.cos(angle)
#             y = radius * math.sin(angle)

#             # # Use object's dimensions to adjust the position
#             # x += obj.dimensions.x * 0.2
#             # y += obj.dimensions.y * 0.2

#             obj.location = Vector((x, y, obj.location.z))
###########################################################################
def get_camera_frame_bounds(camera, surface_to_place,margin=0.9):
    """Calculate safe area within camera frame based on field of view"""
    
    if not camera:
        raise ValueError(f"Camera '{camera.name}' not found")
    
    # Get camera data
    scene = bpy.context.scene
    fov_x = camera.data.angle_x if hasattr(camera.data, 'angle_x') else camera.data.angle
    fov_y = camera.data.angle_y if hasattr(camera.data, 'angle_y') else camera.data.angle
    
    # Calculate visible area at background plane distance
    if not surface_to_place:
        raise ValueError("Background plane not found")
    
    # Distance from camera to background plane
    cam_loc = camera.matrix_world.translation
    plane_loc = surface_to_place.matrix_world.translation
    distance = (plane_loc - cam_loc).length
    
    # Calculate visible width and height at plane distance
    visible_width = 2 * distance * math.tan(fov_x / 2)
    visible_height = 2 * distance * math.tan(fov_y / 2)
    
    # Apply margin to keep objects safely within frame
    safe_width = visible_width * margin
    safe_height = visible_height * margin
    
    return safe_width, safe_height, plane_loc

def calculate_bounding_box(obj):
    """Calculate the bounding box dimensions of an object"""
    local_coords = [Vector(v) for v in obj.bound_box]
    global_coords = [obj.matrix_world @ v for v in local_coords]
    
    min_x = min(v.x for v in global_coords)
    max_x = max(v.x for v in global_coords)
    min_y = min(v.y for v in global_coords)
    max_y = max(v.y for v in global_coords)
    min_z = min(v.z for v in global_coords)
    max_z = max(v.z for v in global_coords)
    
    width = max_x - min_x
    depth = max_y - min_y
    height = max_z - min_z
    
    return width, depth, height

def place_objects_in_circular_arrangement(object_names,camera,surface_to_place, margin=0.8):
    """Place objects in circular arrangement within camera frame on background plane"""
    num_objects = len(object_names)
    if num_objects == 0:
        return
    
    # Get camera frame bounds and plane position
    frame_width, frame_height, plane_center = get_camera_frame_bounds(camera,surface_to_place,margin=margin)
    radius = min(frame_width, frame_height) * 0.4  # Use 40% of smaller dimension
    
    # Calculate angular spacing between objects
    angle_step = 2 * math.pi / num_objects
    
    # Get all objects and their dimensions
    objects = [bpy.data.objects[name] for name in object_names]
    dimensions = [calculate_bounding_box(obj) for obj in objects]
    
    # Find the maximum object footprint to determine minimum spacing
    max_width = max(d[0] for d in dimensions)
    max_depth = max(d[1] for d in dimensions)
    max_dimension = max(max_width, max_depth)
    
    # Adjust radius if needed to prevent collisions
    min_radius = (max_dimension * num_objects) / (2 * math.pi)
    radius = max(radius, min_radius * 1.2)  # Add 20% padding
    
    # Place objects on background plane
    for i, (obj, (width, depth, height)) in enumerate(zip(objects, dimensions)):
        # Calculate angle for this object
        angle = i * angle_step
        
        # Calculate position on plane
        x = plane_center.x + radius * math.cos(angle)
        y = plane_center.y + radius * math.sin(angle)
        z = plane_center.z + height/2  # Place on plane surface
        
        # Set position
        obj.location = (x, y, z)
        
        if camera:
            direction = camera.matrix_world.translation - obj.location
            rot_quat = direction.to_track_quat('-Z', 'Y')
            obj.rotation_euler = rot_quat.to_euler()
        
        # Small random offset to make it look more natural
        obj.location.x += random.uniform(-0.05, 0.05) * radius
        obj.location.y += random.uniform(-0.05, 0.05) * radius
        
        # Small random rotation variation
        obj.rotation_euler.z += random.uniform(-0.2, 0.2)

# def place_objects_randomly(object_names, camera, surface_to_place, min_spacing=0.1, max_attempts=100):
#     """Place objects randomly around center without collisions"""
#     num_objects = len(object_names)
#     if num_objects == 0:
#         return
    
#     # Get camera frame bounds and surface position
#     frame_width, frame_height, surface_center = get_camera_frame_bounds(camera, surface_to_place, margin=0.1)
    
#     # Get all objects and their dimensions
#     objects = [bpy.data.objects[name] for name in object_names]
#     dimensions = [calculate_bounding_box(obj) for obj in objects]
    
#     # Calculate maximum object dimension for spacing
#     max_dimension = max(max(d[0], d[1],d[2]) for d in dimensions)
#     spacing = max_dimension + min_spacing
    
#     # Create grid system for spatial partitioning
#     grid_cell_size = spacing
#     grid = {}
    
#     def get_grid_cell(pos):
#         """Convert position to grid cell coordinates"""
#         return (int(pos[0] // grid_cell_size), int(pos[1] // grid_cell_size))
    
#     def check_collision(pos, radius):
#         """Check for collisions in neighboring grid cells"""
#         cell_x, cell_y = get_grid_cell(pos)
        
#         # Check neighboring cells (3x3 area)
#         for x in range(cell_x - 1, cell_x + 2):
#             for y in range(cell_y - 1, cell_y + 2):
#                 if (x, y) in grid:
#                     for (other_pos, other_radius) in grid[(x, y)]:
#                         distance = math.sqrt((pos[0] - other_pos[0])**2 + (pos[1] - other_pos[1])**2)
#                         if distance < (radius + other_radius):
#                             return True
#         return False
    
#     def add_to_grid(pos, radius):
#         """Add object to the spatial grid"""
#         cell_x, cell_y = get_grid_cell(pos)
#         if (cell_x, cell_y) not in grid:
#             grid[(cell_x, cell_y)] = []
#         grid[(cell_x, cell_y)].append((pos, radius))
    
#     # Place objects
#     for obj, (width, depth, height) in zip(objects, dimensions):
#         placed = False
#         attempts = 0
#         obj_radius = max(width, depth)/2
        
#         while not placed and attempts < max_attempts:
#             attempts += 1
            
#             # Generate random position within frame bounds
#             x = surface_center.x + random.uniform(-frame_width/2, frame_width/2)
#             y = surface_center.y + random.uniform(-frame_height/2, frame_height/2)
#             z = surface_center.z + height/2  # Place on surface
            
#             position = (x, y)
            
#             # Check for collisions using grid system
#             if not check_collision(position, obj_radius):
#                 # Place the object
#                 obj.location = (x, y, z)
                
#                 # Rotate object randomly for natural look
#                 obj.rotation_euler.z = random.uniform(0, 2*math.pi)
                
#                 # Add slight random tilt for realism
#                 obj.rotation_euler.x = random.uniform(-0.1, 0.1)
#                 obj.rotation_euler.y = random.uniform(-0.1, 0.1)
                
#                 # Add to grid system
#                 add_to_grid(position, obj_radius)
#                 placed = True
        
#         if not placed:
#             print(f"Warning: Could not place {obj.name} after {max_attempts} attempts")


###########################################################################

def get_direction_pca(point_cloud):

    cov_matrix = np.cov(point_cloud.T)
    eigen_values, eigen_vectors = np.linalg.eigh(cov_matrix)
    largest_eigen_value = np.argmax(eigen_values)
    eigen_vector_largest = eigen_vectors[:, largest_eigen_value]

    return eigen_vector_largest

def get_pca_direction_centroid_location(obj_name):
    """
    Function returns    PCA data
                        Direction of longitudal axis
                        centroid of the pca
                        location of the object in blender scene.
    """
    # Get the object
    obj = bpy.data.objects[obj_name]
    # Get the camera
    camera_object = bpy.data.objects['Camera']

    # Get the noramlized camera matrix
    matrix = camera_object.matrix_world.normalized().inverted()

    # Get the mesh data of the object object and undo the transformations.
    mesh = obj.to_mesh(preserve_all_data_layers=True)
    mesh.transform(obj.matrix_world)
    mesh.transform(matrix)

    vertices_positions = []
    for v in mesh.vertices:
        vertices_positions.append(list((v.co.x, v.co.y, v.co.z)))

    vertices = np.array(vertices_positions)
    centroid = np.mean(vertices, axis=0)
    direction = get_direction_pca(point_cloud=vertices)
    location_object = obj.location

    return vertices_positions, location_object, direction, centroid

def save_point_cloud_data(self,vertices, file_path):
        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(vertices)

def project_by_object_utils(self, cam, point):
    point = Vector(point)
    scene = bpy.context.scene
    co_2d = bpy_extras.object_utils.world_to_camera_view(scene, cam, point)
    render_scale = scene.render.resolution_percentage / 100
    render_size = (
        int(scene.render.resolution_x * render_scale),
        int(scene.render.resolution_y * render_scale),
    )
    return Vector((co_2d.x * render_size[0], render_size[1] - co_2d.y * render_size[1]))


def clear_scene():
    """
    Delete all existing objects and materials.
    """

    # Delete all objects
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_by_type(type='MESH')
    bpy.ops.object.delete()

    # Delete all materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material)

    # Delete all cameras
    for camera in bpy.data.cameras:
        bpy.data.cameras.remove(camera)

    # Delete all lights
    for light in bpy.data.lights:
        bpy.data.lights.remove(light)

    # Delete all curves (bezier circles, etc.)
    for curve in bpy.data.curves:
        bpy.data.curves.remove(curve)

    # Delete all images
    for image in bpy.data.images:
        bpy.data.images.remove(image)

    # Delete all textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture)

    # Delete all nodes in node trees
    for material in bpy.data.materials:
        if material.use_nodes:
            material.node_tree.nodes.clear()

    # Delete all node groups
    for node_group in bpy.data.node_groups:
        bpy.data.node_groups.remove(node_group)

    return None

def place_objects_in_center(obj_names):
    # Place objects randomly within the circular layout
    for index, obj_name in enumerate(obj_names):
        obj = bpy.context.scene.objects.get(obj_name)
        if obj:
            obj.location = Vector((0.0, 0.0, obj.location.z))


def place_object_randomly(scene,obj, camera, min_distance_factor):
    # Get the current camera field of view in degrees if not provided
    fov_radians = camera.data.angle
    # Calculate camera height and width for the current FOV
    aspect_ratio = camera.data.sensor_width / camera.data.sensor_height
    aspect_ratio = scene.render.resolution_x/scene.render.resolution_y
    camera_z_loc = camera.matrix_world.decompose()[0][2]
    camera_height = 2 * camera_z_loc * tan(fov_radians / 2)
    camera_width = camera_height * aspect_ratio
    
    # Calculate the size of the object within the camera frame (e.g., 20% of camera width)
    object_size_percentage = 0.1
    object_width = object_size_percentage * camera_width
    
    # Calculate the maximum distance for the object based on its size and desired size within the frame
    max_distance = object_width / (2 * tan(fov_radians / 2))
    
    # Calculate the minimum distance based on a factor of the maximum distance
    min_distance = min_distance_factor * max_distance
    
    # Calculate random distance within the range
    random_distance = random.uniform(min_distance, max_distance)
    # Calculate random angle (azimuth) around the camera
    random_angle = random.uniform(0, 2 * np.pi)
    
    # Calculate the object position relative to the camera
    random_x = random_distance * cos(random_angle)
    random_y = random_distance * sin(random_angle)
    random_z = obj.dimensions.z / 2
    
    # Set the object location
    obj.location.x = random_x
    obj.location.y = random_y
    # obj.location.z = random_z

    print("Object's location in random placement function: ",obj.location)

    random_rotation = mathutils.Euler((
        random.uniform(0, 2 * 3.14159),  # Random rotation around X-axis (0 to 360 degrees)
        random.uniform(0, 2 * 3.14159),  # Random rotation around Y-axis (0 to 360 degrees)
        random.uniform(0, 2 * 3.14159)   # Random rotation around Z-axis (0 to 360 degrees)
    ), 'XYZ')
    # Apply the random rotation to the object
    obj.rotation_euler = random_rotation

def place_object_random_postions(scene,obj, camera, min_distance_factor):
    # Get the current camera field of view in degrees if not provided
    fov_radians = camera.data.angle
    # Calculate camera height and width for the current FOV
    aspect_ratio = camera.data.sensor_width / camera.data.sensor_height
    aspect_ratio = scene.render.resolution_x/scene.render.resolution_y
    camera_z_loc = camera.matrix_world.decompose()[0][2]
    camera_height = 2 * camera_z_loc * tan(fov_radians / 2)
    camera_width = camera_height * aspect_ratio
    
    # Calculate the size of the object within the camera frame (e.g., 20% of camera width)
    object_size_percentage = 0.1
    object_width = object_size_percentage * camera_width
    
    # Calculate the maximum distance for the object based on its size and desired size within the frame
    max_distance = object_width / (2 * tan(fov_radians / 2))
    
    # Calculate the minimum distance based on a factor of the maximum distance
    min_distance = min_distance_factor * max_distance
    
    # Calculate random distance within the range
    random_distance = random.uniform(min_distance, max_distance)
    # Calculate random angle (azimuth) around the camera
    random_angle = random.uniform(0, 2 * np.pi)
    
    # Calculate the object position relative to the camera
    random_x = random_distance * cos(random_angle)
    random_y = random_distance * sin(random_angle)
    random_z = obj.dimensions.z / 2
    
    # Set the object location
    obj.location.x = random_x
    obj.location.y = random_y
    # obj.location.z = random_z

    # print("Object's location in random placement function: ",obj.location

##################################################################################

def bounding_boxes_intersect(obj1, obj2):
    def get_bb_world(obj):
        mat = obj.matrix_world
        return [mat @ Vector(corner) for corner in obj.bound_box]

    bb1 = get_bb_world(obj1)
    bb2 = get_bb_world(obj2)

    def get_min_max(bb):
        xs = [v.x for v in bb]
        ys = [v.y for v in bb]
        zs = [v.z for v in bb]
        return (min(xs), max(xs)), (min(ys), max(ys)), (min(zs), max(zs))

    (x1_min, x1_max), (y1_min, y1_max), (z1_min, z1_max) = get_min_max(bb1)
    (x2_min, x2_max), (y2_min, y2_max), (z2_min, z2_max) = get_min_max(bb2)

    overlap_x = x1_min <= x2_max and x1_max >= x2_min
    overlap_y = y1_min <= y2_max and y1_max >= y2_min
    overlap_z = z1_min <= z2_max and z1_max >= z2_min

    return overlap_x and overlap_y and overlap_z

def place_objects_no_collision(obj_names, tries_per_object=100, area_size=5.0):
    placed_objects = []
    
    for name in obj_names:
        obj = bpy.data.objects.get(name)
        if not obj:
            print(f"Object {name} not found.")
            continue

        obj.hide_viewport = False
        obj.hide_render = False

        success = False
        for _ in range(tries_per_object):
            new_pos = Vector((
                random.uniform(-area_size, area_size),
                random.uniform(-area_size, area_size),
                0  # Keep on ground
            ))

            obj.location = new_pos
            bpy.context.view_layer.update()

            collision = any(bounding_boxes_intersect(obj, other) for other in placed_objects)
            if not collision:
                placed_objects.append(obj)
                success = True
                break

        if not success:
            print(f"Failed to place {name} without collision after {tries_per_object} tries.")


def get_bounding_box(obj):
    """Calculate the bounding box of an object in world coordinates."""
    matrix = obj.matrix_world
    bbox = [matrix @ Vector(corner) for corner in obj.bound_box]
    return bbox

def get_object_size(obj):
    """Calculate the maximum dimension of an object's bounding box."""
    bbox = get_bounding_box(obj)
    min_coords = Vector((min(v.x for v in bbox), min(v.y for v in bbox), min(v.z for v in bbox)))
    max_coords = Vector((max(v.x for v in bbox), max(v.y for v in bbox), max(v.z for v in bbox)))
    return (max_coords - min_coords).length

def is_collision(obj1, obj2, buffer=0.1):
    """Check if two objects' bounding boxes intersect with a buffer margin."""
    bbox1 = get_bounding_box(obj1)
    bbox2 = get_bounding_box(obj2)
    
    min1 = Vector((min(v.x for v in bbox1), min(v.y for v in bbox1), min(v.z for v in bbox1)))
    max1 = Vector((max(v.x for v in bbox1), max(v.y for v in bbox1), max(v.z for v in bbox1)))
    min2 = Vector((min(v.x for v in bbox2), min(v.y for v in bbox2), min(v.z for v in bbox2)))
    max2 = Vector((max(v.x for v in bbox2), max(v.y for v in bbox2), max(v.z for v in bbox2)))
    
    # Add buffer to bounding boxes
    min1 -= Vector((buffer, buffer, buffer))
    max1 += Vector((buffer, buffer, buffer))
    min2 -= Vector((buffer, buffer, buffer))
    max2 += Vector((buffer, buffer, buffer))
    
    return not (max1.x < min2.x or min1.x > max2.x or
                max1.y < min2.y or min1.y > max2.y or
                max1.z < min2.z or min1.z > max2.z)

def is_in_camera_view(obj, camera):
    """Check if an object is in the camera's view."""
    cam_matrix = camera.matrix_world
    cam_pos = cam_matrix.translation
    cam_dir = cam_matrix.to_quaternion() @ Vector((0, 0, -1))  # Camera direction (Z-axis)
    
    obj_pos = obj.matrix_world.translation
    to_obj = obj_pos - cam_pos
    distance = to_obj.length
    angle = mathutils.Vector.angle(cam_dir, to_obj.normalized())
    fov = camera.data.angle
    
    return angle < fov / 2 and distance < camera.data.clip_end

def calculate_placement_area(object_names, camera, padding_factor=2.0):
    """Calculate dynamic placement area based on object sizes and camera view."""
    if not object_names:
        return 1.0  # Default size if no objects
    
    # Calculate total area needed based on object sizes
    total_area = 0
    for obj_name in object_names:
        obj = bpy.data.objects.get(obj_name)
        if obj:
            size = get_object_size(obj)
            total_area += size * size  # Approximate area as square of max dimension
    
    # Estimate placement area size (square root of total area, with padding)
    placement_size = (total_area ** 0.5) * padding_factor
    
    # Adjust based on camera's field of view
    cam_pos = camera.matrix_world.translation
    cam_dir = camera.matrix_world.to_quaternion() @ Vector((0, 0, -1))
    fov = camera.data.angle
    distance = camera.data.clip_start + (camera.data.clip_end - camera.data.clip_start) / 2
    fov_width = 2 * distance * (fov / 2).tan()  # Approximate visible width at mid-distance
    
    return min(placement_size, fov_width)

def arrange_objects(object_names, max_attempts=200):
    """Arrange objects without collisions within a dynamic placement area in camera view."""
    scene = bpy.context.scene
    camera = scene.camera
    
    if not camera:
        print("No camera found in the scene!")
        return
    
    # Calculate dynamic placement area
    placement_area_size = calculate_placement_area(object_names, camera)
    print(f"Calculated placement area size: {placement_area_size}")
    
    # Clear existing locations
    for obj_name in object_names:
        obj = bpy.data.objects.get(obj_name)
        if obj:
            obj.location = (0, 0, 0)
    
    placed_objects = []
    
    for obj_name in object_names:
        obj = bpy.data.objects.get(obj_name)
        if not obj:
            print(f"Object {obj_name} not found!")
            continue
        
        attempts = 0
        placed = False
        
        while attempts < max_attempts:
            # Random position within dynamic placement area
            x = random.uniform(-placement_area_size / 2, placement_area_size / 2)
            y = random.uniform(-placement_area_size / 2, placement_area_size / 2)
            z = 0  # Objects on ground plane
            
            obj.location = (x, y, z)
            
            # Check if object is in camera view
            if not is_in_camera_view(obj, camera):
                attempts += 1
                continue
            
            # Check for collisions with buffer
            collision = False
            for placed_obj in placed_objects:
                if is_collision(obj, placed_obj, buffer=0.1):
                    collision = True
                    break
            
            if not collision:
                placed_objects.append(obj)
                placed = True
                break
            
            attempts += 1
        
        if not placed:
            print(f"Could not place {obj_name} without collisions after {max_attempts} attempts.")
            obj.location = (0, 0, 0)  # Reset to origin if placement fails
    
    # Update the scene
    bpy.context.view_layer.update()