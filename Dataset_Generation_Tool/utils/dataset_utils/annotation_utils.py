# import modules
import os 
import json
from typing import List,Tuple,Dict,Any
from pathlib import Path
import time
import numpy as np
import yaml

import bpy
import mathutils

from utils.blender_utils.bpy_utils import *
from utils.blender_utils.render_utils import render_image
from utils.dataset_utils.scene_utils import place_objects_in_circular_arrangement,place_object_randomly


class TimerLogger:
    def __init__(self, log_path):
        self.log_path = log_path
        self.start_times = {}
        self.logs = []

    def start(self, label):
        self.start_times[label] = time.time()

    def stop(self, label):
        if label not in self.start_times:
            raise ValueError(f"No start time found for label '{label}'")
        duration = time.time() - self.start_times[label]
        self.logs.append(f"{label}: {duration:.2f} seconds")
        return duration

    def save(self):
        total_time = sum(float(line.split(":")[1].split()[0]) for line in self.logs)
        self.logs.append(f"\nTotal time: {total_time:.2f} seconds")

        with open(self.log_path, 'w') as f:
            f.write("\n".join(self.logs))
        print(f"Timing log saved to: {self.log_path}")




def save_as_json_file(data:str,file_path:Path):
    """

    Arguments:
        data -- data in dictionary format
        file_path -- str
    """
    with open(file_path,'w') as file:
        json.dump(data, file)

def update_json_file(data, file_path):
    """Updates the json file in real time

    Arguments:
        data -- data to be stored in json file
        file_path -- file path for the json file
    """
    if os.path.isfile(file_path):
        with open(file_path, 'r+') as file:
            try:
                existing_data = json.load(file)
            except json.JSONDecodeError:
                existing_data = {}
            existing_data.update(data)
            file.seek(0)
            json.dump(existing_data, file, indent=4)
            file.truncate()
            file.flush()
            os.fsync(file.fileno())
    else:
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4)
            file.flush()
            os.fsync(file.fileno())

######################## scene_gt_info.json ########################

def format_coordinates(scene,coordinates, object_name,obj_to_class):
    """Format bounding box coordinates in bop format.

    Arguments:
        scene -- Current blender's scene
        coordinates -- coordinates array
        object_name -- name of the blender object 
        obj_to_class -- dict of object names to index eg: {'cracker_box':2,etc..,}

    Returns:
        Dict: Bounding box coordinates in BOP format
    """

    res_x = scene.render.resolution_x
    res_y = scene.render.resolution_y

    if coordinates:
        x1 = (coordinates[0][0])
        x2 = (coordinates[1][0])
        y1 = (1 - coordinates[1][1]) 
        y2 = (1 - coordinates[0][1])
        
        ## Get final bounding box information
        # Calculate the absolute width, height and center of the bounding box
        width = (x2-x1)
        height = (y2-y1)
        cx = x1 + (width/2) 
        cy = y1 + (height/2)

        # get pixel data
        px_count_all = res_x * res_y
        px_count_valid = 7
        px_count_visib = 7
        visib_fract = px_count_visib / px_count_valid if px_count_valid > 0 else 0.0

        obj = bpy.data.objects[object_name]
        object_world_matrix = obj.matrix_world
        position_list = [object_world_matrix.translation.x,
                        object_world_matrix.translation.y,
                        object_world_matrix.translation.z]
        
        cam_R_m2c = object_world_matrix.to_3x3()
        cam_R_m2c_list = [[cam_R_m2c[i][j] for j in range(3)] for i in range(3)]

        rotation_quat = object_world_matrix.to_quaternion()
        # changed_quat = [rotation_quat[1], rotation_quat[2], rotation_quat[3], rotation_quat[0]]
        changed_quat = [rotation_quat[0],rotation_quat[1], rotation_quat[2], rotation_quat[3]]

        pose_data = {
        "location": position_list,
        "rotation_quaternion": changed_quat, # [w, x, y, z] format
        "rotation_euler": cam_R_m2c_list,
        "scale": list(obj.scale),
        "dimensions": list(obj.dimensions)
        }

        ## bounding box top left(x,y) width and height
        # txt_coordinates = str(_class) + ' ' + str(x1) + ' ' + str(y2) + ' ' + str(width) + ' ' + str(height) + '\n'
        txt_coordinates = {
            "obj_id": int(obj_to_class[object_name]),
            "bbox_obj":[x1*res_x,y1*res_y,width*res_x,height*res_y],
            "bbox_visib": [cx,cy,width,height],
            "px_count_all":px_count_all, 
            "px_count_valid":px_count_valid, # TODO:Check this later
            "px_count_visib":px_count_visib, # TODO:Check this later
            "visib_fract":visib_fract,
            "class_label": obj_to_class[object_name],
            "class_name": object_name,
            "3d_pose_data": pose_data
        }
        return txt_coordinates
    # If the current class isn't in view of the camera, then pass
    else:
        pass
    
    return None


def find_bounding_box(scene,obj,camera):
    """Computes camera space bounding box coordinates of the blender's mesh object.

    Arguments:
        scene -- Current blender scene
        obj -- blender object of the model for which the bounding box coordinates are required.
        camera -- blender's camera object

    Returns:
        bounding box coordinates for single object[(min_x,min_y),(max_x,max_y)]
    """

    # Get the inverse transformation matrix. 
    matrix = camera.matrix_world.normalized().inverted()
    #  Create a new mesh object, using the inverse transform matrix to undo any transformations. 
    mesh = obj.to_mesh(preserve_all_data_layers=True)
    mesh.transform(obj.matrix_world)
    mesh.transform(matrix)
    #  Get the world coordinates for the camera frame bounding box, before any transformations.
    frame = [-v for v in camera.data.view_frame(scene=scene)[:3]]

    lx = []
    ly = []

    for v in mesh.vertices:
        co_local = v.co
        z = -co_local.z

        if z <= 0.0:
            #  Vertex is behind the camera; ignore it. 
            continue
        else:
            #  Perspective division 
            frame = [(v / (v.z / z)) for v in frame]
        
        min_x, max_x = frame[1].x, frame[2].x
        min_y, max_y = frame[0].y, frame[1].y
        x = (co_local.x - min_x) / (max_x - min_x)
        y = (co_local.y - min_y) / (max_y - min_y)
        lx.append(x)
        ly.append(y)
    #  Image is not in view if all the mesh verts were ignored 
    if not lx or not ly:
        return None
    
    min_x = np.clip(min(lx), 0.0, 1.0)
    min_y = np.clip(min(ly), 0.0, 1.0)
    max_x = np.clip(max(lx), 0.0, 1.0)
    max_y = np.clip(max(ly), 0.0, 1.0)

    #  Image is not in view if both bounding points exist on the same side 
    if min_x == max_x or min_y == max_y:
        return None

    #  Figure out the rendered image size 
    render = scene.render
    fac = render.resolution_percentage * 0.01
    dim_x = render.resolution_x * fac
    dim_y = render.resolution_y * fac
    
    ## Verify there's no coordinates equal to zero
    coord_list = [min_x, min_y, max_x, max_y]
    if min(coord_list) == 0.0:
        indexmin = coord_list.index(min(coord_list))
        coord_list[indexmin] = coord_list[indexmin] + 0.0000001

    return (min_x, min_y), (max_x, max_y)

def get_point_cloud_data(object_name, camera):
    """
    Function to extract the point cloud data (vertices) of an object in Blender.
    The object will be transformed to the camera's view space.
    """
    obj = bpy.data.objects[object_name]
    # Invert the camera matrix to convert object coordinates to camera coordinates
    matrix = camera.matrix_world.normalized().inverted()

    # Create the mesh from the object, applying transformations to mesh coordinates
    mesh = obj.to_mesh(preserve_all_data_layers=True)
    mesh.transform(obj.matrix_world)  # Apply object's transformations
    mesh.transform(matrix)  # Undo camera transformation

    # Collect the vertices' positions into a list
    vertices_positions = []
    for v in mesh.vertices:
        vertices_positions.append([v.co.x, v.co.y, v.co.z])

    # Free memory by removing the mesh after use
    bpy.data.objects[object_name].to_mesh_clear()

    return vertices_positions

def compute_bbox_coordinates(scene,camera,object_name,obj_to_class):
    """Get all bounding box coordinates for the object

    Arguments:
        scene -- current blender scene
        camera -- blender's camera object
        object_name -- name of the object for which the bounding box coordinates are required 
        obj_to_class -- dict of object names to index eg: {'cracker_box':2,etc..,}

    Returns:
        Dict: Bounding box coordinates for single object in BOP format
    """
    b_box = find_bounding_box(scene=scene,
                              obj=scene.objects[object_name],
                              camera=camera
                              )
    if b_box:
        return format_coordinates(scene=scene,
                                  coordinates=b_box,
                                  object_name=object_name,
                                  obj_to_class=obj_to_class
                                  )
    return ''

def get_scene_gt_info_parameters(scene,object_names,camera,obj_to_class):
    """Computes the scene ground truth parameters (Bounding box coordinates) for all the models/objects in BOP format(scene_gt_info.json)

    Arguments:
        scene -- Current blender scene
        object_names -- List of object names present in the scene
        camera -- blender's camera object
        obj_to_class -- dict of object names to index eg: {'cracker_box':2,etc..,}

    Returns:
        List: [Dict.Dict,Dict,..] Bounding box coordinates for all the models/objects in BOP format
    """
    annotations = {}
    for obj_name in object_names:
        obj_annotations = compute_bbox_coordinates(scene=scene,
                                              camera=camera,
                                              object_name=obj_name,
                                              obj_to_class=obj_to_class
                                              )
        # BOP Annotations for multiple objects in a single image
        annotations[obj_name] = obj_annotations
    return annotations

######################## scene_gt.json ######################## 
def calculate_scene_gt_parameters(scene,obj_name,camera_object,obj_to_class):
    """Calculate the ground truth parameters of a single model/object in BOP format (scene_gt.json)
        **Note** : Also changes the coordinate system of blender's camera to OpenCV coordinate system.
        Blender : +X right, +Y forward, +Z Upwards
        OpenCV : +X right, +Z forward, +Y Upwards

    Arguments:
        scene -- Current blender scene.
        obj_name -- name of the object:str
        camera_object -- blender's camera object.
        obj_to_class -- dict of object names to index eg: {'cracker_box':2,etc..,}

    Returns:
        Dict: ground truth parameters of a single model with respect to camera.
    """       
    obj = scene.objects[obj_name]
    camera_matrix_world = camera_object.matrix_world 
    object_matrix_world = obj.matrix_world

    # Compute object pose in camera frame
    camera_matrix_world_inv = camera_matrix_world.inverted()
    object_pose_camera_frame = camera_matrix_world_inv @ object_matrix_world

    # T_m2w = np.array(obj.matrix_world)
    # T_c2w = np.array(camera_object.matrix_world)
    
    # # For BOP format Camera uses OpenCV coordinate system where +Y is pointing downwards and +Z axis farward
    # # Blender uses right hand coordinate system where +Y axix points farward and +Z axis upwards
    # # Based on observation we need to rotate the coordiante system in blender by -90 deg on X-Axis
    
    # # change_coordinate = np.array([
    # #     [1, 0, 0, 0],
    # #     [0, 0, 1, 0],
    # #     [0, 1, 0, 0],
    # #     [0, 0, 0, 1]
    # #     ])
    # # # Compute the new transformation matrix after rotating on x-axis
    # # T_c2w = change_coordinate @ T_c2w

    # T_m2c = np.dot(np.linalg.inv(T_c2w), T_m2w)
    
    # cam_R_m2c = T_m2c[:3, :3].flatten().tolist()
    # cam_t_m2c = (T_m2c[:3, 3]*1000).flatten().tolist()
    
    # # Extract rotation matrix and convert it to a quaternion
    # rotation_matrix = mathutils.Matrix(T_m2c[:3, :3])
    # cam_q_m2c = rotation_matrix.to_quaternion().normalized()

    # # Convert quaternion to list
    # cam_q_m2c = [cam_q_m2c.w, cam_q_m2c.x, cam_q_m2c.y, cam_q_m2c.z] # w, x, y, z format
    
    # Extract position (translation)
    position = [object_pose_camera_frame.translation.x,
                 object_pose_camera_frame.translation.y,
                 object_pose_camera_frame.translation.z]
    # Extract rotation as quaternion (w,x,y,z)
    rotation_quat = object_pose_camera_frame.to_quaternion()
    rotation_quat = [rotation_quat.w, rotation_quat.x, rotation_quat.y, rotation_quat.z] # w, x, y, z format
    quat_magnitude = math.sqrt(sum(q*q for q in rotation_quat))
    assert abs(quat_magnitude - 1.0) < 1e-6, "Quaternion not normalized"

    # dl_quat = [rotation_quat[1], rotation_quat[2], rotation_quat[3], rotation_quat[0]]

    cam_R_m2c = object_pose_camera_frame.to_3x3()
    cam_R_m2c_list = [[cam_R_m2c[i][j] for j in range(3)] for i in range(3)]

    gt_parameters = {
        "cam_R_m2c":cam_R_m2c_list,
        "cam_t_m2c":position,
        "cam_Q_m2c": rotation_quat, # [w, x, y, z] format
        "obj_id": int(obj_to_class[obj_name]),
        # "obj_name": obj_name
    }
    return gt_parameters

def get_scene_gt_parameters(scene,object_names,camera_object,obj_to_class):
    """Computes the ground truth parameters for all the models/objects in BOP format (scene_gt.json)

    Arguments:
        scene -- Current blender scene
        object_names -- list of object names present in the scene
        camera_object -- blender's camera object
        obj_to_class -- dict of object names to index eg: {'cracker_box':2,etc..,}

    Returns:
        List:[Dict,Dict,Dict,...] ground truth parameters of all models with respect to camera.
    """

    final_parameters = {}
    for obj_name in object_names:
        param = calculate_scene_gt_parameters(scene=scene,
                                              obj_name=obj_name,
                                              camera_object=camera_object,
                                              obj_to_class=obj_to_class
                                              )
        final_parameters[obj_name] = param

    return final_parameters

######################## scene_camera.json ######################## 
def get_k_matrix(scene,camera_object):
    """Computes camera calibration matrix K based on the reference: https://blender.stackexchange.com/questions/38009/3x4-camera-matrix-from-blender-camera/120063#120063

    Arguments:
        scene -- Current blender's scene
        camera_object -- blender's camera object

    Returns:
        List: Camera Calibration Matrix K-(3X3)
    """    
    if camera_object.data.type != 'PERSP':
        raise ValueError('Non-perspective cameras not supported')
    
    scale = scene.render.resolution_percentage / 100
    width = scene.render.resolution_x * scale # px
    height = scene.render.resolution_y * scale # py

    camdata = camera_object.data
    focal_in_mm = camdata.lens # mm
    sensor_width = camdata.sensor_width # mm
    sensor_height = camdata.sensor_height # mm
    pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y

    if (camdata.sensor_fit == 'VERTICAL'):
        # the sensor height is fixed and the sensor width is effectively changed with the pixel aspect ratio
        s_u = width / sensor_width / pixel_aspect_ratio 
        s_v = height / sensor_height
    else: # 'HORIZONTAL' and 'AUTO'
        # the sensor width is fixed and the sensor height is effectively changed with the pixel aspect ratio
        pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
        s_u = width / sensor_width
        s_v = height * pixel_aspect_ratio / sensor_height

    # parameters of intrinsic calibration matrix K
    alpha_u = focal_in_mm * s_u
    alpha_v = focal_in_mm * s_v
    u_0 = width / 2
    v_0 = height / 2
    skew = 0 # only use rectangular pixels
    K = np.array([
        [alpha_u,    skew, u_0],
        [      0, alpha_v, v_0],
        [      0,       0,   1]
    ], dtype=np.float32)
    # s = intrinsics.skew

    cam_K = K.flatten().tolist()
    return cam_K


def get_scene_camera_parameters(scene, camera_object):
    """Computes the scene parameters in BOP format (scene_camera.json)

    Arguments:
        camera_object -- blender's camera object

    Returns:
        Dict: camera parameters for the scene
    """
        
    # Transformation of camera with respect to world
    T_C_W = np.array(camera_object.matrix_world)

    # For BOP format Camera uses OpenCV coordinate system where +Y is pointing downwards and +Z axis farward
    # Blender uses right hand coordinate system where +Y axix points farward and +Z axis upwards
    # Based on observation we need to rotate the coordiante system in blender by -90 deg on X-Axis TODO

    # Correct coordinate system conversion: Rotate by -90 degrees on X-axis
    change_coordinate = np.array([
        [1,  0,  0, 0],  
        [0,  0, -1, 0],  
        [0,  1,  0, 0],  
        [0,  0,  0, 1]
    ])

    # Compute the new transformation matrix T_C_W
    T_C_W = change_coordinate @ T_C_W

    # Compute the transformation of world with respect to camera
    T_W_C = np.linalg.inv(T_C_W)

    # Extract rotation and translation
    cam_R_w2c = T_W_C[:3, :3].flatten().tolist()
    cam_t_w2c = (T_W_C[:3, 3]).flatten().tolist()

    # Return scene parameters in BOP format
    scene_parameters = {
        "cam_K": get_k_matrix(scene, camera_object), 
        "cam_R_w2c": cam_R_w2c, 
        "cam_t_w2c": cam_t_w2c,  
        "depth_scale": 0.1
    }

    return scene_parameters


def get_sequential_step_values(start_value, stop_value,num_images):
    step_value = (stop_value - start_value) / (num_images - 1)
    numbers_list = np.arange(
        start_value, stop_value + step_value, step_value)

    return numbers_list


def get_scene_bounds():
    # Initialize bounds
    min_x = min_y = min_z = float('inf')
    max_x = max_y = max_z = float('-inf')
    
    # Iterate through all objects in scene
    for obj in bpy.data.objects:
        # Skip cameras, lights, etc.
        if obj.type not in ['MESH', 'EMPTY','CURVE']:
            continue
            
        # Get world space corners of object's bounding box
        for corner in obj.bound_box:
            # Transform corner to world space
            world_corner = obj.matrix_world @ mathutils.Vector(corner)
            
            # Update bounds
            min_x = min(min_x, world_corner.x)
            min_y = min(min_y, world_corner.y)
            min_z = min(min_z, world_corner.z)
            max_x = max(max_x, world_corner.x)
            max_y = max(max_y, world_corner.y)
            max_z = max(max_z, world_corner.z)
    
    # Add small padding (10% of total size)
    size_x = max_x - min_x
    size_y = max_y - min_y
    size_z = max_z - min_z
    padding = 0.1  # 10%
    
    min_bound = [
        min_x - size_x * padding,
        min_y - size_y * padding,
        min_z - size_z * padding
    ]
    max_bound = [
        max_x + size_x * padding,
        max_y + size_y * padding,
        max_z + size_z * padding
    ]
    
    return {"min_bound":min_bound, "max_bound":max_bound}


# Save the ground truth blender parameters from blender
def save_gt_blender_parameters(camera,light_source,object_names,deformation_value):
    """
    Distance_offset_track_to_constraint,
    Light_intensity,
    Blur_value,
    Focal_length
    Distance between camera and object
    Deformation_value: Amount of stress or deformation
    """
    print("Object names in blender parameters :", object_names)
    blender_gt_parameters = {"Distance_offset_track_to_constraint": round(camera.constraints['Follow Path'].offset,6),
                             "Light_intensity":round(light_source.data.energy,6),
                             "Blur_value":camera.data.dof.focus_distance,
                             "Focal_length":camera.data.lens,
                             "Camera_sensor_width":camera.data.sensor_width,
                             "Camera_sensor_height":camera.data.sensor_height,
                             "Deformation_value":deformation_value
                             }
    blender_gt_parameters["Distance"] = {} 
    for object_name in object_names:
        object = bpy.data.objects[object_name]
        
        object_world_location = object.matrix_world.translation
        camera_world_location = camera.matrix_world.translation
        
        # Update the distance using world locations
        distance = np.linalg.norm(object_world_location - camera_world_location)
        blender_gt_parameters["Distance"][object_name] = round(distance, 6)

    blender_gt_parameters['Scene_bounds'] = get_scene_bounds()
    

    return blender_gt_parameters


def save_annotation_files_bop(idx,scene,camera,light_source,
                          deformation_value,
                          object_names,class_to_id,
                          output_path:str
                          ):
    """"
    Store all the annotation files

    """
    # Create dictionaries to store BOP annotations.
    scene_camera_params = {}
    scene_gt_params = {}
    scene_gt_info_params = {}
    blender_gt_params = {}

    # Store the json labels
    scene_camera_params[str(idx)] = get_scene_camera_parameters(scene=scene,camera_object=camera) 
    scene_gt_params[str(idx)] = get_scene_gt_parameters(scene,object_names,camera_object=camera,obj_to_class=class_to_id)
    scene_gt_info_params[str(idx)] = get_scene_gt_info_parameters(scene,object_names,camera,obj_to_class=class_to_id)

    # Store blender gt parameters
    blender_gt_params[str(idx)] = save_gt_blender_parameters(camera,
                                                             light_source,
                                                             object_names,
                                                             deformation_value
                                                             )

    scene_cam_path = output_path /'scene_camera.json'
    scene_gt_path = output_path/'scene_gt.json'
    scene_gt_info_path = output_path/'scene_gt_info.json'
    blender_gt_params_path = output_path/'blender_gt.json'

    update_json_file(data=scene_camera_params,file_path=scene_cam_path)
    update_json_file(data=scene_gt_params,file_path=scene_gt_path)
    update_json_file(data=scene_gt_info_params,file_path=scene_gt_info_path)
    update_json_file(data=blender_gt_params,file_path=blender_gt_params_path)


############################# Object Detection and Classification #############################

def calculate_bounding_box_coordinates(scene,coordinates, object_name,obj_to_class,annotation_format):
    """Format bounding box coordinates in bop format.

    Arguments:
        scene -- Current blender's scene
        coordinates -- coordinates array
        object_name -- name of the blender object 
        obj_to_class -- dict of object names to index eg: {'cracker_box':2,etc..,}

    Returns:
        Dict: Bounding box coordinates in BOP format
    """

    res_x = scene.render.resolution_x
    res_y = scene.render.resolution_y

    if coordinates:
        x1 = (coordinates[0][0])
        x2 = (coordinates[1][0])
        y1 = (1 - coordinates[1][1]) 
        y2 = (1 - coordinates[0][1])
        
        ## Get final bounding box information
        # Calculate the absolute width, height and center of the bounding box
        width = (x2-x1)
        height = (y2-y1)
        cx = x1 + (width/2) 
        cy = y1 + (height/2)


        obj = bpy.data.objects[object_name]
        object_world_matrix = obj.matrix_world


        if annotation_format == 'COCO':
            # bounding box top left(x,y) width and height
            txt_coordinates = {
                                "category_id": obj_to_class[object_name],
                                "bbox": [
                                    x1 * res_x,        # top-left x (absolute)
                                    y1 * res_y,        # top-left y (absolute)
                                    width * res_x,     # width (absolute)
                                    height * res_y     # height (absolute)
                                ],
                                "area": (width * res_x) * (height * res_y)
                            }
        elif annotation_format == 'YOLO':
            # txt_coordinates = str(obj_to_class[object_name]) + ' ' + str(x1) + ' ' + str(y2) + ' ' + str(width) + ' ' + str(height) + '\n'
            txt_coordinates = f"{obj_to_class[object_name]} {cx} {cy} {width} {height}\n"

        return txt_coordinates
    # If the current class isn't in view of the camera, then pass
    else:
        pass
    
    return None

def get_all_bbox_coordinates(scene,camera,object_name,obj_to_class,annotation_format):
    """Get all bounding box coordinates for the object

    Arguments:
        scene -- current blender scene
        camera -- blender's camera object
        object_name -- name of the object for which the bounding box coordinates are required 
        obj_to_class -- dict of object names to index eg: {'cracker_box':2,etc..,}

    Returns:
        Dict: Bounding box coordinates for single object in BOP format
    """
    b_box = find_bounding_box(scene=scene,
                              obj=scene.objects[object_name],
                              camera=camera
                              )
    if b_box:
        return calculate_bounding_box_coordinates(scene=scene,
                                  coordinates=b_box,
                                  object_name=object_name,
                                  obj_to_class=obj_to_class,
                                  annotation_format=annotation_format
                                  )
    return ''

def get_bounding_box_info(scene,object_names,camera,obj_to_class,annotation_format):
    """Computes the scene ground truth parameters (Bounding box coordinates) for all the models/objects in BOP format(scene_gt_info.json)

    Arguments:
        scene -- Current blender scene
        object_names -- List of object names present in the scene
        camera -- blender's camera object
        obj_to_class -- dict of object names to index eg: {'cracker_box':2,etc..,}

    Returns:
        List: [Dict.Dict,Dict,..] Bounding box coordinates for all the models/objects in BOP format
    """
    annotations = []
    for obj_name in object_names:
        obj_annotations = get_all_bbox_coordinates(scene=scene,
                                              camera=camera,
                                              object_name=obj_name,
                                              obj_to_class=obj_to_class,
                                              annotation_format=annotation_format
                                              )
        if obj_annotations is not None:
            annotations.append(obj_annotations)
    
    return annotations



def save_annotation_files_object_detection(idx,scene,camera,light_source,
                          deformation_value,
                          object_names,class_to_id,
                          output_path:str,
                          dataset_label_format:str
                          ):
    """"
    Store all the annotation files

    """
    # Create dictionaries to store BOP annotations.
    scene_camera_params = {}
    blender_gt_params = {}

    # Store Scene camera parameters
    scene_camera_params[str(idx).zfill(6)] = get_scene_camera_parameters(scene=scene,camera_object=camera)
    scene_cam_path = output_path /'scene_camera.json'
    update_json_file(data=scene_camera_params,file_path=scene_cam_path)

    # Store blender gt parameters
    blender_gt_params[str(idx).zfill(6)] = save_gt_blender_parameters(camera,
                                                             light_source,
                                                             object_names,
                                                             deformation_value
                                                             )
    blender_gt_params_path = output_path/'blender_gt.json'
    update_json_file(data=blender_gt_params,file_path=blender_gt_params_path)

    # Store bounding box parameters
    if dataset_label_format == 'YOLO':
        labels_filepath = os.path.join(output_path ,'labels')
        os.makedirs(labels_filepath, exist_ok=True)
        
        scene_gt_params = get_bounding_box_info(scene,object_names,camera,obj_to_class=class_to_id,annotation_format=dataset_label_format)
        bbox_labels_path = f"{labels_filepath}/{str(idx).zfill(6)}.txt" 
        with open(bbox_labels_path, 'w') as file:
            for item in scene_gt_params:
                file.write(item)

        # save data.yaml file
        data_yaml_path = output_path / 'data.yaml'
        data_yaml_content = {
            'nc': len(class_to_id),
            'names': {id: name for name, id in class_to_id.items()}
        }
        with open(data_yaml_path, 'w') as file:
            yaml.dump(data_yaml_content, file, default_flow_style=False)
    
    elif dataset_label_format == 'COCO':

        bbox_labels_path = output_path / 'instances_default.json'
        
        # Get image size (from render settings or your resolution)
        res_x = scene.render.resolution_x
        res_y = scene.render.resolution_y
        
        # Get object annotations
        objects_info = get_bounding_box_info(
            scene, object_names, camera,
            obj_to_class=class_to_id,
            annotation_format=dataset_label_format
        )
        
        # Build proper COCO structure
        image_entry, annotations = format_coco_annotation(
            idx, f"{str(idx).zfill(6)}.png", res_x, res_y,
            objects_info, class_to_id
        )
        
        scene_gt_params = {
            "images": [image_entry],
            "annotations": annotations,
            "categories": [{"id": v, "name": k} for k,v in class_to_id.items()]
        }

        update_json_file_coco(scene_gt_params, bbox_labels_path)


def format_coco_annotation(idx, image_name, width, height, objects_info, obj_to_class):
    """
    Format COCO annotations for one image.
    """

    image_entry = {
        "id": idx,
        "file_name": image_name,
        "width": width,
        "height": height
    }

    annotations = []
    ann_id = 1
    for value in objects_info:
        if value is not None and type(value) is not str:
            annotations.append({
                "id": ann_id,
                "image_id": idx,
                "category_id": value["category_id"],
                "bbox":  value["bbox"],
                "area": value["area"],
                "iscrowd": 0
            })
            ann_id += 1
        else:
            annotations.append({
                "id": ann_id,
                "image_id": idx,
                "category_id": '',
                "bbox":  '',
                "area": '',
                "iscrowd": 0
            })
            ann_id += 1

    return image_entry, annotations


def update_json_file_coco(data, file_path):
    """Updates the COCO JSON file incrementally"""
    if os.path.isfile(file_path):
        with open(file_path, 'r+') as file:
            try:
                existing_data = json.load(file)
            except json.JSONDecodeError:
                existing_data = {"images": [], "annotations": [], "categories": []}

            # Append instead of overwrite
            if "images" in data:
                existing_data.setdefault("images", [])
                existing_data["images"].extend(data["images"])

            if "annotations" in data:
                existing_data.setdefault("annotations", [])
                existing_data["annotations"].extend(data["annotations"])

            if "categories" in data:
                existing_data.setdefault("categories", [])
                # Only add new categories
                existing_cats = {c["id"] for c in existing_data["categories"]}
                for c in data["categories"]:
                    if c["id"] not in existing_cats:
                        existing_data["categories"].append(c)

            # Write back
            file.seek(0)
            json.dump(existing_data, file, indent=4)
            file.truncate()
            file.flush()
            os.fsync(file.fileno())
    else:
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4)
            file.flush()
            os.fsync(file.fileno())


############################################################################
def save_annotation_files_classification(file_name,scene,camera,light_source,
                          deformation_value,
                          object_names,class_to_id,
                          output_path:str
                          ):
    """"
    Store all the annotation files

    """
    # Create dictionaries to store BOP annotations.
    blender_gt_params = {}

    # Store blender gt parameters
    blender_gt_params[str(file_name)] = save_gt_blender_parameters(camera,
                                                             light_source,
                                                             object_names,
                                                             deformation_value
                                                             )

    blender_gt_params_path = output_path/'blender_gt.json'

    update_json_file(data=blender_gt_params,file_path=blender_gt_params_path)

###################################################################################################

def generate_6d_scene_format(scene,background_surface,object_names,class_to_id,camera,dataset_name,object_placement,results_dir):
    print("Generating 6D Pose Dataset")

    output_path = Path(results_dir + dataset_name + '/')
    
    for name in object_names:
        object_to_render = bpy.data.objects[name] 
        object_to_render.hide_render = False

    if object_placement == 'circle':
        place_objects_in_circular_arrangement(obj_names=object_names,radius=1.25)
    elif object_placement == 'random':
        raise NotImplementedError("Need to Implement this feature")
    elif object_placement == 'default':
        pass

    scene.use_gravity = True
    scene.view_layers['ViewLayer'].use_pass_z= True
    scene.view_layers['ViewLayer'].use_pass_mist = True
    scene.use_nodes = True
    tree = scene.node_tree
    
    if not tree.nodes['Render Layers']:
        render_layers_node = tree.nodes.new(type='CompositorNodeRLayers')
    else:
        render_layers_node = tree.nodes['Render Layers']

    if not tree.nodes['Composite']:
        output_node = tree.nodes.new(type='Composite')
    else:
        output_node = tree.nodes['Composite']

    links = tree.links

    # Create dictionaries to store BOP annotations.
    scene_camera_params = {}
    scene_gt_params = {}
    scene_gt_info_params = {}

    line_points = get_sequential_step_values(start_value=-50,stop_value=-70) # basic scene

    # line_points = get_sequential_step_values(start_value=-20,stop_value=-60) # lab_environment

    # set_location(scene=scene,obj_name='Sun',location=(0.0,0.0,5.0))
    # set_light_intensity(scene=scene,light_name='Sun',intensity_value=6.0)

    for idx,loc_value in enumerate(line_points):
        scene = bpy.context.scene
        scene.camera = camera
        camera.data.type = 'PERSP'
        camera.data.lens = 150
        set_camera_postion_on_path(camera_object=camera,distance_value=loc_value)
        
        # Randomly rotate the object and place in camera view
        place_object_randomly(scene=scene,obj=object_to_render,camera=camera,min_distance_factor=0.2)  # this is for one object change it later
        
        camera.constraints['Track To'].target = background_surface # check later for better method track to object

        # Update file path for rgb images and render the image.
        scene.render.filepath = os.path.join(output_path /Path('rgb'), str(f"{str(idx).zfill(6)}.png"))
        print("File name of rgb image is  : ", scene.render.filepath)
        links.new(render_layers_node.outputs["Image"], output_node.inputs['Image'])
        bpy.ops.render.render(write_still = True)

        # Store the json labels
        scene_camera_params[str(idx)] = get_scene_camera_parameters(scene=scene,camera_object=camera) 
        scene_gt_params[str(idx)] = get_scene_gt_parameters(scene,object_names,camera_object=camera,obj_to_class=class_to_id)
        scene_gt_info_params[str(idx)] = get_scene_gt_info_parameters(scene,object_names,camera,obj_to_class=class_to_id)

        # Create depth images
        scene.render.filepath = os.path.join(output_path /Path('depth'), str(f"{str(idx).zfill(6)}.png"))
        print("File name of depth image is  : ", scene.render.filepath)
        links.new(render_layers_node.outputs["Depth"], output_node.inputs['Image'])
        bpy.ops.render.render(write_still = True)


    scene_cam_path = output_path /'scene_camera.json'
    scene_gt_path = output_path/'scene_gt.json'
    scene_gt_info_path = output_path/'scene_gt_info.json'

    save_as_json_file(data=scene_camera_params,file_path=scene_cam_path)
    save_as_json_file(data=scene_gt_params,file_path=scene_gt_path)
    save_as_json_file(data=scene_gt_info_params,file_path=scene_gt_info_path)

    print("************************** Completed Rendering process **************************")