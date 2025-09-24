# import modules
import os
import math
import numpy as np
import re
import random

import bpy


# Helper functions
def clear_scene():
    """
    Clear the default scene of blender
    """
    # Delete all objects
    bpy.data.objects.remove(bpy.data.objects['Cube'])

    # Delete all cameras
    for camera in bpy.data.cameras:
        bpy.data.cameras.remove(camera)

    # Delete all lights
    for light in bpy.data.lights:
        bpy.data.lights.remove(light)

    return None

def set_location(scene,obj_name,location:tuple):
    """
    Set the location of the object

    Keyword arguments:
        obj_name -- Name of the mesh object: str 
        location -- tuple(float,float,float)
    """
    scene.objects[str(obj_name)].location = location

def set_rotation(scene,obj_name,rotation):
    """
    Set the rotation of the object

    Keyword arguments:
        obj_name -- Name of the mesh object: str 
        rotation -- tuple(float,float,float)
    """
    scene.objects[str(obj_name)].rotation_euler = rotation

def set_camera_rotation(scene,camera_obj):

    target_location = bpy.data.objects['Camera'].location
    direction = target_location - camera_obj.location
    rotation_matrix = direction.to_track_quat('-Z', 'Y')
    camera_obj.rotation_euler = rotation_matrix.to_euler()


def add_beizer_curve(name, location, scale):
    bpy.ops.curve.primitive_bezier_circle_add(radius = 1,
                                              enter_editmode=False,
                                              align='WORLD',
                                              location=location,
                                              scale=scale
                                              )
    bpy.data.objects['BezierCircle'].name = str(name)
    return bpy.data.objects[str(name)]

def set_random_curve_height(curve_object_name, min_value, max_value):
    """ This function sets the height of the path curve for randomness/variation
    """
    z_value = round(random.uniform(min_value, max_value), 2)
    bpy.data.objects[str(curve_object_name)].location[2] = z_value

def track_object(tracking_object_name, object_to_track):
    """ Function to make camera track the objects
    """
    tracking_object = bpy.data.objects[str(tracking_object_name)]
    if not str('Track To') in tracking_object.constraints.keys():
        tracking_object.constraints.new(type="TRACK_TO")
        tracking_object.constraints['Track To'].target = object_to_track
    else:
        tracking_object.constraints['Track To'].target = object_to_track

def set_light_intensity(scene,light_source,intensity_value:float):
    """
    Set the energy value for the light object.

    Keyword arguments:
        light_name:str
        intensity_value: float
    """
    
    light_source.data.energy = intensity_value

def set_focal_length(camera,value):
    camera.lens = float(value)

def add_blur_dof(camera,blur_value):
    """Adds blur effect to the images using depth of field parameter of the camera.

    Keyword arguments:
        camera: blender camera object
        blur_value: float
    """

    camera.data.dof.use_dof = True
    # camera.data.dof.focus_object = bpy.data.objects[str(focus_background_name)]
    camera.data.dof.focus_distance = blur_value

def track_camera_object(object_to_track):
    camera = bpy.data.objects['Camera']
    if not str('Track To') in camera.constraints.keys():
        camera.constraints.new(type="TRACK_TO")
        camera.constraints['Track To'].target = object_to_track

def track_light_object(object_to_track):
    light = bpy.data.objects['Sun']
    if not str('Track To') in light.constraints.keys():
        light.constraints.new(type="TRACK_TO")
        light.constraints['Track To'].target = object_to_track

def random_camera_position(camera_name):
    camera = bpy.data.objects[str(camera_name)]
    camera.constraints['Follow Path'].offset = random.randint(0, 100)

def random_light_position(light_name):
    light = bpy.data.objects[str(light_name)]
    light.constraints['Follow Path'].offset = random.randint(0, 100)

def reset_obj_location_rotation(obj):
    obj.location = (0.0, 0.0, 0.0)
    obj.rotation_euler = (0.0, 0.0, 0.0)

def set_camera_postion_on_path(camera_object,distance_value):
    camera_object.constraints['Follow Path'].offset = distance_value


def add_camera(camera_name: str,location:tuple,rotation:tuple):
    """Adds Camera object to the scene at desired location and rotation

    Arguments:
        camera_name -- name for the camera object
        location -- (x,y,z)-meters
        rotation -- (x,y,z)-degrees

    Returns:
        blender's camera object
    """
    
    bpy.ops.object.camera_add(location=location,
                              rotation=np.deg2rad(rotation)
                              )
    camera = bpy.context.active_object
    camera.data.lens = 50
    camera.data.name = str(camera_name)
    camera.name = str(camera_name)
    return bpy.data.objects[str(camera_name)]

def add_background_plane(plane_name:str,plane_size:float):
    """Adds a plane object to the scene at world origin

    Arguments:
        plane_name  -- name for the plane object
        plane_size  -- size of the plane

    Returns:
        blender's plane object
    """
    bpy.ops.mesh.primitive_plane_add(size=plane_size,
                                     enter_editmode=False,
                                     align='WORLD',
                                     location=(0, 0, 0)
                                     )
    bpy.data.objects['Plane'].name = plane_name
    return bpy.data.objects[str(plane_name)]

def add_light_source(light_name: str,light_type: str,shadow: bool):
    """Adds a light source with respective type to the origin of the scene.

    Arguments:
        light_name -- name for the light object
        light_type -- type of light source: SUN,POINT,AREA,SPOT
        shadow -- enable or disable shadows:

    Returns:
        blender's light object
    """
    bpy.ops.object.light_add(type=str(light_type).upper(),
                             radius=1,
                             location=(0, 0, 0),
                             )
    bpy.context.object.data.name = str(light_name)
    bpy.context.object.data.energy = 5
    bpy.context.object.data.use_shadow = shadow
    bpy.context.object.data.use_contact_shadow = shadow
    return bpy.data.objects[str(light_name)]

def add_nurbs_path(path_name:str,location:tuple,rotation:tuple,scale:tuple):
    """Adds a nurbs path object (line) to the scene.

    Args:
        path_name : Name for the path
        location : (x,y,z)-meters
        rotation : (x,y,z)-degrees
        scale : (x,y,z)-meters

    Returns:
        blender's nurbs path object 
    """
    bpy.ops.curve.primitive_nurbs_path_add(enter_editmode=False,
                                           align='WORLD',
                                           location=location,
                                           rotation=np.deg2rad(rotation),
                                           scale=scale
                                           )
    bpy.data.objects['NurbsPath'].name = str(path_name)
    return bpy.data.objects[str(path_name)]

def get_texture_map_paths(texture_folder):
    """
    Returns paths for the images which can be used for image textures.

    Keyword arguments:
        texture_folder-- Path for the folder containing the images.
    """
    texture_dict = {
        "normal_map": None,
        "base_color": None,
        "disp_map": None,
        "metal_map": None,
        "roughness_map": None
    }

    files = os.listdir(path=texture_folder)
    for file in files:
        # Match the files and separate
        nrm_match = re.search(r'[\w\-_]*normal', file, re.IGNORECASE) or re.search(r'[\w\-_]*JPG_NormalGL', file, re.IGNORECASE)
        base_match = re.search(r'[\w\-_]*basecolor', file, re.IGNORECASE) or re.search(r'[\w\-_]*COL_VAR1', file, re.IGNORECASE) or \
                     re.search(r'[\w\-_]*diff', file, re.IGNORECASE) or re.search(r'[\w\-_]*col', file, re.IGNORECASE) or \
                     re.search(r'[\w\-_]*JPG_ColorGL', file, re.IGNORECASE)
        disp_match = re.search(r'[\w\-_]*DISP_4K', file, re.IGNORECASE) or re.search(r'[\w\-_]*height', file, re.IGNORECASE) or \
                     re.search(r'[\w\-_]*displacement', file, re.IGNORECASE) or re.search(r'[\w\-_]*disp', file, re.IGNORECASE) or \
                     re.search(r'[\w\-_]*JPG_Displacement', file, re.IGNORECASE)
        metal_match = re.search(r'[\w\-_]*metallic', file, re.IGNORECASE) or re.search(r'[\w\-_]*REFL', file, re.IGNORECASE) or \
                      re.search(r'[\w\-_]*metal', file, re.IGNORECASE) or re.search(r'[\w\-_]*JPG_AmbientOcclusion', file, re.IGNORECASE)
        rough_match = re.search(r'[\w\-_]*roughness', file, re.IGNORECASE) or re.search(r'[\w\-_]*GLOSS', file, re.IGNORECASE) or \
                      re.search(r'[\w\-_]*rough', file, re.IGNORECASE) or re.search(r'[\w\-_]*JPG_Roughness', file, re.IGNORECASE)

        if nrm_match:
            texture_dict['normal_map'] = os.path.join(texture_folder, file)
        if base_match:
            texture_dict['base_color'] = os.path.join(texture_folder, file)
        if disp_match:
            texture_dict['disp_map'] = os.path.join(texture_folder, file)
        if metal_match:
            texture_dict['metal_map'] = os.path.join(texture_folder, file)
        if rough_match:
            texture_dict['roughness_map'] = os.path.join(texture_folder, file)

    return texture_dict

def get_texture_paths(texture_dir):
    """
    Gets the paths for all the texture_folders from the main textures folder and returns them as a list

    Keyword arguments:
        texture_dir-- path for the directory containig the textures.
    """
    bnames = os.listdir(texture_dir)
    for i, cs in enumerate(zip(*bnames)):
        if len(set(cs)) != 1:
            break
    for _i, cs in enumerate(zip(*[b[::-1] for b in bnames])):
        if len(set(cs)) != 1:
            break
    texture_paths = [os.path.join(texture_dir, bname+'/')
                        for bname in bnames if not bname.endswith('.md')]
    return texture_paths

def set_random_pbr_img_textures(textures_path, obj_name, scale=1):
    """
    Applies image textures randomly from the available images to the specified object.

    Keyword arguments:
        texture_paths-- list of paths for the texture folders.
        obj_name-- name of the object to change the materail/texture
    """

    texture_paths = get_texture_paths(textures_path)
    texture_path = random.choice(texture_paths)

    material_name = str(texture_path.split('/')[-2])
    texture_dict = get_texture_map_paths(texture_folder=texture_path)

    # create a new material with the name.
    material = bpy.data.materials.new(
        name=material_name)  # Change name everytime
    material.use_nodes = True

    # Create the nodes for the material

    # Nodes for controlling the texture
    texture_coordinate = material.node_tree.nodes.new(
        type="ShaderNodeTexCoord")
    mapping_node = material.node_tree.nodes.new(type="ShaderNodeMapping")
    mapping_node.inputs['Scale'].default_value = (
        scale, scale, scale)  # Control the scale of the texture

    # Create vector nodes
    normal_map = material.node_tree.nodes.new(type="ShaderNodeNormalMap")
    # displacement_map = material.node_tree.nodes.new(
    #     type="ShaderNodeDisplacement")
    bump_map = material.node_tree.nodes.new(type="ShaderNodeBump")
    invert_node = material.node_tree.nodes.new(type="ShaderNodeInvert")

    # Nodes for principled bsdf and output
    principled_bsdf = material.node_tree.nodes['Principled BSDF']
    material_output = material.node_tree.nodes['Material Output']

    # Connect the nodes
    material.node_tree.links.new(
        texture_coordinate.outputs['UV'], mapping_node.inputs['Vector'])
    # Nodes for image textures
    # Base color
    if texture_dict['base_color'] != None:
        base_color_img = material.node_tree.nodes.new(
            type="ShaderNodeTexImage")
        base_color_img.image = bpy.data.images.load(
            texture_dict['base_color'])
        material.node_tree.links.new(
            mapping_node.outputs['Vector'], base_color_img.inputs['Vector'])
        material.node_tree.links.new(
            base_color_img.outputs['Color'], principled_bsdf.inputs['Base Color'])

    # Normal map
    if texture_dict['normal_map'] != None:
        normal_img = material.node_tree.nodes.new(
            type="ShaderNodeTexImage")
        normal_img.image = bpy.data.images.load(texture_dict['normal_map'])
        material.node_tree.links.new(
            mapping_node.outputs['Vector'], normal_img.inputs['Vector'])
        material.node_tree.links.new(
            normal_img.outputs['Color'], normal_map.inputs['Color'])
        material.node_tree.links.new(
            normal_map.outputs['Normal'], principled_bsdf.inputs['Normal'])

    # Displacement map
    # if texture_dict['disp_map'] != None:
    #     displacement_img = material.node_tree.nodes.new(
    #         type="ShaderNodeTexImage")
    #     displacement_img.image = bpy.data.images.load(
    #         texture_dict['disp_map'])
    #     material.node_tree.links.new(
    #         mapping_node.outputs['Vector'], displacement_img.inputs['Vector'])
    #     material.node_tree.links.new(
    #         displacement_img.outputs['Color'], displacement_map.inputs['Height'])
    #     material.node_tree.links.new(
    #         displacement_map.outputs['Displacement'], material_output.inputs['Displacement'])

    # Roughness map
    if texture_dict['roughness_map'] != None:
        roughness_img = material.node_tree.nodes.new(
            type="ShaderNodeTexImage")
        roughness_img.image = bpy.data.images.load(
            texture_dict['roughness_map'])
        material.node_tree.links.new(
            mapping_node.outputs['Vector'], roughness_img.inputs['Vector'])
        material.node_tree.links.new(
            roughness_img.outputs['Color'], invert_node.inputs['Color'])
        material.node_tree.links.new(
            invert_node.outputs['Color'], principled_bsdf.inputs['Roughness'])

    # Metal map
    if texture_dict['metal_map'] != None:
        metallic_img = material.node_tree.nodes.new(
            type="ShaderNodeTexImage")
        metallic_img.image = bpy.data.images.load(
            texture_dict['metal_map'])
        material.node_tree.links.new(
            mapping_node.outputs['Vector'], metallic_img.inputs['Vector'])
        material.node_tree.links.new(
            metallic_img.outputs['Color'], principled_bsdf.inputs['Metallic'])

    # Set material final output
    material.node_tree.links.new(
        principled_bsdf.outputs['BSDF'], material_output.inputs['Surface'])

    # Set the material to the object
    obj = bpy.context.scene.objects[obj_name]

    if obj.data.materials:
        obj.data.materials.append(material)
        obj.active_material = bpy.data.materials[material_name]
    else:
        obj.data.materials.append(material)
        obj.active_material = material

    return material