import bpy
import numpy as np
import random
import math
from mathutils import Color, Euler, Vector
from pathlib import Path
import os


def get_min_max_values(test_case, constraint_settings_dict):
    """
    This function returns the minimum and maximum values from the json object for the particular test case

    Args:
        test_case: str
        constraint_settings_dict : Dict 

    """
    min_value = constraint_settings_dict[str(test_case)]['min']
    max_value = constraint_settings_dict[str(test_case)]['max']

    return float(min_value), float(max_value)

def get_norm_constraint_light_value(constraint_settings_dict):
    """
    This function returns the min and max values for the lighting based on the normal constraint settings
    
    Args:
        constraint_settings_dict : Dict 
    """
    min_value = constraint_settings_dict['normal']['light_min']
    max_value = constraint_settings_dict['normal']['light_max']

    return (float(min_value)+float(max_value))/2


def get_sequential_step_values(test_case, json_object):

    start_value,stop_value = get_min_max_values(test_case=test_case,constraint_settings_dict=json_object['constraint_settings'])

    num_elements = int(json_object['constraint_settings'][str(test_case)]['num_images'])

    step_value = (stop_value - start_value) / (num_elements - 1)
    numbers_list = np.arange(
        start_value, stop_value + step_value, step_value)

    return numbers_list

def set_random_rotation(obj_to_change):
    """
    Applies a random rotation to the given object.

    Keyword arguments:
    obj_to_change: blender object
    """
    random_rotat_values = [
        random.random()*2*math.pi, random.random()*2*math.pi, random.random()*2*math.pi]
    obj_to_change.rotation_euler = Euler(random_rotat_values, 'XYZ')

def set_random_lighting(light_source_name, min_value, max_value):
    """
    Applies random light intensities to the scene.

    Keyword arguments:
    light_source_name:str
    min_value: float
    max_value: float
    """
    # bpy.data.lights[str(light_source_name)].energy = round(random.uniform(min_value,max_value),2)
    bpy.data.lights[str(light_source_name)].energy = random.uniform(
        min_value, max_value)

def set_random_focal_length(camera_name, min_value, max_value):
    value = random.randint(min_value, max_value)
    bpy.data.cameras[str(camera_name)].lens = float(value)
