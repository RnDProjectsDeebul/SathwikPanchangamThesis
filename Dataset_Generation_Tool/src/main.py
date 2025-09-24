# Import modules
import os
import sys
import json   
import subprocess  
import warnings
import argparse
warnings.filterwarnings("ignore")

import bpy
print("Blender version: ", bpy.app.version_string)
print("Blender API version: ", bpy.app.version)
bpy.context.preferences.view.show_developer_ui = False

# Custom modules
sys.path.append('/home/sathwik/.local/lib/python3.10/site-packages/')
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = root+'/src/'
sys.path.append(src_path)
sys.path.append(root)

from utils.dataset_utils.object_utils import import_models, get_object_names_cls_idx_blender,place_objects_on_target_object
from utils.blender_utils.render_utils import set_render_parameters 
from utils.blender_utils.bpy_utils import clear_scene
from utils.dataset_utils.scene_utils import create_trajectory_scene

from utils.dataset_utils.classification_dataset import generate_classification_dataset
from utils.dataset_utils.object_detection_dataset import generate_object_detection_dataset
from utils.dataset_utils.sixd_pose_dataset import generate_6d_scene_format


#########################################################
# Load config file
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, help='Config file name (without .json extension)')
args = parser.parse_args()

config_dir = os.path.join(root, 'data', 'config_dir')
config_path = os.path.join(config_dir, args.config + '.json')

if not os.path.exists(config_path):
    raise FileNotFoundError(f"Config file not found: {config_path}")

with open(config_path, 'r') as f:
    config_file = json.load(f)
    print("Loaded config:", config_path)

#########################################################
# Get configurations from the loaded JSON file
path_config =config_file['path_settings']
data_config = config_file['dataset_settings']
render_config = config_file['render_settings']
constraint_config = config_file['constraint_settings']


# Create the scene and get the names of the objects.
if data_config.get("scene_type") == 'basic_scene':
    clear_scene()
    surface_to_place, light_source, robot_camera,main_camera, camera_track, light_track = create_trajectory_scene(trajectory_type=config_file['dataset_settings']['camera_trajectory'])
elif data_config.get("scene_type") == 'lab_environment':
    scenes_dir = os.path.join(root, 'data', 'scenes_dir')
    blend_file_path = os.path.join(scenes_dir, 'lab_environment', 'Lab_Environment.blend')
    bpy.ops.wm.open_mainfile(filepath=blend_file_path)
    surface_to_place = bpy.data.objects['Table.006']
    robot_camera = bpy.data.objects['Robot_Camera']
    main_camera = bpy.data.objects['Camera']

# clear_scene()
# background_plane, light_source, camera, camera_track, light_track = create_trajectory_scene(type='line')
scene_objects = bpy.context.scene.objects.keys()
print("Objects Before importing the models ", bpy.data.objects.keys())

#########################################################
# Import the objects based on config
dimensions_path = path_config.get("models_dir") + data_config.get("models_name") + '/models_info.json'
with open(dimensions_path, 'r') as file:
    dimensions_dict = json.load(file)

import_models(dimensions_dict,
              config_file = config_file,
              models_dir= path_config.get("models_dir"),
              models_name=data_config.get("models_name")
              )

#########################################################
# Get the object names,class_to_id, id_to_class
object_names,class_to_id, id_to_class = get_object_names_cls_idx_blender(obj_names_to_remove=scene_objects)
print("Imported objects present in the scene : ", object_names)

# Place the objects on the required surface
place_objects_on_target_object(object_names=object_names,
                               surface_to_place=surface_to_place
                               )

#########################################################
# Generate Datasets based on the configuration file .
dataset_type = data_config.get("dataset_type")
scene = bpy.context.scene

if dataset_type == 'Classification':
    generate_classification_dataset(scene,
                                   surface_to_place,
                                   object_names,
                                   class_to_id,
                                   camera=robot_camera,
                                   config_file=config_file,
                                   light_source=light_source
                                   )
    
elif dataset_type == 'Object Detection':
    generate_object_detection_dataset(scene,
                                      surface_to_place,
                                      object_names,
                                      class_to_id,
                                      camera=robot_camera,
                                      config_file=config_file,
                                      light_source=light_source
                                        )

elif dataset_type =="6D Pose Estimation":
    print(f"Robot camera type : {type(robot_camera)}, {robot_camera.name}")
    generate_6d_scene_format(scene,
                             surface_to_place,
                             object_names,
                             class_to_id,
                             camera=robot_camera,
                             config_file = config_file,
                             light_source=light_source
                             )


###############

# # Set the scene camera to the specified camera
# main_camera =  bpy.data.objects.get('Camera')
# scene = bpy.context.scene
# scene.render.resolution_x = int(1080)
# scene.render.resolution_y = int(720)
# scene.render.resolution_percentage = 200
# scene.cycles.samples = 200
# scene.camera = main_camera
# main_camera.data.lens  = 70
# main_image_path = '/home/sathwik/thesis_sathwik/main_project/dataset_generation/results_dir/t_less_obj_1/main_render.png'
# scene.render.filepath = main_image_path
# bpy.ops.render.render(write_still = True)

print("Object location : ",bpy.data.objects[object_names[0]].location)
print("Table location : ", surface_to_place.location)
print("Object's dimensions : ", bpy.data.objects[object_names[0]].dimensions)

################################
bpy.ops.wm.quit_blender()
sys.exit()


