# import modules
import os
from typing import List,Tuple,Dict,Any
from pathlib import Path

import numpy as np

import bpy

def get_object_names_cls_idx_blender(obj_names_to_remove):
    """Get the names of the objects, class_to_idx and idx_to_class
    Args:
        obj_names_to_remove: list of names to be removed.
    Returns:
        object_names_sorted : list of object_names present in the scene.
        class_to_id: {class_name:id} useful for the annotation.
        id_to_class: {id:class_name} useful for the annotation.
    """
    obj_names = bpy.context.scene.objects.keys()
    if obj_names_to_remove is not None:
        for name in obj_names_to_remove:
            obj_names.remove(str(name))

    object_names_sorted = sorted(obj_names)
    class_to_id = {obj: idx + 1 for idx, obj in enumerate(object_names_sorted)}
    id_to_class = {v: k for k, v in class_to_id.items()}

    return object_names_sorted,class_to_id,id_to_class

def get_object_names_from_config(config_file):
    """
    Returns:
        list of object names from the Config File
    """
    dataset_type = config_file['dataset_settings'].get("dataset_type")

    if dataset_type == "6D Pose Estimation":
        dataset_config_settings = config_file['pose_estimation_settings']
        category = dataset_config_settings['dataset_format']
        if category =='SISO':
            object_names = [dataset_config_settings['selected_objects']]
        elif category == 'MISO':
            object_names = [dataset_config_settings['selected_objects']]
        elif category == 'SIMO':
            object_names = dataset_config_settings['selected_objects']
        elif category == 'MIMO':
            object_names = dataset_config_settings['selected_objects']
        return category,object_names,dataset_config_settings
    
    elif dataset_type == "Classification":
        dataset_config_settings = config_file['classification_settings']
        object_names = dataset_config_settings['selected_objects']
        category = "classification"
        return category,object_names,dataset_config_settings
    elif dataset_type == "Object Detection":
        dataset_config_settings = config_file['object_detection_settings']
        object_names = dataset_config_settings['selected_objects']
        category = "object_detection"
        return category,object_names,dataset_config_settings
    else:
        raise ValueError("Invalid dataset type")
    

def create_cls2idx_obj_file_paths(models_dir,models_name):
    
    models_dir = models_dir+models_name
    object_names = []
    obj_file_path_dict = {}
    for root,dirs,files in os.walk(models_dir):
        for file in files:
            if models_name =='YCB':
                if file.endswith(".obj"):
                    object_file = os.path.join(root, file)
                    object_name = "".join(object_file.split('/')[-3].split('_')[1::])
                    object_names.append(object_name)
                    obj_file_path_dict[object_name] = object_file
            elif models_name =='T_LESS':
                if file.endswith(".ply"):
                    object_file = os.path.join(root, file)
                    object_name = object_file.split('/')[-1].split('.')[0]
                    object_names.append(object_name)
                    obj_file_path_dict[object_name] = object_file
            elif models_name == 'HOPE':
                if file.endswith(".obj"):
                    object_file = os.path.join(root, file)
                    object_name = object_file.split('/')[-1].split('.')[0]
                    object_names.append(object_name)
                    obj_file_path_dict[object_name] = object_file
            elif models_name =='ROBOCUP':
                if file.endswith(".obj"):
                    object_file = os.path.join(root, file)
                    object_name = object_file.split('/')[-1].split('.')[0]
                    object_names.append(object_name)
                    obj_file_path_dict[object_name] = object_file
            
    
    object_names_sorted = sorted(object_names)
    class_to_id = {obj: idx + 1 for idx, obj in enumerate(object_names_sorted)}

    return obj_file_path_dict,object_names_sorted,class_to_id

def import_models(dimensions_dict,
                  config_file,
                  models_dir,
                  models_name
                  ):
    """Imports the 3D models for the specified task."""

    obj_file_path_dict,object_names,class_to_id = create_cls2idx_obj_file_paths(models_dir,models_name)
    dataset_type,instance_object_names,dataset_config_settings = get_object_names_from_config(config_file)
    
    print(obj_file_path_dict)
    print("Instance Object names : ",instance_object_names)
    # Check for SISO,MISO,SIMO,MIMO
    if dataset_type in ['SISO','SIMO','classification']:
        for object_name in instance_object_names:
            if '.obj' in obj_file_path_dict[object_name]:
                import_obj(file_path=obj_file_path_dict[object_name],
                        dimensions_dict=dimensions_dict,
                        obj_name= object_name,
                        obj_key=class_to_id[object_name]
                        )
            elif '.ply' in obj_file_path_dict[object_name]:
                import_ply(file_path=obj_file_path_dict[object_name],
                        dimensions_dict=dimensions_dict,
                        obj_name= object_name,
                        obj_key=class_to_id[object_name]
                        )

    elif dataset_type in ['MISO','MIMO']:
        # print(class_to_id)
        # print("\n \n \n \n")
        # print("*************************************************")
        # print("Object names to be imported : ", object_names)
        # print("Instance Object names : ",instance_object_names)
        # print("*************************************************")
        # print("\n \n \n \n")

        for object_name in object_names:
            if object_name in instance_object_names:
                num_instances_per_obj = int(dataset_config_settings['object_instances'][object_name])
                for instance_idx in range(1,num_instances_per_obj+1):
                    instance_name = f'{object_name}_{instance_idx}'
                    instance_key = f'{class_to_id[object_name]}_{instance_idx}'
                    
                    if '.obj' in obj_file_path_dict[object_name]:
                        import_obj(file_path=obj_file_path_dict[object_name],
                                dimensions_dict=dimensions_dict,
                                obj_name= instance_name,
                                obj_key=class_to_id[object_name]
                                )
                    elif '.ply' in obj_file_path_dict[object_name]:
                        import_ply(file_path=obj_file_path_dict[object_name],
                                dimensions_dict=dimensions_dict,
                                obj_name= instance_name,
                                obj_key=class_to_id[object_name]
                                )
            # else:
            #     if '.obj' in obj_file_path_dict[object_name]:
            #             import_obj(file_path=obj_file_path_dict[object_name],
            #                     dimensions_dict=dimensions_dict,
            #                     obj_name= object_name,
            #                     obj_key=class_to_id[object_name]
            #                     )
            #     elif '.ply' in obj_file_path_dict[object_name]:
            #         import_ply(file_path=obj_file_path_dict[object_name],
            #                 dimensions_dict=dimensions_dict,
            #                 obj_name= object_name,
            #                 obj_key=class_to_id[object_name]
            #                 )
    elif dataset_type == "object_detection":
        for object_name in object_names:
            if object_name in instance_object_names:
                num_instances_per_obj = int(dataset_config_settings['object_instances'][object_name])
                for instance_idx in range(1,num_instances_per_obj+1):
                    if num_instances_per_obj > 1:
                        instance_name = f'{object_name}_{instance_idx}'
                        instance_key = f'{class_to_id[object_name]}_{instance_idx}'
                    else:
                        instance_name = object_name
                        instance_key = class_to_id[object_name]
                    
                    if '.obj' in obj_file_path_dict[object_name]:
                        import_obj(file_path=obj_file_path_dict[object_name],
                                dimensions_dict=dimensions_dict,
                                obj_name= instance_name,
                                obj_key=class_to_id[object_name]
                                )
                    elif '.ply' in obj_file_path_dict[object_name]:
                        import_ply(file_path=obj_file_path_dict[object_name],
                                dimensions_dict=dimensions_dict,
                                obj_name= instance_name,
                                obj_key=class_to_id[object_name]
                                )
    else:
        print(f"Unsupported dataset type: {dataset_type}")

    return None


def import_obj(file_path:str,dimensions_dict:Dict,obj_name:str,obj_key:int):
    """Imports the object from the given file path and sets its dimensions. 
    
    Arguments:
        file_path -- path to the .obj file
        dimensions_dict -- dict containing the dimensions of the objects (models_info.json)
        obj_name -- name of the object to be imported
        obj_key -- key for the object in the dimensions_dict    
    """
    
    # Import the OBJ file
    bpy.ops.wm.obj_import(filepath=file_path)
    imported_object = bpy.context.selected_objects[-1]
    imported_object.name = str(obj_name)

    print(f"Dimensions dictionary values : {dimensions_dict}")

    if imported_object is not None:
        # Set object's Dimensions
        # set_object_dimensions(object_name=obj_name,
        #                       index_key=obj_key,
        #                       dimensions_dict=dimensions_dict,
        #                       obj_format='obj'
        #                       )
        # Set the origin of the object to its center
        imported_object.select_set(True)
        bpy.context.view_layer.objects.active = imported_object
        bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_VOLUME',center='BOUNDS')

        
        
        imported_object.location = (0,0,0)
        imported_object.rotation_euler = (0,0,0)
    else:
        print("Error: Object is None")

    return imported_object

def import_ply(file_path:str,dimensions_dict:Dict,obj_name:str,obj_key:int):
    
    # Import the PLY file
    print(file_path)
    bpy.ops.wm.ply_import(filepath=file_path)
    imported_object = bpy.context.selected_objects[-1]
    imported_object.name = str(obj_name)

    if imported_object is not None:
        # Set the origin of the object to its center
        imported_object.select_set(True)
        bpy.context.view_layer.objects.active = imported_object
        bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_VOLUME',center='BOUNDS')
        bpy.ops.object.modifier_add(type='EDGE_SPLIT')
        bpy.ops.object.modifier_add(type='SUBSURF')
        bpy.context.object.modifiers["Subdivision"].levels = 1
        bpy.context.object.modifiers["Subdivision"].subdivision_type = 'SIMPLE'
        
        imported_object.location = (0,0,0)
        imported_object.rotation_euler = (0,0,0)

        # Set object's Dimensions
        set_object_dimensions(object_name=obj_name,
                              index_key=obj_key,
                              dimensions_dict=dimensions_dict,
                              obj_format='ply'
                              )
        
    else:
        print("Error: Object is None")
    return imported_object

def show_hide_objects(obj_names:List, hide: bool):
    """Shows or Hides the objects present in the scene.
       Helpful while rendering

    Arguments:
        obj_names -- list of object names
        hide -- True/False
    """
    for name in obj_names:
        bpy.context.scene.objects[name].hide_render = hide

def import_ycb_objects_with_dimensions(surface_to_place,models_dir:Path,dimensions_dict:Dict[str, Any],class_to_index:dict):
    """Adds objects in .obj format to the scene along with their materials.

    Arguments:
        surface_to_place -- blender object for placing the models
        models_dir -- path for the models directory
        dimensions_dict -- json object for models dimensions
        class_to_index -- dict_object {class names:idx}, eg: {'cracker_box':2,etc...}

    Returns:
        list of object names
    """

    object_names = []
    obj_files = []
    for root, dirs, files in os.walk(models_dir):
        for file in files:
            if file.endswith(".obj"):
                # create the full path to the OBJ file
                object_file = os.path.join(root, file)
                obj_files.append(object_file)

    for idx, object_file in enumerate(sorted(obj_files)):
        obj_name = "".join(object_file.split('/')[-3].split('_')[1::])
        obj_key = class_to_index[obj_name]
        # Import the OBJ file
        bpy.ops.wm.obj_import(filepath=object_file)
        # Set the active object to the imported object
        imported_object = bpy.context.selected_objects[-1]
        # Rename the new object from textured to the class name.
        imported_object.name = str(obj_name)
        object_names.append(imported_object.name)

        if imported_object is not None:
            # Set the dimensions for the object
            set_object_dimensions(object_name=obj_name,
                                  index_key=obj_key,
                                  dimensions_dict=dimensions_dict,
                                  obj_format='obj'
                                  )
            # Rotate the objects so that they appear in standing pose
            # imported_object.rotation_euler = (-90,0,0)

            # Set the origin of the object to its center
            imported_object.select_set(True)
            bpy.context.view_layer.objects.active = imported_object
            bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_VOLUME',center='BOUNDS')
            
            imported_object.location = (0,0,0)
            imported_object.rotation_euler = (0,0,0)

            #  Adjust the object's position so that they lie just on top of the plane
            place_object_on_surface(obj_to_place=imported_object,
                                    surface_to_place=surface_to_place)
        else:
            print("Error: Object is None")

    return object_names

def place_object_on_surface(obj_to_place,surface_to_place):
    """Moves the blender object on top of another object.

    Arguments:
        obj_to_place -- blender object which needs to be placed
        surface_to_place -- blender object for the surface (eg:table)
    """

    max_dim = np.max(obj_to_place.dimensions)
    # obj_to_place.location.x = surface_to_place.location.x
    # obj_to_place.location.y = surface_to_place.location.y
    obj_to_place.location.z = surface_to_place.location.z+(max_dim/2)+0.001
    return None

def place_objects_on_target_object(object_names,surface_to_place):
    """Moves the blender object on top of another object.

    Arguments:
        object_names -- List of object names
        surface_to_place -- blender object for the surface (eg:table)
    """

    for object_name in object_names:
        obj_to_place = bpy.data.objects[str(object_name)]
        max_dim = np.max(obj_to_place.dimensions)
        obj_to_place.location.x = surface_to_place.location.x
        obj_to_place.location.y = surface_to_place.location.y
        obj_to_place.location.z = surface_to_place.location.z+(max_dim/2)+0.0005
    return None

def set_object_dimensions(object_name,index_key,dimensions_dict,obj_format):
    """Sets the dimensions of the objects from the model_info.json file.

    Arguments:
        object_name -- name of the blender object
        index_key -- key for the object in models_info.json
        dimensions_dict -- models_info.json file which is in dict format
        obj_format -- str: obj or ply

    Returns:
        None
    """
    object_ = bpy.data.objects[object_name]
    
    if obj_format == 'ply':
        object_.scale.x = object_.scale.x*0.001 # Convert mm to meters
        object_.scale.y = object_.scale.y*0.001
        object_.scale.z = object_.scale.z*0.001
        bpy.ops.object.make_single_user(type='SELECTED_OBJECTS', object=True, obdata=True)
        bpy.ops.object.transform_apply(scale=True)
    elif obj_format == "obj":
        object_.dimensions.x = float(dimensions_dict[str(index_key)]["size_x"])*0.001 # Convert mm to meters
        object_.dimensions.y = float(dimensions_dict[str(index_key)]["size_y"])*0.001
        object_.dimensions.z = float(dimensions_dict[str(index_key)]["size_z"])*0.001
    return None

def import_tless_objects_with_dimensions(surface_to_place,models_dir:Path,dimensions_dict:Dict[str, Any],class_to_index:dict):
        """Adds objects in .obj format to the scene along with their materials.

        Arguments:
            surface_to_place -- blender object for placing the models
            models_dir -- path for the models directory
            dimensions_dict -- json object for models dimensions
            class_to_index -- dict_object {class names:idx}, eg: {'cracker_box':2,etc...}

        Returns:
            list of object names
        """
        object_names = []
        ply_files = []
        for root, dirs, files in os.walk(models_dir):
            for file in files:
                if file.endswith(".ply"):
                    # create the full path to the OBJ file
                    object_file = os.path.join(root, file)
                    ply_files.append(object_file)

        for idx, object_file in enumerate(sorted(ply_files)):
            # obj_name = f'obj_{str(idx)}'
            obj_name = object_file.split('/')[-1].split('.')[0] 
            obj_key = class_to_index[obj_name]
            # Import the OBJ file
            # bpy.ops.wm.ply_import(filepath=object_file)
            bpy.ops.import_mesh.ply(filepath=object_file)
            # Set the active object to the imported object
            imported_object = bpy.context.selected_objects[-1]
            # Rename the new object from textured to the class name.
            imported_object.name = str(obj_name)
            object_names.append(imported_object.name)

            if imported_object is not None:
                
                # Rotate the objects so that they appear in standing pose
                # imported_object.rotation_euler = (-90,0,0)

                # Set the origin of the object to its center
                imported_object.select_set(True)
                bpy.context.view_layer.objects.active = imported_object
                bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_VOLUME',center='BOUNDS')
                bpy.ops.object.modifier_add(type='EDGE_SPLIT')
                bpy.ops.object.modifier_add(type='SUBSURF')
                bpy.context.object.modifiers["Subdivision"].levels = 1
                bpy.context.object.modifiers["Subdivision"].subdivision_type = 'SIMPLE'
                
                imported_object.location = (0,0,0)
                imported_object.rotation_euler = (0,0,0)

                # Set object's Dimensions
                set_object_dimensions(object_name=obj_name,
                                      index_key=obj_key,
                                      dimensions_dict=dimensions_dict,
                                      obj_format='ply'
                                      )
                #  Adjust the object's position so that they lie just on top of the plane
                place_object_on_surface(obj_to_place=imported_object,
                                        surface_to_place=surface_to_place)
            else:
                print("Error: Object is None")

        return object_names