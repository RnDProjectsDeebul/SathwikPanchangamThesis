import bpy
from pathlib import Path
from utils.blender_utils.bpy_utils import *
from utils.blender_utils.render_utils import set_render_parameters,render_dataset_images
from utils.dataset_utils.scene_utils import place_objects_in_circular_arrangement,place_object_randomly,place_objects_in_center,place_object_random_postions
from utils.dataset_utils.annotation_utils import save_annotation_files_bop,save_as_json_file,update_json_file,TimerLogger
from utils.dataset_utils.dataset_helpers import get_min_max_values,get_sequential_step_values,set_random_rotation,get_norm_constraint_light_value
from utils.dataset_utils.object_utils import place_object_on_surface,place_objects_on_target_object
from utils.blender_utils.deformation_utils import MaterialDeformer

import time
import os

def generate_6d_scene_format(scene,
                             surface_to_place,
                             object_names,
                             class_to_id,
                             camera,
                             config_file ,
                             light_source
                             ):
    
    """ 
    Args:
        scene : bpy.context.scene
        surface_to_place :  background plane object (bpy format)
        object_names: list
        class_to_idx: dict {"object_name": object_id:int}
        camera: blender camera
        config_file : Dict : generated configuration file

    """
    
    print("Generating 6D Pose Dataset")
    dataset_name=config_file['dataset_name']
    dataset_labels_format = config_file['dataset_settings']['dataset_label_format']
    object_placement=config_file['dataset_settings']['object_placement']
    object_rotation = config_file['dataset_settings']['object_rotation']
    results_dir=config_file['path_settings']['results_dir']
    dataset_type = config_file['dataset_settings']["dataset_type"]

    # Set scene properties
    scene.use_gravity = True
    scene.view_layers['ViewLayer'].use_pass_z= True
    scene.view_layers['ViewLayer'].use_pass_mist = True
    scene.view_layers["ViewLayer"].use_pass_object_index = True  # For Instance Segmentation
    scene.view_layers["ViewLayer"].use_pass_material_index = True  # For Semantic Segmentation
    scene.view_layers["ViewLayer"].use_pass_normal = True  # For Normal Maps
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
    bpy.context.scene.camera = camera
    camera.constraints['Track To'].target = surface_to_place

    # Get Test Constraints and write loop based on test constraints 
    test_constraints = config_file['constraint_settings'].keys()
    if "normal" not in test_constraints:
        raise ValueError("Normal constraint is mandatory for 6D Pose Estimation Dataset Generation")
    
    # Set render parameters 
    set_render_parameters(scene=scene,camera=camera,render_config=config_file['render_settings'])
    img_file_format = config_file['render_settings']['file_format']

    # Place the objects on the required surface
    if object_placement == 'circle':
        place_objects_in_circular_arrangement(object_names, camera,surface_to_place,margin=0.2)
    elif object_placement == 'random':
        for obj_name in object_names:
            place_object_randomly(scene=scene,obj=bpy.data.objects[obj_name] ,camera=camera,min_distance_factor=0.2)
    elif object_placement == 'center':
        place_objects_in_center(object_names)  #spanch2s check this is only for SISO

    # Set object's rotation and show all objects for rendering
    for name in object_names:
        if object_rotation == 'fixed' and dataset_type == 'SISO':
            set_random_rotation(bpy.data.objects[name]) 
        object_to_render = bpy.data.objects[name] 
        object_to_render.hide_render = False
    
    # Log the time
    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(results_dir, str(dataset_name), 'dataset_generation_time_log.txt')
    timer = TimerLogger(log_path)

    for constraint in test_constraints:
        print(f"\n\n Rendering {config_file['constraint_settings'][constraint]['num_images']} images for {constraint} constraint")
        print("--"*30)

        textures_path = config_file['constraint_settings'][constraint]['path'] if constraint == 'constraint_textures' else \
                    config_file['path_settings']['textures_dir']

        dir_name = str(dataset_name) + '/'
        folder_name = str(dataset_name)+'_'+str(constraint)

        if constraint == 'normal': 
            timer.start(f"{constraint}_constraint")
            output_path = Path(results_dir + dir_name + folder_name + '/')
            deformation_value = 0.0
            
            sequential_values = get_sequential_step_values(test_case=constraint,json_object=config_file)
            min_val,max_val = get_min_max_values(test_case=constraint,constraint_settings_dict=config_file['constraint_settings'])

            for idx,value in enumerate(sequential_values):
                for obj_to_render in object_names:
                    # Randomly rotate the object and place in camera view
                    if object_rotation == 'random':
                        set_random_rotation(bpy.data.objects[obj_to_render])
                    if object_placement == 'random' and dataset_type == 'SISO':
                        place_object_random_postions(scene=scene,obj=bpy.data.objects[obj_to_render] ,camera=camera,min_distance_factor=0.2) #spanch2s:TODO For SISO only, Changes for MISO,MIMO,SIMO

                    place_object_on_surface(bpy.data.objects[obj_to_render] ,surface_to_place)

                set_light_intensity(scene=scene,light_source=light_source,
                                    intensity_value=get_norm_constraint_light_value(config_file['constraint_settings'])
                                    )
                set_camera_postion_on_path(camera_object=camera,distance_value=random.uniform(min_val,max_val))
                blur_value = 10
                camera.data.dof.focus_distance = blur_value

                set_random_pbr_img_textures(textures_path=textures_path,obj_name='Background_plane',scale=10)

                print(f'\nRendering image {idx +1} of {len(sequential_values)}')
                
                for image_type in config_file['dataset_settings']['image_types']:
                    render_dataset_images(scene=scene,idx=idx,image_type=image_type,img_file_format=img_file_format,
                                          output_path=output_path,object_names=object_names,camera=camera,links=links,
                                          render_layers_node=render_layers_node,output_node=output_node)

                # Store the annotations in the respective dictionaries
                save_annotation_files_bop(idx,scene,camera,light_source,
                                      deformation_value,
                                      object_names,class_to_id,
                                      output_path=output_path)

            timer.stop(f"{constraint}_constraint")
            print(f"************************** Completed Rendering {constraint} Constraint **************************")
                
        
        elif constraint == 'bright' or constraint == 'dark':
            timer.start(f"{constraint}_constraint")
            output_path = Path(results_dir + dir_name + folder_name + '/')
            deformation_value = 0.0

            norm_min_val,norm_max_val = get_min_max_values(test_case='normal',constraint_settings_dict=config_file['constraint_settings'])

            sequential_values = get_sequential_step_values(test_case=constraint,json_object=config_file)
            for idx,value in enumerate(sequential_values):
                
                for obj_to_render in object_names:
                    if object_rotation == 'random':
                        set_random_rotation(bpy.data.objects[obj_to_render])
                        # place_object_randomly(scene=scene,obj=bpy.data.objects[obj_to_render] ,camera=camera,min_distance_factor=0.2)

                    if object_placement == 'random' and dataset_type == 'SISO':
                        place_object_random_postions(scene=scene,obj=bpy.data.objects[obj_to_render] ,camera=camera,min_distance_factor=0.2) #spanch2s:TODO For SISO only, Changes for MISO,MIMO,SIMO
                
                set_light_intensity(scene=scene,light_source=light_source,intensity_value=value)
                distance_value = (norm_min_val+norm_max_val)/2
                
                set_camera_postion_on_path(camera_object=camera,distance_value=distance_value)
                set_random_pbr_img_textures(textures_path=textures_path,obj_name='Background_plane',scale=10)
                camera.data.dof.focus_distance = 10

                print(f'\nRendering image {idx +1} of {len(sequential_values)}')
                
                for image_type in config_file['dataset_settings']['image_types']:
                    render_dataset_images(scene=scene,idx=idx,image_type=image_type,img_file_format=img_file_format,
                                          output_path=output_path,object_names=object_names,camera=camera,links=links,
                                          render_layers_node=render_layers_node,output_node=output_node)

                # Store the annotations in the respective dictionaries
                save_annotation_files_bop(idx,scene,camera,light_source,
                                      deformation_value,
                                      object_names,class_to_id,
                                      output_path=output_path)

            timer.stop(f"{constraint}_constraint")
            print(f"************************** Completed Rendering {constraint} Constraint **************************")


        elif constraint == 'far' or constraint == 'near':
            timer.start(f"{constraint}_constraint")
            output_path = Path(results_dir + dir_name + folder_name + '/')
            deformation_value = 0.0

            # place_objects_in_center(obj_names=object_names)

            sequential_values = get_sequential_step_values(test_case=constraint,json_object=config_file)
            for idx,value in enumerate(sequential_values):

                for obj_to_render in object_names:
                    if object_rotation == 'random':
                        set_random_rotation(bpy.data.objects[obj_to_render])
                        # place_object_randomly(scene=scene,obj=bpy.data.objects[obj_to_render] ,camera=camera,min_distance_factor=0.2)

                    if object_placement == 'random' and dataset_type == 'SISO':
                        place_object_random_postions(scene=scene,obj=bpy.data.objects[obj_to_render] ,camera=camera,min_distance_factor=0.2) #spanch2s:TODO For SISO only, Changes for MISO,MIMO,SIMO
                    place_object_on_surface(bpy.data.objects[obj_to_render] ,surface_to_place)
                
                set_camera_postion_on_path(camera_object=camera,distance_value = value)

                set_light_intensity(scene=scene,light_source=light_source,
                                    intensity_value=get_norm_constraint_light_value(config_file['constraint_settings'])
                                    )
                set_random_pbr_img_textures(textures_path=textures_path,obj_name='Background_plane',scale=10)
                camera.data.dof.focus_distance = 10

                print(f'\nRendering image {idx +1} of {len(sequential_values)}')

                for image_type in config_file['dataset_settings']['image_types']:
                    render_dataset_images(scene=scene,idx=idx,image_type=image_type,img_file_format=img_file_format,
                                          output_path=output_path,object_names=object_names,camera=camera,links=links,
                                          render_layers_node=render_layers_node,output_node=output_node)

                # Store the annotations in the respective dictionaries
                save_annotation_files_bop(idx,scene,camera,light_source,
                                      deformation_value,
                                      object_names,class_to_id,
                                      output_path=output_path)
                
            timer.stop(f"{constraint}_constraint")
            print(f"************************** Completed Rendering {constraint} Constraint **************************")
                
        elif constraint == 'blur':
            timer.start(f"{constraint}_constraint")
            # TODO: if normal constraint is not choosen then we will get error here and in later distance_value step
            # Put default value -73 or something based on the normal constraint values. Need to check and implement
            norm_min_val,norm_max_val = get_min_max_values(test_case='normal',constraint_settings_dict=config_file['constraint_settings'])
            deformation_value = 0.0
            # place_objects_in_center(obj_names=object_names)
            output_path = Path(results_dir + dir_name + folder_name + '/')

            sequential_values = get_sequential_step_values(test_case=constraint,json_object=config_file)
            camera.constraints['Follow Path'].offset = -30 
            
            for idx,value in enumerate(sequential_values):    
                for obj_to_render in object_names:
                    if object_rotation == 'random':
                        set_random_rotation(bpy.data.objects[obj_to_render])
                        # place_object_randomly(scene=scene,obj=bpy.data.objects[obj_to_render] ,camera=camera,min_distance_factor=0.2)

                    if object_placement == 'random' and dataset_type == 'SISO':
                        place_object_random_postions(scene=scene,obj=bpy.data.objects[obj_to_render] ,camera=camera,min_distance_factor=0.2) #spanch2s:TODO For SISO only, Changes for MISO,MIMO,SIMO
                    place_object_on_surface(bpy.data.objects[obj_to_render] ,surface_to_place)

                distance_value = (norm_min_val+norm_max_val)/2
                set_light_intensity(scene=scene,light_source=light_source,
                                    intensity_value=get_norm_constraint_light_value(config_file['constraint_settings'])
                                    )
                set_camera_postion_on_path(camera_object=camera,distance_value=distance_value)
                set_random_pbr_img_textures(textures_path=textures_path,obj_name='Background_plane',scale=10)
                add_blur_dof(camera=camera,blur_value=value)
                
                print(f'\nRendering image {idx +1} of {len(sequential_values)}')
                
                for image_type in config_file['dataset_settings']['image_types']:
                    render_dataset_images(scene=scene,idx=idx,image_type=image_type,img_file_format=img_file_format,
                                          output_path=output_path,object_names=object_names,camera=camera,links=links,
                                          render_layers_node=render_layers_node,output_node=output_node)

                # Store the annotations in the respective dictionaries
                save_annotation_files_bop(idx,scene,camera,light_source,
                                      deformation_value,
                                      object_names,class_to_id,
                                      output_path=output_path)
                # disable depth of field
                camera.data.dof.use_dof = False
            timer.stop(f"{constraint}_constraint")
            print(f"************************** Completed Rendering {constraint} Constraint **************************")
                

        elif constraint == 'constraint_textures':
            timer.start(f"{constraint}_constraint")
            # TODO: if normal constraint is not choosen then we will get error here and in later distance_value step
            # Put default value -73 or something based on the normal constraint values. Need to check and implement
            norm_min_val,norm_max_val = get_min_max_values(test_case='normal',constraint_settings_dict=config_file['constraint_settings'])
            deformation_value = 0.0
            output_path = Path(results_dir + dir_name + folder_name + '/')

            # place_objects_in_center(obj_names=object_names)

            num_images = config_file['constraint_settings'][constraint]['num_images']
            for idx in range(num_images):
                for obj_to_render in object_names:
                    if object_rotation == 'random':
                        set_random_rotation(bpy.data.objects[obj_to_render])
                        # place_object_randomly(scene=scene,obj=bpy.data.objects[obj_to_render] ,camera=camera,min_distance_factor=0.2)

                    if object_placement == 'random' and dataset_type == 'SISO':
                        place_object_random_postions(scene=scene,obj=bpy.data.objects[obj_to_render] ,camera=camera,min_distance_factor=0.2) #spanch2s:TODO For SISO only, Changes for MISO,MIMO,SIMO
                    place_object_on_surface(bpy.data.objects[obj_to_render] ,surface_to_place)


                # randomize the textures and generate the dataset
                set_light_intensity(scene=scene,light_source=light_source,
                                    intensity_value=get_norm_constraint_light_value(config_file['constraint_settings'])
                                    )
                distance_value = (norm_min_val+norm_max_val)/2
                set_camera_postion_on_path(camera_object=camera,distance_value=distance_value)
                camera.data.dof.focus_distance = 10
                set_random_pbr_img_textures(textures_path=textures_path,obj_name='Background_plane',scale=10)

                print(f'\nRendering image {idx +1} of {num_images}') 
                for image_type in config_file['dataset_settings']['image_types']:
                    render_dataset_images(scene=scene,idx=idx,image_type=image_type,img_file_format=img_file_format,
                                          output_path=output_path,object_names=object_names,camera=camera,links=links,
                                          render_layers_node=render_layers_node,output_node=output_node)

                # Store the annotations in the respective dictionaries
                save_annotation_files_bop(idx,scene,camera,light_source,
                                      deformation_value,
                                      object_names,class_to_id,
                                      output_path=output_path)
            timer.stop(f"{constraint}_constraint")
            print(f"************************** Completed Rendering {constraint} Constraint **************************")
                
        elif constraint == 'deformation':
            timer.start(f"{constraint}_constraint")
            norm_min_val,norm_max_val = get_min_max_values(test_case='normal',constraint_settings_dict=config_file['constraint_settings'])
            output_path = Path(results_dir + dir_name + folder_name + '/')

            # place_objects_in_center(obj_names=object_names)
            # Apply Deformation to the objects present in the scene once and render the images
            deformer = MaterialDeformer()

            num_images = int(config_file['constraint_settings'][constraint]['num_images'])
            for idx in range(num_images):
                
                start_value = 0.12
                stop_value = 0.35
                step_value = (stop_value - start_value) / (num_images - 1)
                numbers_list = np.arange(
                    start_value, stop_value + step_value, step_value)
                
                deformation_value = numbers_list[idx]

                original_mesh_data = {}
                for object in object_names:
                    original_mesh_data[object] = bpy.data.objects[object].data.copy()
                    deformation_type = 'Dent'
                    deformer.assign_material_profile(obj=bpy.data.objects[object],material_type='plastic')
                    if deformation_type == 'Dent':
                        deformer.apply_dent_deformation(obj=bpy.data.objects[object],
                                                        location=bpy.data.objects[object].location,
                                                        radius=0.1, 
                                                        depth=deformation_value 
                                                        )
                    elif deformation_type == 'Bend':
                        deformer.apply_bend_deformation(obj=bpy.data.objects[object],
                                                        axis='x',
                                                        amount=deformation_value,
                                                        origin=(0,0,0)
                                                        )
                    elif deformation_type == 'Twist':
                        deformer.apply_twist_deformation(obj=bpy.data.objects[object],
                                                         axis='z', 
                                                         amount=deformation_value, 
                                                         origin=(0,0,0)
                                                         )
                    elif deformation_type == 'Crush':
                        deformer.apply_crush_deformation(obj=bpy.data.objects[object],
                                                         axis='z',
                                                         amount=deformation_value
                                                         )
                        
                for obj_to_render in object_names:
                    if object_rotation == 'random':
                        set_random_rotation(bpy.data.objects[obj_to_render])
                        # place_object_randomly(scene=scene,obj=bpy.data.objects[obj_to_render] ,camera=camera,min_distance_factor=0.2)

                    if object_placement == 'random' and dataset_type == 'SISO':
                        place_object_random_postions(scene=scene,obj=bpy.data.objects[obj_to_render] ,camera=camera,min_distance_factor=0.2)
                    place_object_on_surface(bpy.data.objects[obj_to_render] ,surface_to_place)

                # Set parameters for environment
                set_light_intensity(scene=scene,light_source=light_source,
                                    intensity_value=get_norm_constraint_light_value(config_file['constraint_settings'])
                                    )
                distance_value = (norm_min_val+norm_max_val)/2
                set_camera_postion_on_path(camera_object=camera,distance_value=distance_value)
                set_random_pbr_img_textures(textures_path=textures_path,obj_name='Background_plane',scale=10)
                camera.data.dof.focus_distance = 10

                print(f'\nRendering image {idx +1} of {num_images}')

                for image_type in config_file['dataset_settings']['image_types']:
                    render_dataset_images(scene=scene,idx=idx,image_type=image_type,img_file_format=img_file_format,
                                          output_path=output_path,object_names=object_names,camera=camera,links=links,
                                          render_layers_node=render_layers_node,output_node=output_node)

                # Store the annotations in the respective dictionaries
                save_annotation_files_bop(idx,scene,camera,light_source,
                                      deformation_value,
                                      object_names,class_to_id,
                                      output_path=output_path)
                
                for object in object_names:
                    bpy.data.objects[object].data = original_mesh_data[object]
            
            timer.stop(f"{constraint}_constraint")
            print(f"************************** Completed Rendering {constraint} Constraint **************************")    
    timer.save()
