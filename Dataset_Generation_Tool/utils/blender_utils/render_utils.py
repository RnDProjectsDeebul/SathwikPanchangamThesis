import os
from pathlib import Path
import numpy as np
import cv2
import csv
import bpy

def set_render_parameters(scene,camera,render_config):
    """Sets the active scene with the specified render parameters.
    
    Args:
        scene : bpy.context.scene
        render_config : Dictionary containing the render parameters
    """

    scene.render.engine = str(render_config['render_engine']).upper()
    scene.cycles.device = str(render_config['device']).upper()
    scene.render.resolution_x = int(render_config['res_x'])
    scene.render.resolution_y = int(render_config['res_y'])
    scene.render.resolution_percentage = 100
    if render_config['render_engine'] == 'CYCLES':
        scene.cycles.samples = render_config['number_samples']
    elif render_config['render_engine'] == 'EEVEE':
        scene.eevee.taa_render_samples = render_config['number_samples']
    scene.render.image_settings.file_format = render_config['file_format']

    camera.data.type = 'PERSP'
    camera.data.lens = render_config['focal_length']

def render_image(scene,file_path):
    """
    Render the image and save it to the desired location

    Args: 
        scene : blender's scene bpy.context.scene
        file_path: str, eg: path/folder/image_name.png
    """

    scene.render.filepath =  file_path
    # Render the current visible scene
    bpy.ops.render.render(write_still=True)


def render_dataset_images(scene,idx,image_type,img_file_format,output_path,object_names,camera,
                          links,render_layers_node,output_node):
    """
    Get requirements from config the file
    
    TODO: Check instance,normal,and semantic images 
    """ 
    if img_file_format == 'PNG':
        img_format = '.png'
    elif img_file_format == 'JPEG':
        img_format = '.jpg'
    else:
        raise ValueError("Unsupported image file format. Use 'PNG' or 'JPEG'.")

    if image_type == 'rgb':
        # Update file path for rgb images and render the image.
        scene.render.filepath = os.path.join(output_path /Path('rgb'), str(f"{str(idx).zfill(6)}{img_format}")) 
        links.new(render_layers_node.outputs["Image"], output_node.inputs['Image'])
        bpy.ops.render.render(write_still = True)
    
    elif image_type == 'depth':
        # Create depth images
        scene.render.filepath = os.path.join(output_path /Path('depth'), str(f"{str(idx).zfill(6)}{img_format}"))
        links.new(render_layers_node.outputs["Depth"], output_node.inputs['Image'])
        bpy.ops.render.render(write_still = True)
        normalize_depth(scene.render.filepath )
        

    elif image_type == 'instance':
        render_instance_masks(scene, idx, output_path,render_layers_node,links,output_node)
    
    elif image_type == 'semantic':
        render_semantic_masks(scene, idx, output_path,links,render_layers_node,output_node)
        # for i, mat in enumerate(bpy.data.materials):
        #             mat.pass_index = i + 1  # Assign unique material index (starting from 1)

        # # Create semantic segmentation images
        # scene.render.filepath = os.path.join(output_path /Path('semantic'), str(f"{str(idx).zfill(6)}.png")) #TODO: automate file format
        # links.new(render_layers_node.outputs["IndexMA"], output_node.inputs['Image'])
        # bpy.ops.render.render(write_still = True)
    
    
    elif image_type == 'normal':
        # Create normal images
        scene.render.filepath = os.path.join(output_path /Path('normal'), str(f"{str(idx).zfill(6)}{img_format}"))
        links.new(render_layers_node.outputs["Normal"], output_node.inputs['Image'])
        bpy.ops.render.render(write_still = True)

    elif image_type == 'point_cloud':
        save_point_cloud_data_all_objects(object_names,camera,
                                          file_dir= output_path /Path('point_cloud'),
                                          file_name =  str(f"{str(idx).zfill(6)}.csv") 
                                          )
                                          
    else:
        print("Invalid image type. Please provide valid image type.")
        # TODO: Write about the ranges and how you are generating and normalizing the depth,instance and segmentation images.

def normalize_depth(depth_img_path):
    depth_img = cv2.imread(depth_img_path, cv2.IMREAD_ANYDEPTH)
    depth_min = np.min(depth_img)
    depth_max = np.max(depth_img)
    depth_normalized = (depth_img - depth_min) / (depth_max - depth_min)
    cv2.imwrite(depth_img_path, (depth_normalized * 255).astype(np.uint8))


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


def save_point_cloud_data_all_objects(object_names, camera, file_dir,file_name):
    """
    Function to save the point cloud data (vertices) for a list of objects in Blender.
    Saves the data of all objects into a single CSV file.
    """
    all_point_cloud_data = []
    os.makedirs(file_dir,exist_ok=True)

    file_path = os.path.join(file_dir, file_name)

    # Iterate over the list of object names and get their point cloud data
    for obj_name in object_names:
        vertices_positions = get_point_cloud_data(obj_name, camera)
        
        # Append the object name to the vertex data for identification (optional)
        for vertex in vertices_positions:
            all_point_cloud_data.append([obj_name] + vertex)  # Adds object name to each point
        
    # Write all point cloud data to a CSV file
    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Object Name', 'X', 'Y', 'Z'])  # Write header
        writer.writerows(all_point_cloud_data)  # Write all points


def render_instance_masks(scene, idx, output_path,render_layers_node,links,output_node):

    """
    Renders and saves instance segmentation masks, where each object is saved as a separate image.
    Each image contains only one visible object at a time.
    """

    output_path = Path(output_path) / "instance"
    output_path.mkdir(parents=True, exist_ok=True)

    # Enable object index pass
    scene.view_layers["ViewLayer"].use_pass_object_index = True  
    scene.use_nodes = True  

    # Get nodes
    # tree = scene.node_tree
    # links = tree.links

    # # Ensure nodes are set up
    # for node in tree.nodes:
    #     tree.nodes.remove(node)

    links.new(render_layers_node.outputs["IndexOB"], output_node.inputs["Image"])

    # Render instance masks (one per object)
    for i, obj in enumerate(bpy.data.objects):
        if obj.type == 'MESH':
            obj.pass_index = i + 1  # Assign unique index

            # Hide all objects except the current one
            for other_obj in bpy.data.objects:
                if other_obj.type == 'MESH':
                    other_obj.hide_render = (other_obj != obj)  # Hide everything except the current object

            # Save each instance with {scene_id}_{instance_id}.png
            file_name = f"{str(idx).zfill(6)}_{i+1}.png" #TODO: automate file format
            scene.render.filepath = str(output_path / file_name)  
            bpy.ops.render.render(write_still=True)

    # Restore visibility of all objects
    for obj in bpy.data.objects:
        obj.hide_render = False  

def render_semantic_masks(scene, idx, output_path,links,render_layers_node,output_node):
    """
    Renders and saves semantic segmentation masks, where each material (semantic class) is assigned a unique index.
    """
    output_path = Path(output_path) / "semantic"
    output_path.mkdir(parents=True, exist_ok=True)  # Ensure output directory exists

    # Enable material index pass
    scene.view_layers["ViewLayer"].use_pass_material_index = True  
    scene.use_nodes = True  

    links.new(render_layers_node.outputs["IndexMA"], output_node.inputs["Image"])

    # Assign unique pass index to materials
    material_mapping = {}  # Store material index mapping
    for i, mat in enumerate(bpy.data.materials):
        mat.pass_index = i + 1
        material_mapping[mat.name] = mat.pass_index  # Save for later use

    # Save semantic segmentation mask
    file_name = f"{str(idx).zfill(6)}.png"
    scene.render.filepath = str(output_path / file_name)  
    bpy.ops.render.render(write_still=True)

    # Save material-label mapping
    mapping_file = output_path / f"{str(idx).zfill(6)}_labels.npy"
    np.save(str(mapping_file), material_mapping)  # Save as NumPy file for reference
