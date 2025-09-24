# Import modules
import streamlit as st
import os
import json
import re
from pathlib import Path

# main code
st.set_page_config("Configuration File Generator", layout="wide")
st.markdown(
    """
    <style>
    .title-container {
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        margin-top: -140px; /* Adjust this value to position the headings vertically */
        height: 20vh; /* Adjust the height of the container */
    }
    .main-title {
        font-size: 32px; /* Adjust font size for the main title */
        font-weight: bold;
        text-align: center;
        margin-bottom: 5px; /* Space between main title and subheading */
    }
    .sub-title {
        font-size: 26px; /* Adjust font size for the subheading */
        font-weight: normal;
        text-align: center;
        color: gray; /* Optional: Adjust color of the subheading */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="title-container">
        <div class="main-title">Blender Based Synthetic Dataset Generation Tool</div>
        <div class="sub-title">Configuration File Generator</div>
    </div>
    """,
    unsafe_allow_html=True,
)

class ConfigGenerator():

    def __init__(self) -> None:

        # Path Settings
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        self.models_dir = root_dir+"/data/models_dir/"
        self.textures_dir = root_dir+"/data/textures_dir/"
        self.constraint_textures_dir = root_dir+ "/data/constraint_textures_dir/"
        self.scenes_dir = root_dir+ "/data/scenes_dir/"
        self.src_dir = root_dir+ "/src/"
        self.config_dir =root_dir+"/data/config_dir/"
        self.example_images_dir = root_dir+"/data/example_images/"

        self.results_dir = root_dir+ "/generated_datasets/"

        # Configuration storage
        self.config = {"path_settings": {}, "dataset_settings": {}, "render_settings": {}}

        # Two column layout for the webpage
        self.column1,self.column2 = st.columns(2)

        self.dataset_settings()
        self.test_constraints_settings()
        self.render_settings()
        self.save_settings()



    def dataset_settings(self): 
        with self.column1.container(height=900):
            st.subheader("Dataset Settings")
            self.dataset_type = st.selectbox("Dataset Type",["Classification", "Object Detection", "6D Pose Estimation"],index=2)
            
            self.scene_type = st.selectbox("Scene Type", ['basic_scene', 'lab_environment'], index=0)
            self.camera_trajectory = st.selectbox("Camera Trajectory", ['LINE', 'CURVE'], index=0)  
            self.models_name = st.selectbox("Models Folder", os.listdir(self.models_dir), index=0)
            
            if self.dataset_type == 'Object Detection':
                self.object_placement = st.selectbox("Object Placement", ['random', 'circle'], index=0)
            elif self.dataset_type == 'Classification': 
                self.object_placement = st.selectbox("Object Placement", ['center', 'random', 'circle'], index=0)
            
            self.object_rotation = st.selectbox("Object Rotation", ['random', 'fixed'], index=0)
            
            
            st.caption("Select the image types to be rendered")
            c1,c2,c3,c4,c5,c6 = st.columns(6)
            rgb = c1.checkbox("RGB", value=True)
            depth = c2.checkbox("Depth", value=False)
            instance = c3.checkbox("Instance", value=False)
            semantic = c4.checkbox("Semantic", value=False)
            normal = c5.checkbox("Normal", value=False)
            point_cloud = c6.checkbox("Point Cloud", value=False)

            # add all selected image types to a list
            self.image_types = [
                                img_type for img_type, flag in 
                                zip(["rgb", "depth", "instance", "semantic", "normal", "point_cloud"], 
                                    [rgb, depth, instance, semantic, normal, point_cloud]) 
                                if flag
                            ]

            # st.warning("Add a gif for LINE and CURVE trajectory")
        
        with self.column2.container(height=900):
            if self.dataset_type == '6D Pose Estimation':
                self.pose_estimation_settings()
            elif self.dataset_type == 'Classification':
                self.classification_settings()
            elif self.dataset_type == 'Object Detection':
                self.object_detection_settings()
            elif self.dataset_type == 'Semantic Segmentation':
                self.semantic_segmentation_settings()

        return None
    
    def print_pose_abbrivations(self):
        if self.dataset_format == 'SISO':
            st.caption("Single Instance of Single Object")
        elif self.dataset_format == 'SIMO':
            st.caption("Single Instance of Multiple Objects")
        elif self.dataset_format == 'MISO':
            st.caption("Multiple Instances of Single Object")
        elif self.dataset_format == 'MIMO':
            st.caption("Multiple Instances of Multiple Objects")

    def pose_estimation_settings(self):
        st.subheader("6D Pose Settings")
        self.dataset_format = st.selectbox("Dataset Type", ['SISO', 'SIMO', 'MISO', 'MIMO'], index=0)
        self.print_pose_abbrivations()
        self.dataset_label_format = st.selectbox("Annotation Format", ['BOP','YCB-V'], index=0)

        allowed_exts = (".obj", ".ply")
        root_dir = os.path.join(self.models_dir, self.models_name, "models")

        files = []
        for dirpath, _, filenames in os.walk(root_dir):
            for f in filenames:
                if f.lower().endswith(allowed_exts):
                    full_path = os.path.join(dirpath, f)
                    rel_path = os.path.relpath(full_path, root_dir)
                    
                    parts = rel_path.split(os.sep)
                    
                    if len(parts) == 1:
                        # Case A: file directly inside /models → use file name
                        display_name = parts[0]
                    else:
                        # Case B: file inside subfolder → use only top-level folder name
                        display_name = parts[0]
                    files.append(display_name)
        files = sorted(set(files))
        
        if self.dataset_format == 'SISO':
            self.object_placement = st.selectbox("Object Placement", ['center','random'], index=0)
            self.object_instances = {}
            include_all_models = None
            self.selected_objects = st.selectbox("Select Object", files)

            clean_name = self.selected_objects.split('.')[0]  # Remove file extension
            if self.models_name == 'YCB':  # Remove leading numeric ID
                object_key = "".join(clean_name.split('_')[1:])
                self.object_instances[object_key] = 1
            else:
                self.object_instances[clean_name] = 1

        elif self.dataset_format == 'SIMO':
            self.object_placement = st.selectbox("Object Placement", ['random', 'circle'], index=0)
            self.object_instances = {}
            include_all_models = st.checkbox("Include All Models", value=False)
            if include_all_models:
                self.selected_objects = files
                st.success(f"All models present in '{self.models_name+'/models/'}' will be included in the dataset")
                # self.selected_objects = os.listdir(self.models_dir+self.models_name+'/models/')
                # st.success(f"All models present in '{self.models_name+'/models/'}' will be included in the dataset")
            else:
                self.selected_objects = st.multiselect("Select Objects", files)
            for obj in self.selected_objects:
                clean_name = obj.split('.')[0]  # Remove file extension
                if self.models_name == 'YCB':  # Remove leading numeric ID
                    object_key = "".join(clean_name.split('_')[1:])
                    self.object_instances[object_key] = 1
                else:
                    self.object_instances[clean_name] = 1

        elif self.dataset_format =='MISO':
            self.object_placement = st.selectbox("Object Placement", ['random', 'circle'], index=0)
            self.object_instances = {}
            include_all_models = None
            self.selected_objects = st.selectbox("Select Object", files)
            num_instances= st.number_input(
                "Number of Instances",
                min_value=1,
                max_value=10,
                value=2,
                key="num_instances",
            )
            clean_name = self.selected_objects.split('.')[0]  # Remove file extension
            if self.models_name == 'YCB':  # Remove leading numeric ID
                object_key = "".join(clean_name.split('_')[1:])
                self.object_instances[object_key] = num_instances
            else:
                self.object_instances[clean_name] = num_instances
            
        elif self.dataset_format == 'MIMO':
            self.object_placement = st.selectbox("Object Placement", ['random', 'circle'], index=0)
            self.object_instances = {}
            include_all_models = st.checkbox("Include All Models", value=False)
            if include_all_models:
                self.selected_objects = files
            else:
                self.selected_objects = st.multiselect("Select Objects", files)

            for obj in self.selected_objects:
                col1,col2 = st.columns(2)
                with col1:
                    st.markdown(obj)
                with col2:
                    num_instances = st.number_input(
                        f"Number of Instances for {obj}",
                        min_value=1,
                        max_value=5,
                        value=2,
                        key=f"num_instances_{obj}",
                    )
                clean_name = obj.split('.')[0]  # Remove file extension
                if self.models_name == 'YCB':  # Remove leading numeric ID
                    object_key = "".join(clean_name.split('_')[1:])
                    self.object_instances[object_key] = num_instances
                else:
                    self.object_instances[clean_name] = num_instances

            # st.success(self.object_instances)

        return
    
    def classification_settings(self):
        st.subheader("Classification Settings")
        self.dataset_label_format = st.selectbox("Dataset Label Format", ['ImageFolder','Folder_CSV'], index=0)

        # Step 2: Define structure strings
        imagefolder_structure = """
        dataset/
        ├── cat/
        │   ├── image1.jpg
        │   └── image2.jpg
        ├── dog/
        │   ├── image3.jpg
        │   └── image4.jpg
        """

        folder_csv_structure = """
        dataset/
        ├── images/
        │   ├── 0001.jpg
        │   ├── 0002.jpg
        │   └── 0003.jpg
        └── labels.csv

        labels.csv:
        filename,label
        0001.jpg,cat
        0002.jpg,dog
        0003.jpg,dog
        """

        # Step 3: Display based on selection
        if self.dataset_label_format == 'ImageFolder':
            st.markdown("### 📂 ImageFolder Structure")
            st.code(imagefolder_structure, language="plaintext")
        elif self.dataset_label_format == 'Folder_CSV':
            st.markdown("### 📂 Folder + CSV Structure")
            st.code(folder_csv_structure, language="plaintext")

        # self.selected_objects = st.multiselect("Select Objects", os.listdir(self.models_dir+self.models_name+'/models/'))
        self.object_instances = {}
        include_all_models = st.checkbox("Include All Models", value=False)

        allowed_exts = (".obj", ".ply")
        files = [
            f for f in os.listdir(self.models_dir + self.models_name + '/models/')
            if f.lower().endswith(allowed_exts)
        ]

        if include_all_models:
            self.selected_objects = files
            st.success(f"All models present in '{self.models_name+'/models/'}' will be included in the dataset")
        else:
            self.selected_objects = st.multiselect("Select Objects", files)

        for obj in self.selected_objects:
            # num_instances = st.number_input(
            #     f"Number of Instances for {obj}",
            #     min_value=1,
            #     max_value=5,
            #     value=2,
            #     key=f"num_instances_{obj}",
            # )
            self.object_instances[obj] = 1

        
        return
    
    def object_detection_settings(self):
        st.subheader("Object Detection Settings")
        self.dataset_label_format = st.selectbox("Dataset Annotations Format", ['YOLO','COCO'], index=0)

        self.object_instances = {}
        include_all_models = st.checkbox("Include All Models", value=False)
        allowed_exts = (".obj", ".ply")
        files = [
            f for f in os.listdir(self.models_dir + self.models_name + '/models/')
            if f.lower().endswith(allowed_exts)
        ]

        if include_all_models:
            self.selected_objects = files
            st.success(f"All models present in '{self.models_name+'/models/'}' will be included in the dataset")
        else:
            self.selected_objects = st.multiselect("Select Objects", files)


        for obj in self.selected_objects:
            num_instances = st.number_input(
                f"Number of Instances for {obj}",
                min_value=1,
                max_value=5,
                value=1,
                key=f"num_instances_{obj}",
            )
            # self.object_instances[obj] = 1
            clean_name = obj.split('.')[0]  # Remove file extension
            if self.models_name == 'YCB':  # Remove leading numeric ID
                object_key = "".join(clean_name.split('_')[1:])
                self.object_instances[object_key] = num_instances
            else:
                self.object_instances[clean_name] = num_instances

        return


    def test_constraints_settings(self):
        # Constraint Settings for bright, dark, far, near,blur, normal,deformation,occluded images, noise,
        with self.column1.container(height=900):
            self.test_constraints = {}
            st.subheader("Constraint Settings")
            # st.caption("Please select one constraint at a time write it clearly in documnentation more training images less testing images")

            self.normal = st.checkbox("Normal Constraint", value=False)
            if self.normal:
                st.caption("Normal Constraint is the default constraint and all parameters are set to optimal values")
                norm1,norm2,norm3 = st.columns(3)
                norm_light_min = norm1.number_input("Light Intensity Min", min_value=3.0, max_value=5.0, value=3.0)
                norm_light_max = norm2.number_input("Light Intensity Max", min_value=3.0, max_value=5.0, value=5.0)

                nor1,nor2,nor3 = st.columns(3)
                normal_min = nor1.number_input("Distance Min", min_value=-100 , max_value= 100, value=-66)
                normal_max = nor2.number_input("Distance Max", min_value=-100, max_value= 100, value=-60)
                num_images_normal = nor3.number_input("Number of Images to Render - Normal", min_value=2, max_value=10000, value=10)
                self.test_constraints["normal"] = {"num_images":num_images_normal,"min": normal_min, "max": normal_max,
                                                   "light_min": norm_light_min, "light_max": norm_light_max}

                
            self.bright = st.checkbox("Bright Constraint", value=False)
            if self.bright:
                st.caption("Recommended values for Light Intensity are between 10 and 15")
                b1,b2,b3 = st.columns(3)
                bright_min = b1.number_input("Light Intensity Min", min_value=10, max_value=15, value=10)
                bright_max = b2.number_input("Light Intensity Max", min_value=10, max_value=15, value=15)
                num_images_bright = b3.number_input("Number of Images to Render - Bright", min_value=2, max_value=10000, value=10)
                self.test_constraints["bright"] = {"num_images":num_images_bright,"min": bright_min, "max": bright_max}

            self.dark = st.checkbox("Dark Constraint", value=False)
            if self.dark:
                st.caption("Recommended values for Light Intensity are between 0.1 and 1.5")
                d1,d2,d3 = st.columns(3)
                dark_min = d1.number_input("Light Intensity Min", min_value=0.1, max_value=1.5, value=0.1)
                dark_max = d2.number_input("Light Intensity Max", min_value=0.1, max_value=1.5, value=1.5)
                num_images_dark = d3.number_input("Number of Images to Render - Dark", min_value=2, max_value=10000, value=10)
                self.test_constraints["dark"] = {"num_images":num_images_dark,"min": dark_min, "max": dark_max}

            self.far = st.checkbox("Far Constraint", value=False)
            if self.far:
                st.caption("Recommended values for Distance are between -55 and -45")
                f1,f2,f3 = st.columns(3)
                far_min = f1.number_input("Distance Min", min_value=-100, max_value=100, value=-55)
                far_max = f2.number_input("Distance Max", min_value=-100 ,max_value=100, value=-45)
                num_images_far = f3.number_input("Number of Images to Render - Far", min_value=2, max_value=10000, value=10)
                self.test_constraints["far"] = {"num_images":num_images_far,"min": far_min, "max": far_max}

            self.near = st.checkbox("Near Constraint", value=False)
            if self.near:
                st.caption("Recommended values for Distance are between -70 and -67")
                n1,n2,n3 = st.columns(3)
                near_min = n1.number_input("Distance Min", min_value=-100, max_value=100, value=-70)
                near_max = n2.number_input("Distance Max", min_value=-100, max_value=100, value=-67)
                num_images_near = n3.number_input("Number of Images to Render - Near", min_value=2, max_value=10000, value=10)
                self.test_constraints["near"] = {"num_images":num_images_near,"min": near_min, "max": near_max}

            self.blur = st.checkbox("Blur Constraint", value=False)
            if self.blur:
                st.caption("Recommended values for Blur are between 0.5 and 2.5")
                st.caption("Aperture is 2.7 and no focus object just vary the distance parameter")
                bl1,bl2,bl3 = st.columns(3)
                blur_min = bl1.number_input("Blur Min", min_value=0.2, max_value=10.0, value=0.5)
                blur_max = bl2.number_input("Blur Max", min_value=0.2, max_value=10.0, value=2.5)
                num_images_blur = bl3.number_input("Number of Images to Render - Blur", min_value=2, max_value=10000, value=10)
                self.test_constraints["blur"] = {"num_images":num_images_blur,"min": blur_min, "max": blur_max}

            self.constraint_textures = st.checkbox("Textures Constraint", value=False)
            if self.constraint_textures:
                st.caption("Provide  Path or Add textures to the constraint textures directory")
                self.constraint_textures_dir = st.text_input("Constraint Textures Directory Path", value=os.path.join(self.constraint_textures_dir))
                num_images_textures = st.number_input("Number of Images to Render - Textures", min_value=2, max_value=10000, value=10)
                self.test_constraints["constraint_textures"] = {"path": self.constraint_textures_dir,"num_images": num_images_textures}
            
            self.deformation = st.checkbox("Deformation Constraint", value=False)
            deformation_parameters = {}
            if self.deformation:
                st.caption("Deformation Constraint is based on the object's mesh properties. Refer Documentation for more details")
                de1,de2,de3 = st.columns(3)
                deformation_type = de1.selectbox("Deformation Type", ['Bend', 'Twist', 'Dent','Crush'], index=0)
                if deformation_type == 'Bend':
                    st.caption("Bend Deformation is applied to the object")
                    bend_col1,bend_col2 = st.columns(2)
                    bend_angle = bend_col1.number_input("Bend Angle", min_value=0, max_value=360, value=90)
                    bend_axis = bend_col2.selectbox("Bend Axis", ['X', 'Y', 'Z'], index=0)
                    
                    deformation_parameters['deformation_type'] = deformation_type
                    deformation_parameters['bend_angle'] = bend_angle
                    deformation_parameters['bend_axis'] = bend_axis

                elif deformation_type == 'Twist':
                    st.caption("Twist Deformation is applied to the object")
                    twist_col1, twist_col2 = st.columns(2)
                    twist_angle = twist_col1.number_input("Twist Angle", min_value=0, max_value=360, value=90)
                    twist_axis = twist_col2.selectbox("Twist Axis", ['X', 'Y', 'Z'], index=0)

                    deformation_parameters['deformation_type'] = deformation_type
                    deformation_parameters['twist_angle'] = twist_angle
                    deformation_parameters['twist_axis'] = twist_axis

                elif deformation_type == 'Dent':
                    st.caption("Dent Deformation is applied to the object")
                    dent_col1, dent_col2 = st.columns(2)
                    dent_depth = dent_col1.number_input("Dent Depth", min_value=0, max_value=10, value=5)
                    dent_radius = dent_col2.number_input("Dent Radius", min_value=0, max_value=10, value=3)

                    deformation_parameters['deformation_type'] = deformation_type
                    deformation_parameters['dent_depth'] = dent_depth
                    deformation_parameters['dent_radius'] = dent_radius

                elif deformation_type == 'Crush':
                    st.caption("Crush Deformation is applied to the object")
                    crush_col1, crush_col2 = st.columns(2)
                    crush_axis = crush_col1.selectbox("Crush Axis", ['X', 'Y', 'Z'], index=0)
                    crush_amount = crush_col2.number_input("amount", min_value=0, max_value=10, value=3)

                    deformation_parameters['deformation_type'] = deformation_type
                    deformation_parameters['crush_axis'] = crush_axis
                    deformation_parameters['crush_amount'] = crush_amount

                st.caption("Recommended values for Subdivision Surface are between 1 and 4")
                # de1,de2 = st.columns(2)
                # subdivision_surface = de1.checkbox("Subdivision Surface", value=False)
                subdivision_surface_val = de2.number_input("Number of subdivisions for mesh",min_value=1,max_value=4,value=1)
                num_images_deformation = de3.number_input("Number of Images to Render - Deformation", min_value=2, max_value=10000, value=10)
                self.test_constraints["deformation"] = {"num_images": num_images_deformation,
                                                        "subdivision_surface_parameter":subdivision_surface_val,
                                                        "deformation_parameters":deformation_parameters
                                                        }

            #self.noise = st.checkbox("Noise Constraint", value=False)
            #self.occluded = st.checkbox("Occluded Constraint", value=False)
            #self.adversarial_pathes, self.something else using compositing tab
        with self.column2.container(height=900):
            st.subheader("Visualize Constraints")
            self.visualize_constraints()

    def visualize_constraints(self):
        st.caption("Example Images for each constraint")
        constraint_images = {
            "normal": "normal_constraint.png",
            "bright": "bright_constraint.png",
            "dark": "dark_constraint.png",
            "blur": "blur_constraint.png",
            "deformation": "deformation_constraint.png",
            "far": "far_constraint.png",
            "near": "near_constraint.png",
            "constraint_textures": "constraint_textures_constraint.png"
        }

        # Collect the constraints to display
        selected_constraints = []
        if self.normal:
            selected_constraints.append("normal")
        if self.bright:
            selected_constraints.append("bright")
        if self.dark:
            selected_constraints.append("dark")
        if self.far:
            selected_constraints.append("far")
        if self.near:
            selected_constraints.append("near")
        if self.blur:
            selected_constraints.append("blur")
        if self.constraint_textures:
            selected_constraints.append("constraint_textures")
        if self.deformation:
            selected_constraints.append("deformation")
        
        for constraint in selected_constraints:
            st.image(
                os.path.join(self.example_images_dir, constraint_images[constraint]),
                caption=constraint.capitalize(),
                use_container_width=True 
            )


    def render_settings(self):

        with self.column1.container(height=400):
            st.subheader("Render settings")
            r1,r2,r3 = st.columns(3)
            self.render_engine = r1.selectbox("Render Engine",['CYCLES','EEVEE'],index=0)
            self.device = r2.selectbox("Device",['CPU','GPU'],index=1)
            self.file_format = r3.selectbox("File Format for the images",['PNG','JPEG'],index=0)      
            
            with r1:
                self.res_x = st.number_input("X resolution of image",min_value=64,max_value=1024,value=128)
            with r2:
                self.res_y = st.number_input("Y resolution of image",min_value=64,max_value=1024,value=128)
            with r3:
                self.focal_length = st.number_input("Focal Length",min_value = 36, max_value = 100,value = 50)
                st.caption("Change in focal length will change the appearance of the object in the image. So change it according to the Object's size and test constraints")
            self.NUMBER_OF_SAMPLES = 200
                

    def save_settings(self):
        """
        Create a JSON file with the configuration settings
        """
        # Path Settings
        self.config["path_settings"]["models_dir"] = self.models_dir
        self.config["path_settings"]["textures_dir"] = self.textures_dir
        self.config["path_settings"]["constraint_textures_dir"] = self.constraint_textures_dir
        self.config["path_settings"]["scenes_dir"] = self.scenes_dir
        self.config["path_settings"]["src_dir"] = self.src_dir
        self.config["path_settings"]["results_dir"] = self.results_dir
        
        # Dataset Settings
        self.config["dataset_settings"]["dataset_type"] = self.dataset_type
        self.config["dataset_settings"]["dataset_label_format"] = self.dataset_label_format
        self.config["dataset_settings"]["scene_type"] = self.scene_type
        self.config["dataset_settings"]["models_name"] = self.models_name
        self.config["dataset_settings"]["object_placement"] = self.object_placement
        self.config["dataset_settings"]["object_rotation"] = self.object_rotation
        self.config["dataset_settings"]["camera_trajectory"] = self.camera_trajectory
        # self.config["dataset_settings"]["annotation_format"] = str(self.annotation_format) # remove
        self.config["dataset_settings"]["image_types"] = self.image_types

        # Render Settings
        self.config["render_settings"]["render_engine"] = self.render_engine
        self.config["render_settings"]["device"] = self.device
        self.config["render_settings"]["res_x"] = self.res_x
        self.config["render_settings"]["res_y"] = self.res_y
        self.config["render_settings"]["number_samples"] = self.NUMBER_OF_SAMPLES
        self.config["render_settings"]["file_format"] = self.file_format
        self.config['render_settings']['focal_length'] = self.focal_length

        if isinstance(self.selected_objects, list):
            clean_names = [obj.split('.')[0] for obj in self.selected_objects]  # Remove file extensions
            self.selected_objects_new = []


            for obj in clean_names:
                if self.models_name == 'YCB':  # Remove leading numeric ID
                    self.selected_objects_new.append("".join(obj.split('_')[1:]))

                elif self.models_name == 'HOPE' or self.models_name == 'T_LESS' or self.models_name == 'ROBOCUP' :  # Keep name as is (just remove extension)
                    self.selected_objects_new.append(obj)
                else:
                    raise ValueError("Need to implement according to the file naming")

        else:
            clean_name = self.selected_objects.split('.')[0]  # Remove file extension
            print("Name before cleaning: ",clean_name)
            if self.models_name == 'YCB':  # Remove leading numeric ID
                self.selected_objects_new = "".join(clean_name.split('_')[1:])
                print("Cleaned name: ",self.selected_objects_new)

            elif self.models_name == 'HOPE' or self.models_name == 'T_LESS' or self.models_name == 'ROBOCUP':
                self.selected_objects_new = clean_name
            else:
                raise ValueError("Need to implement according to the file naming")


        # Save Pose Estimation Settings, Classification Settings, Object Detection Settings
        if self.dataset_type == '6D Pose Estimation':
            self.config["pose_estimation_settings"] = {
                "dataset_format": self.dataset_format,
                "selected_objects": self.selected_objects_new,
                "object_instances": self.object_instances
            }
        elif self.dataset_type == 'Classification':
            self.config["classification_settings"] = {
                "selected_objects": self.selected_objects_new,
                "object_instances": self.object_instances
            }
        elif self.dataset_type == 'Object Detection':
            self.config["object_detection_settings"] = {
                "selected_objects": self.selected_objects_new,
                "object_instances": self.object_instances
                }

        # Save Constraint Settings
        self.config["constraint_settings"] = self.test_constraints
            
        # Save the configuration settings to a JSON file
        with self.column2:
            with st.container(height=400):
                st.subheader("Save Settings")
                self.dataset_name = st.text_input("Provide the name for the dataset folder")
                self.config["dataset_name"] = self.dataset_name
                self.results_dir = st.text_input("Results Directory Path", value=os.path.join(self.results_dir)) # if no input else input value

                is_button_disabled = not bool(self.dataset_name.strip())
                save_button = st.button("Save Configuration Settings", disabled=is_button_disabled)
                if save_button:
                    self.config["path_settings"]["results_dir"] = self.results_dir
                    self.save_json()
        return 
        

    def save_json(self):
        self.json_file_path = os.path.join(self.config_dir, self.dataset_name+"_config.json")
        self.config['config_file_path'] = self.json_file_path
        if not os.path.exists(self.config_dir):
            os.makedirs(self.config_dir)
        with open(self.json_file_path, "w") as f:
            json.dump(self.config, f, indent=4)
        
        return st.success(f"Configuration settings saved to {self.json_file_path} ")


if __name__ == '__main__':
    ConfigGenerator()
    # import time
    # for i in range(20):
    #     print(f"Iteration {i}")
    #     print("pausing for 2 seconds")
    #     time.sleep(2)
    #     print("Resuming")

    # TODO: the above loop is working even if you close the browser. So add the generate dataset button and run the dataset generation script directly here itself


# TODO:  camera sensor size etc later.
# TODO: Most important thing add checkboxes for depth, instance, segmentation and point clouds data
# use this information in the render_dataset_images_function