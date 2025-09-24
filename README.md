# SathwikPanchangamThesis

# Constraint Based Dataset Generation Using Blender API (Bpy)
This repository contains **Blender scripting** and **synthetic data generation** for different constraints like, Normal, Bright, Dark, Far, Near, Blur, Constraint_Textures and Deformation

## 📦 Setup Instructions
### 1. Clone the repository
```
$ git clone git@github.com:RnDProjectsDeebul/SathwikPanchangamThesis.git
$ cd SathwikPanchangamThesis/
```
### Environment Setup
```
$ conda env create -f environment.yml
$ conda activate data_generation
```

### Config File Generation
```
$ cd Dataset_Generation_Tool/src
$ streamlit run generate_config.py
```

### Dataset Generation
```
$ cd Dataset_Generation_Tool/src
$ python3 main.py --config <your_config_file_name>