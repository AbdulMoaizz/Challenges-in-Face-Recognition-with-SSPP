import os
import shutil
import trimesh
import numpy as np

def capture_screenshot_with_trimesh(mesh_path, angles, obj_file_name, pose_name, save_folder):
    try:
        # Load the mesh using trimesh
        mesh = trimesh.load(mesh_path, process=False)

        # Check if the mesh is empty
        if mesh.is_empty:
            print(f"Mesh is empty: {obj_file_name}")
            return

        if len(mesh.vertices) > 100000:
            mesh = mesh.simplify_quadratic_decimation(100000)

        # Create a scene and add the mesh
        scene = trimesh.Scene(mesh)

        # Setup rotation matrix
        rotation_matrix = trimesh.transformations.euler_matrix(
            angles[0], angles[1], angles[2], 'sxyz')

        # Apply rotation
        mesh.apply_transform(rotation_matrix)

        # Render the scene to an image
        image_data = scene.save_image(resolution=(512, 512))

        # Save the image
        image_path = os.path.join(save_folder, f'{pose_name}.png')
        with open(image_path, 'wb') as f:
            f.write(image_data)

        print(f"Captured {obj_file_name} view with pose: {pose_name}")
    except Exception as e:
        print(f"Error capturing screenshot for {obj_file_name} with pose {pose_name}: {e}")

# Define angles for different views
angles_dict = {
    'pose_1': (0, 0, 0),
    'pose_2': (0, 100, 0),
    'pose_3': (0, 170, 0),
    'pose_4': (0, 190, 0),
    'pose_5': (0, 150, 0),
    'pose_6': (0, 20, 0),
    'pose_7': (0, 75, 0),
    'pose_8': (-50, 45, 0),
    'pose_9': (200, -390, -5),
    'pose_10': (45, -45, 20),
    'pose_11': (0, 45, 0),
    'pose_12': (40, -45, 0),
    'pose_13': (0, 0, 30),
    'pose_14': (0, 0, -30),
    'pose_15': (-30, 0, 45),
    'pose_16': (-30, 0, -45),
    'pose_17': (50, -30, 0),
    'pose_18': (-40, -10, 0),
    'pose_19': (120, 30, 0),
    'pose_20': (-120, -30, 0),
}


# Load all .obj files
obj_dir = 'PRNet/newOBJ'
obj_files = [f for f in os.listdir(obj_dir) if f.endswith('.obj')]

# Save folder
save_base_folder = 'newDS'

# Create base save folder if it doesn't exist
if not os.path.exists(save_base_folder):
    os.makedirs(save_base_folder)

# Process .obj file
for obj_file in obj_files:
    obj_path = os.path.join(obj_dir, obj_file)

    if not os.path.exists(obj_path):
        print(f"File does not exist: {obj_file}")
        continue

    # Use the .obj filename as the subdirectory name
    obj_name = os.path.splitext(obj_file)[0]
    person_folder = os.path.join(save_base_folder, obj_name)
    if not os.path.exists(person_folder):
        os.makedirs(person_folder)

    # Capture and save the screenshots for each angle
    for pose_name, angles in angles_dict.items():
        capture_screenshot_with_trimesh(obj_path, angles, obj_file, pose_name, person_folder)

    # Save the .obj file to the corresponding folder
    dst_obj_path = os.path.join(person_folder, obj_file)
    shutil.copy(obj_path, dst_obj_path)
    print(f"Copied {obj_file} to {person_folder}")

# Copy matching 2D images into the corresponding directories
original_images_dir = 'PRNet/newDB'
image_files = [f for f in os.listdir(original_images_dir) if os.path.isfile(os.path.join(original_images_dir, f))]

for image_file in image_files:
    # Assuming image filenames match the subdirectory names
    image_name = os.path.splitext(image_file)[0]
    destination_folder = os.path.join(save_base_folder, image_name)

    if os.path.exists(destination_folder):
        # Copy the image to the corresponding folder
        src_image_path = os.path.join(original_images_dir, image_file)
        dst_image_path = os.path.join(destination_folder, image_file)
        shutil.copy(src_image_path, dst_image_path)
        print(f"Copied {image_file} to {destination_folder}")
    else:
        print(f"No matching directory found for {image_file}")
