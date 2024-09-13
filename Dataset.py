import os
import numpy as np
from PIL import Image
import shutil
import open3d as o3d

def capture_screenshot(mesh, angle, obj_file_name, pose_name, save_folder):
    try:
        # Create a visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)
        vis.add_geometry(mesh)

        # Set the view control parameters
        ctr = vis.get_view_control()

        # Rotate the view control to the specified angle
        ctr.rotate(angle[0], angle[1])

        # Capture the image
        vis.poll_events()
        vis.update_renderer()
        image = vis.capture_screen_float_buffer(do_render=True)

        # Ensure the save folder exists
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        # Save the image to the specified folder with the .obj filename and pose name
        image_path = os.path.join(save_folder, f'{pose_name}.png')
        image = (np.asarray(image) * 255).astype(np.uint8)
        image = Image.fromarray(image)
        image.save(image_path)

        # Clean up
        vis.destroy_window()

        # Debugging output to verify the camera transformation
        print(f"Captured {obj_file_name} view with pose: {pose_name}")
    except Exception as e:
        print(f"Error capturing screenshot for {obj_file_name} with pose {pose_name}: {e}")


# Define angles for different views
angles = {
    'pose_1': (-180, 0),
    'pose_2': (-135, 0),
    'pose_3': (-90, 0),
    'pose_4': (-45, 0),
    'pose_5': (0, 0),
    'pose_6': (45, 0),
    'pose_7': (90, 0),
    'pose_8': (135, 0),
    'pose_9': (180, 0),
    'pose_10': (225, 0),
    'pose_11': (270, 0),
    'pose_12': (315, 0),
    'pose_13': (0, 90),  # Top view
    'pose_14': (0, -90),  # Bottom view
    'pose_15': (0, 45),
    'pose_16': (0, -45),
    'pose_17': (45, 45),
    'pose_18': (-45, -45),
    'pose_19': (135, 45),
    'pose_20': (-135, -45),
}

# Load all .obj files from the directory
obj_dir = 'PRNet/obj_db'
obj_files = [f for f in os.listdir(obj_dir) if f.endswith('.obj')]

# Save folder
save_base_folder = 'Dataset'

# Create base save folder if it doesn't exist
if not os.path.exists(save_base_folder):
    os.makedirs(save_base_folder)

# Process each .obj file
for obj_file in obj_files:
    obj_path = os.path.join(obj_dir, obj_file)
    mesh = o3d.io.read_triangle_mesh(obj_path)

    if mesh.is_empty():
        print(f"Failed to load mesh: {obj_file}")
        continue

    # Use the .obj filename (without extension) as the subdirectory name
    obj_name = os.path.splitext(obj_file)[0]
    person_folder = os.path.join(save_base_folder, obj_name)
    if not os.path.exists(person_folder):
        os.makedirs(person_folder)

    # Capture and save the screenshots for each angle
    for pose_name, angle in angles.items():
        capture_screenshot(mesh, angle, obj_file, pose_name, person_folder)

    # Save the .obj file to the corresponding folder
    dst_obj_path = os.path.join(person_folder, obj_file)
    shutil.copy(obj_path, dst_obj_path)
    print(f"Copied {obj_file} to {person_folder}")

# Copy matching 2D images into the corresponding directories
original_images_dir = 'PRNet/Database'
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
