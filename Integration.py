import os
import cv2
import torch
import subprocess
from newDataset import capture_screenshot_with_trimesh, angles_dict
from LoadModel import recognize_face, load_class_labels
from ModelTraining import train_model, prepare_dataloaders, FaceRecognitionModel, device, class_labels

image_path = "PRNet/newDB/1.jpg"

def search_image_in_database(image_path, model):

    print(f"Checking image {image_path} in the database...")
    matched_class = recognize_face(image_path, model)

    if matched_class:
        print(f"Image found in database. Matched class: {matched_class}")
    else:
        print("No matching image found in the database.")

    return matched_class

def save_to_database(image_path, matched_class):
    # save the matched image to the correct directory
    print(f"Saving image to class {matched_class} directory...")
    target_dir = os.path.join('newDS', matched_class)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    cv2.imwrite(os.path.join(target_dir, os.path.basename(image_path)), cv2.imread(image_path))
    print(f"Image saved in class: {matched_class}")


def handle_unmatched_image(image_path):
    print(f"Handling unmatched image: {image_path}...")
    prnet_path = "/mnt/d/Uni/Dissertation/Face-Recognition-Using-Single-Sample-Per-Person/PRNet"
    demo_script = f"{prnet_path}/demo.py"
    input_dir = os.path.dirname(image_path)
    output_dir = "/mnt/d/Uni/Dissertation/Face-Recognition-Using-Single-Sample-Per-Person/PRNet/newOBJ"

    command = f"wsl python2 {demo_script} -i {input_dir} -o {output_dir} --isDlib True"
    try:
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(result.stdout.decode('utf-8'))
        print("3D model generated successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error running demo.py: {e.stderr.decode('utf-8')}")


def create_3d_screenshots():
    print("Creating 3D screenshots...")
    obj_dir = 'PRNet/newOBJ'
    for obj_file in os.listdir(obj_dir):
        if obj_file.endswith('.obj'):
            obj_path = os.path.join(obj_dir, obj_file)
            if os.path.exists(obj_path):  # Ensure .obj file exists
                person_folder = os.path.join('newDS', os.path.splitext(obj_file)[0])
                if not os.path.exists(person_folder):
                    os.makedirs(person_folder)

                for pose_name, angles in angles_dict.items():
                    capture_screenshot_with_trimesh(obj_path, angles, obj_file, pose_name, person_folder)
                    print(f"Screenshots for {obj_file} captured.")
            else:
                print(f"File {obj_path} does not exist")


def train_or_update_model():
    print("Training or updating the model...")
    train_loader, test_loader = prepare_dataloaders('newDS')

    model = FaceRecognitionModel(num_classes=len(class_labels)).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_model(model, train_loader, criterion, optimizer)

    # Save the updated model
    torch.save(model.state_dict(), 'newDS_Trained_Model.pth')
    print("Model updated with new data.")


def main():
    print(f"Using image: {image_path}")

    model = FaceRecognitionModel(num_classes=len(class_labels)).to(device)

    model.load_state_dict(torch.load('newDS_Trained_Model.pth'))

    model.eval()

    matched_class = search_image_in_database(image_path, model)

    # If a match is found, save the image to the corresponding directory and end the process
    if matched_class:
        save_to_database(image_path, matched_class)
        print("Match found, image saved. Ending process.")
        return

    # If no match is found, handle the unmatched image by generating a 3D model
    print("No match found. Processing unmatched image...")
    handle_unmatched_image(image_path)

    # Capture 3D model screenshots using newDataset.py functionality
    create_3d_screenshots()

    # Update the model with the new dataset
    train_or_update_model()

if __name__ == "__main__":
    main()

