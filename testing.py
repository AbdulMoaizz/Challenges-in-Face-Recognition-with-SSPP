import os
import cv2
import torch
import subprocess
from newDataset import capture_screenshot_with_trimesh, angles_dict
from LoadModel import recognize_face, load_class_labels
from ModelTraining import train_model, prepare_dataloaders, FaceRecognitionModel, device, class_labels

image_path = "PRNet/Database/1.jpg"


def search_image_in_database(image_path, model):
    # Call recognize_face and return the matched class or None if no match
    print(f"Checking image {image_path} in the database...")
    matched_class = recognize_face(image_path, model)  # Pass the model here

    if matched_class:
        print(f"Image found in database. Matched class: {matched_class}")
    else:
        print("No matching image found in the database.")

    print(f"Matched class result: {matched_class}")  # Output the result of the matching
    return matched_class


def save_to_database(image_path, matched_class):
    # Function to save the matched image to the correct directory
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
        print(result.stdout.decode('utf-8'))  # Output from demo.py
        print("3D model generated successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error running demo.py: {e.stderr.decode('utf-8')}")


def create_3d_screenshots():
    print("This function is disabled.")



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
    print("Main function running.")
    # Comment out everything else to ensure only main() runs.

if __name__ == "__main__":
    main()

