import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import os
import trimesh
from PIL import Image

def count_subdirectories(data_dir):
    """Count the number of subdirectories in a given directory."""
    try:
        entries = os.listdir(data_dir)
        subdirs = [entry for entry in entries if os.path.isdir(os.path.join(data_dir, entry))]
        return len(subdirs)
    except FileNotFoundError:
        print(f"The directory '{data_dir}' does not exist.")
        return 0
    except PermissionError:
        print(f"Permission denied to access '{data_dir}'.")
        return 0

# Constants
IMG_HEIGHT, IMG_WIDTH = 150, 150
EPOCHS = 10
DATA_DIR = 'newDS'
BATCH_SIZE = count_subdirectories(DATA_DIR)

# Update data augmentation pipeline
transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
    transforms.GaussianBlur(kernel_size=(3, 3)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create class labels from the dataset subdirectories
class_labels = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
NUM_CLASSES = len(class_labels)

# Define Dataset Class
class FaceDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load 2D image
        image = cv2.imread(self.image_paths[idx])
        if image is None:
            raise FileNotFoundError(f"Image not found: {self.image_paths[idx]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)  # Convert ndarray to PIL Image

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        # Load and process 3D model
        obj_path = self.image_paths[idx].replace('.png', '.obj')
        if os.path.exists(obj_path):
            mesh = trimesh.load(obj_path)
            geometric_features = self.extract_geometric_features(mesh)
        else:
            geometric_features = torch.zeros(3 * 512)  # Ensure this is the correct size

        label = self.labels[idx]
        return image, geometric_features, torch.tensor(label, dtype=torch.long)

    def extract_geometric_features(self, mesh):
        # Extract vertex normals as geometric features
        normals = mesh.vertex_normals

        # Ensuring a fixed size for the features
        if len(normals) > 512:
            normals = normals[:512]  # Truncate if more than 512 vertices
        elif len(normals) < 512:
            padding = 512 - len(normals)
            normals = np.pad(normals, ((0, padding), (0, 0)), mode='constant', constant_values=0)

        normals = normals.flatten()  # Flatten to 1D array
        return torch.tensor(normals, dtype=torch.float32)

# Define Model
class FaceRecognitionModel(nn.Module):
    def __init__(self, num_classes):
        super(FaceRecognitionModel, self).__init__()

        # 2D CNN for image processing (Using ResNet50)
        self.resnet = models.resnet50(weights='DEFAULT')  # Updated for latest weights parameter
        self.resnet.fc = nn.Identity()  # Remove the final classification layer

        # 3D feature processing (Simple MLP as an example)
        self.fc3d = nn.Sequential(
            nn.Linear(3 * 512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5)  # Adding dropout for regularization
        )

        # Final classifier combining both 2D and 3D features
        self.classifier = nn.Sequential(
            nn.Linear(2048 + 512, 256),  # Input: 2048 (ResNet) + 512 (3D features)
            nn.ReLU(),
            nn.Dropout(0.5),  # Adding dropout before the final layer
            nn.Linear(256, num_classes)
        )

    def forward(self, image, geometric_features):
        x2d = self.resnet(image)  # Shape: [batch_size, 2048]
        x3d = self.fc3d(geometric_features)  # Shape: [batch_size, 512]
        combined = torch.cat((x2d, x3d), dim=1)  # Shape: [batch_size, 2560]
        out = self.classifier(combined)  # Output shape: [batch_size, num_classes]
        return out

# Ensure labels are correctly mapped
def get_image_paths_and_labels(data_dir):
    image_paths = []
    labels = []
    label_map = {label: idx for idx, label in enumerate(class_labels)}  # Ensuring correct label indexing
    for label in sorted(os.listdir(data_dir)):
        class_dir = os.path.join(data_dir, label)
        if os.path.isdir(class_dir):
            for img_name in os.listdir(class_dir):
                if img_name.endswith('.png'):
                    image_paths.append(os.path.join(class_dir, img_name))
                    labels.append(label_map[label])
    return image_paths, labels

# Create data loaders
def prepare_dataloaders(data_dir):
    image_paths, labels = get_image_paths_and_labels(data_dir)

    # Train-test split
    train_paths, test_paths, train_labels, test_labels = train_test_split(image_paths, labels, test_size=0.2, random_state=42)

    # Create Dataset and DataLoader
    train_dataset = FaceDataset(train_paths, train_labels, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_dataset = FaceDataset(test_paths, test_labels, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, test_loader

# Train the Model
def train_model(model, train_loader, criterion, optimizer, num_epochs=EPOCHS):
    model.train()
    training_losses = []  # List to store training losses

    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, geom_features, labels in train_loader:
            images = images.to(device)
            geom_features = geom_features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images, geom_features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        average_loss = running_loss / len(train_loader)
        training_losses.append(average_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}")

    # Save the training losses to a file
    np.save('new_training_loss.npy', np.array(training_losses))

# evaluate model
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    validation_losses = []  # List to store validation losses

    with torch.no_grad():
        for images, geom_features, labels in test_loader:
            images = images.to(device)
            geom_features = geom_features.to(device)
            labels = labels.to(device)

            outputs = model(images, geom_features)
            loss = nn.CrossEntropyLoss()(outputs, labels)  # Compute validation loss
            validation_losses.append(loss.item())

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Collecting all labels and predictions for confusion matrix
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    accuracy = 100 * correct / total
    print(f"Accuracy: {accuracy:.2f}%")

    # Save the validation losses to a file
    np.save('new_validation_loss.npy', np.array(validation_losses))


# Function to recognize face from an image (using the trained model)
def recognize_face(model, image_path, class_labels, threshold=0.7):
    model.eval()
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
    img = Image.fromarray(img)  # Convert ndarray to PIL Image
    img = transform(img).unsqueeze(0).to(device)  # Apply the transformations and add batch dimension

    # Dummy 3D feature input
    geom_features = torch.zeros(1, 3 * 512).to(device)

    outputs = model(img, geom_features)

    probs = torch.nn.functional.softmax(outputs, dim=1)
    max_prob, predicted = torch.max(probs.data, 1)

    if max_prob.item() > threshold:
        class_name = class_labels[predicted.item()]
        print(f"The image matches with: {class_name}")
    else:
        print("No matching person found.")

    # Debug: Print confidence scores for each class
    print("Confidence scores:", probs.cpu().data.numpy())

# Main Script
def main():
    # Prepare Data
    train_loader, test_loader = prepare_dataloaders(DATA_DIR)

    # Initialize Model, Criterion, and Optimizer
    model = FaceRecognitionModel(NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the Model
    train_model(model, train_loader, criterion, optimizer, num_epochs=EPOCHS)

    # Evaluate the Model
    evaluate_model(model, test_loader)

    # Save the trained model
    torch.save(model.state_dict(), 'newDS_Trained_Model.pth')
    print("Model trained successfully")

    # Test with a sample image (example usage)
    sample_image_path = 'testImg/moaiz.jpg'
    recognize_face(model, sample_image_path, class_labels)

if __name__ == "__main__":
    main()
