import os
import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
from ModelTraining import FaceRecognitionModel, prepare_dataloaders
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Constants
IMG_HEIGHT, IMG_WIDTH = 150, 150
DATA_DIR = 'newDS'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
    transforms.GaussianBlur(kernel_size=(3, 3)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to load class labels
def load_class_labels():
    return sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])

class_labels = load_class_labels()
NUM_CLASSES = len(class_labels)

# Initialize the model
model = FaceRecognitionModel(NUM_CLASSES).to(device)
model.load_state_dict(torch.load('newDS_Trained_Model.pth'))
model.eval()

# recognize face from an image
def recognize_face(image_path, model, threshold=0.5):
    print(f"Starting recognize_face function for image: {image_path}")

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    print("Image successfully loaded.")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
    img = Image.fromarray(img)
    img = transform(img).unsqueeze(0).to(device)

    # Dummy 3D feature input
    geom_features = torch.zeros(1, 3 * 512).to(device)

    print("Passing image through the model...")
    with torch.no_grad():
        outputs = model(img, geom_features)

    print("Model output received. Computing probabilities...")
    probs = torch.nn.functional.softmax(outputs, dim=1)
    max_prob, predicted = torch.max(probs.data, 1)

    # confidence scores for each class
    for i, class_label in enumerate(class_labels):
        print(f"Class: {class_label}, Confidence: {probs[0][i].item():.4f}")

    # max confidence score
    print(f"Maximum confidence score: {max_prob.item()}, Predicted class: {class_labels[predicted.item()]}")

    if max_prob.item() > threshold:
        class_name = class_labels[predicted.item()]
        print(f"Match found: {class_name} with confidence {max_prob.item():.4f}")
        return class_name
    else:
        print("No match found. Probability too low.")
        return None


def evaluate_model(model, data_loader):
    """Evaluate the model and calculate accuracy."""
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    embeddings = []
    embedding_labels = []

    with torch.no_grad():
        for images, geom_features, labels in data_loader:
            images = images.to(device)
            geom_features = geom_features.to(device)
            labels = labels.to(device)

            outputs = model(images, geom_features)
            _, predicted = torch.max(outputs.data, 1)


            embeddings.extend(outputs.cpu().numpy())
            embedding_labels.extend(labels.cpu().numpy())

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    accuracy = 100 * correct / total
    print(f"Accuracy: {accuracy:.2f}%")

    # determine unique labels present in the test data
    unique_labels = sorted(list(set(all_labels + all_predictions)))

    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_predictions, labels=unique_labels)

    # Plot the confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=[class_labels[i] for i in unique_labels],
                yticklabels=[class_labels[i] for i in unique_labels])
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.title('Confusion Matrix')
    plt.show()

    # Plot t-SNE of embeddings
    plot_tsne(embeddings, embedding_labels)


# Plot training and validation loss curves
def plot_loss_curves(training_loss, validation_loss):
    """Plot the training and validation loss curves."""
    plt.figure()
    plt.plot(training_loss, label='Training Loss')
    plt.plot(validation_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()


def plot_tsne(embeddings, labels):
    """Plot a t-SNE or PCA visualization of the embeddings."""

    # Convert the list of embeddings to a NumPy array
    embeddings_array = np.array(embeddings)

    # Check the shape of the embeddings
    print(f"Shape of embeddings array: {embeddings_array.shape}")

    # Perform t-SNE dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings_array)

    # Plotting
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='jet', alpha=0.7)
    plt.colorbar(scatter, ticks=range(NUM_CLASSES), label='Classes')
    plt.title('t-SNE Plot of Model Embeddings')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.show()

def main():
    # Prepare Data
    _, test_loader = prepare_dataloaders(DATA_DIR)

    # Evaluate the model on the test set
    evaluate_model(model, test_loader)

    # Example of plotting loss curves
    # Load the loss curves data
    try:
        training_loss = np.load('new_training_loss.npy')
        validation_loss = np.load('new_validation_loss.npy')
        plot_loss_curves(training_loss, validation_loss)
    except FileNotFoundError:
        print("Training/Validation loss data not found. Skipping loss curve plotting.")

    # Test with a sample image
    sample_image_path = 'testImg/messi.png'
    recognize_face(sample_image_path, model)

if __name__ == "__main__":
    main()
