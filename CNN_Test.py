import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

# Constants
IMG_HEIGHT, IMG_WIDTH = 150, 150
BATCH_SIZE = 32
EPOCHS = 10
DATA_DIR = 'Dataset'

# create class labels
class_labels = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
NUM_CLASSES = len(class_labels)

# Load and preprocess the images
def preprocess_images(data_dir):
    datagen = ImageDataGenerator(rescale=1. / 255)

    # Setup paths for all subdirectories
    data_gen = datagen.flow_from_directory(
        data_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    # Check if the data generator has loaded the images correctly
    print(f"Found {data_gen.samples} images in {data_gen.num_classes} classes.")

    return data_gen

# Build the CNN model
def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    # Check model summary
    model.summary()

    return model

# Train the model
def train_model(model, train_data):
    model.fit(train_data, epochs=EPOCHS)

# Load and preprocess the test image
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# Predict and compare faces
def recognize_face(model, image_path, class_labels):
    img_array = preprocess_image(image_path)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    class_name = class_labels[predicted_class]

    # Show prediction confidence and result
    print(f"Prediction confidence: {predictions[0][predicted_class]}")
    print(f"The image matches with: {class_name}")

# Main script
def main():
    # generate class labels from the subdirectory names
    class_labels = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])

    train_data = preprocess_images(DATA_DIR)

    model = create_model()
    train_model(model, train_data)

    # Test with a sample image
    test_image_path = 'PRNet/Database/a.jpg'
    recognize_face(model, test_image_path, class_labels)

if __name__ == "__main__":
    main()