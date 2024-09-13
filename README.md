# Face Recognition Using Single Sample Per Person (SSPP)

Here is the README file based on the provided requirements and details of the files.

---

# Face Recognition Using Single Sample Per Person (SSPP)

## Overview

This project is designed to tackle the Single Sample Per Person (SSPP) challenge in face recognition by generating synthetic images from a single 2D face image and leveraging deep learning models for face recognition. The system utilizes **PRNet** for 3D face reconstruction and a hybrid deep learning model to recognize faces with improved accuracy under various conditions.

## Repository Structure

The project consists of the following main components:

- **PRNet/demo.py**: Script for generating 3D face models from a single 2D image.
- **Integration.py**: Handles the integration of checking the face image in the database, and if not available, it sends the image to PRNet for 3D model generation.
- **LoadModel.py**: Contains functions for loading the trained face recognition models.
- **ModelTraining.py**: Code for training the face recognition model using 2D and 3D face images.
- **newDataset.py**: Handles the generation of synthetic screenshots from 3D models, using various angles to augment the dataset.

## Requirements

The project requires two environments to function properly due to the different dependencies used for PRNet and other components.

### PRNet (Python 2.7 Environment)

The PRNet file (`demo.py`) requires **Python 2.7** and **TensorFlow 1.4**, which are outdated versions and not supported on modern Windows systems. To run this, you need to set up **Ubuntu** in your Windows system using **WSL (Windows Subsystem for Linux)**.

### Steps to Set Up PRNet

1. **Install Ubuntu on Windows**  
   Use WSL to install Ubuntu on your Windows system by following [this guide](https://docs.microsoft.com/en-us/windows/wsl/install).

2. **Install Python 2.7**  
   Download and install Python 2.7 from the following link:  
   [Python 2.7 Download](https://www.python.org/ftp/python/2.7.1/python-2.7.1.amd64.msi)

3. **Install pip for Python 2.7**  
   Download and install pip for Python 2.7 using the following command:  
   ```bash
   curl https://bootstrap.pypa.io/pip/2.7/get-pip.py --output get-pip.py
   python2 get-pip.py
   ```

4. **Install TensorFlow 1.4**  
   Install TensorFlow 1.4 with the following command:  
   ```bash
   pip2 install tensorflow==1.4.0
   ```

5. **Install Additional Libraries for PRNet**  
   Install the following libraries needed for PRNet:
   - CMake 3.12.0:  
     ```bash
     pip2 install cmake==3.12.0
     ```
   - Dlib 19.16.0:  
     ```bash
     pip2 install dlib==19.16.0
     ```

### Python 3.12 Environment

The rest of the project files, including `Integration.py`, `LoadModel.py`, `ModelTraining.py`, and `newDataset.py`, are designed to run in **Python 3.12**. Ensure you have the following dependencies installed:

1. **Python 3.12**:  
   Download and install Python 3.12 from [here](https://www.python.org/downloads/).

2. **Install Required Libraries**:  
   Install the required libraries using pip:
   ```bash
   pip install -r requirements.txt
   ```
   Make sure to create a `requirements.txt` file with the necessary libraries such as:
   - **torch** (for PyTorch)
   - **opencv-python**
   - **numpy**
   - **trimesh**

## Running the Code

### Step 1: Generate 3D Model Using PRNet
Run the `demo.py` file from the **PRNet** directory using the following command from the Ubuntu terminal:
```bash
python2 demo.py -i input_directory -o output_directory --isDlib True
```
This command takes a 2D image as input and generates a 3D face model as output.

### Step 2: Capture Synthetic Screenshots
Use the `newDataset.py` script to generate synthetic images from the 3D model. This will create images from multiple angles to enhance the dataset.

### Step 3: Train the Model
Use the `ModelTraining.py` script to train the model using both 2D and synthetic 3D images. This step involves preparing the dataset and training the deep learning model.

### Step 4: Face Recognition
The `Integration.py` script handles checking the input image against the database. If no match is found, it triggers PRNet to generate the 3D model, captures synthetic images, and updates the model for future use.

## Conclusion

This repository provides a comprehensive solution for solving the SSPP challenge by integrating 3D face model generation, synthetic image creation, and deep learning-based face recognition. Ensure that both environments (Python 2.7 and Python 3.12) are correctly set up to run the respective components of the project.
