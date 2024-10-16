# pages/predict_cancer.py
import streamlit as st
import torch
from torch.utils.data import random_split, DataLoader
import os
from PIL import Image
from torchvision import datasets, transforms, models
import torch.optim as optim
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data transformation (same as used during training)
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Define function to load the saved model
def load_model(model_name, num_classes, path_to_weights):
    if model_name == "efficientnet":
        model = models.efficientnet_b0(pretrained=False)
    elif model_name == "mobilenet":
        model = models.mobilenet_v2(pretrained=False)
    else:
        raise ValueError("Invalid model name")

    # Replace the final classification layer
    num_ftrs = model.classifier[-1].in_features
    model.classifier[-1] = torch.nn.Linear(num_ftrs, num_classes)

    # Load the saved model weights
    model.load_state_dict(torch.load(path_to_weights,map_location=device))

    # Move the model to the appropriate device (GPU or CPU)
    model = model.to(device)
    model.eval()  # Set the model to evaluation mode for inference
    return model

# Function to predict the class of a single image
def predict_image(image_path, model, class_names):
    # Load and preprocess the image
    img = Image.open(image_path)
    img = data_transforms(img).unsqueeze(0)  # Add batch dimension (1, 3, 224, 224)

    img = img.to(device)

    # Perform inference
    with torch.no_grad():
        outputs = model(img)
        _, preds = torch.max(outputs, 1)

    return class_names[preds[0]]
def app():
    st.header("Upload Image to Predict Breast Cancer")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    path_to_model = "Breast Cancer_efficientnet_best.pth"
    num_classes = 2
    model = load_model("efficientnet", num_classes, path_to_model)
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        class_names = ['breast_benign','breast_malignant']
        if st.button('Predict'):
            predicted_class = predict_image(uploaded_file, model, class_names)
            st.success(f"Predicted class: {predicted_class}")
