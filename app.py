import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision.models import convnext_base, ConvNeXt_Base_Weights
import torch.nn as nn
from PIL import Image

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class names
class_names = ['A-', 'A+', 'AB-', 'AB+', 'B-', 'B+', 'O-', 'O+']

# Load the model
model = convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT)
num_ftrs = model.classifier[2].in_features
model.classifier[2] = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(num_ftrs, len(class_names))
)
model.load_state_dict(torch.load('convnext_model.pth', map_location=device))
model = model.to(device)
model.eval()

# Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Streamlit app
st.title("Blood Group Detection from Fingerprint")
st.write("Upload a fingerprint image to predict the blood group.")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp", "webp"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Fingerprint", use_column_width=True)
    try:
        img_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(img_tensor)
            _, predicted = torch.max(outputs, 1)
            prediction = class_names[predicted.item()]
        st.success(f"Predicted Blood Group: {prediction}")
    except Exception as e:
        st.error(f"Error: {str(e)}")