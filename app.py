import streamlit as st
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from PIL import Image
import gdown
import os

# === Configuration ===
MODEL_URL = "https://drive.google.com/uc?id=1GjN2Sdi2YpAVwZ06h2eLon4kp4iW2j1S"  # Updated model URL
MODEL_FILENAME = "eff_model_1000.pth"


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_FILENAME):
        gdown.download(MODEL_URL, MODEL_FILENAME, quiet=False)

    # Load the base model with pretrained weights
    model = efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(num_ftrs, 8)  # 8 blood group classes
    )

    # Load saved checkpoint
    checkpoint = torch.load(MODEL_FILENAME, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Get class label mapping
    class_to_idx = checkpoint.get('class_to_idx', {
        'A+': 0, 'A-': 1, 'AB+': 2, 'AB-': 3,
        'B+': 4, 'B-': 5, 'O+': 6, 'O-': 7
    })
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    return model, idx_to_class

# === Load model ===
model, idx_to_class = load_model()

st.title("🩸 Blood Group Prediction (EfficientNet-B3)")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "svg", "bmp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # transform = EfficientNet_B3_Weights.DEFAULT.transforms()
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_label = idx_to_class[predicted.item()]
        st.success(f"Predicted Blood Group: **{predicted_label}**")
