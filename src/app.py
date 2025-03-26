import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
from model import SkinCNN  # Import your CNN model

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define class labels
CLASS_LABELS = ["Atopic Dermatitis", "Eczema", "Melanoma", "Psoriasis"]

# Load trained model
@st.cache_resource
def load_model():
    model_path = "/home/Dock/code/SkinCNN/saved_models/skincnn.pth"  # Change this if your model is in a different location
    model = SkinCNN(num_classes=len(CLASS_LABELS)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# Define preprocessing (same as used during training)
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to match model input
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

# Predict function
def predict(image, model):
    image = image.to(device)
    with torch.no_grad():
        output = model(image)
        _, predicted_class = torch.max(output, 1)
    return CLASS_LABELS[predicted_class.item()]

# Streamlit UI
st.title("Skin Disease Classification")
st.write("Upload an image to classify it as Atopic Dermatitis, Eczema, Melanoma, or Psoriasis.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")  # Convert to RGB
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        model = load_model()
        image_tensor = preprocess_image(image)
        prediction = predict(image_tensor, model)

        st.success(f"Prediction: **{prediction}**")

