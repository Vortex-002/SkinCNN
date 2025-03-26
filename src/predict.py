import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import os
from model import SkinCNN

# Define class index to disease name mapping
CLASS_LABELS = {
    0: "Atopic Dermatitis",
    1: "Eczema",
    2: "Melanoma",
    3: "Psoriasis"
}

# Load the trained model
def load_model(model_path, device):
    model = SkinCNN(num_classes=4)  # Use num_classes=4 as per training
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Preprocess the input image
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Match ResNet input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

# Make a prediction
def predict(image_path, model, device):
    image = preprocess_image(image_path).to(device)
    with torch.no_grad():
        output = model(image)  # Shape: [1, 4] (since 4 classes)
        probs = F.softmax(output, dim=1)  # Convert logits to probabilities
        predicted_class = torch.argmax(probs, dim=1).item()  # Get class index
        confidence = probs[0, predicted_class].item()  # Get confidence of predicted class

    disease_name = CLASS_LABELS.get(predicted_class, "Unknown")
    return disease_name, confidence

if __name__ == "__main__":
    model_path = "/home/Dock/code/SkinCNN/saved_models/skincnn.pth"  # Update if necessary
    image_path = "/home/Dock/code/1_2.jpg"  # Replace with the actual image path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found.")
    elif not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
    else:
        model = load_model(model_path, device)
        disease, confidence = predict(image_path, model, device)
        print(f"Prediction: {disease} (Confidence: {confidence:.4f})")

