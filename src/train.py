import torch
import torch.optim as optim
import torch.nn as nn
import os
from data_loader import load_data
from model import SkinCNN

def train_model(data_dir, save_dir="/home/Dock/code/SkinCNN/saved_models", epochs=10, batch_size=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, _ = load_data(data_dir, batch_size)
    
    # ✅ Check dataset type
    print("Train Dataset Type:", type(train_loader.dataset))
    
    # ✅ Get original dataset if using Subset
    dataset = train_loader.dataset.dataset if isinstance(train_loader.dataset, torch.utils.data.Subset) else train_loader.dataset
    print("Dataset Type:", type(dataset))

    # ✅ Check for class_to_idx
    if hasattr(dataset, 'class_to_idx'):
        print("Class-to-Index Mapping:", dataset.class_to_idx)
        num_classes = len(dataset.class_to_idx)
    else:
        print("Error: class_to_idx not found! Defaulting to num_classes = 4")
        num_classes = 4  # Default fallback
    
    model = SkinCNN(num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4, nesterov=True)
    best_acc = 0.0
    
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, "skincnn.pth")
    
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        train_loss = running_loss / total
        train_acc = correct / total
        
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        val_loss /= val_total
        val_acc = val_correct / val_total
        
        print(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to {model_path}")

if __name__ == "__main__":
    data_dir = "/home/Dock/code/SkinCNN/data"
    train_model(data_dir)

