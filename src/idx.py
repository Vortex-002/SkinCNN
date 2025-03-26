import torch
import os
import json
from data_loader import load_data

def save_class_mapping(data_dir, save_dir="/home/Dock/code/SkinCNN/saved_models"):
    train_loader, _, _ = load_data(data_dir, batch_size=16)

    # ✅ Get dataset object
    dataset = train_loader.dataset.dataset if isinstance(train_loader.dataset, torch.utils.data.Subset) else train_loader.dataset

    # ✅ Extract class-to-index mapping
    os.makedirs(save_dir, exist_ok=True)  # Ensure directory exists
    class_to_idx_path = os.path.join(save_dir, "class_to_idx.json")

    if hasattr(dataset, 'class_to_idx'):
        class_to_idx = dataset.class_to_idx
        print("✅ Class-to-Index Mapping:", class_to_idx)

        # 🔽 Save mapping to a JSON file
        with open(class_to_idx_path, "w") as f:
            json.dump(class_to_idx, f)

        print(f"✅ Class-to-Index mapping saved to {class_to_idx_path}")
    else:
        print("❌ Error: class_to_idx not found!")

if __name__ == "__main__":
    data_dir = "/home/Dock/code/SkinCNN/data"
    save_class_mapping(data_dir)

