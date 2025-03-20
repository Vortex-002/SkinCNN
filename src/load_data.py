import torch.utils
import torch.utils.data.dataloader
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torch.utils.data import DataLoader, random_split


transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

dataset_path = "/home/Dock/code/model/data/Datasets/train"
dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

batch_size = 16
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)



def show_images(dataloader):

    images, labels = next(iter(dataloader))
    images = images * 0.5 + 0.5

    img_grid = make_grid(images[:8], nrow=4)
    plt.figure(figsize=(10,5))
    plt.imshow(img_grid.permute(1,2,0))
    plt.axis("off")
    plt.show()


show_images(train_loader)

print(f"Total Images: {len(dataset)}")
print(f"Training set: {len(train_set)} images")
print(f"Validation set: {len(val_set)} images")
print(f"Test set: {len(test_set)} images")