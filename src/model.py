import torch.nn as nn
import torchvision.models as models

def SkinCNN(num_classes):
    model = models.resnet18(pretrained=True)  # Using ResNet-18
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # Adjust final layer
    return model
