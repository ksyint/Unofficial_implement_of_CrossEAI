import torch
import torch.nn as nn
import torchvision.models as models

class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        self.backbone = models.resnet50(pretrained=True)
        self.backbone.fc = nn.Identity()  # Remove the classification layer
        self.fc = nn.Linear(self.backbone.fc.in_features, 2)  # Add new classification layer

    def forward(self, x):
        features = self.backbone(x)
        output = self.fc(features)
        return features, output
