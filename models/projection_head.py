import torch
import torch.nn as nn

class ProjectionHead(nn.Module):
    def __init__(self, input_dim, projection_dim):
        super(ProjectionHead, self).__init__()
        self.layer1 = nn.Linear(input_dim, projection_dim)
        self.layer2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        return x
