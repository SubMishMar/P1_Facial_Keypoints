import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)  
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)  

        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)  

        # Compute the correct input size for fc1
        self._to_linear = None  # Placeholder
        
        # BatchNorm & Dropout
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.dropout = nn.Dropout(p=0.4)
        
        # Fully connected layers (will be set later)
        self.fc1 = None
        self.fc2 = None

    def _compute_fc_input_size(self):
        """Helper function to determine the correct input size for fc1"""
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 96, 96)  # Example image
            x = self.pool(F.relu(self.bn1(self.conv1(dummy_input))))
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            x = self.pool(F.relu(self.bn3(self.conv3(x))))
            self._to_linear = x.view(1, -1).shape[1]  # Compute flattened size

        # Define fully connected layers now
        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc2 = nn.Linear(512, 136)  # Output 136 (68 keypoints * 2)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        x = x.view(x.size(0), -1)  # Flatten

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  
        
        return x