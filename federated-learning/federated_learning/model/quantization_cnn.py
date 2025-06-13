import torch
import torch.nn as nn
import torch.nn.functional as F 

class Net(nn.Module):
    def __init__(self, num_classes=10):
        super(Net, self).__init__()
        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.fc1 = nn.Linear(64 * 7 * 7, 128, bias=False)  # Assuming input size is 28x28
        self.fc2 = nn.Linear(128, num_classes, bias=False)  # Output layer for classification

    def forward(self, x):
        x = self.quant(x)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.reshape(x.size(0), -1) 
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.dequant(x)
        return x
