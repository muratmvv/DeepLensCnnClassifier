import torch 
import torch.nn as nn 
import torch.nn.functional as F

class CIFAR10_CNN(nn.Module):
    def __init__(self):
        super(CIFAR10_CNN, self).__init__()

        #1.CONVOLUTION BLOCK (Low-Level Features: Edges, Color Gradients)
        # Input: 3 channels (RGB), Output: 32 feature maps, Filter size: 3x3
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size = 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32) #Layeer Normalization: It accelerates training and makes it stable.

        # 2. Convolution Block (Mid-level features: textures, simple shapes)
        # Input: 32 channels, Output: 64 feature maps
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(64)

        #Max Pooling: Reduces dimensions bu half using a 2x2 window
        self.pool = nn.MaxPool2d(kernel_size=2, stride = 2)

        # 3. CLASSIFIER (Fully Connected / Dense Layers)
        # Dimension calculation: 32x32 -> 16x16 after 1st pooling -> 8x8 after 2nd pooling
        # Flattened size: 64 filters * 8 height * 8 width = 4096
        self.fc1 = nn.Linear(64*8*8,512)
        self.dropout = nn.Dropout(0.3) # Randomly turns off 30% of the neurons to prevent overfitting.
        self.fc2 = nn.Linear(512,10) # 10 classes for CIFAR-10


    def forward(self, x):
        #1. Convolution Block 1: Conv-> BatchNorm -> ReLU -> Pooling(32x32 -> 16x16)
        x = self.pool(F.relu(self.bn1(self.conv1(x)))) #Conv -> BatchNorm -> ReLU -> Pooling
        
        #2. Convolution Block 2: Conv-> BatchNorm -> ReLU -> Pooling(16x16 -> 8x8)
        x = self.pool(F.relu(self.bn2(self.conv2(x)))) #Conv -> BatchNorm -> ReLU -> Pooling
        
        #3. Linear (Flatten): (Batch, 64, 8, 8) -> (Batch, 4096)
        x = torch.flatten(x,1) #Flatten the tensor from (Batch, 64, 8, 8) to (Batch, 4096)
        
        
        #4. Classifier: Linear -> Dropout -> Linear
        x = F.relu(self.fc1(x)) #Linear -> ReLU -> Dropout
        x = self.dropout(x)
        x = self.fc2(x) #Linear -> Output Layer (No activation function here, as we'll use CrossEntropyLoss which applies Softmax internally)
        return x


if __name__ == '__main__':
    #Test the model architecture and Tensor dimensions
    model = CIFAR10_CNN()
    dummy_input = torch.randn(1,3,32,32) #Batch size of 1, 3 channels (RGB), 32x32 image
    output = model(dummy_input)
    print(f"Output Shape: {output.shape}") #Expected Output Shape: (1, 10) for 10 classes