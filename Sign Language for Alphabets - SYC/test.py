import torch
from PIL import Image, ImageOps
import torch.nn as nn
import torchvision.transforms as transform

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.linear = nn.Linear(8*8*8, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.view(-1, 8*8*8)
        return self.linear(x)

model = torch.load("Sign Language for Alphabets\checkpoint\checkpoint_950.pth", map_location="cpu")
sample = Image.open("Sign Language for Alphabets\\test\\3_2.jpg")
sample = ImageOps.grayscale(sample)
sample.thumbnail((32, 32))
sample.show()
sample = transform.ToTensor()(sample)
print(nn.functional.softmax(model(sample)))