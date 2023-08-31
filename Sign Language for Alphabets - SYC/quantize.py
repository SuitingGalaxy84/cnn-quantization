import torch
import torch.nn as nn
import json
import os
import argparse
from dataset import SignedAlphabets
import numpy as np
import time
from PIL import Image, ImageOps
from torchvision import transforms
import torch.ao.quantization as quantization




parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, default="E:\GitHub\cnn-quantization\Sign Language for Alphabets - SYC\checkpoint\checkpoint_100.pth")
parser.add_argument("--model_tree", type=str, default="E:\GitHub\cnn-quantization\Sign Language for Alphabets - SYC\model_tree")
args = parser.parse_args()

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.linear = nn.Linear(8*8*8, 29)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.view(-1, 8*8*8)
        return self.linear(x)

class _CNN(nn.Module):
    def __init__(self):
        super(_CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.linear = nn.Linear(8*8*8, 29)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool(x)
        x = x.view(-1, 8*8*8)
        return self.linear(x)
    
test = _CNN()
print(test.state_dict().keys())



def evaluate(dataset, model):
    predictions = []
    with torch.no_grad():
        n_correct = 0
        for i, (feature, label) in enumerate(dataset):
            target = model(feature)
            prediction = torch.max(target, 1)[1] 
            predictions.append(prediction)
            if int(prediction) == label:
                n_correct += 1
        return predictions, n_correct/dataset.__len__()*100



        



checkpoint = torch.load(args.checkpoint, map_location="cpu")



# pth = "H:\Datasets\\asl_alphabet\\asl_alphabet_test"
# inputs = os.listdir(pth)
# for ipt in inputs:
#     file_name = ipt
#     ipt = os.path.join(pth, ipt)
#     ipt = Image.open(ipt)
#     ipt = ImageOps.grayscale(ipt)
#     ipt.thumbnail((32, 32))
#     ipt = transforms.ToTensor()(ipt)

 


# val_dataset = SignedAlphabets(pth="H:\Datasets\\asl_alphabet\\asl_alphabet_train", type="val", split=0.9, transform=None)
# evaluate(val_dataset, checkpoint)







state_dict = dict(checkpoint.state_dict())
parameters_tensor = state_dict.values()
parameters_array = []
for value in list(parameters_tensor):
    value = np.asarray(value)
    parameters_array.append(value)
state_dict = dict(zip(list(state_dict.keys()), parameters_array))
#tree = dict(state_dict)
print(state_dict.keys())
for key in list(state_dict.keys()):
    key_ = key.replace(".", "_")
    np.save(os.path.join(args.model_tree, f"{key_}"), state_dict[key])





checkpoint.eval()
modules_to_fuse = [["conv1" , "relu"], ["conv2", "relu"]]
fused_module = quantization.fuse_modules(model=checkpoint,
                                         modules_to_fuse=modules_to_fuse)
print(fused_module.state_dict())                  