import torch
#torch.set_printoptions(profile="full")
from torch.utils.data import Dataset
import os
from PIL import Image, ImageOps
import string
import shutil
import torchvision.transforms as transforms
import random
from tqdm import tqdm
import argparse
import re


class SignedAlphabets(Dataset):
    def __init__(self, pth=str, type=str, split=float, transform=None):
        super(SignedAlphabets, self).__init__()
        self.pth = pth
        self.type = type
        self.split = split
        self.transform = transform

        self.labels = os.listdir(pth)
        self.samples = []
        self.sample_info = []
        count = 0
        for i in self.labels:
            save_example = 1
            example_pth = os.path.join("Sign Language for Alphabets\example", str(i))
            if os.path.exists("Sign Language for Alphabets\example") is True and count == 0:
                flag = input("example already exist, would you like to build a new example? [Y/N]")
                if flag == "Y" or flag == "y":
                    shutil.rmtree(example_pth)
                    save_example = 1
                elif flag == "N" or flag == "n":
                    save_example = 0
                else:
                    exit()
            pth = os.path.join(self.pth, str(i))
            sample_info =  os.listdir(pth)
            if os.path.exists(example_pth) is False:
                os.makedirs(example_pth)
            for sample_pth in sample_info:
                sample = Image.open(os.path.join(pth, sample_pth))
                sample = ImageOps.grayscale(sample)
                sample.thumbnail((32, 32))
                sample_tensor = transforms.ToTensor()(sample)

                self.samples.append(sample_tensor)
                
                if save_example == 1 and flag != "N" and flag != "n":
                    print(f"saving example: {os.path.join(example_pth, sample_pth)}")
                    sample.save(os.path.join(example_pth, sample_pth))
                    save_example = 0
            count =+ 1

            self.sample_info += sample_info
        
        self.lut = dict(zip(self.samples, self.sample_info))
        self.features = list(self.lut.keys())
        random.shuffle(self.features)
        if self.type == "train":
            self.data = self.features[: int(len(self.features) * split)]
            print(f"load train data! length: {len(self.data)}")
        elif self.type == "val":
            del self.features[: int(len(self.features) * split)]
            self.data = self.features
            print(f"load validation data! length: {len(self.data)}")
        else:
            print("Wrong Dataset Type!")
            exit()
        self.num = list(range(26))
        self.char = string.ascii_lowercase
        self.char_to_num =  dict(zip(self.char, self.num))






    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        feature = self.data[idx]
        label = self.lut[feature].split("_")[1]
        if label in string.ascii_lowercase:
            label = self.char_to_num[label] + 10

        if self.transform is not None:
            feature = self.transform(feature)
        return feature, int(label)
    

class SignedAlphabets(Dataset):
    def __init__(self, pth=str, type=str, split=float, transform=None):
        super(SignedAlphabets, self).__init__()
        self.pth = pth
        self.type = type
        self.split = split
        self.transform = transform

        self.labels = os.listdir(pth)
        self.samples = []
        self.sample_info = []
        count = 0
        for i in tqdm(self.labels):
            save_example = 1
            example_pth = os.path.join("Sign Language for Alphabets\example", str(i))
            
            if os.path.exists("Sign Language for Alphabets\example") is True and count == 0:
                flag = input("example already exist, would you like to build a new example? [Y/N]")
                if flag == "Y" or flag == "y":
                    shutil.rmtree(example_pth)
                    save_example = 1
                elif flag == "N" or flag == "n":
                    save_example = 0
                else:
                    exit()
            elif count == 0:
                os.mkdir("Sign Language for Alphabets\example")
            pth = os.path.join(self.pth, str(i))
            sample_info =  os.listdir(pth)

            if os.path.exists(example_pth) is False:
                os.makedirs(example_pth)
            for sample_pth in sample_info:
                sample = Image.open(os.path.join(pth, sample_pth))
                sample = ImageOps.grayscale(sample)
                sample.thumbnail((32, 32))
                sample_tensor = transforms.ToTensor()(sample)

                self.samples.append(sample_tensor)
                
                if save_example <= 4 and flag != "N" and flag != "n":
                    print(f"saving example: {os.path.join(example_pth, sample_pth)}")
                    sample.save(os.path.join(example_pth, sample_pth))
                    save_example += 1
            count =+ 1

            self.sample_info += sample_info
         
        self.lut = dict(zip(self.samples, self.sample_info))
        self.features = list(self.lut.keys())
        random.shuffle(self.features)
        if self.type == "train":
            self.data = self.features[: int(len(self.features) * split)]
            print(f"load train data! length: {len(self.data)}")
        elif self.type == "val":
            del self.features[: int(len(self.features) * split)]
            self.data = self.features
            print(f"load validation data! length: {len(self.data)}")
        else:
            print("Wrong Dataset Type!")
            exit()
        self.num = list(range(len(os.listdir(pth))))
        self.char = list(string.ascii_uppercase)
        self.char.append("nothing")
        self.char.append("del")
        self.char.append("space")

        self.char_to_num =  dict(zip(self.char, self.num))






    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        feature = self.data[idx]
        label = self.lut[feature]
        label = re.split(r'(\d+)', label)
        label = label[0]
        if label in self.char:
            label = self.char_to_num[label]

        if self.transform is not None:
            feature = self.transform(feature)
        return feature, int(label)
    



            







# SignedN = SignedAlphabets(pth="H:\Datasets\\asl_alphabet\\asl_alphabet_train",
#                          type="train", 
#                          split=0.9,
#                      transform=None)
# for i in tqdm(range(SignedN.__len__())):
#     feature, label = SignedN.__getitem__(i)
#     print(label)





