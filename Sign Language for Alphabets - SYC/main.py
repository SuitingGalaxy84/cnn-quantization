import torch
from torch.utils.data import DataLoader
import multiprocessing as mp
import argparse
from tqdm import tqdm
from dataset import SignedAlphabets
import torch.nn as nn
from torch import optim


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="H:\Datasets\CHESS")
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--num-epochs", type=int, default=1000)
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


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

train = SignedAlphabets(pth="H:\Datasets\\asl_alphabet\\asl_alphabet_train", split=0.9, type="train", transform=None)
val = SignedAlphabets(pth="H:\Datasets\\asl_alphabet\\asl_alphabet_train", split=0.9, type="val", transform=None)

train_loader = DataLoader(train, args.batch_size, True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val, 4, False, num_workers=0, pin_memory=True)


model = CNN().to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")

for epoch in tqdm(range(args.num_epochs)):
    for i, (features, labels) in enumerate(train_loader):
        labels = labels.to(device)
        features = features.to(device)
        features_copy = features.clone().detach()
        pred = model(features_copy)
        loss = criterion(pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 50 == 0:
                #print("one epoch finished, running validation")
            n_correct = 0
            n_sample = 0
            val = []
            with torch.no_grad():
                for j, (features, labels) in enumerate(val_loader):
                    features = features.to(device).clone().detach()
                    labels = labels.to(device)
                    prediction = model(features)
                    val_loss = criterion(prediction,labels)
                    val.append(val_loss)
                val_loss = sum(val)/len(val)
            print(f'for epoch {epoch + 1}/{args.num_epochs}, iteration {i}  loss = {loss:.8f}, ave_val_loss = {val_loss}\n')
    scheduler.step(val_loss)
    if epoch % 50 == 0:
        torch.save(model, f"Sign Language for Alphabets\checkpoint\checkpoint_{epoch}.pth")
            
    



