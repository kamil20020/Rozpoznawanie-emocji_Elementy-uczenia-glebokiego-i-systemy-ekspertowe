import torch
import os
from torch import nn
import random
from PIL import Image
import glob
from pathlib import Path
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchinfo
from torchinfo import summary
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(30 * 30, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.ReLU()
        )

    def forward(self, data):
        data = self.flatten(data)
        logits = self.linear_relu_stack(data)
        return logits

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (data, labels) in enumerate(dataloader):
        data, labels = data.to(device), labels.to(device)
        
        # Compute prediction error
        pred = model(data)
        loss = loss_fn(pred, labels)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(data)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
                
def test(dataloader, model):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for data, labels in dataloader:
            data, labels = data.to(device), labels.to(device)
            pred = model(data)
            test_loss += loss_fn(pred, labels).item()
            correct += (pred.argmax(1) == labels).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

model = NeuralNetwork().to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
learning_rate = 1e-3
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

def walk_through_dir(dir_path):
  for dirpath, dirnames, filenames in os.walk(dir_path):
    print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")
    
image_path = ".\images"
walk_through_dir(image_path)

train_dir = ".\images\\train"
test_dir = ".\images\\test"
print(train_dir, test_dir)

#-------------------------------------------------------------------------------------------------------------------------------------

train_transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(size=30), transforms.Grayscale(num_output_channels=1)])

# Create testing transform (no data augmentation)
test_transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(size=30), transforms.Grayscale(num_output_channels=1)])

# Creating training set
train_data = datasets.ImageFolder(root=train_dir, transform=train_transform)
#Creating test set
test_data = datasets.ImageFolder(root=test_dir, transform=test_transform)

print(f"Train data:\n{train_data}\nTest data:\n{test_data}")

# Get class names as a list
class_names = train_data.classes
print("Class names: ",class_names)

# Can also get class names as a dict
class_dict = train_data.class_to_idx
print("Class names as a dict: ",class_dict)

# Check the lengths
print("The lengths of the training and test sets: ", len(train_data), len(test_data))

BATCH_SIZE = 64

train_dataloader_augmented = DataLoader(train_data, batch_size=BATCH_SIZE)

test_dataloader_augmented = DataLoader(test_data, batch_size=BATCH_SIZE)

epochs = 15
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader_augmented, model, loss_fn, optimizer)
    test(test_dataloader_augmented, model)
print("Done!")