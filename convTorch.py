import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from PIL import Image
import random


transform = torchvision.transforms.Compose(
    [torchvision.transforms.Resize(size=48),torchvision.transforms.ToTensor(), torchvision.transforms.Grayscale(num_output_channels=1)]
)

train_dataset = torchvision.datasets.ImageFolder(
     "./images/train", transform=transform
)

test_dataset = torchvision.datasets.ImageFolder(
     "./images/test", transform=transform
)

batch_size = 32

trainloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)
testloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=True
)


def show_batch(dl):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(images[:batch_size], nrow=8).permute(1, 2, 0))
        #plt.show() #display plot
        break
#show_batch(trainloader)

class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3,3)) #stride=1, padding=1
        self.act1 = nn.ReLU()
        #self.drop1 = nn.Dropout(0.3)
 
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3,3)) #stride=1, padding=1
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
 
        self.flat = nn.Flatten()
 
        self.fc3 = nn.Linear(6272, 512)
        self.act3 = nn.ReLU()
        #self.drop3 = nn.Dropout(0.5)
 
        self.fc4 = nn.Linear(512, 10)
 
    def forward(self, x):
        # input 1x32x32, output 32x32x32
        x = self.act1(self.conv1(x))
        #x = self.drop1(x)
        # input 32x32x32, output 32x32x32
        x = self.act2(self.conv2(x))
        # input 32x32x32, output 32x16x16
        x = self.pool2(x)
        # input 32x16x16, output 6272
        x = self.flat(x)
        # input 6272, output 512
        x = self.act3(self.fc3(x))
        #x = self.drop3(x)
        # input 512, output 10
        x = self.fc4(x)
        return x
    
def trainAndSaveModel():
    
    model = CNNModel()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    n_epochs = 250

    lossResults = []
    accuracyResults = []

    for epoch in range(n_epochs):
        total_loss = 0  # Initialize total loss for the epoch
        for i, (images, labels) in enumerate(trainloader):
            # Forward pass 
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()  # Accumulate the loss for each batch
        average_loss = total_loss / len(trainloader)
        lossResults.append(average_loss)
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in testloader:
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        accuracyResults.append(accuracy)

        print(f'Epoch {epoch + 1}: Accuracy: {accuracy:.0f}%, Loss: {average_loss:.4f}')
        
    #TWORZENIE WYKRESU 
    sub1 = plt.subplot(1, 2, 1)
    sub2 = plt.subplot(1, 2, 2)
    
    # Subplot 1: Accuracy
    sub1.plot(accuracyResults, linestyle='solid', color='r')
    #sub1.plot(accuracy_val, linestyle='solid', color='g')
    sub1.set_xticks(list(range(0, len(accuracyResults)+3)))
    sub1.legend(labels=["train", "val"], loc='best')
    sub1.grid(True)
    #sub1.set_xlim(1,len(accuracyResults)+3)
    #sub1.set_xbound(1, len(lossResults)+3)
    #sub1.plot(accuracyResults, 'or')
    #sub1.plot(accuracy_val, 'og')
    sub1.set_xlabel("Epoch")
    sub1.set_ylabel("Accuracy")
    sub1.set_title("Epoch Accuracy")
    
    # Subplot 2: Loss
    sub2.plot(lossResults, linestyle='solid', color='r')
    #sub2.plot(loss_val, linestyle='solid', color='g')
    sub2.set_xticks(list(range(0, len(lossResults)+3)))
    sub2.legend(labels=["Train", "Val"], loc='best')
    sub2.grid(True)
    #sub2.set_xlim(1,len(accuracyResults)+3)
    #sub2.set_xbound(1, len(lossResults)+3)
    #sub2.plot(lossResults, 'or')
    #sub2.plot(loss_val, 'og')
    sub2.set_xlabel("Epoch")
    sub2.set_ylabel("Loss")
    sub2.set_title("Epoch Loss")
    
    #plt.plot(lossResults, label='train_loss')
    #plt.show()
    #plt.plot(accuracyResults,label='train_accuracy')
    plt.show()
    
    torch.save(model.state_dict(), 'my_model.pth')       

def testOneImage():
    # Define the path to the saved model
    model_path = 'my_model.pth'

    # Load the saved model
    model = CNNModel()
    model.load_state_dict(torch.load(model_path))

    # Set the model to evaluation mode
    model.eval()

    # Load an example image for prediction
    #image_path = './images/test/angry/PrivateTest_1290484.jpg'
    image_path = './images/test/happy/PrivateTest_218533.jpg'
    image = Image.open(image_path)
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Make the prediction
    with torch.no_grad():
        model_output = model(input_tensor)

    # Get the predicted class
    _, predicted_class = torch.max(model_output, 1)

    # Print the predicted class
    print(f'Predicted Class: {predicted_class.item()}')
    
def testMultipleImages():

    # Define the path to the saved model
    model_path = 'my_model.pth'
    # Load the saved model
    model = CNNModel()
    model.load_state_dict(torch.load(model_path))
    # Set the model to evaluation mode
    model.eval()
    
    # Calculate the number of images to use for testing (20% of the test dataset)
    test_size = len(testloader.dataset)
    #num_images_to_test = int(0.5 * test_size)

    # Select a random subset of images for testing
    #random.seed(1)  # Set seed for reproducibility
    #indices_to_test = random.sample(range(test_size), num_images_to_test)

    # Initialize variables to keep track of correct predictions
    correct_predictions = 0
    total_images = 0

    # Loop over the selected subset of the test dataset
    with torch.no_grad():
        for i, (images, labels) in enumerate(testloader):
            if i * batch_size >= test_size:
                break  # Stop when the desired number of images have been tested

            # Forward pass
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            # Update counts
            total_images += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    # Calculate and print the accuracy
    accuracy = correct_predictions / total_images
    print(f'Accuracy on {test_size} randomly selected images: {100 * accuracy:.2f}%')
    
    #TWORZENIE WYKRESU
    #plt.plot(lossResults, label='test_loss')
    #plt.plot(accuracyResults, label='test_accuracy')
    #plt.legend()
    #plt.show()
    
def testMultipleImagesPlot(dl):
    # Define the path to the saved model
    model_path = 'my_model.pth'
    # Load the saved model
    model = CNNModel()
    model.load_state_dict(torch.load(model_path))
    # Set the model to evaluation mode
    model.eval()

    with torch.no_grad():
        for images, labels in dl:
            fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12, 12))
            for i in range(3):
                for j in range(3):
                    # Forward pass
                    output = model(images[i * 3 + j].unsqueeze(0))
                    _, predicted = torch.max(output.data, 1)

                    # Display image
                    axes[i, j].imshow(images[i * 3 + j].permute(1, 2, 0).squeeze(), cmap='gray')
                    axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
                    axes[i, j].set_title(f'Predicted: {predicted.item()}')

            plt.show()
            break

trainAndSaveModel()
#testOneImage()
#testMultipleImages()
#testMultipleImagesPlot(testloader)