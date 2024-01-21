import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from torchsummary import summary
import matplotlib.pyplot as plt
from PIL import Image
import random
import os
from sklearn.metrics import confusion_matrix
import seaborn as sns

if __name__ == '__main__':
    class EmotionCNN(nn.Module):
        def __init__(self):
            super(EmotionCNN, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(32)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(64)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.bn3 = nn.BatchNorm2d(128)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(128 * 6 * 6, 512)
            self.fc2 = nn.Linear(512, 2)  # 2 klasy
            self.dropout = nn.Dropout(0.5)

        def forward(self, x):
            x = self.pool(F.relu(self.bn1(self.conv1(x))))
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            x = self.pool(F.relu(self.bn3(self.conv3(x))))
            x = x.view(-1, 128 * 6 * 6)
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x

    data_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor()
    ])


    train_dataset = datasets.ImageFolder(root="./images/train", transform=data_transform)
    test_dataset = datasets.ImageFolder(root="./images/test", transform=data_transform)

    batchSize = 64

    train_loader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batchSize, shuffle=False, num_workers=4)

    # Inicjalizacja modelu, funkcji straty i optymalizatora
    model = EmotionCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    #print(model)
    #summary(model, (1, 48, 48), batchSize)

    
    # Funkcja do wypisywania dokładności (accuracy) i straty (loss) dla każdej epoki
    def print_accuracy_and_loss_and_save_model():
        #model.eval()
        num_epochs = 200

        # Trening modelu

        lossResults = []
        accuracyResults = []

        for epoch in range(1, num_epochs + 1):
            model.train()
            running_loss = 0.0
            for images, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            average_loss = running_loss / len(train_loader)
            lossResults.append(average_loss)
            correct = 0
            total = 0
            
            with torch.no_grad():
                for images, labels in train_loader:
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            accuracy = correct / total
            accuracyResults.append(accuracy)
            print(f"Epoch {epoch}/{num_epochs}, Loss: {average_loss:.4f}, Training Accuracy: {accuracy * 100:.2f}%")
                    

        torch.save(model.state_dict(), 'my_model.pth') #ZAPISZ MODEL W PLIKU
        
            #TWORZENIE WYKRESU 
        sub1 = plt.subplot(1, 2, 1)
        sub2 = plt.subplot(1, 2, 2)
        
        # Subplot 1: Accuracy
        sub1.plot(range(1, epoch + 1), accuracyResults, linestyle='solid', color='r')
        #sub1.plot(accuracy_val, linestyle='solid', color='g')
        sub1.set_xticks(list(range(0, num_epochs+1, 25)))
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
        sub2.plot(range(1, epoch + 1), lossResults, linestyle='solid', color='r')
        #sub2.plot(loss_val, linestyle='solid', color='g')
        sub2.set_xticks(list(range(0, num_epochs+1, 25)))
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
        
        
    def testOneImage():
        model_path = 'my_model.pth'
        model = EmotionCNN()
        model.load_state_dict(torch.load(model_path))
        model.eval()
        image_path = './images/test/neutral/PublicTest_92867331.jpg'
        image = Image.open(image_path)
        input_tensor = data_transform(image).unsqueeze(0)  # Add batch dimension

        # Make the prediction
        with torch.no_grad():
            model_output = model(input_tensor)

        # Get the predicted class
        _, predicted_class = torch.max(model_output, 1)

        # Print the predicted class
        print(f'Predicted Class: {predicted_class.item()}')
            
    def testMultipleImages():
        model_path = 'my_model.pth'
        model = EmotionCNN()
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        test_size = len(test_loader.dataset)
        #num_images_to_test = int(0.5 * test_size)
        #random.seed(1)  # Set seed for reproducibility
        #indices_to_test = random.sample(range(test_size), num_images_to_test)

        correct_predictions = 0
        total_images = 0

        # Loop over the selected subset of the test dataset
        with torch.no_grad():
            for i, (images, labels) in enumerate(test_loader):
                if i * batchSize >= test_size:
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
        
        model_path = 'my_model.pth'
        model = EmotionCNN()
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        with torch.no_grad():
            size = len(dl.dataset)  # Pobierz pełną liczbę obrazów w zbiorze testowym

            fig, axes = plt.subplots(nrows=6, ncols=6, figsize=(10, 10))
            for i in range(6):
                for j in range(6):
                    # Losowy indeks z pełnej puli danych testowych
                    randomImageIndex = random.randint(0, size-1)
                    images, labels = dl.dataset[randomImageIndex]

                    # Forward pass
                    output = model(images.unsqueeze(0))
                    _, predicted = torch.max(output.data, 1)

                    # Wyświetl obraz
                    axes[i, j].imshow(images.permute(1, 2, 0).squeeze(), cmap='gray')
                    axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
                    axes[i, j].set_title(f'Predicted: {predicted.item()}')

            plt.show()
     
    def test_and_get_confusion_matrix():
        model_path = 'my_model.pth'
        model = EmotionCNN()
        model.load_state_dict(torch.load(model_path))
        model.eval()

        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        cm = confusion_matrix(all_labels, all_predictions)
        return cm
     
    def draw_plot_confusion_matrix(cm, classes):
        plt.figure(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()
    
    #print_accuracy_and_loss_and_save_model()
    #testOneImage()
    #testMultipleImages()
    #testMultipleImagesPlot(test_loader)
    confusion_matrix_result = test_and_get_confusion_matrix()
    draw_plot_confusion_matrix(confusion_matrix_result, classes=['Neutral', 'Surprised'])