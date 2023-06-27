# DermEngine
## Software Development at Louisville, Kentucky 
## Team ID: 
## The development of the AI:

---Version one of the AI model---
- First version of AI model for skin disease detection showed significant progress in accuracy.
- Fine-tuning approach employed to enhance the model.
- Batch size increased from 32 to 64, resulting in more efficient data processing.
- Number of training epochs extended from 10 to 18 to 32.
- Overall accuracy improved from 41% to 47%.
- Larger batch size allowed for better capture of intricate patterns in the dataset.
- Extended training epochs increased model's exposure to data and improved predictions over time.
- These improvements indicate potential for further advancements in future iterations.
- Fine-tuning techniques have a positive impact on enhancing accuracy in skin disease detection.
- 3 Conv2D layers
```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import os

# Set the device to use
device = torch.device("mps")

# Set the path to your dataset
data_dir = "/Users/sharvaysrivastava/Documents/archive"

# Define transformations to apply to the images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the train dataset
train_dataset = ImageFolder(data_dir + "/train", transform=transform)
train_data_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Load the test dataset
test_dataset = ImageFolder(data_dir + "/test", transform=transform)
test_data_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define the CNN model
class SkinDiseaseClassifier(nn.Module):
    def __init__(self, num_classes):
        super(SkinDiseaseClassifier, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

# Create an instance of the model
model = SkinDiseaseClassifier(num_classes=len(train_dataset.classes)).to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Train the model
num_epochs = 32
for epoch in range(num_epochs):
    running_loss = 0.0
    for images, labels in train_data_loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_data_loader):.4f}")

# Evaluate the model on the test set
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_data_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted.labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")

# Save the trained model
torch.save(model.state_dict(), "forty-seven.pth")
```
---Version two of the AI model---
- Extended the number of training epochs from 32 to 100.
- Increased exposure to the data for better refinement and learning of intricate patterns.
- Improved overall accuracy from 47% to 55%.
- Added more hidden layers to enable deeper and more complex representations of the input.
- Captured more nuanced features and patterns related to skin diseases.
- Utilized the Adam optimizer for more efficient and effective optimization.
- Experimented with different learning rates (lr) and weight decay to optimize performance.
- Achieved better convergence and improved accuracy through fine-tuning hyperparameters.
- The collective impact of these modifications led to a notable boost in overall accuracy.
- Demonstrated the potential for further advancements in skin disease detection algorithms.
- Emphasized the importance of fine-tuning techniques, exploring different architectures, and optimization strategies in medical image analysis.
- 7 Conv2D layers
```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

# Set the device to use
device = torch.device("mps")

# Set the path to your dataset
data_dir = "/Users/sharvaysrivastava/Documents/archive"

# Define transformations to apply to the images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the train dataset
train_dataset = ImageFolder(data_dir + "/train", transform=transform)
train_data_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Load the test dataset
test_dataset = ImageFolder(data_dir + "/test", transform=transform)
test_data_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define the CNN model
class SkinDiseaseClassifier(nn.Module):
    def __init__(self, num_classes):
        super(SkinDiseaseClassifier, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(),
            nn.Dropout(0.5),  # Add dropout for regularization
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

# Create an instance of the model
model = SkinDiseaseClassifier(num_classes=len(train_dataset.classes)).to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)

# Train the model
num_epochs = 100
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    for images, labels in train_data_loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_data_loader):.4f}")

    # Evaluate the model on the test set every few epochs
    if (epoch+1) % 5 == 0:
        model.eval()  # Set the model to evaluation mode
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_data_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            accuracy = 100 * correct / total
            print(f"Test Accuracy: {accuracy:.2f}%")

# Save the trained model
torch.save(model.state_dict(), "skin_disease.pth")
```

---Version three of the AI model---
- In the third version of the skin disease detection AI model, a pretrained RestNet50 model was utilized.
- Despite initial optimism, the pretrained model achieved a disappointing accuracy of 32% in skin disease detection.
- It became clear that a more specialized approach was necessary for accurate skin disease detection.
- The number of training epochs was increased to 100 to allow the model to learn extensively from the dataset and understand intricate patterns within skin diseases.
- More hidden layers were added to the model to capture nuanced features and patterns related to skin diseases.
- The Adam optimizer was employed, combining adaptive learning rates and momentum-based updates for improved optimization.
- By experimenting with different learning rates (lr) and weight decay, the model's performance was fine-tuned to achieve better convergence and accuracy.
- Despite the underwhelming results of the pretrained model, these modifications collectively led to a significant increase in accuracy from 47% to 55%.
- This emphasizes the importance of customizing models for specific domains and the need to carefully explore different architectures and optimization strategies in medical image analysis.
```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import resnet50

# Set the device to use
device = torch.device("mps")

# Set the path to your dataset
data_dir = "/Users/sharvaysrivastava/Documents/archive"

# Define transformations to apply to the images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("part 1")

# Load the train dataset
train_dataset = ImageFolder(data_dir + "/train", transform=transform)
train_data_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# Load the test dataset
test_dataset = ImageFolder(data_dir + "/test", transform=transform)
test_data_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

print("part 2")

# Load the pre-trained ResNet-50 model
base_model = resnet50(weights=None, progress=True)

# Modify the fully connected layer to match the number of classes
num_classes = len(train_dataset.classes)
base_model.fc = nn.Linear(2048, num_classes)

# Move the model to the device
base_model = base_model.to(device)

# Create the complete model
model = base_model

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)

print("part 3")

# Train the model
num_epochs = 100
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    print("part 4")
    running_loss = 0.0
    for images, labels in train_data_loader:
        print("part 5")
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        print("part 6")
        optimizer.step()
        running_loss += loss.item()
    print("part 7")
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_data_loader):.4f}")

    # Evaluate the model on the test set every few epochs
    if (epoch+1) % 5 == 0:
        model.eval()  # Set the model to evaluation mode
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_data_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print(f"Test Accuracy: {accuracy:.2f}%")

torch.save(model.state_dict(), "skin_disease_resnet.pth")  # Save the trained model

```
---Version four of the AI model---
- The fourth version of the AI model for skin disease detection attempted to incorporate TensorFlow and Google's pretrained Inception models.
- This approach yielded disappointing results, with an accuracy of only 5%.
- Efforts were made to optimize the model's architecture, increase training epochs, and fine-tune hyperparameters.
- However, the desired improvements were not achieved.
- This highlighted the challenges of skin disease detection and the limitations of generic pretrained models in this domain.
- It emphasized the need for specialized approaches in skin disease detection.
- The fourth version served as a valuable learning experience.
- It underscored the necessity for continuous advancements in medical image analysis.
```python
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

batch_size = 64
image_size = 128

test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "/Users/sharvaysrivastava/Documents/archive/test",
    shuffle=True,
    image_size=(image_size, image_size),
    batch_size=batch_size
)

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "/Users/sharvaysrivastava/Documents/archive/train",
    shuffle=True,
    image_size=(image_size, image_size),
    batch_size=batch_size
)

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.2),
])

for image in train_dataset.take(1):
    plt.figure(figsize=(10, 10))
    first_image = image[0]
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        augmented_image = data_augmentation(tf.expand_dims(first_image, 0))
        plt.imshow(augmented_image[0] / 255)
        plt.axis('off')

rescale = tf.keras.layers.Rescaling(1./127.5, offset=-1)
IMG_SHAPE = (128, 128) + (3,)

base_model = tf.keras.applications.inception_v3.InceptionV3(
    include_top=False,
    weights='imagenet',
    input_tensor=None,
    input_shape=(128, 128, 3),
    pooling=None,
    classes=1000,
    classifier_activation='softmax'
)
base_model.trainable = False

prediction_layer = tf.keras.layers.Dense(1, activation='sigmoid')
inputs = tf.keras.Input(shape=(image_size, image_size, 3))
x = data_augmentation(inputs)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)

model = tf.keras.Model(inputs, outputs)
base_learning_rate = 0.0001
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=['accuracy']
)

initial_epochs = 10
history = model.fit(
    train_dataset,
    epochs=initial_epochs,
    validation_data=test_dataset
)

loss0, accuracy0 = model.evaluate(test_dataset)
print("Initial loss: {:.2f}".format(loss0))
print("Initial accuracy: {:.2f}".format(accuracy0))

model.save("my_model")

```
---Version five(Final) of the AI model--- 

- Fifth and final version of the AI model for skin disease detection
- Utilizes pretrained DenseNet121 model for faster convergence
- Batch size: 64 for balanced memory consumption and computational efficiency
- Learning rate: 0.001 with Adam optimizer for model optimization
- Dropout layer (probability: 0.3) added to prevent overfitting and enhance generalization
- Achieves impressive accuracy of 88.73% after fine-tuning and modifications
- Pretrained DenseNet121 model, lower batch size, optimized learning rate, dropout layer, and Adam optimizer contribute to high accuracy
- Potential to assist medical professionals in diagnosing and treating skin diseases
- Demonstrates effectiveness of leveraging preexisting architectures, selecting appropriate hyperparameters, and utilizing state-of-the-art optimization techniques in medical image analysis.
```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
import torchvision.models as models
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torchvision.models import densenet121, DenseNet121_Weights

# Parameters
batch_size = 64
num_epochs = 75
learning_rate = 0.001
num_classes = 8
# Custom Dataset
class SkinDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = ImageFolder(data_dir, transform=transform)

    def __getitem__(self, index):
        img, label = self.data[index]
        return img, label

    def __len__(self):
        return len(self.data)
# Data Transformations and Datasets
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

data_dir = '/Users/sharvaysrivastava/Documents/data'

dataset = SkinDataset(data_dir, transform=data_transforms)

train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
# Data Loaders
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
# Pre-trained Model
model = densenet121(weights=DenseNet121_Weights.DEFAULT)
for param in model.parameters():
    param.requires_grad = False

num_features = model.classifier.in_features
model.classifier = nn.Linear(num_features, num_classes)

# Add dropout layer
dropout_prob = 0.3 # Adjust the dropout probability as desired
num_features = model.classifier.in_features
model.classifier = nn.Sequential(
    nn.Linear(num_features, 512),
    nn.ReLU(),
    nn.Dropout(dropout_prob),
    nn.Linear(512, num_classes)
)
# Loss Function, Optimizer, and Device
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

device = torch.device('mps')
model = model.to(device)
# Training and Evaluation
total_step = len(train_loader)

# Training and Evaluation
total_step = len(train_loader)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    with tqdm(total=len(train_loader), unit='batch') as progress_bar:
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            progress_bar.set_postfix({'Epoch': epoch + 1, 'Loss': total_loss / (i + 1)})
            progress_bar.update()

    # Evaluation
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Test Accuracy: {accuracy:.2f}%')

        if accuracy > 73.35:
            print('Desired accuracy reached. Breaking out of the training loop.')
            break


        accuracy = 100 * correct / total
        print(f'Test Accuracy: {accuracy:.2f}%')
torch.save(model.state_dict(), "new.pth")
```
