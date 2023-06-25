
from django.shortcuts import render
import os
from django.shortcuts import render
from django.conf import settings
from torchvision.transforms import transforms
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from torchvision.datasets import ImageFolder

data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define the root directory of your dataset
data_dir = '/Users/sharvaysrivastava/Documents/data'

# Use ImageFolder to load the dataset and infer class labels
dataset = ImageFolder(data_dir, transform=data_transform)

# Get the class-to-index mapping
class_to_idx = dataset.class_to_idx

model = models.densenet121()
model.classifier = nn.Linear(model.classifier.in_features, 8)
model.load_state_dict(torch.load('/Users/sharvaysrivastava/PycharmProjects/TSANationals/HomeTSA/Main.pth'))
model.eval()


def predict_disease(request):
    if request.method == 'POST' and request.FILES['image']:
        # Get the uploaded image from the request
        image = request.FILES['image']

        # Save the image temporarily to a folder
        temp_image_path = os.path.join(settings.MEDIA_ROOT, 'temp_image.jpg')
        with open(temp_image_path, 'wb+') as destination:
            for chunk in image.chunks():
                destination.write(chunk)

        # Process the image using your AI model
        image = Image.open(temp_image_path)
        image = data_transform(image)
        image = torch.unsqueeze(image, 0)  # Add batch dimension

        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)
            predicted_class = predicted.item()
            predicted_label = dataset.classes[predicted_class]

        # Remove the temporary image file
        os.remove(temp_image_path)

        return render(request, 'prediction.html', {'predicted_label': predicted_label})

    return render(request, 'upload.html')


def homepage(request):
    return render(request, 'index.html')




