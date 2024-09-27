import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from PIL import Image

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Directories
data_dir = 'dataset'  # Ensure this directory has subfolders for each class

# Define transforms
train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Load dataset
full_dataset = datasets.ImageFolder(root=os.path.join(data_dir), transform=train_transform)

# Get the number of classes
num_classes = len(full_dataset.classes)

# Split dataset into training and validation
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Apply test_transform to validation dataset
val_dataset.dataset.transform = test_transform

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

# Load pre-trained model and modify final layer
model = models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False  # Freeze early layers

# Replace the final fully connected layer
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, num_classes)
)

model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Training function with validation
def train(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs):
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total

        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for val_images, val_labels in val_loader:
                val_images, val_labels = val_images.to(device), val_labels.to(device)
                
                val_outputs = model(val_images)
                val_loss = criterion(val_outputs, val_labels)
                
                val_running_loss += val_loss.item()
                _, val_predicted = torch.max(val_outputs.data, 1)
                val_total += val_labels.size(0)
                val_correct += (val_predicted == val_labels).sum().item()
        
        val_loss = val_running_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total

        # Step the scheduler
        scheduler.step()

        print(f'Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.2f}%')
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%\n')

# Prediction function
def predict_image(image_path, model, transform, device):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    image = image.to(device)
    model.eval()
    
    with torch.no_grad():
        outputs = model(image)
        probabilities = nn.functional.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)
    
    class_label = full_dataset.classes[predicted_class.item()]
    confidence = confidence.item() * 100  # Convert to percentage

    return class_label, confidence

# Train the model
num_epochs = 25  # Start with fewer epochs to prevent overfitting
train(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs)

# Save the model
torch.save(model.state_dict(), 'genshin_detector_resnet18_model.pth')
print("Training complete and model saved.")

# Test images
test_images = [
    'Hu_Tao_Test.jpg',
    'Kokomi_Test.jpg',
    'Ayaka_Test.jpg',
    'Raiden_Test.jpg',
    'Ryu_Test.jpg',
    'X_Test.jpg',
    'Terry_Bogard_Test.jpg'
]

for image_path in test_images:
    predicted_class, confidence = predict_image(image_path, model, test_transform, device)
    print(f'The predicted class for the image {image_path} is: {predicted_class} (Confidence: {confidence:.2f}%)')
