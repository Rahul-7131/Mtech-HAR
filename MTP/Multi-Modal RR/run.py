import torchvision
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchsummary import summary
from CIFAR import PyramidCNNModel

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Define model, loss function, and optimizer
model = PyramidCNNModel(input_channels=3, n_classes=10, num_splits=2).to(device)

summary(model, input_size=(3, 32, 32))
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total number of trainable parameters: {total_params}")
'''
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# Training loop
epochs = 1
for epoch in range(epochs):
    print(f"training started ")
    model.train()
    running_loss = 0.0
    print(f"Epoch {epoch+1}")
    for i, (images, labels) in enumerate(trainloader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        print(f"\nBatch {i+1}, Input shape: {images.shape}")
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'\nEpoch {epoch+1}, Loss: {running_loss/len(trainloader)}')

# Testing loop
start = time.time()
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

end = time.time()
print(f'Accuracy on CIFAR-10 test set: {100 * correct / total:.2f}%')
#print(f"Trained for {epochs} Epochs")
print(f"Inference time:{(end-start):.2f} sec")
'''