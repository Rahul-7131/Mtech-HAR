import numpy as np
import torch
import optuna
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from torch.utils.data import DataLoader, TensorDataset
from UCI_model import time_series_to_spectrogram, PyramidAttentionModel  # Use the updated correction.py

# Load datasets
X_train_path = "/home/rahul/ML-Mtech/HAR/UCI HAR Dataset/UCI HAR Dataset/train/X_train.txt"
y_train_path = "/home/rahul/ML-Mtech/HAR/UCI HAR Dataset/UCI HAR Dataset/train/y_train.txt"
X_test_path = "/home/rahul/ML-Mtech/HAR/UCI HAR Dataset/UCI HAR Dataset/test/X_test.txt"
y_test_path = "/home/rahul/ML-Mtech/HAR/UCI HAR Dataset/UCI HAR Dataset/test/y_test.txt"

X_train = np.loadtxt(X_train_path)
y_train = np.loadtxt(y_train_path, dtype=int)
X_test = np.loadtxt(X_test_path)
y_test = np.loadtxt(y_test_path, dtype=int)

subset_ratio = 0.001
subset_size = int(len(X_train) * subset_ratio)
X_train = X_train[:subset_size]
y_train = y_train[:subset_size]

# Convert time-series to spectrograms
def convert_to_spectrograms(X):
    spectrograms = [time_series_to_spectrogram(ts) for ts in X]
    spectrograms = np.expand_dims(np.array(spectrograms), axis=1)  # Add channel dimension
    spectrograms = np.mean(spectrograms, axis=1, keepdims=True)  # Average across 5 channels
    # Add padding here to increase width dimension
    spectrograms = np.pad(spectrograms, ((0, 0), (0, 0), (0, 0), (0, 3)), mode='constant')  # Padding width by 3 to make (1, 129, 8)
    return torch.tensor(spectrograms, dtype=torch.float32)

train_spectrograms = convert_to_spectrograms(X_train)
test_spectrograms = convert_to_spectrograms(X_test)
print(f"Train spectrograms shape: {train_spectrograms.shape}")

# Adjust labels
y_train_zero_indexed = torch.tensor(y_train - 1, dtype=torch.long)
y_test_zero_indexed = torch.tensor(y_test - 1, dtype=torch.long)

# Prepare DataLoader
train_dataset = TensorDataset(train_spectrograms, y_train_zero_indexed)
test_dataset = TensorDataset(test_spectrograms, y_test_zero_indexed)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# Set up device and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on {device}")

# Global variable to keep track of the run number
run_number = 1

def objective(trial):
    global run_number  # Declare the global variable
    
    # Hyperparameter search space
    lr = trial.suggest_loguniform("lr", 1e-4, 1e-2)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-2)
    
    # Update DataLoader with trial batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Define model, loss, optimizer
    model = PyramidAttentionModel(input_channels=1, n_classes=6).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Training loop
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        print(f"running epoch number:{epoch+1}")
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            gap_outputs, dct_outputs = model(inputs)
            loss_gap = criterion(gap_outputs, labels)
            loss_dct = criterion(dct_outputs, labels)
            loss = loss_gap + 2.5 * loss_dct
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
    # Evaluation
    model.eval()
    
    print(f"run number: {run_number}")
    correct_gap, total = 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            gap_outputs, _ = model(inputs)
            _, predicted_gap = torch.max(gap_outputs, 1)
            total += labels.size(0)
            correct_gap += (predicted_gap == labels).sum().item()
    
    # Increment the run number for the next trial
    run_number += 1
    
    # Return accuracy for optimization
    accuracy = 100 * correct_gap / total
    print("finished Nas search")
    return accuracy

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)

# Print the best hyperparameters
print("Best hyperparameters: ", study.best_params)
best_params = study.best_params

# Use best_params['lr'], best_params['batch_size'], and best_params['weight_decay']
train_loader = DataLoader(train_dataset, batch_size=best_params["batch_size"], shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=best_params["batch_size"], shuffle=False)

model = PyramidAttentionModel(input_channels=1, n_classes=6).to(device)
optimizer = optim.Adam(model.parameters(), lr=best_params["lr"], weight_decay=best_params["weight_decay"])

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)  # Reduce LR every 5 epochs

# Training loop
num_epochs = 10
print("training loop started")
for epoch in range(num_epochs):
    scheduler.step()
    model.train()
    print(f"Epoch {epoch+1}/{num_epochs} training started...")  
    running_loss_gap = 0.0
    running_loss_dct = 0.0
    epoch_dct_features = []
    
    for i, (inputs, labels) in enumerate(train_loader):
        print(f"Processing batch {i+1}/{len(train_loader)}...")
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()  
        print("Forward pass starting...")
        gap_outputs, dct_outputs = model(inputs)  # Forward pass
        print("Forward pass completed.")
        epoch_dct_features.append(dct_outputs.detach().cpu().numpy())
        # Compute losses
        loss_gap = criterion(gap_outputs, labels)
        loss_dct = criterion(dct_outputs, labels)
        print("Backward pass starting...")
        total_loss = loss_gap + 2.5*loss_dct  # Combined loss
        
        total_loss.backward()  # Backward pass
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        print("Backward pass completed.")
        optimizer.step()  # Optimization step
        
        running_loss_gap += loss_gap.item()
        running_loss_dct += loss_dct.item()
        
        if (i + 1) % 5 == 0:  # Print every 5 batches
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], "
            f"Loss GAP: {running_loss_gap / 10:.4f}, Loss DCT: {running_loss_dct / 10:.4f}")
            running_loss_gap = 0.0
            running_loss_dct = 0.0

print("Training complete!")

# Evaluation loop
model.eval()
correct_gap = 0
correct_dct = 0
total = 0

with torch.no_grad():  # Disable gradient calculations
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        gap_outputs, dct_outputs = model(inputs)  # Forward pass
        
        _, predicted_gap = torch.max(gap_outputs, 1)
        _, predicted_dct = torch.max(dct_outputs, 1)
        
        total += labels.size(0)
        correct_gap += (predicted_gap == labels).sum().item()
        correct_dct += (predicted_dct == labels).sum().item()

# Accuracy calculations
accuracy_gap = 100 * correct_gap / total
accuracy_dct = 100 * correct_dct / total

print(f"Test Accuracy (GAP): {accuracy_gap:.2f}%")
print(f"Test Accuracy (DCT): {accuracy_dct:.2f}%")

