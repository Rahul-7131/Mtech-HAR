import numpy as np
import time
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from skopt import gp_minimize
from skopt.space import Real, Categorical
from skopt.utils import use_named_args
from model import time_series_to_spectrogram, PyramidAttentionModel  

# Load datasets
X_train_path = "/home/rahul/ML-Mtech/HAR/UCI HAR Dataset/UCI HAR Dataset/train/X_train.txt"
y_train_path = "/home/rahul/ML-Mtech/HAR/UCI HAR Dataset/UCI HAR Dataset/train/y_train.txt"
X_test_path = "/home/rahul/ML-Mtech/HAR/UCI HAR Dataset/UCI HAR Dataset/test/X_test.txt"
y_test_path = "/home/rahul/ML-Mtech/HAR/UCI HAR Dataset/UCI HAR Dataset/test/y_test.txt"

X_train = np.loadtxt(X_train_path)
y_train = np.loadtxt(y_train_path, dtype=int)
X_test = np.loadtxt(X_test_path)
y_test = np.loadtxt(y_test_path, dtype=int)

# Use only a percentage subset of data
percentage = 0.1
train_subset_size = int(len(X_train) * percentage)
test_subset_size = int(len(X_test) * percentage)

X_train = X_train[:train_subset_size]
y_train = y_train[:train_subset_size]
X_test = X_test[:test_subset_size]
y_test = y_test[:test_subset_size]

# Convert time-series to spectrograms
def convert_to_spectrograms(X):
    spectrograms = [time_series_to_spectrogram(ts) for ts in X]
    spectrograms = np.expand_dims(np.array(spectrograms), axis=1)
    spectrograms = np.mean(spectrograms, axis=1, keepdims=True)
    spectrograms = np.pad(spectrograms, ((0, 0), (0, 0), (0, 0), (0, 3)), mode='constant')
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

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Set up device and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on {device}")

# Search space
search_space = [
    Real(1e-4, 1e-2, prior="log-uniform", name="lr"),
    Categorical([8, 16, 32, 64], name="batch_size"),
    Real(1e-6, 1e-2, prior="log-uniform", name="weight_decay"),
    Categorical([2, 4, 6, 8], name="num_splits"),
    Categorical([16, 32, 64, 128], name="k_channels"),
]

# Global variable to track trials
run_number = 0

@use_named_args(search_space)
def objective(lr, batch_size, weight_decay, num_splits, k_channels):
    global run_number

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model = PyramidAttentionModel(input_channels=1, n_classes=6, num_splits=num_splits, k_channels=k_channels).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    num_epochs = 1
    for epoch in range(num_epochs):
        print('working on epoch:', epoch,'Trial:', run_number)
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            gap_outputs = model(inputs)
            loss = criterion(gap_outputs, labels)
            loss.backward()
            optimizer.step()

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            gap_outputs = model(inputs)
            _, predicted = torch.max(gap_outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    run_number += 1

    return -accuracy

# Run Bayesian Optimization
res = gp_minimize(
    objective, 
    search_space, 
    n_calls=5,
    n_random_starts=5,
    random_state=42
)

# Print best hyperparameters
best_params = {dim.name: value for dim, value in zip(search_space, res.x)}
print("Bayesian Optimization completed")
print("Best hyperparameters:", best_params)

best_split = best_params["num_splits"]
best_k_channels = best_params["k_channels"]

print(f"Training model with best split: {best_split} and best k_channels: {best_k_channels}")

# Initialize and train the model using the best hyperparameters
model = PyramidAttentionModel(input_channels=1, n_classes=6, num_splits=best_split, k_channels=best_k_channels).to(device)
optimizer = optim.Adam(model.parameters(), lr=best_params["lr"], weight_decay=best_params["weight_decay"])
criterion = nn.CrossEntropyLoss()

num_epochs = 1
print("Training loop started")
for epoch in range(num_epochs):
    model.train()
    print(f"Epoch {epoch+1}/{num_epochs} training started...")  

    running_loss_gap = 0.0
    correct_gap = 0
    total_samples = 0

    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()  

        gap_outputs = model(inputs)
        loss_gap = criterion(gap_outputs, labels)
        total_loss = loss_gap 
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss_gap += loss_gap.item()
        _, predicted_gap = torch.max(gap_outputs, 1)
        correct_gap += (predicted_gap == labels).sum().item()
        total_samples += labels.size(0)

        if (i + 1) % 50 == 0:
            running_loss_gap = 0.0

    accuracy_gap = 100 * correct_gap / total_samples
    print(f"Epoch [{epoch+1}/{num_epochs}] completed. "
          f"Loss GAP: {running_loss_gap / len(train_loader):.4f}"
          f"Accuracy GAP: {accuracy_gap:.2f}%")

print("Training complete!")

# Evaluation loop
model.eval()
correct_gap = 0
total = 0

all_labels = []
all_gap_predictions = []
start_time = time.time()
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        gap_outputs = model(inputs)
        _, predicted_gap = torch.max(gap_outputs, 1)
        
        total += labels.size(0)
        correct_gap += (predicted_gap == labels).sum().item()
        
        all_labels.extend(labels.cpu().numpy())
        all_gap_predictions.extend(predicted_gap.cpu().numpy())

end_time = time.time()
accuracy_gap = 100 * correct_gap / total
f1_gap = f1_score(all_labels, all_gap_predictions, average='weighted')
inf = end_time - start_time
print(f"Split number = {model.num_splits}")
print(f"Test Accuracy (GAP): {accuracy_gap:.2f}%")
print(f"Test F1-Score (GAP): {f1_gap:.2f}")
print(f"Inference time: {inf:.2f} seconds")
