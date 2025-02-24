import numpy as np
import time
import optuna
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from UCI_model import time_series_to_spectrogram, PyramidAttentionModel  # Updated with DWA

# Load datasets (unchanged)
X_train = np.loadtxt(X_train_path)
y_train = np.loadtxt(y_train_path, dtype=int)
X_test = np.loadtxt(X_test_path)
y_test = np.loadtxt(y_test_path, dtype=int)

# Convert time-series to spectrograms (unchanged)
# ... [same conversion code] ...

# Set up device and model (modified)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on {device}")

# Hyperparameter optimization objective (modified)
def objective(trial):
    global run_number
    
    # Hyperparameters (unchanged)
    lr = trial.suggest_loguniform("lr", 1e-4, 1e-2)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-2)
    
    # Model with Dual Wavelet Attention
    model = PyramidAttentionModel(
        input_channels=1, 
        n_classes=6,
        num_splits=2,
        use_wavelet=True  # Add wavelet attention flag
    ).to(device)
    
    # Training loop (modified)
    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            # Forward pass with wavelet attention
            gap_outputs, wavelet_outputs = model(inputs)
            loss = criterion(gap_outputs, labels) + 2.5 * criterion(wavelet_outputs, labels)
            # ... [rest of training loop] ...

    # Evaluation (modified)
    correct_wavelet = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            _, wavelet_outputs = model(inputs)
            _, predicted_wavelet = torch.max(wavelet_outputs, 1)
            correct_wavelet += (predicted_wavelet == labels).sum().item()
    
    return 100 * correct_wavelet / total

# Post-NAS training (modified)
model = PyramidAttentionModel(
    input_channels=1,
    n_classes=6,
    num_splits=12,
    use_wavelet=True  # Enable wavelet attention
).to(device)

# Training loop with wavelet metrics (modified)
for epoch in range(num_epochs):
    # ... [same setup] ...
    gap_outputs, wavelet_outputs = model(inputs)
    
    # Wavelet-based loss calculation
    loss_wavelet = criterion(wavelet_outputs, labels)
    total_loss = loss_gap + 2.5 * loss_wavelet
    
    # Track wavelet accuracy
    _, predicted_wavelet = torch.max(wavelet_outputs, 1)
    correct_wavelet += (predicted_wavelet == labels).sum().item()

# Evaluation (modified)
all_wavelet_predictions = []
with torch.no_grad():
    for inputs, labels in test_loader:
        _, wavelet_outputs = model(inputs)
        _, predicted_wavelet = torch.max(wavelet_outputs, 1)
        correct_wavelet += (predicted_wavelet == labels).sum().item()
        all_wavelet_predictions.extend(predicted_wavelet.cpu().numpy())

# Reporting (modified)
f1_wavelet = f1_score(all_labels, all_wavelet_predictions, average='weighted')
print(f"Test Accuracy (Wavelet): {accuracy_wavelet:.2f}%")
print(f"Test F1-Score (Wavelet): {f1_wavelet:.2f}")