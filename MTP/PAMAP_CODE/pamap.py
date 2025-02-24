import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pamap_model import time_series_to_spectrogram, PyramidAttentionModel

# Define constants
DATA_PATHS = [
    f"/home/rahul/ML-Mtech/HAR/pamap2/PAMAP2_Dataset/Protocol/subject10{i}.dat" for i in range(1, 10)
]
ACTIVITY_ID_MAP = {
    0: "transient", 1: "lying", 2: "sitting", 3: "standing", 4: "walking",
    5: "running", 6: "cycling", 7: "Nordic_walking", 9: "watching_TV",
    10: "computer_work", 11: "car driving", 12: "ascending_stairs", 13: "descending_stairs",
    16: "vacuum_cleaning", 17: "ironing", 18: "folding_laundry", 19: "house_cleaning",
    20: "playing_soccer", 24: "rope_jumping"
}
COLUMNS = [
    "timestamp", "activityID", "heartrate"
] + [
    f"{part}{sensor}" for part in ["hand", "chest", "ankle"] for sensor in [
        "Temperature", "Acc16_1", "Acc16_2", "Acc16_3", "Acc6_1", "Acc6_2", "Acc6_3",
        "Gyro1", "Gyro2", "Gyro3", "Magne1", "Magne2", "Magne3",
        "Orientation1", "Orientation2", "Orientation3", "Orientation4"
    ]
]

# Load datasets
def load_datasets(paths):
    combined_data = []
    for path in paths:
        data = pd.read_csv(path, sep='\s+', header=None)
        data.columns = COLUMNS
        data["subject_id"] = int(path.split("subject10")[1][0])
        combined_data.append(data)
    return pd.concat(combined_data, ignore_index=True)

# Clean data
def clean_data(df):
    orientation_cols = [col for col in df.columns if "Orientation" in col]
    df = df.drop(columns=orientation_cols + ["timestamp", "subject_id"], errors="ignore")
    df = df[df.activityID != 0]  # Remove transient activities
    df = df.apply(pd.to_numeric, errors="coerce").interpolate()  # Handle NaNs
    return df

# Scale data
def scale_data(train, test):
    scaler = RobustScaler()
    train_scaled = scaler.fit_transform(train)
    test_scaled = scaler.transform(test)
    return train_scaled, test_scaled

# Convert time-series to spectrograms
def convert_to_spectrograms(X):
    def preprocess_time_series(ts):
        ts = np.nan_to_num(ts, nan=0.0, posinf=0.0, neginf=0.0)
        return ts

    spectrograms = [time_series_to_spectrogram(preprocess_time_series(ts)) for ts in X]
    spectrograms = np.expand_dims(spectrograms, axis=1)  # Add channel dimension
    spectrograms = np.pad(spectrograms, ((0, 0), (0, 0), (0, 0), (0, 3)), mode='constant')  # Pad width
    return torch.tensor(spectrograms, dtype=torch.float32)


# Main processing
raw_data = load_datasets(DATA_PATHS)
data = clean_data(raw_data)
data.reset_index(drop=True, inplace=True)

# Split dataset
train_data, test_data = train_test_split(data, test_size=0.2, random_state=1)
print("split done")
# Separate features and labels
X_train, y_train = train_data.drop("activityID", axis=1).values, train_data["activityID"].values
X_test, y_test = test_data.drop("activityID", axis=1).values, test_data["activityID"].values

subset_ratio = 0.05
subset_size = int(len(X_train) * subset_ratio)
X_train = X_train[:subset_size]
y_train = y_train[:subset_size]

# Scale features
X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

# Convert to spectrograms
print("conversion started")
train_spectrograms = convert_to_spectrograms(X_train_scaled)
test_spectrograms = convert_to_spectrograms(X_test_scaled)
'''
plt.imshow(train_spectrograms[0][0].numpy())
plt.show()
'''
print("conversion complete")
# Adjust labels
y_train_tensor = torch.tensor(y_train - 1, dtype=torch.long)
y_test_tensor = torch.tensor(y_test - 1, dtype=torch.long)

# Prepare DataLoaders
train_dataset = TensorDataset(train_spectrograms, y_train_tensor)
test_dataset = TensorDataset(test_spectrograms, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# Map activity IDs to sequential indices
valid_ids = sorted(ACTIVITY_ID_MAP.keys())
id_to_label = {id_: idx for idx, id_ in enumerate(valid_ids)}
data["activityID"] = data["activityID"].map(id_to_label)
n_classes = len(valid_ids)
# Model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PyramidAttentionModel(input_channels=1, n_classes=n_classes).to(device)
print("Model n_classes:", model.n_classes)
# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

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
