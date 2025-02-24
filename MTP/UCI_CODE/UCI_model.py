import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
from dct import multi_frequency_compression, channel_attention  

# Function to convert time-series data to spectrogram
def time_series_to_spectrogram(time_series, sample_rate=50, n_fft=256, hop_length=128):
    time_series = np.array(time_series, dtype=np.float32) 
    spectrogram = librosa.stft(time_series, n_fft=n_fft, hop_length=hop_length)
    spectrogram = np.abs(spectrogram)
    spectrogram = librosa.amplitude_to_db(spectrogram, ref=np.max)  # Normalize
    return spectrogram

class PyramidMultiScaleCNN(nn.Module):
    def __init__(self, input_channels):
        super(PyramidMultiScaleCNN, self).__init__()

        # Shared layers
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1, groups=input_channels),
            nn.Conv2d(input_channels, 32, kernel_size=1)
        )
        self.conv5x5 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=5, padding=2, groups=32),
            nn.Conv2d(32, 64, kernel_size=1)
        )
        self.conv7x7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=7, padding=3, groups=64),
            nn.Conv2d(64, 128, kernel_size=1)
        )

    def forward(self, x):
        # First branch
        branch1_x3x3 = F.relu(self.conv3x3(x))  # Output of 3x3
        branch1_x = torch.cat([branch1_x3x3, x], dim=1)  # Concatenate with input
        branch1_x = branch1_x[:, :32, :, :]  # Keep only first 32 channels
        branch1_x5x5 = F.relu(self.conv5x5(branch1_x))  # Output of 5x5
        branch1_x = torch.cat([branch1_x5x5, x], dim=1)  # Concatenate with input
        branch1_x = branch1_x[:, :64, :, :]  # Keep only first 64 channels
        branch1_x7x7 = F.relu(self.conv7x7(branch1_x))  # Output of 7x7
        
        # Concatenate outputs of all three layers for branch 1
        branch1_output = torch.cat([branch1_x3x3, branch1_x5x5, branch1_x7x7], dim=1)

        # Second branch (reuse same layers)
        branch2_x3x3 = F.relu(self.conv3x3(x))  # Output of 3x3
        branch2_x = torch.cat([branch2_x3x3, x], dim=1)  # Concatenate with input
        branch2_x = branch2_x[:, :32, :, :]  # Keep only first 32 channels
        branch2_x5x5 = F.relu(self.conv5x5(branch2_x))  # Output of 5x5
        branch2_x = torch.cat([branch2_x5x5, x], dim=1)  # Concatenate with input
        branch2_x = branch2_x[:, :64, :, :]  # Keep only first 64 channels
        branch2_x7x7 = F.relu(self.conv7x7(branch2_x))  # Output of 7x7
        
        # Concatenate outputs of all three layers for branch 2
        branch2_output = torch.cat([branch2_x3x3, branch2_x5x5, branch2_x7x7], dim=1)

        # Final concatenation of both branches
        final_output = torch.cat([branch1_output, branch2_output], dim=1)

        return final_output


class PyramidAttentionModel(nn.Module):
    def __init__(self, input_channels, n_classes, num_splits):
        super(PyramidAttentionModel, self).__init__()
        assert 2 <= num_splits and num_splits % 2 == 0, "num_splits must be even and atleast 2"
        self.num_splits = num_splits
        self.dropout = nn.Dropout(p=0.3)
        
        # Save number of classes
        self.n_classes = n_classes
        
        # Create a PyramidMultiScaleCNN for each split
        self.groups = nn.ModuleList([PyramidMultiScaleCNN(input_channels) for _ in range(num_splits)])
        
        # GAP branch
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # DCT branch using the multi-frequency compression from dct.py
        self.k = 2  # Number of parts for DCT
        self.max_u = 5  # Maximum frequency component along height
        self.max_v = 5  # Maximum frequency component along width
        
        feature_size = int((self.num_splits/2)*1792/2)
        # Final classification layers for both branches
        self.classifier_gap = nn.Linear(feature_size, n_classes)
        self.classifier_dct = nn.Linear(input_channels, n_classes)  # Placeholder value
        self.batch_norm_dct = nn.BatchNorm1d(num_features=feature_size)  # Placeholder size; updated dynamically if needed

    def forward(self, x):
        # Ensure the number of channels is evenly divisible by num_splits
        batch_size, channels, height, width = x.size()
        if channels % self.num_splits != 0:
            padding = (0, 0, 0, 0, 0, self.num_splits - (channels % self.num_splits))
            x = F.pad(x, padding)
            channels += self.num_splits - (channels % self.num_splits)
        
        split_channels = channels // self.num_splits
        
        # Split the input into `num_splits` parts
        split_inputs = torch.split(x, split_channels, dim=1)
        
        # Pass each part through its corresponding PyramidMultiScaleCNN
        outputs = [group(split_input) for group, split_input in zip(self.groups, split_inputs)]
        
        # Concatenate the outputs from all groups along the channel dimension
        concat_out = torch.cat(outputs, dim=1)
        
        # GAP Branch
        gap_out = self.gap(concat_out)
        gap_out = gap_out.view(gap_out.size(0), -1)  # Flatten for fully connected layer
        gap_pred = self.classifier_gap(gap_out)
        
        # DCT Branch
        dct_out_list = []
        for i in range(concat_out.size(0)):  # For each batch
            dct_input = concat_out[i]
            compressed_dct = multi_frequency_compression(dct_input, self.k, self.max_u, self.max_v)
            dct_out_list.append(compressed_dct)
        
        dct_out = torch.stack([torch.tensor(item) if isinstance(item, np.ndarray) else item for item in dct_out_list], dim=0)
        dct_out = dct_out.to(x.device)
        dct_out = dct_out.view(dct_out.size(0), -1)  # Flatten for batch normalization

        # Dynamically adjust the classifier_dct and batch_norm_dct layers if needed
        if dct_out.size(1) != self.classifier_dct.in_features:
            self.classifier_dct = nn.Linear(dct_out.size(1), self.n_classes).to(x.device)
            self.batch_norm_dct = nn.BatchNorm1d(num_features=dct_out.size(1)).to(x.device)

        # Apply batch normalization and dropout
        dct_out = self.batch_norm_dct(dct_out)
        dct_out = self.dropout(dct_out)
        
        # Pass through classifier_dct
        dct_pred = self.classifier_dct(dct_out)
        
        return gap_pred, dct_pred
