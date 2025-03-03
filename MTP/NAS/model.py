import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
#from dct import multi_frequency_compression, channel_attention  

# Function to convert time-series data to spectrogram
def time_series_to_spectrogram(time_series, sample_rate=50, n_fft=256, hop_length=128):
    time_series = np.array(time_series, dtype=np.float32) 
    spectrogram = librosa.stft(time_series, n_fft=n_fft, hop_length=hop_length)
    spectrogram = np.abs(spectrogram)
    spectrogram = librosa.amplitude_to_db(spectrogram, ref=np.max)  # Normalize
    return spectrogram

class PyramidMultiScaleCNN(nn.Module):
    def __init__(self, input_channels, k):
        super(PyramidMultiScaleCNN, self).__init__()

        # Compute output channels based on k (ratio of input to output channels)
        output_channels = int(input_channels * k)
        
        # Shared layers with dynamic output channels
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1, groups=input_channels),
            nn.Conv2d(input_channels, output_channels // 4, kernel_size=1)  # Adjust channels
        )
        self.conv5x5 = nn.Sequential(
            nn.Conv2d(output_channels // 4, output_channels // 4, kernel_size=5, padding=2, groups=output_channels // 4),
            nn.Conv2d(output_channels // 4, output_channels // 2, kernel_size=1)
        )
        self.conv7x7 = nn.Sequential(
            nn.Conv2d(output_channels // 2, output_channels // 2, kernel_size=7, padding=3, groups=output_channels // 2),
            nn.Conv2d(output_channels // 2, output_channels, kernel_size=1)
        )

    def forward(self, x):
        # First branch
        branch1_x3x3 = F.relu(self.conv3x3(x))  # Output of 3x3
        branch1_x = torch.cat([branch1_x3x3, x], dim=1)  # Concatenate with input
        branch1_x = branch1_x[:, :branch1_x3x3.size(1), :, :]  # Keep only required channels
        branch1_x5x5 = F.relu(self.conv5x5(branch1_x))  # Output of 5x5
        branch1_x = torch.cat([branch1_x5x5, x], dim=1)
        branch1_x = branch1_x[:, :branch1_x5x5.size(1), :, :]
        branch1_x7x7 = F.relu(self.conv7x7(branch1_x))  # Output of 7x7
        
        # Concatenate outputs of all three layers for branch 1
        branch1_output = torch.cat([branch1_x3x3, branch1_x5x5, branch1_x7x7], dim=1)

        # Second branch (reuse same layers)
        branch2_x3x3 = F.relu(self.conv3x3(x))
        branch2_x = torch.cat([branch2_x3x3, x], dim=1)
        branch2_x = branch2_x[:, :branch2_x3x3.size(1), :, :]
        branch2_x5x5 = F.relu(self.conv5x5(branch2_x))
        branch2_x = torch.cat([branch2_x5x5, x], dim=1)
        branch2_x = branch2_x[:, :branch2_x5x5.size(1), :, :]
        branch2_x7x7 = F.relu(self.conv7x7(branch2_x))

        # Concatenate outputs of all three layers for branch 2
        branch2_output = torch.cat([branch2_x3x3, branch2_x5x5, branch2_x7x7], dim=1)

        # Final concatenation of both branches
        final_output = torch.cat([branch1_output, branch2_output], dim=1)

        return final_output

class PyramidAttentionModel(nn.Module):
    def __init__(self, input_channels, n_classes, num_splits, k_channels):
        super(PyramidAttentionModel, self).__init__()
        assert num_splits % 2 == 0 and num_splits >= 2, "num_splits must be even and at least 2"
        self.num_splits = num_splits
        self.n_classes = n_classes
        k = k_channels
        # Pyramid CNN groups
        self.groups = nn.ModuleList([
            PyramidMultiScaleCNN(input_channels, k) for _ in range(num_splits)
        ])
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Feature size placeholder (set dynamically in forward)
        self.feature_size = None
        self.classifier_gap = None  # Will be initialized dynamically

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        # Ensure num_splits does not exceed available channels
        self.num_splits = min(self.num_splits, channels)  
        split_channels = max(1, channels // self.num_splits)

        # Split the input into `num_splits` parts
        split_inputs = torch.split(x, split_channels, dim=1)
        
        # Process each split with PyramidMultiScaleCNN
        outputs = [group(split_input) for group, split_input in zip(self.groups, split_inputs)]
        
        concat_out = torch.cat(outputs, dim=1)

        # GAP Branch
        gap_out = self.gap(concat_out).view(batch_size, -1)  # Flatten

        # Ensure classifier matches feature size
        if self.feature_size is None:
            self.feature_size = gap_out.shape[1]
            self.classifier_gap = nn.Linear(self.feature_size, self.n_classes).to(x.device)
        
        return self.classifier_gap(gap_out)


