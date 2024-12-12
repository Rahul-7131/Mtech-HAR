import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
from dct_old import multi_frequency_compression, channel_attention  

# Function to convert time-series data to spectrogram
def time_series_to_spectrogram(time_series, sample_rate=50, n_fft=28, hop_length=128):
    time_series = np.array(time_series, dtype=np.float32) 
    spectrogram = librosa.stft(time_series, n_fft=n_fft, hop_length=hop_length)
    spectrogram = np.abs(spectrogram)
    spectrogram = librosa.amplitude_to_db(spectrogram, ref=np.max)  # Normalize
    return spectrogram

class PyramidMultiScaleCNN(nn.Module):
    def __init__(self, input_channels):
        super(PyramidMultiScaleCNN, self).__init__()
        self.conv3x3 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.conv5x5 = nn.Conv2d(64 + input_channels, 128, kernel_size=5, padding=2)
        self.conv7x7 = nn.Conv2d(128 + input_channels, 256, kernel_size=7, padding=3)

        self.branch2_conv3x3 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.branch2_conv5x5 = nn.Conv2d(64 + input_channels, 128, kernel_size=5, padding=2)
        self.branch2_conv7x7 = nn.Conv2d(128 + input_channels, 256, kernel_size=7, padding=3)

    def forward(self, x):
        # First branch
        branch1_x3x3 = F.relu(self.conv3x3(x))  # Output of 3x3
        branch1_x = torch.cat([branch1_x3x3, x], dim=1)  # Concatenate with input
        branch1_x5x5 = F.relu(self.conv5x5(branch1_x))  # Output of 5x5
        branch1_x = torch.cat([branch1_x5x5, x], dim=1)  # Concatenate with input
        branch1_x7x7 = F.relu(self.conv7x7(branch1_x))  # Output of 7x7
        
        # Concatenate outputs of all three layers for branch 1
        branch1_output = torch.cat([branch1_x3x3, branch1_x5x5, branch1_x7x7], dim=1)

        # Second branch
        branch2_x3x3 = F.relu(self.branch2_conv3x3(x))  # Output of 3x3
        branch2_x = torch.cat([branch2_x3x3, x], dim=1)  # Concatenate with input
        branch2_x5x5 = F.relu(self.branch2_conv5x5(branch2_x))  # Output of 5x5
        branch2_x = torch.cat([branch2_x5x5, x], dim=1)  # Concatenate with input
        branch2_x7x7 = F.relu(self.branch2_conv7x7(branch2_x))  # Output of 7x7
        
        # Concatenate outputs of all three layers for branch 2
        branch2_output = torch.cat([branch2_x3x3, branch2_x5x5, branch2_x7x7], dim=1)

        # Final concatenation of both branches
        final_output = torch.cat([branch1_output, branch2_output], dim=1)

        return final_output

class PyramidAttentionModel(nn.Module):
    def __init__(self, input_channels, n_classes):
        super(PyramidAttentionModel, self).__init__()
        self.dropout = nn.Dropout(p=0.3)
        
        # Save number of classes
        self.n_classes = n_classes
        
        # Split into two groups and pass each group through Pyramid CNN
        self.group1 = PyramidMultiScaleCNN(input_channels)
        self.group2 = PyramidMultiScaleCNN(input_channels)
        
        # GAP branch
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # DCT branch using the multi-frequency compression from dct.py
        self.k = 2  # Number of parts for DCT
        self.max_u = 5  # Maximum frequency component along height
        self.max_v = 5  # Maximum frequency component along width
        
        # Final classification layers for both branches
        self.classifier_gap = nn.Linear(1792, n_classes)
        self.classifier_dct = nn.Linear(input_channels, n_classes)  # Placeholder value
        self.batch_norm_dct = nn.BatchNorm1d(num_features=1792)  # Placeholder size; updated dynamically if needed
    def forward(self, x):
        # Ensure the number of channels is even (pad if necessary)
        batch_size, channels, height, width = x.size()
    
        # If channels is odd, pad the input to make the channels even
        if channels % 2 != 0:
            padding = (0, 0, 0, 0, 0, 1)  # Pad 1 channel along the channel dimension
            x = F.pad(x, padding)
            channels += 1  # Update the channel count after padding
    
        half_channels = channels // 2  # Now, channels is even
    
        # Split the input into two parts: each group gets half of the input channels
        x1 = x[:, :half_channels, :, :]  # First half of channels
        x2 = x[:, half_channels:, :, :]  # Second half of channels
    
        # Pass each half through the Pyramid CNNs
        out1 = self.group1(x1)
        out2 = self.group2(x2)
    
        # Concatenate the outputs from both groups along the channel dimension
        concat_out = torch.cat([out1, out2], dim=1)
    
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
        