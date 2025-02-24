import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa

class PyramidMultiScaleCNN(nn.Module):
    def __init__(self, input_channels):
        super(PyramidMultiScaleCNN, self).__init__()

        # Shared layers
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1, groups=1),   #NOTE: changed Groups = input channels here
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
    
class PyramidCNNModel(nn.Module):
    def __init__(self, input_channels, n_classes, num_splits):
        super(PyramidCNNModel, self).__init__()
        assert num_splits >= 2 and num_splits % 2 == 0, "num_splits must be even and at least 2"
        self.num_splits = num_splits
        
        self.groups = nn.ModuleList([PyramidMultiScaleCNN(input_channels) for _ in range(num_splits)])
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        feature_size = (self.num_splits // 2) * 1792 // 2                                                                   #NOTE:changes here
        self.classifier = nn.Linear(feature_size, n_classes)
        
    def forward(self, x):                                                                                                      #REFACTORED for CIFAR
        batch_size, channels, height, width = x.size()
        split_channels = max(3, channels // self.num_splits)                                                                #NOTE: changes here 

        # Adjust number of splits if channels are insufficient
        if channels % self.num_splits != 0:
            split_channels = channels // self.num_splits + 1

        split_inputs = torch.split(x, split_channels, dim=1)

        split_inputs = [F.pad(split_input, (0, 0, 0, 0, 0, max(0, 3 - split_input.size(1)))) for split_input in split_inputs]

        outputs = [group(split_input) for group, split_input in zip(self.groups, split_inputs)]
        concat_out = torch.cat(outputs, dim=1)

        gap_out = self.gap(concat_out).view(concat_out.size(0), -1)
        predictions = self.classifier(gap_out)

        return predictions
