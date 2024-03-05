import torch.nn as nn
import torch.nn.functional as F

class ARModel(nn.Module):
    def __init__(self):
        super(ARModel, self).__init__()
        
        # Assume the dimension of metadata is 5
        # Will change later
        metadata_dimension = 5
        
        self.conv_layer = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # ...
            # ...
            # ...
            # May add more layers...
        )
        self.rnn_layer = nn.LSTM(input_size=16*112*112, hidden_size=128, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(metadata_dimension, 64)  
        self.fc3 = nn.Linear(128, 4)  # Final classification

    def forward(self, frames, metadata):
        batch_size, timesteps, C, H, W = frames.size()
        c_in = frames.view(batch_size * timesteps, C, H, W)
        c_out = self.conv_layer(c_in)
        r_in = c_out.view(batch_size, timesteps, -1)
        r_out, (h_n, h_c) = self.rnn_layer(r_in)
        r_out2 = self.fc1(r_out[:, -1, :])
        
        m_out = self.fc2(metadata)
        combined = torch.cat((r_out2, m_out), 1)
        output = self.fc3(combined)
        return output
