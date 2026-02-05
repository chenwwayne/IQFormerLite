import torch
import torch.nn as nn

class MCLDNN(nn.Module):
    """
    MCLDNN PyTorch implementation based on Keras code.
    Input shape: (Batch, 2, 128) -> (Batch, I/Q, Time)
    """
    def __init__(self, frame_length=128, num_classes=11):
        super(MCLDNN, self).__init__()
        
        # Part-A: Multi-channel Inputs and Spatial Characteristics Mapping Section
        
        # x1 path: Conv2D on I/Q combined
        # Keras: Conv2D(50, (2,8), padding='same') on (N, 2, 128, 1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 50, kernel_size=(2, 8), padding='same'),
            nn.ReLU()
        )
        
        # x2 path: Conv1D on I channel
        # Keras: Conv1D(50, 8, padding='causal') on (N, 128, 1)
        self.conv2 = nn.Sequential(
            nn.ConstantPad1d((7, 0), 0), # Causal padding: pad left by kernel_size - 1
            nn.Conv1d(1, 50, kernel_size=8),
            nn.ReLU()
        )
        
        # x3 path: Conv1D on Q channel
        # Keras: Conv1D(50, 8, padding='causal') on (N, 128, 1)
        self.conv3 = nn.Sequential(
            nn.ConstantPad1d((7, 0), 0),
            nn.Conv1d(1, 50, kernel_size=8),
            nn.ReLU()
        )
        
        # x path after concat: Conv2D
        # Keras: Conv2D(50, (1,8), padding='same')
        self.conv4 = nn.Sequential(
            nn.Conv2d(50, 50, kernel_size=(1, 8), padding='same'),
            nn.ReLU()
        )
        
        # Final Conv2D before LSTM
        # Keras: Conv2D(100, (2,5), padding='valid')
        # Input (N, 100, 2, 128) -> Output (N, 100, 1, 124)
        self.conv5 = nn.Sequential(
            nn.Conv2d(100, 100, kernel_size=(2, 5), padding=0), # Valid padding
            nn.ReLU()
        )
        
        # Part-B: Temporal Characteristics Extraction Section
        # LSTM
        # Keras: LSTM(128, return_sequences=True), then LSTM(128)
        # This is equivalent to a 2-layer LSTM where the last layer returns only the last step?
        # No, Keras stacked LSTMs are usually separate layers.
        # Layer 1: returns sequence. Layer 2: returns last step.
        # PyTorch LSTM num_layers=2 returns output for all steps for the last layer.
        # To match Keras exactly:
        # We can use a 2-layer LSTM and take the last time step of the output (which comes from the last layer).
        self.lstm = nn.LSTM(input_size=100, hidden_size=128, num_layers=2, batch_first=True)
        
        # DNN
        self.classifier = nn.Sequential(
            nn.Linear(128, 128),
            nn.SELU(),
            nn.Dropout(0.5),
            nn.Linear(128, 128),
            nn.SELU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # x: (Batch, 2, 128)
        
        # Input 1: (Batch, 1, 2, 128) for Conv2D
        input1 = x.unsqueeze(1)
        x1 = self.conv1(input1) # (Batch, 50, 2, 128)
        
        # Input 2 (I) and 3 (Q): (Batch, 1, 128) for Conv1D
        input2 = x[:, 0:1, :]
        input3 = x[:, 1:2, :]
        
        x2 = self.conv2(input2) # (Batch, 50, 128)
        x3 = self.conv3(input3) # (Batch, 50, 128)
        
        # Reshape to stack: (Batch, 50, 1, 128)
        x2_reshape = x2.unsqueeze(2)
        x3_reshape = x3.unsqueeze(2)
        
        # Concatenate on Height (dim 2) -> (Batch, 50, 2, 128)
        x_concat1 = torch.cat([x2_reshape, x3_reshape], dim=2)
        
        x_conv4 = self.conv4(x_concat1) # (Batch, 50, 2, 128)
        
        # Concatenate x1 and x_conv4 on Channel (dim 1) -> (Batch, 100, 2, 128)
        x_concat2 = torch.cat([x1, x_conv4], dim=1)
        
        x_conv5 = self.conv5(x_concat2) # (Batch, 100, 1, 124)
        
        # Reshape for LSTM
        # (Batch, 100, 1, 124) -> Squeeze H -> (Batch, 100, 124) -> Permute -> (Batch, 124, 100)
        x_lstm_in = x_conv5.squeeze(2).permute(0, 2, 1)
        
        lstm_out, _ = self.lstm(x_lstm_in) # (Batch, 124, 128)
        
        # Take the last time step
        x_last = lstm_out[:, -1, :] # (Batch, 128)
        
        out = self.classifier(x_last)
        
        return out

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MCLDNN(frame_length=128, num_classes=11).to(device)
    x = torch.randn(32, 2, 128).to(device)
    y = model(x)
    print("Output shape:", y.shape)
