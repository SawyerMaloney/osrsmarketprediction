import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

class TimeSeriesDataset(Dataset):
    def __init__(self, arr, seq_len=32, horizon=5):
        """
        arr: numpy array of shape (timesteps, items, features)
        seq_len: how many past steps to feed into the model
        """
        self.arr = arr
        self.seq_len = seq_len
        self.horizon = horizon

        # Use first item's first feature as price series
        prices = arr[:, 0, 0]  # shape: (timesteps,)
        self.log_returns = np.diff(np.log(prices + 1e-8), prepend=np.nan)  # avoid log(0)

    def __len__(self):
        return len(self.arr) - self.seq_len - self.horizon + 1

    def __getitem__(self, idx):
        x = self.arr[idx:idx+self.seq_len]
        y = self.log_returns[idx+self.seq_len + (self.horizon-1)]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)




class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        """
        input_size = number of features per item * number of items
        """
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape: (batch, seq_len, items, features)
        b, t, i, f = x.shape
        x = x.view(b, t, i*f)  # flatten items/features into one vector

        out, _ = self.lstm(x)   # out: (batch, seq_len, hidden)
        out = out[:, -1, :]     # last timestep
        return self.fc(out).squeeze(-1)  # (batch,)
    
# check if cuda is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
    
# Load data
arr = np.load("timeseries.npy")   # shape (timesteps, items, features)

# Split into train/test
split = int(0.8 * len(arr))
train_data = TimeSeriesDataset(arr[:split])
test_data  = TimeSeriesDataset(arr[split:])

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_data, batch_size=64, shuffle=False)

# Model, optimizer, loss
items, features = arr.shape[1], arr.shape[2]
input_size = items * features

model = LSTMModel(input_size)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# Training loop
for epoch in range(30):
    model.train()
    train_loss = 0
    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        preds = model(x)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(x)

    # Eval
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            loss = criterion(preds, y)
            test_loss += loss.item() * len(x)

    print(f"Epoch {epoch+1} | Train Loss {train_loss/len(train_data):.6f} | Test Loss {test_loss/len(test_data):.6f}")
