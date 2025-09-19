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

        # Use first item's first feature as price series
        prices = arr[:, 0, 0]  # shape: (timesteps,)
        self.log_returns = np.diff(np.log(prices + 1e-8))  # avoid log(0)

    def __len__(self):
        # -1 because log_return has one fewer element
        return len(self.arr) - self.seq_len - 1  

    def __getitem__(self, idx):
        # Input sequence (seq_len timesteps of all items/features)
        x = self.arr[idx:idx+self.seq_len]

        # Target = log return at the NEXT step
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
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# Training loop
for epoch in range(30):
    model.train()
    train_loss = 0
    for x, y in train_loader:
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
            preds = model(x)
            loss = criterion(preds, y)
            test_loss += loss.item() * len(x)

    print(f"Epoch {epoch+1} | Train Loss {train_loss/len(train_data):.6f} | Test Loss {test_loss/len(test_data):.6f}")
