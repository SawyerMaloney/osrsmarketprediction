import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import itertools

class TimeSeriesDataset(Dataset):
    def __init__(self, arr, seq_len=32, horizon=5, target_scale=100.0, clip_inputs=True, clip_value=5.0):
        """
        arr: numpy array of shape (timesteps, items, features)
        seq_len: how many past steps to feed into the model
        """
        self.arr = arr.astype(np.float32)
        self.seq_len = seq_len
        self.horizon = horizon

        # compute log returns for target
        # with unnormalized data
        prices = self.arr[:, 0, 0]
        prices = np.clip(prices, a_min=1e-8, a_max=None)
        log_prices = np.log(prices)
        self.log_returns = np.diff(log_prices, prepend=log_prices[0])

        # multiply log returns by constant
        self.log_returns *= target_scale

        self.log_returns = np.clip(self.log_returns, -5*target_scale, 5*target_scale)

        # normalize input features
        # compute mean/std across timesteps and items for each feature
        timesteps, items, features = self.arr.shape
        arr_2d = self.arr.reshape(-1, features)
        mean = arr_2d.std(axis=0) + 1e-8
        std = arr_2d.std(axis=0) + 1e-8
        arr_normalized = (arr_2d - mean) / std
        arr_normalized = arr_normalized.reshape(timesteps, items, features)

        # Optional clipping to remove extreme outliers
        if clip_inputs:
            arr_normalized = np.clip(arr_normalized, -clip_value, clip_value)

        self.arr = arr_normalized
        
    def __len__(self):
        return len(self.arr) - self.seq_len - self.horizon + 1

    def __getitem__(self, idx):
        x = self.arr[idx:idx+self.seq_len]
        y = self.log_returns[idx+self.seq_len + (self.horizon-1)]

        # Runtime checks
        if torch.isnan(torch.tensor(x)).any() or torch.isnan(torch.tensor(y)):
            print(f"[WARNING] NaN detected in batch idx {idx}")
        if torch.isinf(torch.tensor(x)).any() or torch.isinf(torch.tensor(y)):
            print(f"[WARNING] Inf detected in batch idx {idx}")


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
# Training loop with target stats monitoring
for epoch in range(30):
    model.train()
    train_loss = 0
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        preds = model(x)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(x)

        # Runtime monitoring of targets and predictions
        if batch_idx == 0:  # only first batch to avoid too much output
            print(f"Batch {batch_idx} | x min {x.min():.3f}, max {x.max():.3f}, mean {x.mean():.3f}")
            print(f"Batch {batch_idx} | y min {y.min():.3f}, max {y.max():.3f}, mean {y.mean():.3f}")
            print(f"Batch {batch_idx} | preds min {preds.min():.3f}, max {preds.max():.3f}, mean {preds.mean():.3f}")

    # Evaluation
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):
            x, y = x.to(device), y.to(device)
            preds = model(x)
            loss = criterion(preds, y)
            test_loss += loss.item() * len(x)

            # Check first batch stats
            if batch_idx == 0:
                print(f"[Eval] Batch {batch_idx} | x min {x.min():.3f}, max {x.max():.3f}, mean {x.mean():.3f}")
                print(f"[Eval] Batch {batch_idx} | y min {y.min():.3f}, max {y.max():.3f}, mean {y.mean():.3f}")
                print(f"[Eval] Batch {batch_idx} | preds min {preds.min():.3f}, max {preds.max():.3f}, mean {preds.mean():.3f}")

    print(f"Epoch {epoch+1} | Train Loss {train_loss/len(train_data):.6f} | "
          f"Test Loss {test_loss/len(test_data):.6f}")