import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt

# ---------------------------
# Dataset
# ---------------------------
class TimeSeriesDataset(Dataset):
    def __init__(self, arr, seq_len=32, horizon=5, target_scale=100.0,
                 clip_inputs=True, clip_value=5.0, target_item=0):
        """
        arr: numpy array of shape (timesteps, items, features)
        target_item: which item (0..N-1) to predict
        """
        self.arr = arr.astype(np.float32)
        self.seq_len = seq_len
        self.horizon = horizon
        self.target_item = target_item

        # ---------------------------
        # Compute log returns for target item
        # ---------------------------
        prices = self.arr[:, target_item, 0]
        prices = np.clip(prices, 1e-8, None)
        log_prices = np.log(prices)
        self.log_returns = np.diff(log_prices, prepend=log_prices[0])
        self.log_returns *= target_scale
        self.log_returns = np.clip(self.log_returns, -5*target_scale, 5*target_scale)

        # ---------------------------
        # Normalize input features
        # ---------------------------
        timesteps, items, features = self.arr.shape
        arr_2d = self.arr.reshape(-1, features)
        mean = arr_2d.mean(axis=0)
        std = arr_2d.std(axis=0) + 1e-8
        arr_normalized = (arr_2d - mean) / std
        arr_normalized = arr_normalized.reshape(timesteps, items, features)

        if clip_inputs:
            arr_normalized = np.clip(arr_normalized, -clip_value, clip_value)
        self.arr = arr_normalized

        # Save stats for rescaling later
        self.y_mean = np.mean(self.log_returns)
        self.y_std  = np.std(self.log_returns) + 1e-8

    def __len__(self):
        return len(self.arr) - self.seq_len - self.horizon + 1

    def __getitem__(self, idx):
        x = self.arr[idx:idx+self.seq_len]
        y = self.log_returns[idx+self.seq_len + (self.horizon-1)]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)



# ---------------------------
# LSTM Model
# ---------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=4):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        b, t, i, f = x.shape
        x = x.view(b, t, i*f)  # flatten items/features
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out).squeeze(-1)


# ------------------------
#   Setting up model and datasets
# ------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
arr = np.load("timeseries.npy")

# ---------------------------
# Hyperparameters
# ---------------------------
target_item = 1  # second item
horizon = 2      # based on top performing results
seq_len = 32
num_epochs = 30
batch_size = 32
hidden_size = 128
num_layers = 4
lr = 1e-3
threshold = 2.0
spike_weight = 5.0

# ---------------------------
# Dataset / DataLoader
# ---------------------------
train_dataset = TimeSeriesDataset(arr[:int(0.8*len(arr))],
                                  horizon=horizon,
                                  target_item=target_item,
                                  seq_len=seq_len)
test_dataset  = TimeSeriesDataset(arr[int(0.8*len(arr)):],
                                  horizon=horizon,
                                  target_item=target_item,
                                  seq_len=seq_len)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ---------------------------
# Model / optimizer / criterion
# ---------------------------
input_size = arr.shape[1] * arr.shape[2]
model = LSTMModel(input_size, hidden_size=hidden_size, num_layers=num_layers).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# ---------------------------
# Training loop with threshold-weighted MSE
# ---------------------------
for epoch in range(num_epochs):
    model.train()
    train_loss_total = 0.0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        preds = model(x)

        weights = torch.ones_like(y)
        weights[torch.abs(y) > threshold] = spike_weight
        loss = (weights * (preds - y)**2).mean()
        loss.backward()
        optimizer.step()
        train_loss_total += loss.item() * x.size(0)

    train_loss_epoch = train_loss_total / len(train_dataset)

    # ---------------------------
    # Evaluation
    # ---------------------------
    model.eval()
    test_loss_total = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x)

            weights = torch.ones_like(y)
            weights[torch.abs(y) > threshold] = spike_weight
            loss = (weights * (preds - y)**2).mean()
            test_loss_total += loss.item() * x.size(0)

    test_loss_epoch = test_loss_total / len(test_dataset)
    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss_epoch:.6f} | Test Loss: {test_loss_epoch:.6f}")

# ---------------------------
# Plot predictions after training
# ---------------------------
model.eval()
all_preds = []
all_true = []

with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        preds = model(x)
        # Rescale to original target scale
        preds_real = preds * train_dataset.y_std + train_dataset.y_mean
        y_real = y * train_dataset.y_std + train_dataset.y_mean
        all_preds.append(preds_real.cpu().numpy())
        all_true.append(y_real.cpu().numpy())

all_preds = np.concatenate(all_preds)
all_true = np.concatenate(all_true)

plt.figure(figsize=(12,5))
plt.plot(all_true, label="True")
plt.plot(all_preds, label="Predicted")
plt.title(f"Item {target_item} Predictions (Horizon {horizon})")
plt.xlabel("Time step")
plt.ylabel("Target (rescaled)")
plt.legend()
plt.show()
