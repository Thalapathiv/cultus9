# =========================================================
# Advanced Time Series Forecasting with Uncertainty
# =========================================================

# -----------------------------
# STEP 1: Imports
# -----------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import mean_squared_error, mean_absolute_error

# -----------------------------
# STEP 2: Data Generation
# -----------------------------
def generate_time_series(n_samples=10000, seed=42):
    """
    Generates a multivariate time series with:
    - Trend
    - Multiple seasonalities
    - Heteroscedastic noise
    """
    np.random.seed(seed)
    t = np.arange(n_samples)

    trend = 0.0005 * t
    daily = np.sin(2 * np.pi * t / 24)
    weekly = np.sin(2 * np.pi * t / (24 * 7))
    yearly = np.sin(2 * np.pi * t / (24 * 365))

    noise_scale = 0.1 + 0.0001 * t
    noise = np.random.normal(0, noise_scale)

    target = trend + daily + weekly + yearly + noise

    df = pd.DataFrame({
        "target": target,
        "trend": trend,
        "daily_seasonality": daily,
        "weekly_seasonality": weekly,
        "yearly_seasonality": yearly,
        "noise_scale": noise_scale
    })

    return df

df = generate_time_series()
print("Dataset shape:", df.shape)
df.head()

# -----------------------------
# STEP 3: Dataset Preparation
# -----------------------------
SEQ_LEN = 48
HORIZON = 10

class TimeSeriesDataset(Dataset):
    def __init__(self, data):
        values = data.values
        self.X, self.y = [], []

        for i in range(len(values) - SEQ_LEN - HORIZON):
            self.X.append(values[i:i+SEQ_LEN])
            self.y.append(values[i+SEQ_LEN:i+SEQ_LEN+HORIZON, 0])

        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dataset = TimeSeriesDataset(df)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64)

# -----------------------------
# STEP 4: LSTM Model
# -----------------------------
class LSTMForecaster(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, HORIZON)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])

device = "cuda" if torch.cuda.is_available() else "cpu"

model = LSTMForecaster(input_size=df.shape[1]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
criterion = nn.MSELoss()

# -----------------------------
# STEP 5: Model Training
# -----------------------------
EPOCHS = 10

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        preds = model(x)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} | Train MSE: {total_loss/len(train_loader):.4f}")

# -----------------------------
# STEP 6: Deterministic Evaluation
# -----------------------------
model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for x, y in val_loader:
        x = x.to(device)
        preds = model(x).cpu()
        y_true.append(y)
        y_pred.append(preds)

y_true = torch.cat(y_true)
y_pred = torch.cat(y_pred)

rmse = mean_squared_error(y_true, y_pred, squared=False)
mae = mean_absolute_error(y_true, y_pred)

print("\nDeterministic Model Performance")
print("--------------------------------")
print(f"RMSE : {rmse:.4f}")
print(f"MAE  : {mae:.4f}")

# -----------------------------
# STEP 7: MC Dropout (Probabilistic Forecasting)
# -----------------------------
def mc_dropout_predictions(model, x, n_samples=100):
    model.train()  # enable dropout
    preds = []
    for _ in range(n_samples):
        preds.append(model(x).unsqueeze(0))
    return torch.cat(preds)

x_sample, y_sample = next(iter(val_loader))
x_sample = x_sample.to(device)

mc_preds = mc_dropout_predictions(model, x_sample)

mean_pred = mc_preds.mean(dim=0)

lower_80 = torch.quantile(mc_preds, 0.10, dim=0)
upper_80 = torch.quantile(mc_preds, 0.90, dim=0)

lower_95 = torch.quantile(mc_preds, 0.05, dim=0)
upper_95 = torch.quantile(mc_preds, 0.95, dim=0)

# -----------------------------
# STEP 8: Uncertainty Metrics
# -----------------------------
def coverage(y, lower, upper):
    return ((y >= lower) & (y <= upper)).float().mean().item()

cp_80 = coverage(y_sample, lower_80, upper_80)
cp_95 = coverage(y_sample, lower_95, upper_95)

miw_80 = (upper_80 - lower_80).mean().item()
miw_95 = (upper_95 - lower_95).mean().item()

print("\nProbabilistic Model Evaluation")
print("--------------------------------")
print(f"80% Coverage Probability : {cp_80:.3f}")
print(f"95% Coverage Probability : {cp_95:.3f}")
print(f"80% Mean Interval Width  : {miw_80:.3f}")
print(f"95% Mean Interval Width  : {miw_95:.3f}")

# -----------------------------
# STEP 9: Visualization
# -----------------------------
plt.figure(figsize=(12,5))

plt.plot(y_sample[0].numpy(), label="Actual", marker="o")
plt.plot(mean_pred[0].detach().numpy(), label="Prediction", marker="x")

plt.fill_between(
    range(HORIZON),
    lower_80[0].detach().numpy(),
    upper_80[0].detach().numpy(),
    alpha=0.3,
    label="80% Interval"
)

plt.fill_between(
    range(HORIZON),
    lower_95[0].detach().numpy(),
    upper_95[0].detach().numpy(),
    alpha=0.2,
    label="95% Interval"
)

plt.title("Multi-step Forecast with Uncertainty Quantification")
plt.xlabel("Forecast Horizon")
plt.ylabel("Value")
plt.legend()
plt.show()
