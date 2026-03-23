import os
import time
import random
import joblib
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from plot_data import plot_predictions, plot_training_history
from event_based_data import process_all_months
torch.manual_seed(42)
np.random.seed(42)
torch.set_num_threads(8)


INPUT_STEPS = 500
OUTPUT_STEPS = 20
BATCH_SIZE = 32
EPOCHS = 15
HIDDEN_SIZE1 = 128
HIDDEN_SIZE2 = 64 
HIDDEN_SIZE3 = 32
LEARNING_RATE = 5e-4
PATIENCE = 5
Dropout = 0
Dynamic_threshold = 2
Downsample_size = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = pd.read_csv(r"SPC_output_1y.csv")
Time_col= ['Time']
ref_cols = ['P_I_ref', 'Q_I_ref', 'P_I_ref_int', 'Q_I_ref_int']
weather_cols = ['eta_PV']
meas_cols = ['P_I_meas', 'Q_I_meas' ]
feature_cols = ref_cols + weather_cols + meas_cols

X_train, y_train, X_val, y_val, X_test, y_test, downsampled_data = process_all_months(
    data,
    feature_cols,
    meas_cols,
    number_of_months=12,
    dynamic_threshold=Dynamic_threshold,
    stable_keep_step=Downsample_size
)

feature_scaler = StandardScaler()
target_scaler = {}

X_train_scaled = feature_scaler.fit_transform(X_train)
X_val_scaled   = feature_scaler.transform(X_val)
X_test_scaled  = feature_scaler.transform(X_test)

y_train_scaled = np.zeros_like(y_train)
y_val_scaled   = np.zeros_like(y_val)
y_test_scaled  = np.zeros_like(y_test)

for i, col in enumerate(meas_cols):
    scaler = StandardScaler()
    y_train_scaled[:, i:i+1] = scaler.fit_transform(y_train[:, i:i+1])
    y_val_scaled[:, i:i+1]   = scaler.transform(y_val[:, i:i+1])
    y_test_scaled[:, i:i+1]  = scaler.transform(y_test[:, i:i+1])
    target_scaler[col] = scaler

print("Scaled shapes -> Train:", X_train_scaled.shape,
      "Val:", X_val_scaled.shape,
      "Test:", X_test_scaled.shape)

class SequenceDataset(Dataset):
    def __init__(self, X, y, input_steps, output_steps, n_non_targets):
        self.X = X
        self.y = y
        self.input_steps = input_steps
        self.output_steps = output_steps
        self.n_non_targets = n_non_targets
        self.indices = list(range(0, len(X) - input_steps - output_steps, output_steps))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        s = self.indices[idx]
        x_history = torch.tensor(self.X[s:s+INPUT_STEPS], dtype=torch.float32)
        x_future_refs = torch.tensor(self.X[s+INPUT_STEPS:s+INPUT_STEPS+OUTPUT_STEPS, :self.n_non_targets], dtype=torch.float32)  # measured values are excluded from input window
        y_target = torch.tensor(self.y[s+INPUT_STEPS : s+INPUT_STEPS+OUTPUT_STEPS], dtype=torch.float32)
        return x_history, x_future_refs, y_target

n_non_targets = X_train_scaled.shape[1] - len(meas_cols)
print(f"Verified: {X_train_scaled.shape[1]} total features - {len(meas_cols)} targets = {n_non_targets} non-targets")

train_loader = DataLoader(
    SequenceDataset(X_train_scaled, y_train_scaled, INPUT_STEPS, OUTPUT_STEPS, n_non_targets),
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    SequenceDataset(X_val_scaled, y_val_scaled, INPUT_STEPS, OUTPUT_STEPS, n_non_targets),
    batch_size=BATCH_SIZE,
    shuffle=False
)

class SpcLSTM(nn.Module):
    def __init__(self, n_features, n_targets):
        super().__init__()
        self.n_targets = n_targets
        self.lstm1 = nn.LSTM(n_features, HIDDEN_SIZE1, batch_first=True)
        self.lstm2 = nn.LSTM(HIDDEN_SIZE1, HIDDEN_SIZE2, batch_first=True)
        self.fc_dense = nn.Linear(HIDDEN_SIZE2, HIDDEN_SIZE3)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(Dropout)
        self.fc_final = nn.Linear(HIDDEN_SIZE3, n_targets)
        # New: Learnable bias to correct systematic offsets
        self.output_bias = nn.Parameter(torch.zeros(n_targets))
        

    def forward(self, x_history, x_future_refs, y_target=None, output_steps=OUTPUT_STEPS, teacher_forcing_ratio=0.0):
        n_non_targets = x_history.size(2) - self.n_targets
        
        # 1. ENCODER: Process history ONCE
        out1, (h1, c1) = self.lstm1(x_history)              # We get the hidden states (h, c) which contain the "dynamics"
        _, (h2, c2) = self.lstm2(out1)
        current_pred = x_history[:, -1, n_non_targets:]     # Start the recursive chain with the last known values from history
        preds = []

        # 2. DECODER: Loop only over the output steps      # Combine future reference (weather/setpoints) with the last prediction
        for t in range(output_steps):
            step_ref = x_future_refs[:, t, :]
            step_input = torch.cat([step_ref, current_pred], dim=1).unsqueeze(1)
            out1, (h1, c1) = self.lstm1(step_input, (h1, c1))
            out2, (h2, c2) = self.lstm2(out1, (h2, c2))
            out_dense = self.relu(self.fc_dense(out2.squeeze(1)))
            out_dense = self.dropout(out_dense)
            pred = self.fc_final(out_dense) + self.output_bias
            preds.append(pred.unsqueeze(1))

            # Teacher Forcing
            if y_target is not None and np.random.rand() < teacher_forcing_ratio:
                current_pred = y_target[:, t, :]
            else:
                current_pred = pred

        return torch.cat(preds, dim=1)

n_features = X_train_scaled.shape[1]
n_targets = y_train_scaled.shape[1]

model = SpcLSTM(n_features, n_targets).to(device)
criterion = nn.MSELoss(reduction="none")
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3)
time_weights = torch.linspace(1, 1.5, OUTPUT_STEPS).to(device)
W_P_MSE = 1.5    # heavily penalize P offset
W_Q_MSE = 2.5    # standard penalty for Q
W_TRACK = 0.15   # 0.15 applied to the sum of tracking losses

train_losses, val_losses = [], []
train_loss_per_target = np.zeros(n_targets)
val_loss_per_target = np.zeros(n_targets)
best_val_loss = np.inf
counter = 0

def get_teacher_forcing_ratio(epoch, max_epochs):
    return max(0, 0.3 * (1 - epoch / max_epochs))

start_train = time.time()
for epoch in range(EPOCHS):
    # --------------------------
    # Training
    # --------------------------
    model.train()
    train_loss = 0
    train_loss_per_target[:] = 0
    tf_ratio = get_teacher_forcing_ratio(epoch, EPOCHS)  # decaying TF

    for x_hist, x_fref, y_batch in train_loader:
        x_hist, x_fref, y_batch = x_hist.to(device), x_fref.to(device), y_batch.to(device)
        optimizer.zero_grad()
        y_out = model(x_hist, x_fref, y_target=y_batch, output_steps=OUTPUT_STEPS, teacher_forcing_ratio=tf_ratio) 
        y_out_slice = y_out[:, -OUTPUT_STEPS:, :] 
        loss_mse_raw = criterion(y_out_slice, y_batch)
        # Index 0 = P (Active Power), Index 1 = Q (Reactive Power)
        loss_p = (loss_mse_raw[:, :, 0] * time_weights[None, :]).mean() * W_P_MSE
        loss_q = (loss_mse_raw[:, :, 1] * time_weights[None, :]).mean() * W_Q_MSE

        # --- TRACKING (GRADIENT) LOSS ---
        # This calculates the difference between consecutive steps
        diff_pred = y_out_slice[:, 1:, :] - y_out_slice[:, :-1, :]
        diff_true = y_batch[:, 1:, :] - y_batch[:, :-1, :]
        loss_tracking_p = nn.MSELoss()(diff_pred[:, :, 0], diff_true[:, :, 0])
        loss_tracking_q = nn.MSELoss()(diff_pred[:, :, 1], diff_true[:, :, 1])
        # 3. Horizon Mean Loss (Magnitude/Plateau)
        mean_pred = y_out_slice.mean(dim=1)
        mean_true = y_batch.mean(dim=1)
        loss_mean = nn.MSELoss()(mean_pred, mean_true)
        loss = (loss_p + loss_q + W_TRACK * (loss_tracking_p + loss_tracking_q) + (0.05 * loss_mean))
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
        train_loss += loss.item() * x_hist.size(0)
        train_loss_per_target += loss_mse_raw.detach().mean(dim=(0,1)).cpu().numpy() * x_hist.size(0)

    train_loss /= len(train_loader.dataset)
    train_loss_per_target /= len(train_loader.dataset)
    # --------------------------
    # Validation
    # --------------------------
    model.eval()
    val_loss = 0
    val_loss_per_target[:] = 0

    with torch.no_grad():
        for x_hist, x_fref, y_batch in val_loader:
            x_hist, x_fref, y_batch = x_hist.to(device), x_fref.to(device), y_batch.to(device) 
            y_out = model(x_hist, x_fref, output_steps=OUTPUT_STEPS, teacher_forcing_ratio=0.0)   # Pass future refs
            y_out_slice = y_out[:, -OUTPUT_STEPS:, :]
            loss_mse_raw_val = criterion(y_out_slice, y_batch)
            
            loss_p_val = (loss_mse_raw_val[:, :, 0] * time_weights[None, :]).mean() * W_P_MSE
            loss_q_val = (loss_mse_raw_val[:, :, 1] * time_weights[None, :]).mean() * W_Q_MSE
            
            diff_pred_v = y_out_slice[:, 1:, :] - y_out_slice[:, :-1, :]
            diff_true_v = y_batch[:, 1:, :] - y_batch[:, :-1, :]
            
            loss_tracking_p_v = nn.MSELoss()(diff_pred_v[:, :, 0], diff_true_v[:, :, 0])
            loss_tracking_q_v = nn.MSELoss()(diff_pred_v[:, :, 1], diff_true_v[:, :, 1])
            
            mean_pred_v = y_out_slice.mean(dim=1)
            mean_true_v = y_batch.mean(dim=1)
            loss_mean_v = nn.MSELoss()(mean_pred_v, mean_true_v)

            
            current_total_val_loss = (loss_p_val + loss_q_val + W_TRACK * (loss_tracking_p_v + loss_tracking_q_v) + (0.05 * loss_mean_v))
            
            val_loss += current_total_val_loss.item() * x_hist.size(0)
            val_loss_per_target += loss_mse_raw_val.detach().mean(dim=(0,1)).cpu().numpy() * x_hist.size(0)
    val_loss /= len(val_loader.dataset)
    val_loss_per_target /= len(val_loader.dataset)
    scheduler.step(val_loss)

    # --------------------------
    # Logging
    # --------------------------
    end_train = time.time()
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch+1}: Learning Rate: {current_lr:.2e} : Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}")
    print(f"Train MSE per target={train_loss_per_target}, Val MSE per target={val_loss_per_target}")
    print(f"Total training time: {end_train - start_train:.2f} seconds")

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    # --------------------------
    # Early stopping
    # --------------------------
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "dynamic_response_lstm_final_recurrsive.pth")
        counter = 0
    else:
        counter += 1
        if counter >= PATIENCE:
            print("Early stopping triggered")
            break

def autoregressive_test_function(
    model, X_test_scaled, y_test_scaled,
    target_scaler, input_len, output_len,
    device, start_idx, total_steps,
    feature_cols, meas_cols
 ):
    model.eval()
    window = X_test_scaled[start_idx:start_idx+input_len].copy()
    preds = []
    n_targets = len(meas_cols)
    n_non_targets = window.shape[1] - n_targets
    max_steps = min(total_steps, len(X_test_scaled) - start_idx - input_len)

    for t in range(0, max_steps, output_len):
        x_hist = torch.tensor(window).unsqueeze(0).float().to(device)
        f_start = start_idx + input_len + t
        f_end = f_start + output_len
        
        if f_end > len(X_test_scaled): 
            break
        x_fref = torch.tensor(X_test_scaled[f_start:f_end, :n_non_targets]).unsqueeze(0).float().to(device)
        with torch.no_grad():
            yp = model(x_hist, x_fref, output_steps=output_len, teacher_forcing_ratio=0.0)
        yp = yp.squeeze(0).cpu().numpy() 
        preds.append(yp)
        window = np.roll(window, -output_len, axis=0)
        window[-output_len:, :n_non_targets] = X_test_scaled[f_start:f_end, :n_non_targets]
        window[-output_len:, n_non_targets:] = yp
    pred_scaled = np.concatenate(preds, axis=0)
    true_scaled = y_test_scaled[start_idx + input_len : start_idx + input_len + len(pred_scaled)]

    pred_inv = np.zeros_like(pred_scaled)
    true_inv = np.zeros_like(true_scaled)

    for i, col in enumerate(meas_cols):
        pred_inv[:, i] = target_scaler[col].inverse_transform(pred_scaled[:, i:i+1]).squeeze()
        true_inv[:, i] = target_scaler[col].inverse_transform(true_scaled[:, i:i+1]).squeeze()

    return true_inv, pred_inv

model.load_state_dict(torch.load("dynamic_response_lstm_final_recurrsive.pth"))
start_test = time.time()
true_inv, pred_inv = autoregressive_test_function(
    model,
    X_test_scaled,
    y_test_scaled,
    target_scaler,
    INPUT_STEPS,
    OUTPUT_STEPS,
    device,
    start_idx=0,
    total_steps=60000,
    feature_cols=feature_cols,
    meas_cols=meas_cols
)
end_test = time.time()
print(f"Total testing time: {end_test - start_test:.2f} seconds")

def compute_metrics(true, pred):
    mse = mean_squared_error(true, pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(true, pred)
    epsilon = 1e-9
    mape = np.mean(np.abs((true - pred) / ( true + epsilon)), axis=0) * 100
    overall_mape = np.mean(mape)
    # Overall WAPE (Standard percentage metric for power systems)
    overall_wape = (np.sum(np.abs(true - pred)) / np.sum(np.abs(true))) * 100
    target_titles = [
        "Active power (P) - (TSO-SPC)",
        "Reactive power (Q) - (TSO-SPC)"]
    scale = 1e6  # convert to M
    print("\n=== Overall Metrics ===")
    print(f"RMSE : {rmse/ scale:.4f}M, R2 : {r2:.4f}, WAPE : {overall_wape:.2f}%") #, MAPE : {overall_mape:.2f}%

    print("\n=== Per Target Metrics ===")
    for i, name in enumerate(target_titles):
        y_true = true[:, i]
        y_pred = pred[:, i]
        t_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        t_r2 = r2_score(y_true, y_pred)
        t_mae = mean_absolute_error(y_true, y_pred)
        t_wape = (np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))) * 100
        t_mape = mape[i]

        print(f"{name}:")
        print(f"  RMSE: {t_rmse/scale:.4f}M, MAE: {t_mae/scale:.4f}M,  R2: {t_r2:.4f}, WAPE: {t_wape:.2f}%")
        

compute_metrics(true_inv, pred_inv)
plot_predictions(true_inv, pred_inv, ["MW", "MVar"])


def save_trained_model(model, feature_scaler, target_scaler, save_dir="lstm_auto_recurrsive", model_name="spc_lstm_recurrsive"):
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, f"{model_name}.pth"))
    joblib.dump(feature_scaler, os.path.join(save_dir, f"{model_name}_feature_scaler.pkl"))
    joblib.dump(target_scaler, os.path.join(save_dir, f"{model_name}_target_scaler.pkl"))
    print(f"Model and scalers saved to '{save_dir}'")
save_trained_model(model, feature_scaler, target_scaler)

plot_training_history(train_losses, val_losses, patience=None, counter=None)
