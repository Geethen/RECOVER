import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.base import BaseEstimator, RegressorMixin

# --- Model Architecture (Must match train_regression_model.py) ---

class EmbeddingAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.attn(x)

class GaussianNoise(nn.Module):
    def __init__(self, std=0.01):
        super().__init__()
        self.std = std
    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x) * self.std
            return x + noise
        return x

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(x + self.block(x))

class MultiHeadRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.noise = GaussianNoise(std=0.01)
        self.input_gate = EmbeddingAttention(input_dim)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            ResidualBlock(hidden_dim, dropout),
            ResidualBlock(hidden_dim, dropout),
            ResidualBlock(hidden_dim, dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU()
        )
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim // 2, hidden_dim // 4),
                nn.ReLU(),
                nn.Dropout(dropout / 2),
                nn.Linear(hidden_dim // 4, 1),
                nn.Tanh()
            ) for _ in range(output_dim)
        ])
    def forward(self, x):
        x = self.noise(x)
        gated_input = self.input_gate(x)
        shared_features = self.encoder(gated_input)
        return torch.cat([head(shared_features) for head in self.heads], dim=1)

# --- Sklearn Wrapper (Required for loading LO-CART pickles) ---
class PyTorchModelWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, model, device, target_idx=0):
        self.model = model
        self.device = device
        self.target_idx = target_idx
        self.is_fitted_ = True 

    def fit(self, X, y):
        return self

    def predict(self, X):
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            preds_scaled = self.model(X_tensor).cpu().numpy()
        return preds_scaled[:, self.target_idx].flatten()

def main():
    # --- Paths ---
    base_dir = r"C:\Users\coach\myfiles\postdoc2\code"
    model_checkpoint_path = os.path.join(base_dir, "models", "best_model.pth")
    locart_model_dir = os.path.join(base_dir, "models", "locart_comparison")
    input_csv = os.path.join(base_dir, "data", "extracted_indices.csv")
    labels_csv = os.path.join(base_dir, "data", "extracted_natural_labels.csv")
    output_csv = os.path.join(base_dir, "data", "reference_departure_with_intervals.csv")

    # --- Load Data ---
    print(f"Loading data from {input_csv} and labels from {labels_csv}...")
    df = pd.read_csv(input_csv)
    df_labels = pd.read_csv(labels_csv)
    
    # Merge to identify natural/transformed areas
    # Labels file has 'id', 'natural', 'geo'
    df = pd.merge(df, df_labels[['id', 'natural', 'geo']], on='id', how='left')
    
    Xcols = [f'A{i:02d}' for i in range(64)]
    ycols = ['NBR', 'NDMI', 'NDWI']
    
    X = df[Xcols].values
    # Handle NaNs
    if np.isnan(X).any():
        print("Warning: NaNs found in features. Filling with 0.")
        X = np.nan_to_num(X)

    # --- Load Main Model & Scalers ---
    print(f"Loading model checkpoint and scalers...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_checkpoint_path, map_location=device, weights_only=False)
    
    scaler_X = checkpoint['scaler_X']
    scaler_y = checkpoint['scaler_y']
    
    model = MultiHeadRegressionModel(len(Xcols), len(ycols))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Preprocess Inputs
    X_scaled = scaler_X.transform(X)

    # --- 1. Predict Reference States (Base Model) ---
    print("Predicting reference states...")
    X_tensor = torch.FloatTensor(X_scaled).to(device)
    with torch.no_grad():
        batch_size = 20000
        all_preds_scaled = []
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i : i + batch_size]
            preds = model(batch)
            all_preds_scaled.append(preds.cpu().numpy())
    
    y_ref_scaled = np.vstack(all_preds_scaled)
    y_ref = scaler_y.inverse_transform(y_ref_scaled)

    # --- 2. Predict Intervals (LO-CART Models) ---
    print("Predicting uncertainty intervals (LO-CART)...")
    intervals_result = {}
    
    for i, col in enumerate(ycols):
        # Load the best LO-CART model for this target
        # Based on comparison, LO-CART was best for all
        locart_path = os.path.join(locart_model_dir, f"best_model_{col}_LO-CART.pkl")
        if not os.path.exists(locart_path):
            print(f"Warning: LO-CART model for {col} not found at {locart_path}. Skipping intervals for this target.")
            continue
            
        with open(locart_path, 'rb') as f:
            locart_model = pickle.load(f)
        
        print(f"  Processing intervals for {col}...")
        # Predict intervals in batches
        all_intervals_scaled = []
        for start in range(0, len(X_scaled), batch_size):
            end = start + batch_size
            batch_X = X_scaled[start:end]
            ints = locart_model.predict(batch_X) # Returns [n, 2]
            all_intervals_scaled.append(ints)
        
        ints_scaled = np.vstack(all_intervals_scaled)
        
        # Inverse transform bounds
        # Lower bound
        lower_full = np.zeros_like(y_ref_scaled)
        lower_full[:, i] = ints_scaled[:, 0]
        lower_orig = scaler_y.inverse_transform(lower_full)[:, i]
        
        # Upper bound
        upper_full = np.zeros_like(y_ref_scaled)
        upper_full[:, i] = ints_scaled[:, 1]
        upper_orig = scaler_y.inverse_transform(upper_full)[:, i]
        
        intervals_result[col] = (lower_orig, upper_orig)

    # --- 3. Compile Results & Calculate Departures ---
    print("Finalizing results...")
    for i, col in enumerate(ycols):
        df[f'{col}_ref'] = y_ref[:, i]
        df[f'{col}_diff'] = df[col] - df[f'{col}_ref']
        
        if col in intervals_result:
            df[f'{col}_lower'] = intervals_result[col][0]
            df[f'{col}_upper'] = intervals_result[col][1]
            df[f'{col}_width'] = df[f'{col}_upper'] - df[f'{col}_lower']
            
    # --- Save ---
    print(f"Saving combined results to {output_csv}...")
    df.to_csv(output_csv, index=False)
    
    # Summary Table for console
    print("\nPrediction Summary (Means):")
    summary = []
    for col in ycols:
        summary.append({
            'Target': col,
            'Observed': df[col].mean(),
            'Reference': df[f'{col}_ref'].mean(),
            'Departure': df[f'{col}_diff'].abs().mean(),
            'Avg Width': df[f'{col}_width'].mean() if f'{col}_width' in df else np.nan
        })
    print(pd.DataFrame(summary))
    
    # Report summary by natural status
    print("\nSummary by Status (natural=1 vs natural=0):")
    status_summary = df.groupby('natural')[[f'{c}_width' for c in ycols if f'{c}_width' in df]].mean()
    print(status_summary)

    print("\nProcess Complete!")

if __name__ == "__main__":
    main()
