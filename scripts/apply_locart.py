import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, RegressorMixin
from clover.locart import LocartSplit
from clover.scores import RegressionScore
import pickle
from tqdm import tqdm

# --- Model Architecture ---
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

# --- Sklearn Wrapper for PyTorch Model ---
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

def load_data(base_dir, natural_data_path, full_data_path, labels_data_path):
    print("--- Data Loading & Splitting ---")
    
    # 1. Load Natural Data (Used for Training & LOCART)
    print(f"Loading natural data from {natural_data_path}...")
    df_nat = pd.read_csv(natural_data_path)
    Xcols = [f'A{i:02d}' for i in range(64)]
    ycols = ['NBR', 'NDMI', 'NDWI']
    
    X_nat = df_nat[Xcols].values
    y_nat = df_nat[ycols].values
    
    # Clean NaNs (Match training logic)
    mask = ~(np.isnan(X_nat).any(axis=1) | np.isnan(y_nat).any(axis=1))
    X_nat = X_nat[mask]
    y_nat = y_nat[mask]
    
    # Replicate Training Splits
    X_train_orig, X_temp_orig, y_train_orig, y_temp_orig = train_test_split(X_nat, y_nat, test_size=0.3, random_state=42)
    X_val_orig, X_test_orig, y_val_orig, y_test_orig = train_test_split(X_temp_orig, y_temp_orig, test_size=0.5, random_state=42)
    
    print(f"Total Natural Samples: {len(X_nat)}")
    print(f"Original Test Set (Available for LOCART): {len(X_test_orig)}")

    # 2. Load Transformed Data (For Width prediction only)
    print("\nLoading full dataset to identify transformed areas...")
    df_full = pd.read_csv(full_data_path)
    df_labels = pd.read_csv(labels_data_path)
    
    if 'geo' in df_full.columns and 'geo' in df_labels.columns:
        df_merged = pd.merge(df_full, df_labels[['geo', 'natural']], on='geo', how='inner')
    else:
        df_merged = df_full

    if 'natural' not in df_merged.columns:
        raise ValueError("Critical: 'natural' column missing. Cannot identify transformed areas.")

    # Filter for Transformed areas (natural == 0)
    df_transformed = df_merged[df_merged['natural'] == 0]
    X_trans = df_transformed[Xcols].values
    X_trans = np.nan_to_num(X_trans)
    
    print(f"Transformed Data (For Width Est.): {len(X_trans)}")

    return X_test_orig, y_test_orig, X_trans

def evaluate_method(method_name, cart_type, weighting, X_cal_sc, y_cal_sc_target, X_eval_sc, y_eval_sc_target, wrapper, alpha=0.1):
    """
    Trains and evaluates a specific LO-CART variant.
    """
    print(f"    Evaluating {method_name}...")
    
    # Init LO-CART Model with specific params
    # cart_type='CART' or 'RF'
    # weighting=True (A-LOCART) or False (LOCART)
    locart = LocartSplit(RegressionScore, wrapper, alpha=alpha, is_fitted=True, cart_type=cart_type, weighting=weighting)
    
    # Fit/Calibrate
    try:
        locart.fit(X_cal_sc, y_cal_sc_target)
        locart.calib(X_cal_sc, y_cal_sc_target)
    except AttributeError:
        # If calib is integrated
        locart.fit(X_cal_sc, y_cal_sc_target)
        
    # Eval on Natural
    intervals = locart.predict(X_eval_sc)
    lower = intervals[:, 0]
    upper = intervals[:, 1]
    
    cov = np.mean((y_eval_sc_target >= lower) & (y_eval_sc_target <= upper))
    width = np.mean(upper - lower)
    
    return cov, width, locart

def predict_transformed_width(locart_model, X_trans, scaler_X, batch_size=50000):
    """
    Calculates average width on transformed data in batches.
    """
    widths = []
    for b_start in range(0, len(X_trans), batch_size):
        b_end = min(b_start + batch_size, len(X_trans))
        X_batch = X_trans[b_start:b_end]
        if np.isnan(X_batch).any():
            X_batch = np.nan_to_num(X_batch)
        
        X_batch_sc = scaler_X.transform(X_batch)
        int_batch = locart_model.predict(X_batch_sc)
        widths.append(int_batch[:, 1] - int_batch[:, 0])
        
    return np.mean(np.concatenate(widths))

def main():
    # --- Paths ---
    base_dir = r"C:\Users\coach\myfiles\postdoc2\code"
    model_path = os.path.join(base_dir, "models", "best_model.pth")
    natural_data_path = os.path.join(base_dir, "data", "dfsubsetNatural.csv")
    full_data_path = os.path.join(base_dir, "data", "extracted_indices.csv")
    labels_data_path = os.path.join(base_dir, "data", "extracted_natural_labels.csv")
    output_dir = os.path.join(base_dir, "models", "locart_comparison")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(base_dir, "plots", "locart_comparison"), exist_ok=True)

    # --- Data Prep ---
    X_nat_unseen, y_nat_unseen, X_trans = load_data(base_dir, natural_data_path, full_data_path, labels_data_path)
    
    # Split Natural Unseen into LOCART Calibration and LOCART Evaluation Sets
    X_cal, X_eval, y_cal, y_eval = train_test_split(X_nat_unseen, y_nat_unseen, test_size=0.5, random_state=42)
    
    # --- Load Model & Scaling ---
    print(f"\nLoading model checkpoint from {model_path}...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    scaler_X = checkpoint['scaler_X']
    scaler_y = checkpoint['scaler_y']
    
    input_dim = X_cal.shape[1]
    ycols = ['NBR', 'NDMI', 'NDWI']
    output_dim = len(ycols)
    
    model = MultiHeadRegressionModel(input_dim, output_dim)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Scale Data
    X_cal_scaled = scaler_X.transform(X_cal)
    X_eval_scaled = scaler_X.transform(X_eval)
    y_cal_scaled = scaler_y.transform(y_cal)
    y_eval_scaled = scaler_y.transform(y_eval)

    # --- Define Methods to Compare ---
    methods = {
        # Name: (cart_type, weighting)
        "LO-CART":    ("CART", False),
        "A-LOCART":   ("CART", True),
        "LO-FOREST":  ("RF", False), 
        "A-LOFOREST": ("RF", True)
    }

    alpha = 0.1 # 90% confidence
    all_results = []

    for i, col in enumerate(ycols):
        print(f"\n--- Processing Target: {col} ---")
        wrapper = PyTorchModelWrapper(model, device, target_idx=i)
        
        target_results = []
        
        for method_name, (c_type, w_bool) in methods.items():
            # 1. Evaluate on Natural Data
            cov, width, trained_model = evaluate_method(
                method_name, c_type, w_bool, 
                X_cal_scaled, y_cal_scaled[:, i], 
                X_eval_scaled, y_eval_scaled[:, i], 
                wrapper, alpha
            )
            
            # 2. Estimate Transformed Width
            print(f"    Estimating transformed width for {method_name}...")
            trans_width = predict_transformed_width(trained_model, X_trans, scaler_X)
            
            print(f"      -> Cov: {cov*100:.2f}%, Nat Width: {width:.4f}, Trans Width: {trans_width:.4f}")
            
            res_entry = {
                'Target': col,
                'Method': method_name,
                'Nat_Coverage': cov,
                'Nat_Width': width,
                'Trans_Width': trans_width,
                'Model_Obj': trained_model
            }
            all_results.append(res_entry)
            target_results.append(res_entry)

        # 3. Select Best Method for this Target
        # Criteria: Must meet coverage >= (1-alpha) AND minimize Nat_Width
        # Allowing small tolerance e.g. 0.89 for 0.90 target? Strict for now.
        valid_methods = [r for r in target_results if r['Nat_Coverage'] >= (1.0 - alpha - 0.01)] # 1% tolerance
        
        if not valid_methods:
            print("    Warning: No method met strict coverage. Selecting highest coverage.")
            best_method = max(target_results, key=lambda x: x['Nat_Coverage'])
        else:
            # Select narrowest width among valid
            best_method = min(valid_methods, key=lambda x: x['Nat_Width'])
            
        print(f"    BEST SELECTED: {best_method['Method']} (Width: {best_method['Nat_Width']:.4f})")
        
        # 4. Save Best Model & Plot
        best_model = best_method['Model_Obj']
        with open(os.path.join(output_dir, f"best_model_{col}_{best_method['Method']}.pkl"), 'wb') as f:
            pickle.dump(best_model, f)
            
        # Plot Best Method on Natural Eval (100 samples)
        intervals = best_model.predict(X_eval_scaled)
        plot_idx = np.random.choice(len(y_eval), 100, replace=False)
        
        # Inverse & Plot Logic
        y_real = scaler_y.inverse_transform(y_eval_scaled)[:, i][plot_idx]
        y_pred = wrapper.predict(X_eval_scaled[plot_idx])
        # Need to reshape pred for inverse
        p_full = np.zeros((100, output_dim)); p_full[:, i] = y_pred
        p_real = scaler_y.inverse_transform(p_full)[:, i]
        
        # Inverse Interval
        int_sub = intervals[plot_idx]
        l_full = np.zeros((100, output_dim)); l_full[:, i] = int_sub[:, 0]
        u_full = np.zeros((100, output_dim)); u_full[:, i] = int_sub[:, 1]
        l_real = scaler_y.inverse_transform(l_full)[:, i]
        u_real = scaler_y.inverse_transform(u_full)[:, i]

        # Calc Errors
        yerr_l = np.maximum(0, p_real - l_real)
        yerr_u = np.maximum(0, u_real - p_real)

        plt.figure(figsize=(12, 6))
        x_ax = np.arange(100)
        plt.errorbar(x_ax, p_real, yerr=[yerr_l, yerr_u], fmt='none', ecolor='gray', alpha=0.6, label='90% PI')
        plt.scatter(x_ax, p_real, color='red', s=25, label='Predicted')
        plt.scatter(x_ax, y_real, color='green', s=25, label='Actual')
        plt.title(f"{col} - Best Method: {best_method['Method']}\nCov: {best_method['Nat_Coverage']*100:.1f}%, Width: {best_method['Nat_Width']:.4f}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(base_dir, "plots", "locart_comparison", f"best_{col}.png"))
        plt.close()

    # Save Comparison CSV
    df_res = pd.DataFrame(all_results).drop(columns=['Model_Obj'])
    print("\n--- FINAL COMPARISON ---")
    print(df_res)
    df_res.to_csv(os.path.join(output_dir, "method_comparison.csv"), index=False)

if __name__ == "__main__":
    main()
