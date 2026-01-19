import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Define the regression model (same as training)
class RegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[128, 64, 32]):
        super(RegressionModel, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim
        
        # Output layer (no activation for regression)
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class RegressionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Load data
import os
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(base_dir, "data", "extracted_indices.csv")
df = pd.read_csv(data_path)

Xcols = ['A00', 'A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08',
         'A09', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18',
         'A19', 'A20', 'A21', 'A22', 'A23', 'A24', 'A25', 'A26', 'A27', 'A28',
         'A29', 'A30', 'A31', 'A32', 'A33', 'A34', 'A35', 'A36', 'A37', 'A38',
         'A39', 'A40', 'A41', 'A42', 'A43', 'A44', 'A45', 'A46', 'A47', 'A48',
         'A49', 'A50', 'A51', 'A52', 'A53', 'A54', 'A55', 'A56', 'A57', 'A58',
         'A59', 'A60', 'A61', 'A62', 'A63']
ycols = ['NBR', 'NDMI', 'NDWI']

X = df[Xcols].values
y = df[ycols].values

# Remove NaN
mask = ~(np.isnan(X).any(axis=1) | np.isnan(y).any(axis=1))
X = X[mask]
y = y[mask]

# Split (same seed as training)
np.random.seed(42)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Load checkpoint
model_path = os.path.join(base_dir, "models", "best_model.pth")
checkpoint = torch.load(model_path, weights_only=False)
scaler_X = checkpoint['scaler_X']
scaler_y = checkpoint['scaler_y']

# Transform test data
X_test_scaled = scaler_X.transform(X_test)
y_test_scaled = scaler_y.transform(y_test)

# Create dataset
test_dataset = RegressionDataset(X_test_scaled, y_test_scaled)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RegressionModel(64, 3, hidden_dims=[128, 64, 32]).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Make predictions
all_predictions = []
all_targets = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        predictions = model(X_batch)
        all_predictions.append(predictions.cpu().numpy())
        all_targets.append(y_batch.numpy())

predictions = np.vstack(all_predictions)
targets = np.vstack(all_targets)

# Calculate test loss
test_loss = np.mean((predictions - targets) ** 2)

# Inverse transform
predictions_original = scaler_y.inverse_transform(predictions)
targets_original = scaler_y.inverse_transform(targets)

# Calculate baselines
y_train_original = scaler_y.inverse_transform(scaler_y.transform(y_train))
y_train_mean_original = y_train_original.mean(axis=0)
y_train_median_original = np.median(y_train_original, axis=0)

mean_predictions_original = np.tile(y_train_mean_original, (len(targets_original), 1))
median_predictions_original = np.tile(y_train_median_original, (len(targets_original), 1))

# Calculate baseline MSE
y_test_scaled_baseline = scaler_y.transform(y_test)
y_train_mean_scaled = scaler_y.transform(y_train).mean(axis=0)
y_train_median_scaled = scaler_y.transform(y_train)
y_train_median_scaled = np.median(y_train_median_scaled, axis=0)

mean_predictions_scaled = np.tile(y_train_mean_scaled, (len(y_test_scaled_baseline), 1))
median_predictions_scaled = np.tile(y_train_median_scaled, (len(y_test_scaled_baseline), 1))

mean_mse = np.mean((y_test_scaled_baseline - mean_predictions_scaled) ** 2)
median_mse = np.mean((y_test_scaled_baseline - median_predictions_scaled) ** 2)

# Print results
print("="*60)
print("MODEL PERFORMANCE")
print("="*60)
print(f"Test Loss (MSE): {test_loss:.6f}")
print(f"Test RMSE: {np.sqrt(test_loss):.6f}")

print(f"\n{'='*60}")
print("BASELINE COMPARISONS")
print(f"{'='*60}")

print(f"\nMean Baseline:")
print(f"  MSE:  {mean_mse:.6f}")
print(f"  RMSE: {np.sqrt(mean_mse):.6f}")

print(f"\nMedian Baseline:")
print(f"  MSE:  {median_mse:.6f}")
print(f"  RMSE: {np.sqrt(median_mse):.6f}")

improvement_over_mean = ((mean_mse - test_loss) / mean_mse) * 100
improvement_over_median = ((median_mse - test_loss) / median_mse) * 100

print(f"\nModel Improvement:")
print(f"  vs Mean Baseline:   {improvement_over_mean:+.2f}%")
print(f"  vs Median Baseline: {improvement_over_median:+.2f}%")

print("\n" + "="*80)
print("PER-TARGET PERFORMANCE")
print("="*80)
print(f"{'Target':<10} {'Metric':<8} {'Model':<12} {'Mean BL':<12} {'Median BL':<12} {'Improvement':<15}")
print("-" * 80)

for i, col in enumerate(ycols):
    # Model metrics
    model_mse = mean_squared_error(targets_original[:, i], predictions_original[:, i])
    model_mae = mean_absolute_error(targets_original[:, i], predictions_original[:, i])
    model_r2 = r2_score(targets_original[:, i], predictions_original[:, i])
    
    # Mean baseline
    mean_bl_mse = mean_squared_error(targets_original[:, i], mean_predictions_original[:, i])
    mean_bl_mae = mean_absolute_error(targets_original[:, i], mean_predictions_original[:, i])
    mean_bl_r2 = r2_score(targets_original[:, i], mean_predictions_original[:, i])
    
    # Median baseline
    median_bl_mse = mean_squared_error(targets_original[:, i], median_predictions_original[:, i])
    median_bl_mae = mean_absolute_error(targets_original[:, i], median_predictions_original[:, i])
    median_bl_r2 = r2_score(targets_original[:, i], median_predictions_original[:, i])
    
    # Improvements
    mse_imp_mean = ((mean_bl_mse - model_mse) / mean_bl_mse) * 100
    mse_imp_median = ((median_bl_mse - model_mse) / median_bl_mse) * 100
    
    print(f"{col:<10} {'MSE':<8} {model_mse:<12.6f} {mean_bl_mse:<12.6f} {median_bl_mse:<12.6f} {mse_imp_mean:>+6.2f}% / {mse_imp_median:>+6.2f}%")
    print(f"{'':<10} {'MAE':<8} {model_mae:<12.6f} {mean_bl_mae:<12.6f} {median_bl_mae:<12.6f}")
    print(f"{'':<10} {'R²':<8} {model_r2:<12.6f} {mean_bl_r2:<12.6f} {median_bl_r2:<12.6f}")
    print()

# Create visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, col in enumerate(ycols):
    axes[i].scatter(targets_original[:, i], predictions_original[:, i], alpha=0.5, s=1)
    axes[i].plot([targets_original[:, i].min(), targets_original[:, i].max()],
                 [targets_original[:, i].min(), targets_original[:, i].max()],
                 'r--', lw=2)
    axes[i].set_xlabel(f'Actual {col}')
    axes[i].set_ylabel(f'Predicted {col}')
    axes[i].set_title(f'{col} Predictions')
    axes[i].grid(True, alpha=0.3)
    
    r2 = r2_score(targets_original[:, i], predictions_original[:, i])
    rmse = np.sqrt(mean_squared_error(targets_original[:, i], predictions_original[:, i]))
    axes[i].text(0.05, 0.95, f'R² = {r2:.4f}\nRMSE = {rmse:.4f}', 
                transform=axes[i].transAxes,
                verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plot_path = os.path.join(base_dir, "plots", "model_predictions.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"\nPlot saved as '{plot_path}'")
print("\nEvaluation complete!")

