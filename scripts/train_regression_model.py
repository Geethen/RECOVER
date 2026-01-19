import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import matplotlib.pyplot as plt
import wandb

# Seed all random number generators
def seed_all(seed=42):
    """Set random seeds for reproducibility"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Make PyTorch deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Define the regression model
class RegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[128, 64, 32]):
        super(RegressionModel, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
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


# Early stopping class
class EarlyStopping:
    def __init__(self, patience=15, min_delta=0.0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        score = -val_loss if self.mode == 'min' else val_loss
        
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        
        return self.early_stop


# Custom Dataset class
class RegressionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Training function
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


# Validation function
def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


# Function to create prediction plots
def create_prediction_plots(model, dataloader, scaler_y, ycols, device, epoch, max_samples=5000):
    """Create prediction vs actual plots and return figure"""
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            predictions = model(X_batch)
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(y_batch.numpy())
            
            # Limit samples for faster plotting
            if sum(len(p) for p in all_predictions) >= max_samples:
                break
    
    predictions = np.vstack(all_predictions)[:max_samples]
    targets = np.vstack(all_targets)[:max_samples]
    
    # Inverse transform to original scale
    predictions_original = scaler_y.inverse_transform(predictions)
    targets_original = scaler_y.inverse_transform(targets)
    
    # Create plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, col in enumerate(ycols):
        axes[i].scatter(targets_original[:, i], predictions_original[:, i], alpha=0.3, s=5)
        
        # Add perfect prediction line
        min_val = min(targets_original[:, i].min(), predictions_original[:, i].min())
        max_val = max(targets_original[:, i].max(), predictions_original[:, i].max())
        axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        axes[i].set_xlabel(f'Actual {col}', fontsize=10)
        axes[i].set_ylabel(f'Predicted {col}', fontsize=10)
        axes[i].set_title(f'{col} - Epoch {epoch}', fontsize=12)
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()
        
        # Calculate and display metrics
        from sklearn.metrics import r2_score, mean_squared_error
        r2 = r2_score(targets_original[:, i], predictions_original[:, i])
        rmse = np.sqrt(mean_squared_error(targets_original[:, i], predictions_original[:, i]))
        
        # Show value ranges
        pred_min, pred_max = predictions_original[:, i].min(), predictions_original[:, i].max()
        actual_min, actual_max = targets_original[:, i].min(), targets_original[:, i].max()
        
        axes[i].text(0.05, 0.95, 
                    f'R² = {r2:.4f}\nRMSE = {rmse:.4f}\nPred range: [{pred_min:.3f}, {pred_max:.3f}]\nActual range: [{actual_min:.3f}, {actual_max:.3f}]',
                    transform=axes[i].transAxes,
                    verticalalignment='top',
                    fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    return fig


# Main training script
def main():
    # Set all random seeds
    seed_all(42)
    
    # Initialize wandb
    wandb.init(
        project="expected_Ref_Model",
        config={
            "architecture": "MLP",
            "hidden_dims": [128, 64, 32],
            "learning_rate": 0.001,
            "epochs": 100,
            "batch_size": 64,
            "optimizer": "Adam",
            "dropout": 0.2,
            "early_stopping_patience": 15,
        }
    )
    config = wandb.config
    
    # Load data
    import os
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "extracted_indices.csv")
    df = pd.read_csv(data_path)
    
    # Define feature and target columns
    Xcols = ['A00', 'A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08',
             'A09', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18',
             'A19', 'A20', 'A21', 'A22', 'A23', 'A24', 'A25', 'A26', 'A27', 'A28',
             'A29', 'A30', 'A31', 'A32', 'A33', 'A34', 'A35', 'A36', 'A37', 'A38',
             'A39', 'A40', 'A41', 'A42', 'A43', 'A44', 'A45', 'A46', 'A47', 'A48',
             'A49', 'A50', 'A51', 'A52', 'A53', 'A54', 'A55', 'A56', 'A57', 'A58',
             'A59', 'A60', 'A61', 'A62', 'A63']
    ycols = ['NBR', 'NDMI', 'NDWI']
    
    # Extract features and targets
    X = df[Xcols].values
    y = df[ycols].values
    
    # Remove any rows with NaN values
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y).any(axis=1))
    X = X[mask]
    y = y[mask]
    
    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    
    # Split data into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Standardize features
    scaler_X = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_val = scaler_X.transform(X_val)
    X_test = scaler_X.transform(X_test)
    
    # Optionally standardize targets (recommended for regression)
    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(y_train)
    y_val = scaler_y.transform(y_val)
    y_test = scaler_y.transform(y_test)
    
    # Create datasets and dataloaders
    train_dataset = RegressionDataset(X_train, y_train)
    val_dataset = RegressionDataset(X_val, y_val)
    test_dataset = RegressionDataset(X_test, y_test)
    
    batch_size = config.batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    wandb.config.update({"device": str(device)})
    
    # Initialize model
    input_dim = len(Xcols)
    output_dim = len(ycols)
    model = RegressionModel(input_dim, output_dim, hidden_dims=config.hidden_dims).to(device)
    
    print(f"\nModel architecture:\n{model}")
    wandb.watch(model, log="all", log_freq=100)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    early_stopping = EarlyStopping(patience=config.early_stopping_patience, min_delta=1e-6, mode='min')
    
    # Training loop
    num_epochs = config.epochs
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    print("\nStarting training...")
    for epoch in tqdm(range(num_epochs), desc="Training"):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate_epoch(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log to wandb
        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'learning_rate': current_lr,
        })
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(base_dir, "models", "best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'scaler_X': scaler_X,
                'scaler_y': scaler_y,
            }, best_model_path)
            wandb.save(best_model_path)
        
        if (epoch + 1) % 10 == 0:
            print(f"\nEpoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {current_lr:.6f}")
            
            # Create and log prediction plots to wandb
            fig = create_prediction_plots(model, val_loader, scaler_y, ycols, device, epoch+1)
            wandb.log({f"predictions_epoch_{epoch+1}": wandb.Image(fig)})
            plt.close(fig)
        
        # Early stopping check
        if early_stopping(val_loss):
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            wandb.log({'early_stopped_epoch': epoch})
            break
    
    # Load best model for testing
    checkpoint = torch.load(best_model_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate on test set
    test_loss = validate_epoch(model, test_loader, criterion, device)
    print(f"\n{'='*60}")
    print("MODEL PERFORMANCE")
    print(f"{'='*60}")
    print(f"Test Loss (MSE): {test_loss:.6f}")
    print(f"Test RMSE: {np.sqrt(test_loss):.6f}")
    
    # Calculate baseline metrics
    print(f"\n{'='*60}")
    print("BASELINE COMPARISONS")
    print(f"{'='*60}")
    
    # Mean baseline: predict the mean of training set for all samples
    y_train_mean = y_train.mean(axis=0)
    mean_predictions = np.tile(y_train_mean, (len(y_test), 1))
    mean_mse = np.mean((y_test - mean_predictions) ** 2)
    mean_rmse = np.sqrt(mean_mse)
    
    # Median baseline: predict the median of training set for all samples
    y_train_median = np.median(y_train, axis=0)
    median_predictions = np.tile(y_train_median, (len(y_test), 1))
    median_mse = np.mean((y_test - median_predictions) ** 2)
    median_rmse = np.sqrt(median_mse)
    
    print(f"\nMean Baseline:")
    print(f"  MSE:  {mean_mse:.6f}")
    print(f"  RMSE: {mean_rmse:.6f}")
    
    print(f"\nMedian Baseline:")
    print(f"  MSE:  {median_mse:.6f}")
    print(f"  RMSE: {median_rmse:.6f}")
    
    # Calculate improvement over baselines
    improvement_over_mean = ((mean_mse - test_loss) / mean_mse) * 100
    improvement_over_median = ((median_mse - test_loss) / median_mse) * 100
    
    print(f"\nModel Improvement:")
    print(f"  vs Mean Baseline:   {improvement_over_mean:+.2f}%")
    print(f"  vs Median Baseline: {improvement_over_median:+.2f}%")
    

    
    # Plot training history
    plt.grid(True)
    history_plot_path = os.path.join(base_dir, "plots", "training_history.png")
    plt.savefig(history_plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Make predictions on test set
    model.eval()
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
    
    # Inverse transform to get original scale
    predictions_original = scaler_y.inverse_transform(predictions)
    targets_original = scaler_y.inverse_transform(targets)
    
    # Calculate baselines in original scale
    y_train_original = scaler_y.inverse_transform(y_train)
    y_train_mean_original = y_train_original.mean(axis=0)
    y_train_median_original = np.median(y_train_original, axis=0)
    
    mean_predictions_original = np.tile(y_train_mean_original, (len(targets_original), 1))
    median_predictions_original = np.tile(y_train_median_original, (len(targets_original), 1))
    
    # Per-target metrics
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    print("\nPer-Target Performance:")
    print(f"{'Target':<10} {'Metric':<8} {'Model':<12} {'Mean BL':<12} {'Median BL':<12} {'Improvement':<12}")
    print("-" * 80)
    
    # Dictionary to store metrics for wandb
    wandb_metrics = {}
    
    for i, col in enumerate(ycols):
        # Model metrics
        model_mse = mean_squared_error(targets_original[:, i], predictions_original[:, i])
        model_mae = mean_absolute_error(targets_original[:, i], predictions_original[:, i])
        model_r2 = r2_score(targets_original[:, i], predictions_original[:, i])
        
        # Mean baseline metrics
        mean_bl_mse = mean_squared_error(targets_original[:, i], mean_predictions_original[:, i])
        mean_bl_mae = mean_absolute_error(targets_original[:, i], mean_predictions_original[:, i])
        mean_bl_r2 = r2_score(targets_original[:, i], mean_predictions_original[:, i])
        
        # Median baseline metrics
        median_bl_mse = mean_squared_error(targets_original[:, i], median_predictions_original[:, i])
        median_bl_mae = mean_absolute_error(targets_original[:, i], median_predictions_original[:, i])
        median_bl_r2 = r2_score(targets_original[:, i], median_predictions_original[:, i])
        
        # Calculate improvements
        mse_imp_mean = ((mean_bl_mse - model_mse) / mean_bl_mse) * 100
        mse_imp_median = ((median_bl_mse - model_mse) / median_bl_mse) * 100
        
        # Log to wandb
        wandb_metrics[f'{col}_mse'] = model_mse
        wandb_metrics[f'{col}_mae'] = model_mae
        wandb_metrics[f'{col}_r2'] = model_r2
        wandb_metrics[f'{col}_improvement_vs_mean'] = mse_imp_mean
        wandb_metrics[f'{col}_improvement_vs_median'] = mse_imp_median
        
        print(f"{col:<10} {'MSE':<8} {model_mse:<12.6f} {mean_bl_mse:<12.6f} {median_bl_mse:<12.6f} {mse_imp_mean:>+6.2f}% / {mse_imp_median:>+6.2f}%")
        print(f"{'':<10} {'MAE':<8} {model_mae:<12.6f} {mean_bl_mae:<12.6f} {median_bl_mae:<12.6f}")
        print(f"{'':<10} {'R²':<8} {model_r2:<12.6f} {mean_bl_r2:<12.6f} {median_bl_r2:<12.6f}")
        print()
    
    # Log final metrics to wandb
    wandb.log(wandb_metrics)
    
    # Plot predictions vs actual for each target
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, col in enumerate(ycols):
        axes[i].scatter(targets_original[:, i], predictions_original[:, i], alpha=0.5)
        axes[i].plot([targets_original[:, i].min(), targets_original[:, i].max()],
                     [targets_original[:, i].min(), targets_original[:, i].max()],
                     'r--', lw=2)
        axes[i].set_xlabel(f'Actual {col}')
        axes[i].set_ylabel(f'Predicted {col}')
        axes[i].set_title(f'{col} Predictions')
        axes[i].grid(True)
        
        # Calculate R² score
        r2 = r2_score(targets_original[:, i], predictions_original[:, i])
        axes[i].text(0.05, 0.95, f'R² = {r2:.4f}', transform=axes[i].transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    pred_plot_path = os.path.join(base_dir, "plots", "predictions_vs_actual.png")
    plt.savefig(pred_plot_path, dpi=300, bbox_inches='tight')
    wandb.log({"predictions_vs_actual": wandb.Image(pred_plot_path)})
    plt.show()
    
    print("\nTraining complete! Model saved as 'best_model.pth'")
    print("Plots saved as 'training_history.png' and 'predictions_vs_actual.png'")
    
    # Finish wandb run
    wandb.finish()


if __name__ == "__main__":
    main()
