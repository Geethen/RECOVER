import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm import tqdm
import matplotlib.pyplot as plt
import wandb
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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

def init_weights(m):
    """Weight initialization for better convergence (Kaiming for ReLU)"""
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

class EmbeddingAttention(nn.Module):
    """Learns which parts of the 64D embedding are most important for the tasks"""
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.attn(x)


class GaussianNoise(nn.Module):
    """Adds noise to embeddings to improve robustness during training"""
    def __init__(self, std=0.01):
        super().__init__()
        self.std = std

    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x) * self.std
            return x + noise
        return x


class MultiTaskUncertaintyLoss(nn.Module):
    """Learns the relative weights of multiple tasks using homoscedastic uncertainty"""
    def __init__(self, num_tasks=3):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, predictions, targets):
        loss = 0
        for i in range(len(self.log_vars)):
            precision = torch.exp(-self.log_vars[i])
            diff = (predictions[:, i] - targets[:, i])**2
            loss += precision * diff.mean() + self.log_vars[i]
        return loss.mean()


class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim), # LayerNorm often superior for embeddings
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
        
        # Robustness: Add Small Jitter to Embeddings
        self.noise = GaussianNoise(std=0.01)
        
        # Feature Focusing: Weight dimensions of the embedding
        self.input_gate = EmbeddingAttention(input_dim)
        
        # Shared Context Encoder: Learns general environmental patterns
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            ResidualBlock(hidden_dim, dropout),
            ResidualBlock(hidden_dim, dropout),
            ResidualBlock(hidden_dim, dropout), # Added depth
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU()
        )
        
        # Individual heads for each target (NBR, NDMI, NDWI)
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
def train_epoch(model, dataloader, criterion, optimizer, device, clip_grad=1.0):
    model.train()
    total_loss = 0
    total_mse = 0
    
    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        mse = torch.nn.functional.mse_loss(predictions, y_batch)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping to ensure stability and prevent extreme updates
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        
        optimizer.step()
        
        total_loss += loss.item()
        total_mse += mse.item()
    
    return total_loss / len(dataloader), total_mse / len(dataloader)


# Validation function
def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_mse = 0
    
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            mse = torch.nn.functional.mse_loss(predictions, y_batch)
            total_loss += loss.item()
            total_mse += mse.item()
    
    return total_loss / len(dataloader), total_mse / len(dataloader)


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
        r2_val = r2_score(targets_original[:, i], predictions_original[:, i])
        rmse_val = np.sqrt(mean_squared_error(targets_original[:, i], predictions_original[:, i]))
        
        # Show value ranges
        pred_min, pred_max = predictions_original[:, i].min(), predictions_original[:, i].max()
        actual_min, actual_max = targets_original[:, i].min(), targets_original[:, i].max()
        
        axes[i].text(0.05, 0.95, 
                    f'R² = {r2_val:.4f}\nRMSE = {rmse_val:.4f}\nPred range: [{pred_min:.3f}, {pred_max:.3f}]\nActual range: [{actual_min:.3f}, {actual_max:.3f}]',
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
    
    # Data source definition
    dataset_name = "dfsubsetNatural.parquet"
    
    # Initialize wandb
    wandb.init(
        project="expected_Ref_Model",
        config={
            "architecture": "UncertaintyAttentionalResidualMLP",
            "hidden_dim": 256,
            "learning_rate": 0.001,
            "epochs": 150,
            "batch_size": 64,
            "optimizer": "AdamW",
            "dropout": 0.2,
            "weight_decay": 1e-2, # AdamW typically uses higher weight decay (e.g., 0.01)
            "input_noise_std": 0.01,
            "early_stopping_patience": 25,
            "data_source": dataset_name,
            "output_activation": "tanh",
            "encoder_norm": "LayerNorm",
            "scheduler": "CosineAnnealingWarmRestarts",
            "warmup_epochs": 5,
            "warmup_start_lr_factor": 0.01, # Start LR at 1% of base LR
        }
    )
    config = wandb.config
    
    # Load data
    import os
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Ensure necessary directories exist
    os.makedirs(os.path.join(base_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "plots"), exist_ok=True)
    
    data_path = os.path.join(base_dir, "data", dataset_name)
    df = pd.read_parquet(data_path)
    
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
    
    # Use MinMaxScaler for targets to match Tanh output range [-1, 1]
    # Scaling to [-0.95, 0.95] helps avoid Tanh saturation at the edges
    scaler_y = MinMaxScaler(feature_range=(-0.95, 0.95))
    y_train = scaler_y.fit_transform(y_train)
    y_val = scaler_y.transform(y_val)
    y_test = scaler_y.transform(y_test)
    
    # Create datasets and dataloaders
    train_dataset = RegressionDataset(X_train, y_train)
    val_dataset = RegressionDataset(X_val, y_val)
    test_dataset = RegressionDataset(X_test, y_test)
    
    batch_size = config.batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    wandb.config.update({"device": str(device)})
    
    input_dim = len(Xcols)
    output_dim = len(ycols)
    
    # Initialize Multi-Head model and apply weights
    model = MultiHeadRegressionModel(
        input_dim, 
        output_dim, 
        hidden_dim=config.hidden_dim, 
        dropout=config.dropout
    ).to(device)
    model.apply(init_weights)
    
    print(f"\nModel architecture:\n{model}")
    wandb.watch(model, log="all", log_freq=100)
    
    # Multi-Task Loss and AdamW Optimizer
    criterion = MultiTaskUncertaintyLoss(num_tasks=output_dim).to(device)
    # Separate params to avoid weight decay on uncertainty log_vars (which should move freely)
    optimizer = optim.AdamW([
        {'params': model.parameters(), 'weight_decay': config.weight_decay},
        {'params': criterion.parameters(), 'weight_decay': 0.0}
    ], lr=config.learning_rate)
    
    # Modern Scheduler: Cosine Annealing with Warm Restarts
    # This helps the model jump out of local minima
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=10,    # First restart after 10 epochs
        T_mult=2,  # Double the cycle length each time
        eta_min=1e-6
    )
    early_stopping = EarlyStopping(patience=config.early_stopping_patience, min_delta=1e-6, mode='min')
    
    # Training loop
    num_epochs = config.epochs
    best_val_mse = float('inf')
    train_losses = []
    val_losses = []
    
    print("\nStarting training...")
    for epoch in tqdm(range(num_epochs), desc="Training"):
        train_loss, train_mse = train_epoch(model, train_loader, criterion, optimizer, device, clip_grad=1.0)
        val_loss, val_mse = validate_epoch(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log to wandb
        log_payload = {
            'epoch': epoch,
            'total_train_loss': train_loss,
            'train_mse': train_mse,
            'val_loss': val_loss,
            'val_mse': val_mse,
            'learning_rate': current_lr,
        }
        
        # Log learned weights (variances) for each task
        with torch.no_grad():
            for i, col in enumerate(ycols):
                log_payload[f'task_weight_{col}'] = torch.exp(-criterion.log_vars[i]).item()
        
        wandb.log(log_payload)
        
        # Learning rate scheduling
        # Learning rate scheduling (CosineAnnealingWarmRestarts uses internal epoch count)
        scheduler.step()
        
        # Save best model based on MSE
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_model_path = os.path.join(base_dir, "models", "best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_mse': val_mse,
                'scaler_X': scaler_X,
                'scaler_y': scaler_y,
                'data_source': config.data_source,
            }, best_model_path)
            wandb.save(best_model_path)
        
        if (epoch + 1) % 10 == 0:
            print(f"\nEpoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Train MSE: {train_mse:.6f}, Val MSE: {val_mse:.6f}, LR: {current_lr:.6f}")
            
            # Create and log prediction plots to wandb
            # Using a buffer to avoid temp file issues on Windows with wandb.Image(fig)
            from io import BytesIO
            from PIL import Image
            fig = create_prediction_plots(model, val_loader, scaler_y, ycols, device, epoch+1)
            buf = BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            img = Image.open(buf)
            wandb.log({f"predictions_epoch_{epoch+1}": wandb.Image(img)})
            plt.close(fig)
        
        # Early stopping check based on MSE
        if early_stopping(val_mse):
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            wandb.log({'early_stopped_epoch': epoch})
            break
    
    # Load best model for testing
    checkpoint = torch.load(best_model_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate on test set
    test_loss, test_mse = validate_epoch(model, test_loader, criterion, device)
    print(f"\n{'='*60}")
    print("MODEL PERFORMANCE (DEBUG-FIX-V1)")
    print(f"{'='*60}")
    print(f"Test Loss (Uncertainty): {test_loss:.6f}")
    print(f"Test MSE: {test_mse:.6f}")
    print(f"Test RMSE: {np.sqrt(test_mse):.6f}")
    
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
    
    # Linear Probe Baseline (OLS)
    print(f"\n{'='*60}")
    print("LINEAR PROBE BASELINE (OLS)")
    print(f"{'='*60}")
    linear_probe = LinearRegression()
    linear_probe.fit(X_train, y_train)
    y_test_linear = linear_probe.predict(X_test)
    linear_mse = mean_squared_error(y_test, y_test_linear)
    linear_rmse = np.sqrt(linear_mse)
    print(f"Linear Probe MSE: {linear_mse:.6f}")
    print(f"Linear Probe RMSE: {linear_rmse:.6f}")
    
    # Calculate improvement over Baselines (using MSE)
    improvement_over_mean = ((mean_mse - test_mse) / mean_mse) * 100
    improvement_over_median = ((median_mse - test_mse) / median_mse) * 100
    improvement_over_linear = ((linear_mse - test_mse) / linear_mse) * 100
    
    print(f"\nModel Improvement (based on MSE) [FIXED]:")
    print(f"  vs Mean Baseline:   {improvement_over_mean:+.2f}%")
    print(f"  vs Median Baseline: {improvement_over_median:+.2f}%")
    print(f"  vs Linear Probe:    {improvement_over_linear:+.2f}%")
    
    wandb.run.summary["improvement_vs_linear_probe"] = improvement_over_linear
    

    
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
    print("\nPer-Target Performance:")
    print(f"{'Target':<10} {'Metric':<8} {'Model':<12} {'Mean BL':<12} {'Median BL':<12} {'Improvement':<12}")
    print("-" * 80)
    
    # List for summary table
    table_data = []
    
    # Calculate linear probe baseline in original scale (using pre-calculated multi-target results)
    y_test_linear_original = scaler_y.inverse_transform(y_test_linear)
    
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
        
        # Per-target improvements
        mse_imp_mean = ((mean_bl_mse - model_mse) / mean_bl_mse) * 100
        mse_imp_median = ((median_bl_mse - model_mse) / median_bl_mse) * 100
        
        # Per-target Linear Probe performance
        lp_mse = mean_squared_error(targets_original[:, i], y_test_linear_original[:, i])
        mse_imp_linear = ((lp_mse - model_mse) / lp_mse) * 100

        # Add to summary table data
        table_data.append([
            col,
            model_mse,
            model_mae,
            model_r2,
            mse_imp_mean,
            mse_imp_median,
            mse_imp_linear
        ])
        
        # Log to wandb summary (this makes them appear in the W&B Run Table/Grid)
        wandb.run.summary[f"test_{col}_mse"] = model_mse
        wandb.run.summary[f"test_{col}_mae"] = model_mae
        wandb.run.summary[f"test_{col}_r2"] = model_r2
        wandb.run.summary[f"test_{col}_imp_vs_linear"] = mse_imp_linear
        
        print(f"{col:<10} {'MSE':<8} {model_mse:<12.6f} {mean_bl_mse:<12.6f} {lp_mse:<12.6f} {mse_imp_linear:>+6.2f}%")
        print(f"{'':<10} {'MAE':<8} {model_mae:<12.6f} {mean_bl_mae:<12.6f} {median_bl_mae:<12.6f}")
        print(f"{'':<10} {'R²':<8} {model_r2:<12.6f} {mean_bl_r2:<12.6f} {median_bl_r2:<12.6f}")
        print()
    
    # Log Media Table to W&B
    table_columns = ["Target", "MSE", "MAE", "R2", "Imp. vs Mean %", "Imp. vs Median %", "Imp. vs Linear %"]
    metrics_table = wandb.Table(columns=table_columns, data=table_data)
    wandb.log({"final_performance_summary": metrics_table})
    
    # Small pause to ensure wandb syncs the final heavy logs
    import time
    time.sleep(2)
    
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
