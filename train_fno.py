import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model.FNO import FNO2d
from utils.load import NpyReader
from utils.Loss import LpLoss
from timeit import default_timer
import os
import matplotlib.pyplot as plt
from datetime import datetime
from torch.amp import autocast, GradScaler  # Mixed precision training


def count_params(model):
    """Calculate the number of trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate_model(model, data_loader, criterion, device):
    """Evaluate model performance on a given data loader"""
    model.eval()
    total_loss = 0.0
    count = 0
    
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            # Fix: Model output is (batch, height, width, 1), need to squeeze the last dimension
            out = model(x).squeeze(-1)  # Remove the last channel dimension
            loss = criterion.abs(out.view(x.shape[0], -1), y.view(x.shape[0], -1))
            total_loss += loss.item()
            count += 1
    
    return total_loss / count if count > 0 else float('inf')


def generate_plots(train_losses, checkpoint_dir):
    """Generate training charts - Optimized version"""
    
    # 🔧 Optimization 1: For large datasets, only plot a sampled subset
    max_points = 1000  # Plot 1000 points at most
    if len(train_losses) > max_points:
        indices = np.linspace(0, len(train_losses)-1, max_points, dtype=int)
        epochs_list = [i+1 for i in indices]
        losses_to_plot = [train_losses[i] for i in indices]
    else:
        epochs_list = list(range(1, len(train_losses) + 1))
        losses_to_plot = train_losses
    
    plt.figure(figsize=(10, 5))
    
    # Loss curve - Log scale
    plt.subplot(1, 2, 1)
    plt.plot(epochs_list, losses_to_plot, 'b-', label='Train Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.title(f'Training Progress (Epoch {len(train_losses)})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Loss curve - Linear scale
    plt.subplot(1, 2, 2)
    plt.plot(epochs_list, losses_to_plot, 'b-', label='Train Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (linear scale)')
    plt.title('Training Progress (Linear)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    # 🔧 Optimization 2: Reduce DPI to decrease file size and saving time
    plt.savefig(f'{checkpoint_dir}/training_progress.png', dpi=100)
    plt.close()


def save_prediction_visualizations(true_data, pred_data, output_dir, n_vis):
    """Save visualization images of prediction results"""
    
    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    print(f"Generating visualizations for {n_vis} samples...")
    
    for i in range(n_vis):
        if i % 10 == 0:
            print(f"  Completed {i}/{n_vis} samples")
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Use range of True values as a unified standard for colorbar
        vmin_unified = true_data[i].min()
        vmax_unified = true_data[i].max()
        
        # Ground Truth
        im1 = axes[0].imshow(true_data[i], cmap='viridis', origin='lower', 
                            vmin=vmin_unified, vmax=vmax_unified)
        axes[0].set_title(f'True (Sample {i+1})')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('y')
        plt.colorbar(im1, ax=axes[0])
        
        # Prediction (using same color range)
        im2 = axes[1].imshow(pred_data[i], cmap='viridis', origin='lower',
                            vmin=vmin_unified, vmax=vmax_unified)
        axes[1].set_title(f'Predicted (Sample {i+1})')
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('y')
        plt.colorbar(im2, ax=axes[1])
        
        # Error (using independent range)
        error = np.abs(true_data[i] - pred_data[i])
        im3 = axes[2].imshow(error, cmap='Reds', origin='lower')
        axes[2].set_title(f'Absolute Error (Sample {i+1})')
        axes[2].set_xlabel('x')
        axes[2].set_ylabel('y')
        plt.colorbar(im3, ax=axes[2])
        
        plt.tight_layout()
        
        # Save image
        save_path = os.path.join(vis_dir, f'prediction_sample_{i+1:03d}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"All visualization images saved to: {vis_dir}")
    return vis_dir


def evaluate_test_set(model, test_data_path, sigma_mean, sigma_std, 
                      g_lift_mean, g_lift_std, criterion, device, checkpoint_dir, N_grid, n_vis=50):
    """Test set evaluation"""
    
    print("\nEvaluating test set...")
    
    # Load test data - if a directory is passed, look for test_data.npy within it
    if os.path.isdir(test_data_path):
        test_data_file = os.path.join(test_data_path, 'test_data.npy')
        if not os.path.exists(test_data_file):
            # If test_data.npy is not in checkpoint_dir, use default path
            test_data_file = './data/test_data.npy'
    else:
        test_data_file = test_data_path
    
    test_data = np.load(test_data_file, allow_pickle=True).item()
    sigma_test = torch.from_numpy(test_data['sigma']).float()
    u_test = torch.from_numpy(test_data['u']).float()
    g_lift_test = torch.from_numpy(test_data['g_lift']).float()
    
    # Process test data (no sampling)
    sigma_test = (sigma_test - sigma_mean) / sigma_std
    g_lift_test = (g_lift_test - g_lift_mean) / g_lift_std

    x_test = torch.stack([sigma_test, g_lift_test], dim=-1)

    # Load best model
    best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, weights_only=False))
        print(f"Loaded best model from: {best_model_path}")
    
    model.eval()

    # Test set evaluation
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_test, u_test), 
        batch_size=16, shuffle=False
    )
    
    test_loss = evaluate_model(model, test_loader, criterion, device)
    
    # Generate predictions
    ntest = len(u_test)  # Use actual test data size
    predict = np.zeros((ntest, N_grid+1, N_grid+1))
    true = np.zeros((ntest, N_grid+1, N_grid+1))

    with torch.no_grad():
        batch_idx = 0
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x).reshape(x.shape[0], N_grid+1, N_grid+1)
            
            batch_size_current = x.shape[0]
            end_idx = min(batch_idx + batch_size_current, ntest)
            actual_size = end_idx - batch_idx
            
            predict[batch_idx:end_idx] = out[:actual_size].cpu().numpy()
            true[batch_idx:end_idx] = y[:actual_size].cpu().numpy()
            
            batch_idx += actual_size
            if batch_idx >= ntest:
                break

    # Error calculation
    abs_errors = np.abs(predict - true)
    rel_errors = abs_errors / (np.abs(true) + 1e-8)
    
    print(f"Test Results:")
    print(f"   Test Loss: {test_loss:.6f}")
    print(f"   Mean Absolute Error: {np.mean(abs_errors):.6f}")
    print(f"   Mean Relative Error: {np.mean(rel_errors[rel_errors < 10]):.6f}")
    
    # Save prediction results
    predict_data = {
        'true': true, 
        'predict': predict, 
        'abs_errors': abs_errors,
        'test_loss': test_loss
    }
    
    # Create test_results subfolder under checkpoint_dir
    output_dir = os.path.join(checkpoint_dir, 'test_results')
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'predict_data_final.npy')
    
    np.save(save_path, predict_data)
    print(f"Final prediction data saved to: {save_path}")
    
    # Generate visualization images - use n_vis parameter to control quantity
    vis_dir = save_prediction_visualizations(true, predict, output_dir, min(n_vis, ntest))
    
    return test_loss


def main(model, TRAIN_PATH, TEST_PATH, N_grid, epochs, batch_size, learning_rate, mode='train', test_dir=None, n_vis=50):

    if mode == 'train':
        # Create saving directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_dir = f"/amax/haibo/Diffusion4IP/work_dir/fno/{timestamp}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"Checkpoint directory: {checkpoint_dir}")

        # Load training data
        reader = NpyReader(TRAIN_PATH)
        u_train = reader.read_field('u')
        sigma_train = reader.read_field('sigma')
        g_lift_train = reader.read_field('g_lift')  

        print(f"Data Statistics:")
        print(f"u_train: [{u_train.min():.4f}, {u_train.max():.4f}], Mean: {u_train.mean():.4f}")
        print(f"sigma_train: [{sigma_train.min():.4f}, {sigma_train.max():.4f}], Mean: {sigma_train.mean():.4f}")
        print(f"g_lift_train: [{g_lift_train.min():.4f}, {g_lift_train.max():.4f}], Mean: {g_lift_train.mean():.4f}")

        # Load pre-processed normalization parameters
        norm_params = np.load('./data/normalization_params.npy', allow_pickle=True).item()
        sigma_mean = norm_params['sigma']['mean']
        sigma_std = norm_params['sigma']['std']
        g_lift_mean = norm_params['g_lift']['mean']
        g_lift_std = norm_params['g_lift']['std']
        
        print(f"Loaded Normalization Parameters:")
        print(f"sigma: mean={sigma_mean:.6f}, std={sigma_std:.6f}")
        print(f"g_lift: mean={g_lift_mean:.6f}, std={g_lift_std:.6f}")

        # Normalization
        sigma_train = (sigma_train - sigma_mean) / sigma_std
        g_lift_train = (g_lift_train - g_lift_mean) / g_lift_std

        # Prepare data
        x_train = torch.stack([sigma_train, g_lift_train], dim=-1)
        y_train = torch.from_numpy(u_train.cpu().numpy() if isinstance(u_train, torch.Tensor) else u_train).float()

        print(f"Training data: {len(x_train)} samples")

        # Data loader - optimize memory usage
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x_train, y_train), 
            batch_size=batch_size, shuffle=True
        )

        # Optimization strategy
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        
        # For FNO models, mixed precision is disabled because complex operations are not supported
        # scaler = GradScaler('cuda')
        use_mixed_precision = False

        criterion = LpLoss(size_average=True)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Training history
        train_losses = []

        print(f"\nStarting training - Model parameters: {count_params(model):,}")

        best_train_loss = float('inf')

        for ep in range(epochs):
            # Training phase
            model.train()
            epoch_train_loss = 0.0
            train_count = 0

            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()

                # Mixed precision training
                # Disable autocast for FNO as complex operations are unsupported
                with autocast('cuda', enabled=False):
                    # Fix: Model output is (batch, height, width, 1), squeeze the last dimension
                    out = model(x).squeeze(-1)  # Remove the last channel dimension
                    
                    # Debugging info
                    if ep == 0 and train_count == 0:
                        print(f"Input x shape: {x.shape}")
                        print(f"Target y shape: {y.shape}")
                        print(f"Model out shape: {out.shape}")
                        print(f"Shape after out.view: {out.view(x.shape[0], -1).shape}")
                        print(f"Shape after y.view: {y.view(x.shape[0], -1).shape}")
                    
                    loss = criterion.abs(out.view(x.shape[0], -1), y.view(x.shape[0], -1))
                
                # Standard backpropagation (no mixed precision)
                loss.backward()
                optimizer.step()

                epoch_train_loss += loss.item()
                train_count += 1
                
            # Log loss
            avg_train_loss = epoch_train_loss / train_count if train_count > 0 else float('inf')
            train_losses.append(avg_train_loss)
            
            # Update learning rate
            scheduler.step()

            # Save best model
            status = ""
            if avg_train_loss < best_train_loss:
                best_train_loss = avg_train_loss
                best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
                torch.save(model.state_dict(), best_model_path)
                status = f"best ↓ ({avg_train_loss:.6f})"
            
            # Print progress
            current_lr = optimizer.param_groups[0]['lr']
                
            print(f'Epoch {ep+1:3d}/{epochs} | Train: {avg_train_loss:.6f} | '
                    f'LR: {current_lr:.7f} | {status}')
            
            # Generate charts and save model - 🔧 Optimization: Change to every 50 epochs
            if (ep + 1) % 50 == 0 or ep == epochs - 1:
                generate_plots(train_losses, checkpoint_dir)
                # Save model for current epoch
                epoch_model_path = os.path.join(checkpoint_dir, f'model_epoch_{ep+1}.pth')
                torch.save(model.state_dict(), epoch_model_path)
    
        # Return results
        results = {
            'train_losses': train_losses,
            'best_train_loss': best_train_loss,
            'epochs_trained': len(train_losses)
        }

        return results

    if mode == 'test':

        print("=== Test Mode: Evaluating on test set only ===")
        
        # Load normalization parameters
        norm_params_path = './data/normalization_params.npy'
        if not os.path.exists(norm_params_path):
            raise FileNotFoundError(f"Normalization parameters file not found: {norm_params_path}")
        
        norm_params = np.load(norm_params_path, allow_pickle=True).item()
        sigma_mean = norm_params['sigma']['mean']
        sigma_std = norm_params['sigma']['std']
        g_lift_mean = norm_params['g_lift']['mean']
        g_lift_std = norm_params['g_lift']['std']
        
        print(f"Loaded Normalization Parameters:")
        print(f"  sigma: mean={sigma_mean:.6f}, std={sigma_std:.6f}")
        print(f"  g_lift: mean={g_lift_mean:.6f}, std={g_lift_std:.6f}")
        
        # Determine model directory
        if test_dir is not None:
            # Use specified test directory
            latest_checkpoint_dir = test_dir
        else:
            # Find the latest model file
            work_dir = "/amax/haibo/Diffusion4IP/work_dir/fno"
            if not os.path.exists(work_dir):
                raise FileNotFoundError(f"Working directory does not exist: {work_dir}")
            
            # Get the latest checkpoint directory
            checkpoint_dirs = [d for d in os.listdir(work_dir) if os.path.isdir(os.path.join(work_dir, d))]
            if not checkpoint_dirs:
                raise FileNotFoundError(f"No checkpoint directories found in: {work_dir}")
            
            latest_checkpoint_dir = os.path.join(work_dir, sorted(checkpoint_dirs)[-1])
        
        best_model_path = os.path.join(latest_checkpoint_dir, 'best_model.pth')
        
        if not os.path.exists(best_model_path):
            raise FileNotFoundError(f"Best model file not found: {best_model_path}")
        
        print(f"Loading model from: {best_model_path}")
        model.load_state_dict(torch.load(best_model_path, weights_only=False))
        
        # Evaluate on test set
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        criterion = LpLoss(size_average=True)
        
        test_loss = evaluate_test_set(model, TEST_PATH, 
                                      sigma_mean, sigma_std, g_lift_mean, g_lift_std, 
                                      criterion, device, latest_checkpoint_dir, N_grid, n_vis=n_vis)
        
        return {
            'test_loss': test_loss,
            'mode': 'test'
        }


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    parser = argparse.ArgumentParser(description='Optimized FNO training with faster convergence')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='Run mode: train or test')
    parser.add_argument('--N_grid', type=int, default=128)
    parser.add_argument('--ntrain', type=int, default=2000)
    parser.add_argument('--ntest', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)  # Increased batch size for efficiency
    parser.add_argument('--epochs', type=int, default=1500)   # Reduced epochs due to easier convergence
    parser.add_argument('--learning_rate', type=float, default=0.001)  # Slightly increased LR for faster convergence
    parser.add_argument('--width', type=int, default=20)  
    parser.add_argument('--modes1', type=int, default=8)  
    parser.add_argument('--modes2', type=int, default=8)  
    parser.add_argument('--layers', type=int, default=4)  

    parser.add_argument('--test_dir', type=str, help='Path to model directory in test mode')
    parser.add_argument('--n_vis', type=int, default=50, help='Number of sample visualizations to generate in test mode')
    args = parser.parse_args()

    print(f"Training Arguments: {args}")

    # Use optimized FNO configuration - faster training speed
    print("=== Using optimized FNO configuration ===")
    model = FNO2d(modes1=args.modes1, modes2=args.modes2, width=args.width, layers=args.layers, in_channels=2, 
                         include_grid=True).to(device)
    print(f"Optimized model parameters: {count_params(model):,}")
    print("Expected improvements: ~70% parameter reduction, ~2-3x training acceleration")
    
    # Data paths
    TRAIN_PATH = './data/train_data.npy'
    TEST_PATH = './data/test_data.npy'
    
    results = main(
        model, TRAIN_PATH, TEST_PATH, 
        N_grid=args.N_grid, epochs=args.epochs,
        batch_size=args.batch_size, learning_rate=args.learning_rate,
        mode=args.mode, test_dir=args.test_dir, n_vis=args.n_vis
    )