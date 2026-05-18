import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np
import torch.nn as nn

# ============================================================================
# PLOTTING FUNCTIONS FOR ENTROPY AND LOSS TRACKING
# ============================================================================

def plot_training_curves(entropy_history, has_entropy_reg, save_path=None):
    """
    Plot training curves including loss, entropy, and reconstruction.
    
    Args:
        entropy_history: Dictionary containing training history
        has_entropy_reg: Boolean indicating if entropy regularization was used
        save_path: Optional path to save the figure
    """
    if has_entropy_reg and len(entropy_history.get('train_entropy', [])) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Training and Validation Loss
        ax1 = axes[0, 0]
        ax1.plot(entropy_history['epochs'], entropy_history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        ax1.plot(entropy_history['epochs'], entropy_history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training and Validation Loss', fontsize=14)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Entropy over time (should DECREASE)
        ax2 = axes[0, 1]
        ax2.plot(entropy_history['epochs'], entropy_history['train_entropy'], 'g-', label='Train Entropy', linewidth=2)
        ax2.plot(entropy_history['epochs'], entropy_history['val_entropy'], 'orange', label='Val Entropy', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Entropy (per sample avg)', fontsize=12)
        ax2.set_title('Entropy Minimization Progress ↓', fontsize=14)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        # Highlight if entropy is decreasing
        if entropy_history['train_entropy'][0] > entropy_history['train_entropy'][-1]:
            ax2.text(0.02, 0.95, '✅ Entropy Decreasing (Good!)', transform=ax2.transAxes, 
                    fontsize=11, color='green', bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen'))
        else:
            ax2.text(0.02, 0.95, '⚠️ Entropy NOT Decreasing', transform=ax2.transAxes, 
                    fontsize=11, color='red', bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral'))
        
        # Plot 3: Reconstruction Loss
        ax3 = axes[1, 0]
        ax3.plot(entropy_history['epochs'], entropy_history['reconstruction_loss'], 'purple', linewidth=2)
        ax3.set_xlabel('Epoch', fontsize=12)
        ax3.set_ylabel('Reconstruction Loss (L1)', fontsize=12)
        ax3.set_title('Reconstruction Loss', fontsize=14)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Entropy vs Loss relationship
        ax4 = axes[1, 1]
        scatter = ax4.scatter(entropy_history['train_entropy'], entropy_history['train_loss'], 
                              c=entropy_history['epochs'], cmap='viridis', s=50, alpha=0.7)
        ax4.set_xlabel('Entropy', fontsize=12)
        ax4.set_ylabel('Train Loss', fontsize=12)
        ax4.set_title('Entropy vs Loss (color = epoch)', fontsize=14)
        cbar = plt.colorbar(scatter, ax=ax4)
        cbar.set_label('Epoch', fontsize=10)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
    else:
        # Plot just loss curves for non-entropy models
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1 = axes[0]
        ax1.plot(entropy_history['epochs'], entropy_history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        ax1.plot(entropy_history['epochs'], entropy_history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training and Validation Loss', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[1]
        ax2.plot(entropy_history['epochs'], entropy_history['reconstruction_loss'], 'purple', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Reconstruction Loss', fontsize=12)
        ax2.set_title('Reconstruction Loss', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def print_entropy_summary(entropy_history):
    """
    Print summary statistics for entropy minimization.
    
    Args:
        entropy_history: Dictionary containing training history
    """
    print("\n" + "="*70)
    print("📊 ENTROPY MINIMIZATION SUMMARY")
    print("="*70)
    print(f"Initial Train Entropy: {entropy_history['train_entropy'][0]:.6f}")
    print(f"Final Train Entropy:   {entropy_history['train_entropy'][-1]:.6f}")
    print(f"Entropy Reduction:     {entropy_history['train_entropy'][0] - entropy_history['train_entropy'][-1]:.6f}")
    print(f"Reduction Percentage:  {(1 - entropy_history['train_entropy'][-1]/entropy_history['train_entropy'][0])*100:.2f}%")
    
    if entropy_history['train_entropy'][-1] < 0.01:
        print("\n⚠️ WARNING: Very low entropy detected (< 0.01)!")
        print("   Possible rule collapse. Consider REDUCING entropy_coef.")
    elif entropy_history['train_entropy'][-1] > 1.0:
        print("\n⚠️ WARNING: High entropy detected (> 1.0)!")
        print("   Rules may not be specializing. Consider INCREASING entropy_coef.")
    else:
        print(f"\n✅ Healthy entropy level: {entropy_history['train_entropy'][-1]:.6f}")
        print("   Rules are specializing without collapsing!")
    
    if entropy_history['train_entropy'][-1] < entropy_history['train_entropy'][0]:
        print("\n✅ SUCCESS: Entropy decreased as desired (minimization working)")
    else:
        print("\n❌ ISSUE: Entropy increased! Check entropy_coef sign (should be ADDED to loss)")
    
    print("="*70)


def plot_entropy_distribution(model, data_loader, device, save_path=None):
    """
    Plot distribution of per-sample entropy.
    
    Args:
        model: Trained model
        data_loader: DataLoader for evaluation
        device: Device to run on
        save_path: Optional path to save the figure
    """
    model.eval()
    all_entropies = []
    
    with torch.no_grad():
        for batch_X, _ in data_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            if len(outputs) == 3:
                _, _, entropy_penalty = outputs
            else:
                # For regular GIFTSHIFT, compute entropy manually
                y = model.encode(batch_X)
                entropy_per_rule = -y * torch.log(y + 1e-10)
                entropy_penalty = entropy_per_rule.sum(dim=1).mean()
            all_entropies.append(entropy_penalty.item())
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(all_entropies, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
    ax.set_xlabel('Per-Sample Entropy', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'Entropy Distribution (Mean: {np.mean(all_entropies):.4f}, Std: {np.std(all_entropies):.4f})', fontsize=14)
    ax.axvline(np.mean(all_entropies), color='red', linestyle='--', label=f'Mean: {np.mean(all_entropies):.4f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return all_entropies


def plot_rule_activation_heatmap(model, data_loader, device, save_path=None, max_samples=100):
    """
    Plot rule activation patterns across samples.
    
    Args:
        model: Trained model
        data_loader: DataLoader for evaluation
        device: Device to run on
        save_path: Optional path to save the figure
        max_samples: Maximum number of samples to display
    """
    model.eval()
    all_activations = []
    
    with torch.no_grad():
        for batch_X, _ in data_loader:
            batch_X = batch_X.to(device)
            y = model.encode(batch_X)
            if model.rules_count > 1:
                y = torch.nn.functional.normalize(y, p=1, dim=1)
            all_activations.append(y.cpu().numpy())
    
    activations = np.concatenate(all_activations, axis=0)
    activations = activations[:max_samples]  # Limit for visualization
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Heatmap of rule activations
    ax1 = axes[0]
    im = ax1.imshow(activations.T, aspect='auto', cmap='viridis', origin='lower')
    ax1.set_xlabel('Sample Index', fontsize=12)
    ax1.set_ylabel('Rule Number', fontsize=12)
    ax1.set_title(f'Rule Activation Heatmap (first {len(activations)} samples)', fontsize=14)
    plt.colorbar(im, ax=ax1, label='Activation Strength')
    
    # Bar plot of average rule usage
    ax2 = axes[1]
    mean_activations = activations.mean(axis=0)
    std_activations = activations.std(axis=0)
    rules = np.arange(1, len(mean_activations) + 1)
    ax2.bar(rules, mean_activations, yerr=std_activations, capsize=5, 
            alpha=0.7, color='steelblue', edgecolor='black')
    ax2.set_xlabel('Rule Number', fontsize=12)
    ax2.set_ylabel('Average Activation', fontsize=12)
    ax2.set_title('Average Rule Usage Across All Samples', fontsize=14)
    ax2.set_xticks(rules)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return activations


def plot_entropy_vs_coefficient(results_df, save_path=None):
    """
    Plot entropy vs different entropy_coefficient values.
    Useful for hyperparameter tuning.
    
    Args:
        results_df: DataFrame with columns ['entropy_coef', 'final_entropy', 'test_accuracy']
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Entropy vs Coefficient
    ax1 = axes[0]
    ax1.plot(results_df['entropy_coef'], results_df['final_entropy'], 'o-', color='blue', linewidth=2, markersize=8)
    ax1.set_xlabel('Entropy Coefficient', fontsize=12)
    ax1.set_ylabel('Final Entropy', fontsize=12)
    ax1.set_title('Effect of Coefficient on Final Entropy', fontsize=14)
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Accuracy vs Coefficient
    ax2 = axes[1]
    ax2.plot(results_df['entropy_coef'], results_df['test_accuracy'], 'o-', color='green', linewidth=2, markersize=8)
    ax2.set_xlabel('Entropy Coefficient', fontsize=12)
    ax2.set_ylabel('Test Accuracy', fontsize=12)
    ax2.set_title('Effect of Coefficient on Test Accuracy', fontsize=14)
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()