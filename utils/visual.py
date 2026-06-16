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


# ============================================================================

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def plot_antecedent_rules(
    model,
    image_shape=(8, 8),
    figsize=(14, 6),
    cmap="viridis"
):
    """
    Visualize antecedent fuzzy laws/rules as images.

    Shows:
    - Gaussian means
    - Literal gates
    - Combination weights

    Args:
        model: trained GIFTSHIFT/GIFT/ANFIS-like model
        image_shape: shape of digit image
    """

    means = _to_numpy(model.mean)
    literal = _to_numpy(torch.sigmoid(model.literal))
    weights = _to_numpy(torch.sigmoid(model.comb_weight))

    rules = means.shape[1]

    fig, axes = plt.subplots(
        rules,
        3,
        figsize=figsize
    )

    if rules == 1:
        axes = np.expand_dims(axes, axis=0)

    for r in range(rules):

        mean_img = means[:, r].reshape(image_shape)
        literal_img = literal[:, r].reshape(image_shape)
        weight_img = weights[:, r].reshape(image_shape)

        ax = axes[r, 0]
        im = ax.imshow(mean_img, cmap=cmap)
        ax.set_title(f"Rule {r} Mean")
        ax.axis("off")

        ax = axes[r, 1]
        im = ax.imshow(literal_img, cmap=cmap)
        ax.set_title(f"Rule {r} Literal")
        ax.axis("off")

        ax = axes[r, 2]
        im = ax.imshow(weight_img, cmap=cmap)
        ax.set_title(f"Rule {r} Comb Weight")
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def plot_consequent_rules(
    model,
    image_shape=(8, 8),
    class_index=0,
    figsize=(12, 4),
    cmap="coolwarm"
):
    """
    Visualize consequent TSK laws as images.

    Each rule has a local slope vector.
    We reshape it into digit form.

    Args:
        class_index: which output class to visualize
    """

    slopes = _to_numpy(model.local_slopes)

    rules = slopes.shape[0]

    fig, axes = plt.subplots(
        1,
        rules,
        figsize=figsize
    )

    if rules == 1:
        axes = [axes]

    vmax = np.abs(slopes).max()

    for r in range(rules):

        slope_img = slopes[r, :, class_index].reshape(image_shape)

        ax = axes[r]

        im = ax.imshow(
            slope_img,
            cmap=cmap,
            vmin=-vmax,
            vmax=vmax
        )

        ax.set_title(f"Rule {r}")
        ax.axis("off")

    fig.colorbar(im, ax=axes)
    plt.suptitle(f"Consequent Laws - Class {class_index}")
    plt.tight_layout()
    plt.show()


def plot_rule_activation(
    model,
    sample,
    image_shape=(8, 8),
    cmap="magma"
):
    """
    Show how strongly each rule activates for a sample.
    """

    model.eval()

    with torch.no_grad():

        if sample.ndim == 1:
            sample = sample.unsqueeze(0)

        activations = model.encode(sample)

        if model.rules_count > 1:
            activations = F.normalize(activations, p=1, dim=1)

    activations = activations[0].cpu().numpy()

    fig, axes = plt.subplots(
        1,
        len(activations) + 1,
        figsize=(3 * (len(activations) + 1), 3)
    )

    sample_img = sample[0].cpu().numpy().reshape(image_shape)

    axes[0].imshow(sample_img, cmap="gray")
    axes[0].set_title("Input")
    axes[0].axis("off")

    for i, a in enumerate(activations):

        img = np.ones(image_shape) * a

        axes[i + 1].imshow(img, cmap=cmap, vmin=0, vmax=1)
        axes[i + 1].set_title(f"Rule {i}\n{a:.3f}")
        axes[i + 1].axis("off")

    plt.tight_layout()
    plt.show()


def plot_full_rule_interpretation(
    model,
    sample=None,
    image_shape=(8, 8),
    class_index=0
):
    """
    Full interpretability visualization.
    """

    plot_antecedent_rules(
        model,
        image_shape=image_shape
    )

    plot_consequent_rules(
        model,
        image_shape=image_shape,
        class_index=class_index
    )

    if sample is not None:
        plot_rule_activation(
            model,
            sample,
            image_shape=image_shape
        )




# model/law_visualization.py

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from typing import Optional, List, Tuple, Dict, Any
import seaborn as sns

def plot_fuzzy_laws_complete(
    model,
    feature_names: Optional[List[str]] = None,
    class_names: Optional[List[str]] = None,
    rule_indices: Optional[List[int]] = None,
    figsize: Tuple[int, int] = (20, 12),
    save_path: Optional[str] = None
):
    """
    Plot complete fuzzy laws including both antecedent (membership functions)
    and consequent (TSK or Mamdani output) for each rule.
    
    Args:
        model: Trained GIFTSHIFT, GIFTSHIFTENTROPY, or LITANFIS model
        feature_names: Names of input features
        class_names: Names of output classes
        rule_indices: Which rules to plot (if None, plots all rules)
        figsize: Figure size
        save_path: Path to save the figure
    """
    model.eval()
    
    with torch.no_grad():
        # Extract model parameters
        if hasattr(model, 'mean'):
            # GIFTSHIFT family
            means = model.mean.cpu().numpy()  # (in_features, rules)
            stds = torch.nn.functional.softplus(model.std).cpu().numpy()
            literal_probs = torch.sigmoid(model.literal).cpu().numpy()
            comb_weights = torch.sigmoid(model.comb_weight).cpu().numpy()
            slopes = model.sigmoid_slope.cpu().numpy()
            has_tsk = hasattr(model, 'local_slopes')
            
            if has_tsk:
                tsk_slopes = model.local_slopes.cpu().numpy()  # (rules, in_features, out_features)
                tsk_biases = model.local_biases.cpu().numpy()  # (rules, out_features)
        else:
            raise ValueError("Model type not supported for law visualization")
        
        in_features = model.in_features
        rules = model.rules_count
        out_features = model.out_features if hasattr(model, 'out_features') else 1
        
        # Determine which rules to plot (limit to 4 for readability)
        if rule_indices is None:
            rule_indices = list(range(min(rules, 4)))
        else:
            rule_indices = [r for r in rule_indices if r < rules]
        
        n_rules = len(rule_indices)
        
        # Create feature and class names
        if feature_names is None:
            feature_names = [f'x{i}' for i in range(min(in_features, 6))]
        else:
            feature_names = feature_names[:min(in_features, 6)]
        
        if class_names is None:
            if out_features == 1:
                class_names = ['Output']
            else:
                class_names = [f'Class {i}' for i in range(out_features)]
        else:
            class_names = class_names[:out_features]
        
        n_features = len(feature_names)
        
        # Create figure with GridSpec for better layout
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(n_rules, 3, figure=fig, width_ratios=[2, 2, 1.5], hspace=0.3, wspace=0.3)
        
        for rule_idx, rule_id in enumerate(rule_indices):
            # === ANTECEDENT PART (IF) ===
            ax_antecedent = fig.add_subplot(gs[rule_idx, 0])
            
            # Plot membership functions for selected features
            x_range = np.linspace(-3, 3, 500)
            
            for feat_idx in range(n_features):
                mu = means[feat_idx, rule_id]
                sigma = stds[feat_idx, rule_id]
                literal_prob = literal_probs[feat_idx, rule_id]
                comb_weight = comb_weights[feat_idx, rule_id]
                slope = slopes[feat_idx, rule_id]
                
                # Calculate combined membership
                gaussian = np.exp(-((x_range - mu) ** 2) / (2 * sigma ** 2))
                literal_mf = gaussian * literal_prob + (1 - gaussian) * (1 - literal_prob)
                sigmoidal = 1 / (1 + np.exp(-(x_range - mu) * slope))
                temp_mf = sigmoidal * literal_prob + (1 - sigmoidal) * (1 - literal_prob)  # Simplified
                combined = comb_weight * literal_mf + (1 - comb_weight) * temp_mf
                
                # Plot with color gradient
                color = plt.cm.viridis(feat_idx / n_features)
                ax_antecedent.plot(x_range, combined, color=color, linewidth=2.5, 
                                 label=feature_names[feat_idx])
                
                # Mark the center
                ax_antecedent.axvline(x=mu, color=color, linestyle=':', alpha=0.5, linewidth=1)
            
            ax_antecedent.set_ylim([-0.05, 1.05])
            ax_antecedent.set_xlim([-3, 3])
            ax_antecedent.set_xlabel('Input Value (normalized)', fontsize=9)
            ax_antecedent.set_ylabel('Membership', fontsize=9)
            ax_antecedent.set_title(f'Rule {rule_id} - Antecedent (IF)', fontsize=11, fontweight='bold')
            ax_antecedent.grid(True, alpha=0.3)
            ax_antecedent.legend(loc='upper right', fontsize=7, ncol=2)
            
            # === CONSEQUENT PART (THEN) ===
            ax_consequent = fig.add_subplot(gs[rule_idx, 1])
            
            if has_tsk:
                # TSK-style consequent: linear function
                biases = tsk_biases[rule_id]
                slopes_rule = tsk_slopes[rule_id]  # (in_features, out_features)
                
                # Create a 2D grid for visualization if 1D output
                if out_features == 1:
                    # Plot linear function for each input feature
                    for feat_idx in range(n_features):
                        mu = means[feat_idx, rule_id]
                        slope_val = slopes_rule[feat_idx, 0] if slopes_rule.ndim > 1 else slopes_rule[feat_idx]
                        
                        x_vals = np.linspace(-3, 3, 100)
                        y_vals = biases[0] + slope_val * (x_vals - mu)
                        
                        color = plt.cm.plasma(feat_idx / n_features)
                        ax_consequent.plot(x_vals, y_vals, color=color, linewidth=2, 
                                         label=f'{feature_names[feat_idx]} contribution')
                    
                    # Add bias line
                    ax_consequent.axhline(y=biases[0], color='red', linestyle='--', 
                                         alpha=0.7, label=f'Bias = {biases[0]:.3f}')
                    
                    ax_consequent.set_xlabel('Input Value', fontsize=9)
                    ax_consequent.set_ylabel('Output Contribution', fontsize=9)
                    ax_consequent.set_title(f'Rule {rule_id} - Consequent (THEN)', fontsize=11, fontweight='bold')
                    ax_consequent.grid(True, alpha=0.3)
                    ax_consequent.legend(loc='best', fontsize=7)
                    
                else:
                    # Multiple outputs - show bar chart of biases
                    x_pos = np.arange(len(class_names))
                    ax_consequent.bar(x_pos, biases, color=plt.cm.Set3(np.linspace(0, 1, len(class_names))))
                    ax_consequent.set_xticks(x_pos)
                    ax_consequent.set_xticklabels(class_names, rotation=45, ha='right', fontsize=8)
                    ax_consequent.set_ylabel('Bias Weight', fontsize=9)
                    ax_consequent.set_title(f'Rule {rule_id} - Output Biases', fontsize=11, fontweight='bold')
                    ax_consequent.grid(True, alpha=0.3, axis='y')
                    
                    # Add value labels on bars
                    for i, v in enumerate(biases):
                        ax_consequent.text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=8)
            
            else:
                # Mamdani-style consequent
                ax_consequent.text(0.5, 0.5, 'Mamdani Output\n(see class mapping)', 
                                 ha='center', va='center', transform=ax_consequent.transAxes,
                                 fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat'))
                ax_consequent.set_title(f'Rule {rule_id} - Consequent (THEN)', fontsize=11, fontweight='bold')
                ax_consequent.axis('off')
            
            # === RULE SUMMARY ===
            ax_summary = fig.add_subplot(gs[rule_idx, 2])
            ax_summary.axis('off')
            
            # Create a text box with rule summary
            summary_text = f"RULE {rule_id}\n\n"
            summary_text += "IF:\n"
            for feat_idx in range(min(n_features, 4)):
                mu = means[feat_idx, rule_id]
                lit_prob = literal_probs[feat_idx, rule_id]
                comb_weight = comb_weights[feat_idx, rule_id]
                
                if comb_weight > 0.6:
                    mf_type = "≈"
                elif comb_weight < 0.4:
                    mf_type = "≠" if lit_prob > 0.5 else ">" if lit_prob > 0.5 else "<"
                else:
                    mf_type = "~"
                
                summary_text += f"  {feature_names[feat_idx]} {mf_type} {mu:.2f}\n"
            
            if has_tsk and out_features == 1:
                summary_text += f"\nTHEN:\n"
                summary_text += f"  y = {tsk_biases[rule_id, 0]:.3f}"
                for feat_idx in range(min(n_features, 3)):
                    slope_val = tsk_slopes[rule_id, feat_idx, 0] if tsk_slopes.ndim > 2 else tsk_slopes[rule_id, feat_idx]
                    if abs(slope_val) > 0.01:
                        summary_text += f"\n    + {slope_val:.3f}·({feature_names[feat_idx]}-{means[feat_idx, rule_id]:.2f})"
            
            # Add the text box
            ax_summary.text(0.1, 0.95, summary_text, transform=ax_summary.transAxes,
                          fontsize=9, verticalalignment='top', fontfamily='monospace',
                          bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.suptitle('Complete Fuzzy Laws: Antecedent (IF) and Consequent (THEN) Parts', 
                    fontsize=14, fontweight='bold', y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Complete fuzzy laws saved to {save_path}")
        else:
            plt.show()

def plot_law_activation_map(
    model,
    dataloader,
    feature_names: Optional[List[str]] = None,
    class_names: Optional[List[str]] = None,
    num_samples: int = 200,
    figsize: Tuple[int, int] = (16, 10),
    save_path: Optional[str] = None
):
    """
    Plot how fuzzy laws activate for different input samples.
    Shows which rules fire and how they contribute to final output.
    
    Args:
        model: Trained GIFTSHIFT model
        dataloader: DataLoader with samples
        feature_names: Names of features
        class_names: Names of classes
        num_samples: Number of samples to plot
        figsize: Figure size
        save_path: Path to save figure
    """
    model.eval()
    
    with torch.no_grad():
        # Collect data
        all_activations = []
        all_outputs = []
        all_labels = []
        
        for batch_X, batch_y in dataloader:
            # Get rule activations
            if hasattr(model, 'encode'):
                activations = model.encode(batch_X)
                if model.rules_count > 1:
                    activations = torch.nn.functional.normalize(activations, p=1, dim=1)
            else:
                raise ValueError("Model must have encode method")
            
            # Get outputs
            outputs = model(batch_X)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            all_activations.append(activations.cpu().numpy())
            all_outputs.append(outputs.cpu().numpy())
            all_labels.append(batch_y.cpu().numpy())
            
            if len(np.vstack(all_activations)) >= num_samples:
                break
        
        activations = np.vstack(all_activations)[:num_samples]
        outputs = np.vstack(all_outputs)[:num_samples]
        labels = np.concatenate(all_labels)[:num_samples]
        
        # Create figure
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Plot 1: Rule activation heatmap
        ax1 = fig.add_subplot(gs[0, :2])
        im = ax1.imshow(activations.T, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
        ax1.set_xlabel('Sample Index', fontsize=11)
        ax1.set_ylabel('Rule Index', fontsize=11)
        ax1.set_title('Rule Activation Patterns (Which laws fire?)', fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax1, label='Activation Strength')
        
        # Add class labels as colored background
        unique_labels = np.unique(labels)
        label_colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(labels):
            color_idx = np.where(unique_labels == label)[0][0]
            ax1.add_patch(Rectangle((i-0.5, -0.5), 1, activations.shape[1], 
                                   facecolor=label_colors[color_idx], alpha=0.15, edgecolor='none'))
        
        # Plot 2: Average activation per rule per class
        ax2 = fig.add_subplot(gs[0, 2])
        class_activations = {}
        for label in unique_labels:
            mask = labels == label
            class_activations[label] = activations[mask].mean(axis=0)
        
        x = np.arange(model.rules_count)
        width = 0.8 / len(unique_labels)
        
        for i, (label, avg_acts) in enumerate(class_activations.items()):
            offset = (i - len(unique_labels)/2 + 0.5) * width
            ax2.bar(x + offset, avg_acts, width, label=f'Class {label}', alpha=0.7)
        
        ax2.set_xlabel('Rule Index', fontsize=11)
        ax2.set_ylabel('Average Activation', fontsize=11)
        ax2.set_title('Which rules specialize in which classes?', fontsize=12, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=9)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_xticks(x)
        
        # Plot 3: Rule contribution to output (for first 2 rules)
        ax3 = fig.add_subplot(gs[1, :])
        
        # If binary classification, plot decision boundary illustration
        if outputs.shape[1] == 1 or (len(np.unique(labels)) == 2):
            # Select two most active rules
            avg_activations = activations.mean(axis=0)
            top_rules = np.argsort(avg_activations)[-2:]
            
            # Scatter plot of activation space
            rule1_acts = activations[:, top_rules[0]]
            rule2_acts = activations[:, top_rules[1]]
            
            scatter = ax3.scatter(rule1_acts, rule2_acts, c=labels, cmap='coolwarm', 
                                 alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
            ax3.set_xlabel(f'Activation of Rule {top_rules[0]}', fontsize=11)
            ax3.set_ylabel(f'Activation of Rule {top_rules[1]}', fontsize=11)
            ax3.set_title(f'Rule Activation Space (Rules {top_rules[0]} and {top_rules[1]})', 
                         fontsize=12, fontweight='bold')
            plt.colorbar(scatter, ax=ax3, label='Class Label')
            ax3.grid(True, alpha=0.3)
        
        plt.suptitle('How Fuzzy Laws Activate for Different Inputs', 
                    fontsize=14, fontweight='bold', y=1.02)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Law activation map saved to {save_path}")
        else:
            plt.show()

def visualize_complete_law_structure(
    model,
    dataloader=None,
    feature_names: Optional[List[str]] = None,
    class_names: Optional[List[str]] = None,
    rule_indices: Optional[List[int]] = None,
    save_dir: Optional[str] = None
):
    """
    Complete visualization of fuzzy law structure including both antecedent and consequent.
    
    Args:
        model: Trained GIFTSHIFT/GIFTSHIFTENTROPY model
        dataloader: DataLoader for activation analysis (optional)
        feature_names: Names of input features
        class_names: Names of output classes
        rule_indices: Which rules to plot
        save_dir: Directory to save figures
    """
    print("\n" + "="*70)
    print("VISUALIZING COMPLETE FUZZY LAW STRUCTURE")
    print("Antecedent (IF) + Consequent (THEN) Parts")
    print("="*70)
    
    # Plot 1: Complete laws with both parts
    print("\n📊 Plotting complete fuzzy laws (antecedent + consequent)...")
    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)
        plot_fuzzy_laws_complete(
            model, feature_names, class_names, rule_indices,
            save_path=f"{save_dir}/complete_fuzzy_laws.png"
        )
    else:
        plot_fuzzy_laws_complete(model, feature_names, class_names, rule_indices)
    
    # Plot 2: Law activation patterns (if dataloader provided)
    if dataloader is not None:
        print("\n📊 Plotting law activation patterns...")
        if save_dir:
            plot_law_activation_map(
                model, dataloader, feature_names, class_names,
                save_path=f"{save_dir}/law_activation_map.png"
            )
        else:
            plot_law_activation_map(model, dataloader, feature_names, class_names)
    
    # Print human-readable rules
    print_fuzzy_rules_structured(model, feature_names, class_names)
    
    print("\n✅ Complete law visualization finished!")
    print("="*70)

def print_fuzzy_rules_structured(
    model,
    feature_names: Optional[List[str]] = None,
    class_names: Optional[List[str]] = None
):
    """
    Print structured fuzzy rules with both antecedent and consequent parts.
    
    Args:
        model: Trained GIFTSHIFT model
        feature_names: Names of input features
        class_names: Names of output classes
    """
    with torch.no_grad():
        means = model.mean.cpu().numpy()
        stds = torch.nn.functional.softplus(model.std).cpu().numpy()
        literal_probs = torch.sigmoid(model.literal).cpu().numpy()
        comb_weights = torch.sigmoid(model.comb_weight).cpu().numpy()
        
        in_features = model.in_features
        rules = model.rules_count
        out_features = model.out_features if hasattr(model, 'out_features') else 1
        
        has_tsk = hasattr(model, 'local_slopes')
        if has_tsk:
            tsk_slopes = model.local_slopes.cpu().numpy()
            tsk_biases = model.local_biases.cpu().numpy()
        
        if feature_names is None:
            feature_names = [f'x{i}' for i in range(min(in_features, 8))]
        else:
            feature_names = feature_names[:min(in_features, 8)]
        
        if class_names is None:
            if out_features == 1:
                class_names = ['Output']
            else:
                class_names = [f'C{i}' for i in range(out_features)]
        
        print("\n" + "="*80)
        print("EXTRACTED FUZZY LAWS (IF-THEN RULES)")
        print("="*80)
        
        for rule_idx in range(rules):
            print(f"\n{'='*80}")
            print(f"📜 LAW #{rule_idx + 1}")
            print(f"{'='*80}")
            
            # Antecedent (IF part)
            print("\n🔍 IF (Antecedent):")
            print("-" * 60)
            
            conditions = []
            for feat_idx in range(len(feature_names)):
                mu = means[feat_idx, rule_idx]
                sigma = stds[feat_idx, rule_idx]
                lit_prob = literal_probs[feat_idx, rule_idx]
                weight = comb_weights[feat_idx, rule_idx]
                
                # Determine the linguistic term
                if weight > 0.7:
                    # Gaussian dominant
                    term = f"is approximately {mu:.3f}"
                elif weight < 0.3:
                    # Sigmoidal dominant
                    if lit_prob > 0.7:
                        term = f"is NOT near {mu:.3f}"
                    elif lit_prob > 0.5:
                        term = f"is greater than {mu:.3f}"
                    else:
                        term = f"is less than {mu:.3f}"
                else:
                    # Hybrid
                    term = f"is around {mu:.3f}"
                
                conditions.append(f"    {feature_names[feat_idx]} {term}")
            
            print("\n".join(conditions))
            
            # Consequent (THEN part)
            print("\n🎯 THEN (Consequent):")
            print("-" * 60)
            
            if has_tsk:
                if out_features == 1:
                    # Single output
                    bias = tsk_biases[rule_idx, 0]
                    print(f"    y = {bias:.4f}")
                    
                    # Add linear terms
                    for feat_idx in range(len(feature_names)):
                        slope = tsk_slopes[rule_idx, feat_idx, 0] if tsk_slopes.ndim > 2 else tsk_slopes[rule_idx, feat_idx]
                        if abs(slope) > 0.01:
                            sign = "+" if slope > 0 else "-"
                            print(f"        {sign} {abs(slope):.4f} · ({feature_names[feat_idx]} - {means[feat_idx, rule_idx]:.3f})")
                else:
                    # Multiple outputs
                    for class_idx in range(out_features):
                        bias = tsk_biases[rule_idx, class_idx]
                        print(f"    {class_names[class_idx]}: {bias:.4f}")
                        
                        for feat_idx in range(len(feature_names)):
                            slope = tsk_slopes[rule_idx, feat_idx, class_idx] if tsk_slopes.ndim > 2 else tsk_slopes[rule_idx, feat_idx, class_idx]
                            if abs(slope) > 0.01:
                                sign = "+" if slope > 0 else "-"
                                print(f"        {sign} {abs(slope):.4f} · ({feature_names[feat_idx]} - {means[feat_idx, rule_idx]:.3f})")
            else:
                print("    (Mamdani-style output - see model configuration)")
            
            # Add interpretation
            print("\n💡 Interpretation:")
            print("-" * 60)
            
            # Find strongest features
            strengths = comb_weights[range(len(feature_names)), rule_idx] * means[range(len(feature_names)), rule_idx]
            top_features = np.argsort(np.abs(strengths))[-3:]
            
            print(f"    This rule focuses most on: {', '.join([feature_names[i] for i in top_features])}")
            
            if has_tsk and out_features == 1:
                if tsk_biases[rule_idx, 0] > 0:
                    print(f"    This rule contributes POSITIVELY to the output")
                else:
                    print(f"    This rule contributes NEGATIVELY to the output")
        
        print("\n" + "="*80)


import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_consequent_digit_laws(
    model,
    image_shape=(8, 8),
    figsize_per_rule=(16, 12),
    cmap="coolwarm",
    save_path=None
):
    """
    Visualize consequent TSK laws as digit-like images.

    Each:
        local_slopes[rule, :, class]
    becomes one image.

    Positive:
        increases class confidence

    Negative:
        suppresses class confidence
    """

    slopes = model.local_slopes.detach().cpu().numpy()

    rules = slopes.shape[0]
    in_features = slopes.shape[1]
    classes = slopes.shape[2]

    vmax = np.abs(slopes).max()

    fig, axes = plt.subplots(
        rules,
        classes,
        figsize=(classes * 2.2, rules * 2.2)
    )

    if rules == 1:
        axes = np.expand_dims(axes, axis=0)

    if classes == 1:
        axes = np.expand_dims(axes, axis=1)

    for r in range(rules):

        for c in range(classes):

            ax = axes[r, c]

            slope_img = slopes[r, :, c].reshape(image_shape)

            im = ax.imshow(
                slope_img,
                cmap=cmap,
                vmin=-vmax,
                vmax=vmax
            )

            ax.set_title(f"R{r} → C{c}", fontsize=9)
            ax.axis("off")

    cbar = fig.colorbar(
        im,
        ax=axes.ravel().tolist(),
        shrink=0.6
    )

    cbar.set_label("Consequent Weight")

    plt.suptitle(
        "Consequent Fuzzy Laws (THEN Part)",
        fontsize=16
    )

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

def plot_antecedent_digit_laws(
    model,
    image_shape=(8, 8),
    cmap="viridis",
    save_path=None
):
    """
    Visualize antecedent centers as images.
    """

    means = model.mean.detach().cpu().numpy()

    rules = means.shape[1]

    fig, axes = plt.subplots(
        1,
        rules,
        figsize=(rules * 3, 3)
    )

    if rules == 1:
        axes = [axes]

    for r in range(rules):

        ax = axes[r]

        img = means[:, r].reshape(image_shape)

        ax.imshow(img, cmap=cmap)

        ax.set_title(f"Rule {r}")
        ax.axis("off")

    plt.suptitle("Antecedent Fuzzy Laws (IF Part)")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def visualize_full_fuzzy_system(
    model,
    image_shape=(8, 8),
    output_dir="./law_visualizations"
):
    """
    Visualize BOTH:
        IF part
        THEN part
    """

    import os

    os.makedirs(output_dir, exist_ok=True)

    print("\n📊 Plotting antecedent laws...")
    
    plot_antecedent_digit_laws(
        model,
        image_shape=image_shape,
        save_path=f"{output_dir}/antecedent_laws.png"
    )

    print("✅ Antecedent laws saved")

    print("\n📊 Plotting consequent laws...")

    plot_consequent_digit_laws(
        model,
        image_shape=image_shape,
        save_path=f"{output_dir}/consequent_laws.png"
    )

    print("✅ Consequent laws saved")