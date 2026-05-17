"""
Fuzzy Rule Visualization Utilities for GIFT-based Models
Author: Your Name
Description: Visualize fuzzy logic rules, membership functions, and activations
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple, Union
import os


class FuzzyRuleVisualizer:
    """
    Visualizer for fuzzy logic rules in GIFT, GIFTSHIFT, and GIFTSHIFTENTROPY models
    """
    
    def __init__(self, model: torch.nn.Module, device: torch.device, feature_names: Optional[List[str]] = None):
        """
        Initialize the visualizer
        
        Args:
            model: Trained GIFT/GIFTSHIFT/GIFTSHIFTENTROPY model
            device: torch device (cpu/cuda)
            feature_names: Optional list of feature names
        """
        self.model = model
        self.device = device
        self.model.eval()
        
        # Extract model parameters
        with torch.no_grad():
            if hasattr(model, 'mean'):
                self.means = model.mean.cpu().numpy()
                self.stds = F.softplus(model.std).cpu().numpy()
                self.literal = torch.sigmoid(model.literal).cpu().numpy()
                self.sigmoid_slope = model.sigmoid_slope.cpu().numpy()
                self.weight = torch.sigmoid(model.comb_weight).cpu().numpy()
                
                self.n_features = self.means.shape[0]
                self.n_rules = self.means.shape[1]
                
                # For TSK models
                if hasattr(model, 'local_slopes'):
                    self.slopes = model.local_slopes.cpu().numpy()
                    self.biases = model.local_biases.cpu().numpy()
            else:
                raise AttributeError("Model doesn't have required GIFT/GIFTSHIFT attributes")
        
        # Set default feature names if not provided
        if feature_names is None:
            self.feature_names = [f'feature_{i}' for i in range(self.n_features)]
        else:
            self.feature_names = feature_names
    
    def visualize_membership_functions(self, save_path: Optional[str] = None, 
                                       max_features: int = 10) -> None:
        """
        Visualize fuzzy membership functions for each rule and feature
        
        Args:
            save_path: Path to save the figure (optional)
            max_features: Maximum number of features to show (to avoid overcrowding)
        """
        n_features_to_show = min(self.n_features, max_features)
        
        fig, axes = plt.subplots(self.n_rules, n_features_to_show, 
                                 figsize=(4*n_features_to_show, 3*self.n_rules))
        
        if self.n_rules == 1:
            axes = axes.reshape(1, -1)
        if n_features_to_show == 1:
            axes = axes.reshape(-1, 1)
        
        x_range = np.linspace(-3, 3, 100)
        
        for rule_idx in range(self.n_rules):
            for feat_idx in range(n_features_to_show):
                ax = axes[rule_idx, feat_idx] if self.n_rules > 1 else axes[feat_idx]
                
                # Get parameters
                mu = self.means[feat_idx, rule_idx]
                sigma = self.stds[feat_idx, rule_idx]
                slope = self.sigmoid_slope[feat_idx, rule_idx]
                w = self.weight[feat_idx, rule_idx]
                lit = self.literal[feat_idx, rule_idx]
                
                # Compute membership functions
                gaussian_mf = np.exp(-((x_range - mu) ** 2) / (2 * sigma ** 2))
                sigmoid_mf = 1 / (1 + np.exp(-(x_range - mu) * slope))
                
                # Combined membership
                gaussian_neg = (gaussian_mf * lit) + (1 - gaussian_mf) * (1 - lit)
                sigmoid_neg = (sigmoid_mf * lit) + (1 - sigmoid_mf) * (1 - lit)
                combined_mf = w * gaussian_neg + (1 - w) * sigmoid_neg
                
                # Plot
                ax.plot(x_range, gaussian_mf, 'b--', alpha=0.5, linewidth=1)
                ax.plot(x_range, sigmoid_mf, 'g--', alpha=0.5, linewidth=1)
                ax.plot(x_range, combined_mf, 'r-', linewidth=2)
                ax.axvline(mu, color='k', linestyle=':', alpha=0.3)
                ax.set_title(f'R{rule_idx+1}, {self.feature_names[feat_idx]}\nμ={mu:.2f}')
                ax.set_xlabel('Input')
                ax.set_ylabel('μ(x)')
                
                if rule_idx == 0 and feat_idx == 0:
                    ax.legend(['Gaussian', 'Sigmoidal', 'Combined'], fontsize=6)
        
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def visualize_rule_activations(self, X_samples: np.ndarray, y_samples: np.ndarray,
                                   n_samples: int = 5, save_path: Optional[str] = None) -> None:
        """
        Visualize rule activations for sample inputs
        
        Args:
            X_samples: Input samples (numpy array)
            y_samples: Labels for samples
            n_samples: Number of samples to visualize
            save_path: Path to save the figure
        """
        n_samples = min(n_samples, len(X_samples))
        X_tensor = torch.tensor(X_samples[:n_samples], dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            rule_activations = self.model.encode(X_tensor).cpu().numpy()
            if rule_activations.shape[1] > 1:
                rule_activations = rule_activations / rule_activations.sum(axis=1, keepdims=True)
        
        fig, axes = plt.subplots(n_samples, self.n_rules + 1, 
                                 figsize=(3*(self.n_rules+1), 3*n_samples))
        
        for i in range(n_samples):
            # Original data (reshape if it's image data like Digits)
            ax = axes[i, 0] if n_samples > 1 else axes[0]
            
            # Check if it's likely image data (64 features for Digits)
            if X_samples.shape[1] == 64:  # Digits dataset (8x8)
                img = X_samples[i].reshape(8, 8)
                ax.imshow(img, cmap='gray')
                ax.set_title(f'Label: {int(y_samples[i])}')
            else:
                ax.bar(range(min(10, X_samples.shape[1])), X_samples[i, :10])
                ax.set_title(f'Sample {i+1}')
            ax.axis('off')
            
            # Rule activations
            for rule_idx in range(self.n_rules):
                ax_idx = axes[i, rule_idx + 1] if n_samples > 1 else axes[rule_idx + 1]
                ax_idx.bar([f'R{rule_idx+1}'], [rule_activations[i, rule_idx]], 
                          color='steelblue', alpha=0.7)
                ax_idx.set_ylim(0, 1)
                ax_idx.set_ylabel('Activation')
                ax_idx.set_title(f'Rule {rule_idx+1}')
        
        plt.suptitle('Rule Activations for Sample Inputs', fontsize=14)
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_rule_activation_heatmap(self, X: np.ndarray, y: np.ndarray,
                                     save_path: Optional[str] = None) -> np.ndarray:
        """
        Plot heatmap of rule activations vs classes
        
        Args:
            X: Input features
            y: Labels
            save_path: Path to save the figure
            
        Returns:
            Array of class activation patterns
        """
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            rule_acts = self.model.encode(X_tensor).cpu().numpy()
            if rule_acts.shape[1] > 1:
                rule_acts = rule_acts / rule_acts.sum(axis=1, keepdims=True)
        
        classes = np.unique(y)
        class_activations = []
        for cls in classes:
            mask = y == cls
            class_activations.append(rule_acts[mask].mean(axis=0))
        
        class_activations = np.array(class_activations)
        
        # Plot heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        im = ax.imshow(class_activations.T, cmap='viridis', aspect='auto')
        
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels([f'Class {int(c)}' for c in classes])
        ax.set_xlabel('Class')
        ax.set_ylabel('Rule Index')
        ax.set_title('Rule Activation Patterns by Class')
        
        plt.colorbar(im, ax=ax, label='Average Activation')
        
        # Add annotations
        for i in range(class_activations.shape[1]):
            for j in range(class_activations.shape[0]):
                text = ax.text(j, i, f'{class_activations[j, i]:.2f}',
                              ha="center", va="center", color="white" if class_activations[j, i] < 0.5 else "black",
                              fontsize=9)
        
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        return class_activations
    
    def analyze_rule_specialization(self, X: np.ndarray, y: np.ndarray,
                                    save_path: Optional[str] = None) -> dict:
        """
        Analyze which rules specialize in which classes
        
        Args:
            X: Input features
            y: Labels
            save_path: Path to save the figure
            
        Returns:
            Dictionary with rule specialization information
        """
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            rule_acts = self.model.encode(X_tensor).cpu().numpy()
            if rule_acts.shape[1] > 1:
                rule_acts = rule_acts / rule_acts.sum(axis=1, keepdims=True)
        
        classes = np.unique(y)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Left: Activation distribution
        ax = axes[0]
        for rule_idx in range(rule_acts.shape[1]):
            ax.hist(rule_acts[:, rule_idx], bins=20, alpha=0.5, 
                    label=f'Rule {rule_idx+1}', density=True)
        ax.set_xlabel('Activation Strength')
        ax.set_ylabel('Density')
        ax.set_title('Distribution of Rule Activations')
        ax.legend()
        
        # Right: Dominant class per rule
        ax = axes[1]
        specialization = {}
        
        for rule_idx in range(rule_acts.shape[1]):
            class_means = []
            for cls in classes:
                mask = y == cls
                class_means.append(rule_acts[mask, rule_idx].mean())
            dominant_class = classes[np.argmax(class_means)]
            max_activation = np.max(class_means)
            specialization[rule_idx] = {
                'dominant_class': int(dominant_class),
                'max_activation': float(max_activation),
                'class_means': {int(cls): float(mean) for cls, mean in zip(classes, class_means)}
            }
            
            ax.bar(rule_idx, max_activation, 
                   label=f'Class {int(dominant_class)}', alpha=0.7)
        
        ax.set_xlabel('Rule Index')
        ax.set_ylabel('Max Avg Activation')
        ax.set_title('Rule Specialization (Most Activated Class)')
        ax.legend()
        
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        # Print summary
        print("\n" + "="*50)
        print("RULE SPECIALIZATION SUMMARY")
        print("="*50)
        for rule_idx, info in specialization.items():
            print(f"Rule {rule_idx+1}: Specializes in Class {info['dominant_class']} "
                  f"(avg activation: {info['max_activation']:.3f})")
        
        return specialization
    
    def extract_fuzzy_rules(self, top_k_features: int = 10) -> List[str]:
        """
        Extract human-readable fuzzy rules
        
        Args:
            top_k_features: Number of top features to show per rule
            
        Returns:
            List of rule strings
        """
        rules_text = []
        
        for rule_idx in range(self.n_rules):
            rule_str = f"\n{'='*60}\nRULE {rule_idx + 1}:\n{'='*60}\n"
            rule_str += "IF\n"
            
            # Get feature importances for this rule (using slopes as importance)
            if hasattr(self, 'slopes'):
                importances = np.abs(self.slopes[rule_idx, :, 0]) if self.slopes.shape[-1] == 1 else np.abs(self.slopes[rule_idx, :, :]).mean(axis=1)
                top_features = np.argsort(importances)[-top_k_features:][::-1]
            else:
                top_features = range(min(top_k_features, self.n_features))
            
            antecedents = []
            for feat_idx in top_features:
                mu = self.means[feat_idx, rule_idx]
                lit = self.literal[feat_idx, rule_idx]
                w = self.weight[feat_idx, rule_idx]
                
                if w > 0.6:
                    mf_type = "Gaussian"
                elif w < 0.4:
                    mf_type = "Sigmoidal"
                else:
                    mf_type = "Hybrid"
                
                if lit > 0.6:
                    relation = "is approximately"
                elif lit < 0.4:
                    relation = "is NOT approximately"
                else:
                    relation = "is around"
                
                antecedents.append(f"    {self.feature_names[feat_idx]} {relation} {mu:.2f} [{mf_type}]")
            
            rule_str += "\n".join(antecedents[:top_k_features])
            
            # Consequent
            if hasattr(self, 'slopes') and hasattr(self, 'biases'):
                rule_str += "\nTHEN\n"
                rule_str += f"    output = {self.biases[rule_idx, 0]:.4f}"
                
                for feat_idx in top_features[:5]:  # Show top 5 slopes
                    slope = self.slopes[rule_idx, feat_idx, 0] if self.slopes.shape[-1] == 1 else self.slopes[rule_idx, feat_idx, :].mean()
                    if abs(slope) > 0.01:
                        sign = "+" if slope > 0 else "-"
                        rule_str += f" {sign} ({abs(slope):.4f} × {self.feature_names[feat_idx]})"
            
            rules_text.append(rule_str)
        
        return rules_text
    
    def print_fuzzy_rules(self, top_k_features: int = 10) -> None:
        """Print extracted fuzzy rules"""
        rules = self.extract_fuzzy_rules(top_k_features)
        for rule in rules:
            print(rule)
    
    def save_fuzzy_rules(self, filepath: str, top_k_features: int = 10) -> None:
        """Save fuzzy rules to text file"""
        rules = self.extract_fuzzy_rules(top_k_features)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            f.write("\n".join(rules))
        print(f"Rules saved to {filepath}")
    
    def visualize_pixel_maps(self, image_shape: Tuple[int, int] = (8, 8),
                            save_path: Optional[str] = None) -> None:
        """
        Visualize fuzzy parameters as pixel maps (useful for image datasets like Digits)
        
        Args:
            image_shape: Shape of the image (height, width)
            save_path: Path to save the figure
        """
        if self.n_features != image_shape[0] * image_shape[1]:
            print(f"Warning: Features ({self.n_features}) don't match image shape {image_shape}")
            return
        
        fig, axes = plt.subplots(3, self.n_rules, figsize=(4*self.n_rules, 12))
        
        if self.n_rules == 1:
            axes = axes.reshape(3, 1)
        
        for rule_idx in range(self.n_rules):
            # Mean values
            ax = axes[0, rule_idx] if self.n_rules > 1 else axes[0, rule_idx]
            mean_img = self.means[:, rule_idx].reshape(image_shape)
            im = ax.imshow(mean_img, cmap='RdBu', vmin=-2, vmax=2)
            ax.set_title(f'Rule {rule_idx+1} - Centers (μ)')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)
            
            # Literal values
            ax = axes[1, rule_idx] if self.n_rules > 1 else axes[1, rule_idx]
            lit_img = self.literal[:, rule_idx].reshape(image_shape)
            ax.imshow(lit_img, cmap='gray', vmin=0, vmax=1)
            ax.set_title(f'Rule {rule_idx+1} - Literal (¬)')
            ax.axis('off')
            
            # Combination weights
            ax = axes[2, rule_idx] if self.n_rules > 1 else axes[2, rule_idx]
            weight_img = self.weight[:, rule_idx].reshape(image_shape)
            ax.imshow(weight_img, cmap='hot', vmin=0, vmax=1)
            ax.set_title(f'Rule {rule_idx+1} - Weight')
            ax.axis('off')
        
        plt.suptitle('Fuzzy Logic Parameters as Pixel Maps', fontsize=14)
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def create_comprehensive_report(self, X: np.ndarray, y: np.ndarray,
                                   save_dir: str = 'rule_visualizations',
                                   image_shape: Optional[Tuple[int, int]] = None) -> None:
        """
        Create a comprehensive visualization report
        
        Args:
            X: Input features
            y: Labels
            save_dir: Directory to save visualizations
            image_shape: Optional image shape for pixel map visualization
        """
        os.makedirs(save_dir, exist_ok=True)
        
        print("Generating comprehensive fuzzy rule visualization report...")
        print("="*60)
        
        # 1. Membership functions
        print("\n1. Visualizing membership functions...")
        self.visualize_membership_functions(save_path=f'{save_dir}/membership_functions.png')
        
        # 2. Rule activations on samples
        print("2. Visualizing rule activations on samples...")
        self.visualize_rule_activations(X, y, n_samples=5, 
                                       save_path=f'{save_dir}/rule_activations.png')
        
        # 3. Rule activation heatmap
        print("3. Generating rule activation heatmap...")
        self.plot_rule_activation_heatmap(X, y, 
                                         save_path=f'{save_dir}/rule_heatmap.png')
        
        # 4. Rule specialization analysis
        print("4. Analyzing rule specialization...")
        self.analyze_rule_specialization(X, y,
                                        save_path=f'{save_dir}/rule_specialization.png')
        
        # 5. Pixel maps (if image data)
        if image_shape is not None and self.n_features == image_shape[0] * image_shape[1]:
            print("5. Visualizing pixel maps...")
            self.visualize_pixel_maps(image_shape, save_path=f'{save_dir}/pixel_maps.png')
        
        # 6. Save fuzzy rules
        print("6. Extracting and saving fuzzy rules...")
        self.save_fuzzy_rules(f'{save_dir}/fuzzy_rules.txt')
        self.print_fuzzy_rules()
        
        print("\n" + "="*60)
        print(f"✅ Comprehensive report saved to '{save_dir}/'")
        print("="*60)


# Convenience function for quick visualization
def quick_visualize(model: torch.nn.Module, 
                   device: torch.device,
                   X_train: np.ndarray,
                   y_train: np.ndarray,
                   feature_names: Optional[List[str]] = None,
                   save_dir: str = 'quick_vis',
                   is_image_data: bool = False,
                   image_shape: Tuple[int, int] = (8, 8)) -> FuzzyRuleVisualizer:
    """
    Quick visualization function for fuzzy rules
    
    Args:
        model: Trained GIFT/GIFTSHIFT model
        device: torch device
        X_train: Training features
        y_train: Training labels
        feature_names: Optional feature names
        save_dir: Directory to save visualizations
        is_image_data: Whether the data is image data (like Digits)
        image_shape: Shape of images if is_image_data=True
        
    Returns:
        FuzzyRuleVisualizer instance
    """
    visualizer = FuzzyRuleVisualizer(model, device, feature_names)
    
    if is_image_data:
        visualizer.create_comprehensive_report(X_train, y_train, save_dir, image_shape)
    else:
        visualizer.create_comprehensive_report(X_train, y_train, save_dir)
    
    return visualizer