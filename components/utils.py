import torch
import numpy as np
import matplotlib.pyplot as plt

def evaluate_anfis_model(
    model, X_train, y_train, X_test, y_test, 
    device="cpu", s_mode_choice="alpha_beta", centers_init=None,
    dataset_name=None  # Added to know which dataset we're working with
):
    """
    Evaluate ANFIS model with metrics, MF contributions, and automatic dataset plotting.

    Works for binary, multi-class, regression datasets.
    Handles 1D, 2D, and multi-feature datasets.
    """

    model.eval()
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_test_tensor  = torch.tensor(X_test,  dtype=torch.float32).to(device)

    with torch.no_grad():
        y_train_pred = model(X_train_tensor)
        y_test_pred  = model(X_test_tensor)

    # Task type detection
    n_outputs = y_train_pred.shape[1] if len(y_train_pred.shape) > 1 else 1
    if n_outputs == 1 and len(np.unique(y_train)) <= 2:
        task_type = "binary"
    elif n_outputs > 1 or len(np.unique(y_train)) > 2:
        task_type = "multiclass"
    else:
        task_type = "regression"

    # Apply activation
    if task_type == "binary":
        final_train_prob = torch.sigmoid(y_train_pred).cpu().numpy().reshape(-1)
        final_test_prob  = torch.sigmoid(y_test_pred).cpu().numpy().reshape(-1)
    elif task_type == "multiclass":
        final_train_prob = torch.softmax(y_train_pred, dim=1).cpu().numpy()
        final_test_prob  = torch.softmax(y_test_pred, dim=1).cpu().numpy()
    else:
        final_train_prob = y_train_pred.cpu().numpy().reshape(-1)
        final_test_prob  = y_test_pred.cpu().numpy().reshape(-1)

    # ---------------------------
    # Summarize results - FIXED VERSION
    # ---------------------------
    def summarize_split(name, probs, labels):
        if task_type == "binary":
            preds = (probs >= 0.5).astype(np.int64)
        elif task_type == "multiclass":
            preds = probs.argmax(axis=1)
        else:
            preds = probs  # regression

        total = len(preds)  # Use preds length instead of labels length
        if task_type == "regression":
            rmse = np.sqrt(np.mean((preds - labels)**2))
            print(f"{name}: RMSE={rmse:.4f}")
        else:
            # Ensure labels and preds have same shape for comparison
            if len(labels) != len(preds):
                # If shapes don't match, try to reshape or truncate
                min_len = min(len(labels), len(preds))
                labels = labels[:min_len]
                preds = preds[:min_len]
                total = min_len
            
            correct = int((preds == labels).sum())
            wrong = total - correct
            accuracy = correct / total
            error = wrong / total
            print(f"{name}: accuracy rate {accuracy:.3f} ({correct}/{total}) | error {error:.3f} ({wrong})")

    print("\nFinal Results:")
    print("Mode:", s_mode_choice)
    summarize_split("Train", final_train_prob, y_train)
    summarize_split("Test", final_test_prob, y_test)

    # ---------------------------
    # MF contributions (Advanced MF only)
    # ---------------------------
    if hasattr(model, 'mf_layer') and hasattr(model.mf_layer, "c") and hasattr(model.mf_layer, "s"):
        try:
            return_parts = hasattr(model.mf_layer, "alpha") or hasattr(model.mf_layer, "beta")
            if return_parts:
                w, mu_g, mu_s, blend, mu_blend = model.mf_layer(X_test_tensor, return_parts=True)
                gauss_contrib = (blend * mu_g).mean(dim=0).detach().cpu().numpy()
                sig_contrib   = ((1 - blend) * mu_s).mean(dim=0).detach().cpu().numpy()
                gauss_weight  = blend.mean(dim=0).detach().cpu().numpy()
                sig_weight    = (1 - blend).mean(dim=0).detach().cpu().numpy()

                n_rules, n_features = gauss_contrib.shape
                print("\nPer-rule, per-feature MF contributions and raw participation:")
                for r in range(n_rules):
                    for f in range(n_features):
                        print(f"Rule {r}, Feature {f}: "
                              f"Raw weight -> Gaussian={gauss_weight[r,f]:.4f}, Sigmoid={sig_weight[r,f]:.4f}")
        except Exception as e:
            print("MF contributions not available:", e)

    # ---------------------------
    # Plot dataset results
    # ---------------------------
    n_features = X_train.shape[1]
    n_classes = len(np.unique(y_train)) if task_type != "regression" else 1

    # 3D dataset (exactly 3 features, e.g., swiss-roll)
    if n_features == 3:
        fig = plt.figure(figsize=(18,5))
        
        # Train ground truth
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.scatter(X_train[:,0], X_train[:,1], X_train[:,2], c=y_train, cmap="bwr", s=20)
        ax1.set_title("Train Ground Truth")
        ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")

        # Train predictions
        ax2 = fig.add_subplot(132, projection='3d')
        ax2.scatter(X_train[:,0], X_train[:,1], X_train[:,2], c=final_train_prob if task_type=="regression" else final_train_prob, cmap="RdBu_r", s=20)
        ax2.set_title("Train Predictions")
        ax2.set_xlabel("X"); ax2.set_ylabel("Y"); ax2.set_zlabel("Z")
        
        # Test predictions
        ax3 = fig.add_subplot(133, projection='3d')
        ax3.scatter(X_test[:,0], X_test[:,1], X_test[:,2], c=final_test_prob if task_type=="regression" else final_test_prob, cmap="RdBu_r", s=20)
        ax3.set_title("Test Predictions")
        ax3.set_xlabel("X"); ax3.set_ylabel("Y"); ax3.set_zlabel("Z")

        plt.tight_layout()
        plt.show()

    # IRIS dataset special handling (multi-class with 4 features)
    elif task_type == "multiclass":
        # Plot petal features (2,3) as in your example
        petal_idx = (2, 3)
        baseline = X_train.mean(axis=0)

        xx, yy = np.meshgrid(
            np.linspace(X_train[:, petal_idx[0]].min() - 0.5, X_train[:, petal_idx[0]].max() + 0.5, 250),
            np.linspace(X_train[:, petal_idx[1]].min() - 0.5, X_train[:, petal_idx[1]].max() + 0.5, 250),
        )

        grid = np.tile(baseline, (xx.size, 1)).astype(np.float32)
        grid[:, petal_idx[0]] = xx.ravel()
        grid[:, petal_idx[1]] = yy.ravel()

        grid_tensor = torch.from_numpy(grid).to(device)
        with torch.no_grad():
            grid_logits = model(grid_tensor)
            grid_pred = torch.argmax(grid_logits, dim=1).cpu().numpy().reshape(xx.shape)

        plt.figure(figsize=(8, 6))
        plt.contourf(
            xx,
            yy,
            grid_pred,
            levels=np.arange(n_classes + 1) - 0.5,
            cmap="tab10",
            alpha=0.6,
        )
        plt.colorbar(ticks=range(n_classes), label="Predicted class")
        plt.scatter(
            X_train[:, petal_idx[0]],
            X_train[:, petal_idx[1]],
            c=y_train,
            cmap="tab10",
            edgecolor="k",
            alpha=0.6,
            label="Train",
        )
        plt.scatter(
            X_test[:, petal_idx[0]],
            X_test[:, petal_idx[1]],
            c=y_test,
            cmap="tab10",
            marker="x",
            label="Test",
        )
        plt.title(f"ANFIS μ+ Decision Boundary (mode={s_mode_choice})")
        plt.xlabel("Petal length (standardized)")
        plt.ylabel("Petal width (standardized)")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # 2D datasets (binary classification with 2 features) - Moons, circles, etc.
    elif n_features == 2 and task_type == "binary":
        plt.figure(figsize=(8, 6))
        
        # Create grid for visualization
        xx, yy = np.meshgrid(
            np.linspace(X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5, 300),
            np.linspace(X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5, 300)
        )
        
        grid = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)
        grid_tensor = torch.from_numpy(grid).to(device)
        
        with torch.no_grad():
            prob_grid = torch.sigmoid(model(grid_tensor)).cpu().numpy().reshape(xx.shape)
            contour = plt.contourf(xx, yy, prob_grid, levels=30, cmap="RdBu_r", alpha=0.6)
            plt.colorbar(contour, label="P(class=1)")
        
        # Training data
        plt.scatter(
            X_train[:, 0], X_train[:, 1], 
            c=y_train, cmap="bwr", edgecolor="k", s=40, label="Train"
        )
        
        # Test data
        plt.scatter(
            X_test[:, 0], X_test[:, 1], 
            c=y_test, cmap="cool", edgecolor="k", s=60, marker="^", label="Test"
        )
        
        # Optional: show rule centers
        if centers_init is not None:
            plt.scatter(
                centers_init[:, 0], centers_init[:, 1],
                c="yellow", s=200, edgecolor="black", marker="X",
                label="Rule Centers"
            )
        
        title = f"ANFIS Decision Boundary"
        if s_mode_choice:
            title += f"  |  Mode = {s_mode_choice}"
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    # Multi-class classification with 2 features
    elif n_features == 2 and task_type == "multiclass":
        plt.figure(figsize=(8, 6))
        
        # Create grid for visualization
        xx, yy = np.meshgrid(
            np.linspace(X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5, 300),
            np.linspace(X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5, 300)
        )
        
        grid = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)
        grid_tensor = torch.from_numpy(grid).to(device)
        
        with torch.no_grad():
            logits_grid = model(grid_tensor)
            pred_grid = torch.argmax(logits_grid, dim=1).cpu().numpy().reshape(xx.shape)
            contour = plt.contourf(xx, yy, pred_grid, alpha=0.6, 
                                  levels=np.arange(n_classes + 1) - 0.5,
                                  cmap="tab10")
            plt.colorbar(contour, label="Predicted class")
        
        # Training data
        scatter_train = plt.scatter(
            X_train[:, 0], X_train[:, 1], 
            c=y_train, cmap="tab10", edgecolor="k", s=40, label="Train"
        )
        
        # Test data
        scatter_test = plt.scatter(
            X_test[:, 0], X_test[:, 1], 
            c=y_test, cmap="tab10", edgecolor="k", s=60, marker="^", label="Test"
        )
        
        # Optional: show rule centers
        if centers_init is not None:
            plt.scatter(
                centers_init[:, 0], centers_init[:, 1],
                c="yellow", s=200, edgecolor="black", marker="X",
                label="Rule Centers"
            )
        
        title = f"ANFIS Decision Boundary"
        if s_mode_choice:
            title += f"  |  Mode = {s_mode_choice}"
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    # Other multi-feature datasets (non-iris)
    elif n_features > 2 and task_type != "regression":
        # For other datasets, show first 2 features
        f1, f2 = 0, 1
        baseline = X_train.mean(axis=0)
        
        xx, yy = np.meshgrid(
            np.linspace(X_train[:, f1].min() - 0.5, 
                       X_train[:, f1].max() + 0.5, 250),
            np.linspace(X_train[:, f2].min() - 0.5, 
                       X_train[:, f2].max() + 0.5, 250),
        )
        
        grid = np.tile(baseline, (xx.size, 1)).astype(np.float32)
        grid[:, f1] = xx.ravel()
        grid[:, f2] = yy.ravel()
        
        grid_tensor = torch.from_numpy(grid).to(device)
        with torch.no_grad():
            grid_logits = model(grid_tensor)
            grid_pred = torch.argmax(grid_logits, dim=1).cpu().numpy().reshape(xx.shape)
        
        plt.figure(figsize=(8, 6))
        plt.contourf(
            xx, yy, grid_pred,
            levels=np.arange(n_classes + 1) - 0.5,
            cmap="tab10",
            alpha=0.6,
        )
        plt.colorbar(ticks=range(n_classes), label="Predicted class")
        
        plt.scatter(
            X_train[:, f1], X_train[:, f2],
            c=y_train, cmap="tab10", edgecolor="k", alpha=0.6, label="Train"
        )
        plt.scatter(
            X_test[:, f1], X_test[:, f2],
            c=y_test, cmap="tab10", marker="x", label="Test"
        )
        
        if centers_init is not None and centers_init.shape[1] >= 2:
            plt.scatter(
                centers_init[:, f1], centers_init[:, f2],
                c="yellow", s=200, edgecolor="black", marker="X",
                label="Rule Centers"
            )
        
        title = f"ANFIS Decision Boundary (Features {f1}, {f2})"
        if s_mode_choice:
            title += f" (mode={s_mode_choice})"
        plt.title(title)
        plt.xlabel(f"Feature {f1} (standardized)")
        plt.ylabel(f"Feature {f2} (standardized)")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # 1D regression or single-feature dataset
    elif n_features == 1:
        plt.figure(figsize=(6,4))
        plt.scatter(X_train[:,0], y_train, label="Train", color="blue")
        plt.scatter(X_test[:,0], y_test, label="Test", color="red", marker="^")
        
        # Sort for clean plotting
        sort_idx = np.argsort(X_train[:,0])
        X_sorted = X_train[sort_idx, 0]
        
        if task_type == "binary":
            prob_sorted = final_train_prob[sort_idx]
            plt.plot(X_sorted, prob_sorted, 'k--', label="Predicted Probability")
        elif task_type == "multiclass":
            # For multiclass, show the predicted class probabilities for each class
            prob_sorted = final_train_prob[sort_idx]
            n_classes = prob_sorted.shape[1]
            for i in range(n_classes):
                plt.plot(X_sorted, prob_sorted[:, i], '--', label=f"P(class={i})")
        else:  # regression
            pred_sorted = final_train_prob[sort_idx]
            plt.plot(X_sorted, pred_sorted, 'k--', label="Predicted")
            
        plt.xlabel("Feature 0")
        plt.ylabel("Output")
        plt.title("1D Dataset Visualization")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # ---------------------------
    # Plot MFs per rule (Advanced MF)
    # ---------------------------
    if hasattr(model, 'mf_layer') and hasattr(model.mf_layer, "c") and hasattr(model.mf_layer, "s"):
        try:
            # Get number of rules
            n_rules = model.K if hasattr(model, 'K') else model.mf_layer.K if hasattr(model.mf_layer, 'K') else len(model.mf_layer.c)
            
            def plot_mfs_rule(rule, x_range=(-20,20), n=400):
                x = np.linspace(*x_range, n, dtype=np.float32)
                # Create proper input tensor for MF layer
                # For 1D input, we need to unsqueeze, for multi-dimensional we need to create dummy features
                if n_features == 1:
                    x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(1).to(device)
                else:
                    # Create a dummy input with the right dimensions
                    x_tensor = torch.zeros(n, n_features, dtype=torch.float32).to(device)
                    for f in range(n_features):
                        x_tensor[:, f] = torch.tensor(x, dtype=torch.float32)
                
                with torch.no_grad():
                    # Check if the MF layer supports return_parts
                    try:
                        w, mu_g, mu_s, blend, mu_blend = model.mf_layer(x_tensor, return_parts=True)
                    except:
                        # If return_parts not supported, skip plotting
                        print(f"MF layer doesn't support return_parts for rule {rule}")
                        return
                
                fig, axes = plt.subplots(1, n_features, figsize=(5*n_features,4))
                if n_features==1: 
                    axes = [axes]
                
                for f, ax in enumerate(axes):
                    mu_gf = mu_g[:, rule, f].detach().cpu().numpy()
                    mu_sf = mu_s[:, rule, f].detach().cpu().numpy()
                    mu_bf = mu_blend[:, rule, f].detach().cpu().numpy()
                    
                    ax.plot(x, mu_gf, label='Gaussian MF', color='yellow')
                    ax.plot(x, mu_sf, label='Sigmoid MF', color='lightblue')
                    ax.plot(x, mu_bf, '--', label='Blended MF', color='black')
                    
                    # Add vertical line at center if available
                    try:
                        center = model.mf_layer.c[rule, f].detach().cpu().numpy()
                        ax.axvline(x=center, color='red', linestyle=':', alpha=0.5, label=f'Center={center:.2f}')
                    except:
                        pass
                    
                    ax.set_title(f"Rule {rule}, Feature {f}")
                    ax.set_ylim(0, 1.05)
                    ax.grid(True)
                    ax.legend(loc='upper right')
                
                plt.suptitle(f"Membership Functions for Rule {rule}")
                plt.tight_layout()
                plt.show()
            
            # Plot only first few rules to avoid too many plots
            max_rules_to_plot = min(n_rules, 5)
            for r in range(max_rules_to_plot):
                plot_mfs_rule(r)
            
            if n_rules > max_rules_to_plot:
                print(f"\nNote: Showing only first {max_rules_to_plot} rules out of {n_rules} total rules.")
                
        except Exception as e:
            print("MF plotting not available:", e)