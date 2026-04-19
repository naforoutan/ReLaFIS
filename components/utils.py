from operator import lt
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch.nn.functional as F
import matplotlib.pyplot as plt
from components.models import GIFT


def evaluate_anfis_model(
    model,
    X_train, y_train,
    X_test, y_test,
    device="cpu",
    s_mode_choice="alpha_beta",
    centers_init=None,
    dataset_name=None,
    grid_res=250,
):

    model.eval()
    device = torch.device(device)

    X_train = np.asarray(X_train)
    X_test  = np.asarray(X_test)
    y_train = np.asarray(y_train)
    y_test  = np.asarray(y_test)

    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    X_test_t  = torch.tensor(X_test,  dtype=torch.float32, device=device)

    # ---------------------------
    # Forward pass
    # ---------------------------
    with torch.no_grad():
        out_train = model(X_train_t)
        out_test  = model(X_test_t)

    unique_y = np.unique(y_train)
    is_classification = (
        np.issubdtype(y_train.dtype, np.integer) and len(unique_y) >= 2
    )

    # ---------------------------
    # Metrics
    # ---------------------------
    print("\nFinal Results:")
    print(f"Mode: {s_mode_choice}")

    def summarize(split_name, logits_or_preds, y_true):
        if not is_classification:
            pred = logits_or_preds.detach().cpu().numpy().reshape(-1)
            rmse = np.sqrt(np.mean((pred - y_true.reshape(-1)) ** 2))
            print(f"{split_name}: RMSE={rmse:.4f}")
            return

        logits = logits_or_preds
        if logits.ndim == 1 or logits.shape[1] == 1:
            probs = torch.sigmoid(logits.view(-1))
            pred = (probs >= 0.5).long()
        else:
            pred = torch.argmax(logits, dim=1)

        pred_np = pred.detach().cpu().numpy()
        correct = int((pred_np == y_true).sum())
        total = len(y_true)
        acc = correct / total
        err = 1 - acc

        print(f"{split_name}: accuracy rate {acc:.3f} ({correct}/{total}) | error {err:.3f} ({total-correct})")

    summarize("Train", out_train, y_train)
    summarize("Test",  out_test,  y_test)

    # ==========================================================
    # =================== VISUALIZATION =========================
    # ==========================================================

    n_features = X_train.shape[1]

    # ----------------------------------------------------------
    # 1D Regression
    # ----------------------------------------------------------
    if not is_classification and n_features == 1:
        with torch.no_grad():
            pred_tr = model(X_train_t).cpu().numpy().reshape(-1)
            pred_te = model(X_test_t).cpu().numpy().reshape(-1)

        order = np.argsort(X_train[:, 0])
        plt.figure(figsize=(8, 4))
        plt.scatter(X_train[:, 0], y_train, s=25, label="Train True")
        plt.scatter(X_test[:, 0], y_test, marker="^", s=40, label="Test True")
        plt.plot(X_train[order, 0], pred_tr[order], "--", label="Train Pred")
        plt.title("1D Regression")
        plt.legend()
        plt.tight_layout()
        plt.show()
        return

    # ----------------------------------------------------------
    # 3D Regression (Swiss Roll)
    # ----------------------------------------------------------
    if not is_classification and n_features == 3:
        fig = plt.figure(figsize=(12, 5))
        ax1 = fig.add_subplot(121, projection="3d")
        ax2 = fig.add_subplot(122, projection="3d")

        ax1.scatter(X_train[:, 0], X_train[:, 1], X_train[:, 2], c=y_train, s=10)
        ax1.set_title("Train (true target)")

        with torch.no_grad():
            pred_te = model(X_test_t).cpu().numpy().reshape(-1)

        ax2.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c=pred_te, s=10)
        ax2.set_title("Test (predicted)")

        plt.tight_layout()
        plt.show()
        return


    # ----------------------------------------------------------
    # 2D Classification (enhanced)
    # ----------------------------------------------------------
    if is_classification and n_features == 2:

        x_min, x_max = X_train[:, 0].min()-0.5, X_train[:, 0].max()+0.5
        y_min, y_max = X_train[:, 1].min()-0.5, X_train[:, 1].max()+0.5

        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, grid_res),
            np.linspace(y_min, y_max, grid_res)
        )

        grid = np.c_[xx.ravel(), yy.ravel()]
        grid_t = torch.tensor(grid, dtype=torch.float32, device=device)

        with torch.no_grad():
            logits = model(grid_t)

        if logits.ndim == 1 or logits.shape[1] == 1:
            probs = torch.sigmoid(logits.view(-1)).cpu().numpy()
            Z = probs.reshape(xx.shape)
        else:
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            Z = pred.reshape(xx.shape)

        plt.figure(figsize=(8, 6))

        if logits.ndim == 1 or logits.shape[1] == 1:
            plt.contourf(xx, yy, Z, levels=30, alpha=0.5)
            plt.colorbar(label="P(class=1)")
        else:
            levels = np.arange(Z.max() + 2) - 0.5
            plt.contourf(xx, yy, Z, levels=levels, alpha=0.5)
            plt.colorbar(label="Predicted class")

        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)


        # Train
        plt.scatter(
            X_train[:, 0], X_train[:, 1],
            c=y_train,
            edgecolor="black",
            s=40,
            label="Train"
        )

        # Test (larger triangle markers)
        plt.scatter(
            X_test[:, 0], X_test[:, 1],
            c=y_test,
            marker="^",
            edgecolor="black",
            s=70,
            label="Test"
        )

        # Plot rule centers (K)
        if centers_init is not None:
            plt.scatter(
                centers_init[:, 0],
                centers_init[:, 1],
                marker="X",
                s=250,
                c="white",
                edgecolor="black",
                linewidth=2,
                label="Rule Centers (K)"
            )

            # Annotate rule index
            for i, (cx, cy) in enumerate(centers_init):
                plt.text(cx, cy, f"K{i}", fontsize=10, weight="bold")

        plt.title(f"Decision Boundary | dataset={dataset_name}")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # ----------------------------------------------------------
    # High-D Classification → PCA projection (FIXED with scaling)
    # ----------------------------------------------------------
    if is_classification and n_features > 2:

        from sklearn.preprocessing import StandardScaler

        # Scale first
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled  = scaler.transform(X_test)

        # PCA on scaled data
        pca = PCA(n_components=2, svd_solver="full")
        X_train_2d = pca.fit_transform(X_train_scaled)
        X_test_2d  = pca.transform(X_test_scaled)

        x_min, x_max = X_train_2d[:, 0].min()-0.5, X_train_2d[:, 0].max()+0.5
        y_min, y_max = X_train_2d[:, 1].min()-0.5, X_train_2d[:, 1].max()+0.5

        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, grid_res),
            np.linspace(y_min, y_max, grid_res)
        )

        grid_2d = np.c_[xx.ravel(), yy.ravel()]

        #  Reverse PCA → scaled space
        grid_scaled = pca.inverse_transform(grid_2d)

        # Reverse scaling → original feature space
        grid_orig = scaler.inverse_transform(grid_scaled)

        grid_orig = np.clip(
            grid_orig,
            X_train.min(axis=0),
            X_train.max(axis=0)
        )


        grid_t = torch.tensor(grid_orig, dtype=torch.float32, device=device)

        with torch.no_grad():
            logits = model(grid_t)

        if logits.ndim == 1 or logits.shape[1] == 1:
            probs = torch.sigmoid(logits.view(-1)).cpu().numpy()
            Z = probs.reshape(xx.shape)
        else:
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            Z = pred.reshape(xx.shape)

        plt.figure(figsize=(8, 6))
        if logits.ndim == 1 or logits.shape[1] == 1:
            # Binary → probability map
            plt.contourf(xx, yy, Z, levels=30, alpha=0.5)
            plt.colorbar(label="P(class=1)")
        else:
            # Multiclass → discrete regions
            levels = np.arange(Z.max() + 2) - 0.5
            plt.contourf(xx, yy, Z, levels=levels, alpha=0.5)
            plt.colorbar(label="Predicted class")

        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)  

        # Train
        plt.scatter(
            X_train_2d[:, 0], X_train_2d[:, 1],
            c=y_train,
            edgecolor="black",
            s=40,
            label="Train"
        )

        # Test
        plt.scatter(
            X_test_2d[:, 0], X_test_2d[:, 1],
            c=y_test,
            marker="^",
            edgecolor="black",
            s=70,
            label="Test"
        )

        # Rule centers
        if centers_init is not None:
            centers_scaled = scaler.transform(centers_init)
            centers_2d = pca.transform(centers_scaled)

            plt.scatter(
                centers_2d[:, 0],
                centers_2d[:, 1],
                marker="X",
                s=250,
                c="white",
                edgecolor="black",
                linewidth=2,
                label="Rule Centers (K)"
            )

            for i, (cx, cy) in enumerate(centers_2d):
                plt.text(cx, cy, f"K{i}", fontsize=10, weight="bold")

        plt.title(f"PCA Projection | dataset={dataset_name}")
        plt.legend()
        plt.tight_layout()
        plt.show()

    plot_gift_memberships(model, X_train, device=device)



def plot_gift_memberships(model, X_train, device="cpu", grid_res=200):
    import torch
    import numpy as np
    import matplotlib.pyplot as plt

    if not hasattr(model, "mf_layer"):
        print("Model has no mf_layer.")
        return

    mf = model.mf_layer

    if not hasattr(mf, "alpha_logits"):
        print("Not a hierarchical GIFT model.")
        return

    model.eval()

    X_np = X_train
    n_inputs = X_np.shape[1]
    K = mf.K

    for i in range(K):  # one figure per rule

        fig, axes = plt.subplots(1, n_inputs, figsize=(5 * n_inputs, 4))
        if n_inputs == 1:
            axes = [axes]

        for j in range(n_inputs):
            ax = axes[j]

            x_min, x_max = X_np[:, j].min()*2, X_np[:, j].max()*2
            x_vals = np.linspace(x_min, x_max, grid_res)

            # vary only dimension j
            x_tensor = torch.zeros((grid_res, n_inputs), dtype=torch.float32, device=device)
            x_tensor[:, j] = torch.tensor(x_vals, device=device)

            with torch.no_grad():
                outputs = mf(x_tensor, return_parts=True)

            # unpack
            mu_pos  = outputs[1]
            mu_neg  = outputs[2]
            mu_g    = outputs[3]
            mu_l    = outputs[4]

            mu_sym  = outputs[5]
            mu_dir  = outputs[6]
            mu_comb = outputs[7]
            mu      = outputs[8]

            alpha   = outputs[9]
            beta    = outputs[10]
            gamma   = outputs[11]
            delta   = outputs[12]

            # select rule i, dim j
            mu_pos = mu_pos[:, i, j].cpu().numpy()
            mu_neg = mu_neg[:, i, j].cpu().numpy()
            mu_g   = mu_g[:, i, j].cpu().numpy()
            mu_l   = mu_l[:, i, j].cpu().numpy()

            #mu_sym  = mu_sym[:, i, j].cpu().numpy()
            #mu_dir  = mu_dir[:, i, j].cpu().numpy()
            #mu_comb = mu_comb[:, i, j].cpu().numpy()
            mu      = mu[:, i, j].cpu().numpy()

            alpha_val = torch.sigmoid(alpha[0, i, j]).item()
            beta_val  = torch.sigmoid(beta[0, i, j]).item()
            gamma_val = torch.sigmoid(gamma[0, i, j]).item()
            delta_val = torch.sigmoid(delta[0, i, j]).item()

            # ---- plot per dimension (subplot)
            ax.plot(x_vals, mu_pos, label="mu_pos")
            ax.plot(x_vals, mu_neg, label="mu_neg")

            ax.plot(x_vals, mu_g, label="mu_g")
            ax.plot(x_vals, mu_l, label="mu_l")

            #ax.plot(x_vals, mu_sym, "--", label=f"mu_sym α={alpha_val:.2f}")
            #ax.plot(x_vals, mu_dir, "--", label=f"mu_dir β={beta_val:.2f}")

            #ax.plot(x_vals, mu_comb, label=f"mu_comb γ={gamma_val:.2f}")
            ax.plot(x_vals, mu, "--", label=f"final μ δ={delta_val:.2f}", linewidth=2)

            ax.set_title(f"Rule {i}, Dim {j}")
            ax.set_xlabel(f"x[{j}]")
            ax.set_ylabel("Membership")
            ax.grid(True)
            ax.legend()

        plt.tight_layout()
        plt.show()

