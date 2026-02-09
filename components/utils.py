import numpy as np
import torch
import matplotlib.pyplot as plt


def evaluate_anfis_model(
    model,
    X_train, y_train,
    X_test, y_test,
    device="cpu",
    s_mode_choice="alpha_beta",
    centers_init=None,
    dataset_name=None,
    max_rules_to_plot=5,
    grid_res=300,
):
    """
    Clean evaluation utility:
    - prints train/test error (classification acc + error count; regression RMSE)
    - prints per-rule, per-feature Gaussian/Sigmoid participation (raw weights from blend)
    - plots dataset + decision regions (when possible) + rule centers
    - plots μ parts for advanced MF (Gaussian / Sigmoid / Blended) if supported
    """

    model.eval()
    device = torch.device(device)

    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    X_test_t  = torch.tensor(X_test,  dtype=torch.float32, device=device)

    # ---------------------------
    # 1) Infer task type
    # ---------------------------
    with torch.no_grad():
        out_train = model(X_train_t)
        out_test  = model(X_test_t)

    # If output is (B, C) => classification; if (B, 1) could be regression or binary
    out_dim = 1 if out_train.ndim == 1 else out_train.shape[1]
    unique_y = np.unique(y_train)

    # Prefer explicit interpretation:
    # - If y looks integer labels and #classes >= 2 => classification
    # - else regression
    is_int_labels = np.issubdtype(np.asarray(y_train).dtype, np.integer)
    if is_int_labels and len(unique_y) >= 2:
        task_type = "classification"
        n_classes = out_dim  # should match your n_outputs
    else:
        task_type = "regression"
        n_classes = 1

    # ---------------------------
    # 2) Predictions + metrics
    # ---------------------------
    def _summarize_split(split_name, logits_or_preds, y_true):
        if task_type == "regression":
            pred = logits_or_preds.detach().cpu().numpy().reshape(-1)
            y_true = np.asarray(y_true).reshape(-1)
            rmse = float(np.sqrt(np.mean((pred - y_true) ** 2)))
            print(f"{split_name}: RMSE={rmse:.4f}")
            return {"rmse": rmse}

        # classification
        logits = logits_or_preds
        if logits.ndim == 1:
            # rare: (B,) treat as 2-class with threshold (not expected in your setup)
            probs1 = torch.sigmoid(logits)
            pred = (probs1 >= 0.5).long()
        else:
            pred = torch.argmax(logits, dim=1)

        pred_np = pred.detach().cpu().numpy().astype(np.int64)
        y_true_np = np.asarray(y_true).astype(np.int64)

        total = len(y_true_np)
        correct = int((pred_np == y_true_np).sum())
        wrong = total - correct
        acc = correct / max(total, 1)
        err = wrong / max(total, 1)

        print(f"{split_name}: accuracy rate {acc:.3f} ({correct}/{total}) | error {err:.3f} ({wrong})")
        return {"acc": acc, "error_rate": err, "correct": correct, "total": total, "wrong": wrong}

    print("\nFinal Results:")
    print(f"Mode: {s_mode_choice}")
    train_stats = _summarize_split("Train", out_train, y_train)
    test_stats  = _summarize_split("Test",  out_test,  y_test)

    # ---------------------------
    # 3) MF participation (advanced MF only)
    #    Prints raw blend weights: Gaussian=blend, Sigmoid=1-blend
    # ---------------------------
    def _print_mf_participation():
        if not hasattr(model, "mf_layer"):
            return
        mf = model.mf_layer

        # We need alpha/beta or C AND return_parts support
        supports_parts = hasattr(mf, "forward")
        has_blend_params = hasattr(mf, "alpha") or hasattr(mf, "beta") or hasattr(mf, "C")
        if not (supports_parts and has_blend_params):
            return

        # Must call mf_layer(..., return_parts=True)
        try:
            with torch.no_grad():
                w, mu_g, mu_s, blend_exp, mu_blend = mf(X_test_t, return_parts=True)
        except Exception:
            return

        # blend_exp: (B, K, n_inputs)
        # raw participation (weights): gaussian = blend, sigmoid = 1 - blend
        gauss_w = blend_exp.mean(dim=0).detach().cpu().numpy()          # (K, n_inputs)
        sig_w   = (1.0 - blend_exp).mean(dim=0).detach().cpu().numpy()  # (K, n_inputs)

        K, D = gauss_w.shape
        print("\nPer-rule, per-feature MF contributions and raw participation:")
        for r in range(K):
            for f in range(D):
                print(f"Rule {r}, Feature {f}: Raw weight -> Gaussian={gauss_w[r, f]:.4f}, Sigmoid={sig_w[r, f]:.4f}")

    _print_mf_participation()

    # ---------------------------
    # 4) Plot dataset + decision region (when possible)
    # ---------------------------
    def _plot_dataset_and_boundary():
        Xtr = np.asarray(X_train)
        Xte = np.asarray(X_test)
        ytr = np.asarray(y_train)
        yte = np.asarray(y_test)

        n_features = Xtr.shape[1]

        # ---- helper: predict class on a grid
        def _predict_grid(grid_np):
            grid_t = torch.tensor(grid_np, dtype=torch.float32, device=device)
            with torch.no_grad():
                logits = model(grid_t)
            if logits.ndim == 1:
                # not expected for your n_outputs=2 binary
                probs1 = torch.sigmoid(logits).detach().cpu().numpy()
                return probs1, None
            else:
                probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
                pred = np.argmax(probs, axis=1)
                return probs, pred

        # ---- 3D regression (swiss-roll)
        if task_type == "regression" and n_features == 3:
            fig = plt.figure(figsize=(12, 5))
            ax1 = fig.add_subplot(121, projection="3d")
            ax2 = fig.add_subplot(122, projection="3d")

            # true t/y
            ax1.scatter(Xtr[:, 0], Xtr[:, 1], Xtr[:, 2], c=ytr.reshape(-1), s=10)
            ax1.set_title("Train (true target)")
            ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")

            # predicted t/y on test
            with torch.no_grad():
                pred_te = model(torch.tensor(Xte, dtype=torch.float32, device=device)).detach().cpu().numpy().reshape(-1)
            ax2.scatter(Xte[:, 0], Xte[:, 1], Xte[:, 2], c=pred_te, s=10)
            ax2.set_title("Test (predicted target)")
            ax2.set_xlabel("X"); ax2.set_ylabel("Y"); ax2.set_zlabel("Z")

            plt.tight_layout()
            plt.show()
            return

        # ---- 1D regression
        if task_type == "regression" and n_features == 1:
            with torch.no_grad():
                pred_tr = model(torch.tensor(Xtr, dtype=torch.float32, device=device)).detach().cpu().numpy().reshape(-1)
                pred_te = model(torch.tensor(Xte, dtype=torch.float32, device=device)).detach().cpu().numpy().reshape(-1)

            order = np.argsort(Xtr[:, 0])
            plt.figure(figsize=(8, 4))
            plt.scatter(Xtr[:, 0], ytr.reshape(-1), label="Train true", s=25)
            plt.scatter(Xte[:, 0], yte.reshape(-1), label="Test true",  s=40, marker="^")
            plt.plot(Xtr[order, 0], pred_tr[order], "--", label="Train pred")
            plt.title("1D Regression")
            plt.xlabel("x"); plt.ylabel("y")
            plt.legend()
            plt.tight_layout()
            plt.show()
            return

        # ---- Iris special: show petal features (2,3)
        if dataset_name == "iris" and task_type == "classification" and n_features >= 4:
            f1, f2 = 2, 3
            baseline = Xtr.mean(axis=0)

            x_min, x_max = Xtr[:, f1].min() - 0.5, Xtr[:, f1].max() + 0.5
            y_min, y_max = Xtr[:, f2].min() - 0.5, Xtr[:, f2].max() + 0.5
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_res),
                                 np.linspace(y_min, y_max, grid_res))

            grid = np.tile(baseline, (xx.size, 1)).astype(np.float32)
            grid[:, f1] = xx.ravel()
            grid[:, f2] = yy.ravel()

            _, pred = _predict_grid(grid)
            Z = pred.reshape(xx.shape)

            plt.figure(figsize=(8, 6))
            plt.contourf(xx, yy, Z, levels=np.arange(n_classes + 1) - 0.5, alpha=0.6)
            plt.scatter(Xtr[:, f1], Xtr[:, f2], c=ytr, edgecolor="k", s=35, label="Train")
            plt.scatter(Xte[:, f1], Xte[:, f2], c=yte, edgecolor="k", s=55, marker="^", label="Test")

            if centers_init is not None and centers_init.shape[1] >= max(f1, f2) + 1:
                plt.scatter(centers_init[:, f1], centers_init[:, f2], s=200, marker="X", edgecolor="k", label="Rule Centers")

            plt.title(f"Iris decision regions | mode={s_mode_choice}")
            plt.xlabel(f"Feature {f1}"); plt.ylabel(f"Feature {f2}")
            plt.legend()
            plt.tight_layout()
            plt.show()
            return

        # ---- Generic 2D classification plot (moons/circles/spiral)
        if task_type == "classification" and n_features == 2:
            x_min, x_max = Xtr[:, 0].min() - 0.5, Xtr[:, 0].max() + 0.5
            y_min, y_max = Xtr[:, 1].min() - 0.5, Xtr[:, 1].max() + 0.5
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_res),
                                 np.linspace(y_min, y_max, grid_res))
            grid = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)

            probs, pred = _predict_grid(grid)

            plt.figure(figsize=(8, 6))
            if n_classes == 2 and probs is not None:
                # show P(class=1)
                p1 = probs[:, 1].reshape(xx.shape)
                plt.contourf(xx, yy, p1, levels=30, alpha=0.6)
                plt.colorbar(label="P(class=1)")
            else:
                Z = pred.reshape(xx.shape)
                plt.contourf(xx, yy, Z, levels=np.arange(n_classes + 1) - 0.5, alpha=0.6)
                plt.colorbar(label="Predicted class")

            plt.scatter(Xtr[:, 0], Xtr[:, 1], c=ytr, edgecolor="k", s=35, label="Train")
            plt.scatter(Xte[:, 0], Xte[:, 1], c=yte, edgecolor="k", s=55, marker="^", label="Test")

            if centers_init is not None:
                plt.scatter(centers_init[:, 0], centers_init[:, 1], s=200, marker="X", edgecolor="k", label="Rule Centers")

            plt.title(f"Decision regions | dataset={dataset_name} | mode={s_mode_choice}")
            plt.xlabel("Feature 0"); plt.ylabel("Feature 1")
            plt.legend()
            plt.tight_layout()
            plt.show()
            return

        # ---- Fallback for higher-D classification: plot first 2 dims
        if task_type == "classification" and n_features > 2:
            f1, f2 = 0, 1
            baseline = Xtr.mean(axis=0)

            x_min, x_max = Xtr[:, f1].min() - 0.5, Xtr[:, f1].max() + 0.5
            y_min, y_max = Xtr[:, f2].min() - 0.5, Xtr[:, f2].max() + 0.5
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_res),
                                 np.linspace(y_min, y_max, grid_res))

            grid = np.tile(baseline, (xx.size, 1)).astype(np.float32)
            grid[:, f1] = xx.ravel()
            grid[:, f2] = yy.ravel()

            _, pred = _predict_grid(grid)
            Z = pred.reshape(xx.shape)

            plt.figure(figsize=(8, 6))
            plt.contourf(xx, yy, Z, levels=np.arange(n_classes + 1) - 0.5, alpha=0.6)
            plt.colorbar(label="Predicted class")

            plt.scatter(Xtr[:, f1], Xtr[:, f2], c=ytr, edgecolor="k", s=35, label="Train")
            plt.scatter(Xte[:, f1], Xte[:, f2], c=yte, edgecolor="k", s=55, marker="^", label="Test")

            if centers_init is not None and centers_init.shape[1] >= 2:
                plt.scatter(centers_init[:, f1], centers_init[:, f2], s=200, marker="X", edgecolor="k", label="Rule Centers")

            plt.title(f"Decision regions (proj) | mode={s_mode_choice}")
            plt.xlabel(f"Feature {f1}"); plt.ylabel(f"Feature {f2}")
            plt.legend()
            plt.tight_layout()
            plt.show()
            return

    _plot_dataset_and_boundary()

    # ---------------------------
    # 5) Plot μ parts (advanced MF only)
    # ---------------------------
    def _plot_mu_parts_advanced():
        if not hasattr(model, "mf_layer"):
            return
        mf = model.mf_layer

        has_blend_params = hasattr(mf, "alpha") or hasattr(mf, "beta") or hasattr(mf, "C")
        if not has_blend_params:
            return

        # Need access to centers to select x-range per feature
        if not hasattr(mf, "c"):
            return

        K = getattr(model, "K", None) or getattr(mf, "K", None) or mf.c.shape[0]
        D = mf.c.shape[1]

        # Use feature-wise ranges from training data if possible
        Xtr = np.asarray(X_train)
        if Xtr.ndim == 2 and Xtr.shape[1] == D:
            mins = Xtr.min(axis=0)
            maxs = Xtr.max(axis=0)
        else:
            mins = np.full(D, -2.0, dtype=np.float32)
            maxs = np.full(D,  2.0, dtype=np.float32)

        K_to_plot = min(int(K), int(max_rules_to_plot))

        for r in range(K_to_plot):
            fig, axes = plt.subplots(1, D, figsize=(5 * D, 4))
            if D == 1:
                axes = [axes]

            for f in range(D):
                x = np.linspace(mins[f] - 0.5, maxs[f] + 0.5, 400, dtype=np.float32)

                # Build input where only feature f varies, others fixed at their centers for this rule
                x_in = np.zeros((len(x), D), dtype=np.float32)
                with torch.no_grad():
                    centers_rf = mf.c[r].detach().cpu().numpy()
                x_in[:] = centers_rf
                x_in[:, f] = x

                x_t = torch.tensor(x_in, dtype=torch.float32, device=device)

                with torch.no_grad():
                    w, mu_g, mu_s, blend_exp, mu_blend = mf(x_t, return_parts=True)

                mu_gf = mu_g[:, r, f].detach().cpu().numpy()
                mu_sf = mu_s[:, r, f].detach().cpu().numpy()
                mu_bf = mu_blend[:, r, f].detach().cpu().numpy()

                ax = axes[f]
                ax.plot(x, mu_gf, label="Gaussian")
                ax.plot(x, mu_sf, label="Sigmoid")
                ax.plot(x, mu_bf, "--", label="Blended")

                center_val = centers_rf[f]
                ax.axvline(center_val, linestyle=":", alpha=0.7, label=f"center={center_val:.2f}")

                ax.set_title(f"Rule {r}, Feature {f}")
                ax.set_ylim(0, 1.05)
                ax.grid(True)
                ax.legend(loc="upper right")

            plt.suptitle(f"Membership parts (mode={s_mode_choice}) | Rule {r}")
            plt.tight_layout()
            plt.show()

        if K > K_to_plot:
            print(f"\nNote: Plotted μ parts for first {K_to_plot} rules out of {K} total rules.")

    _plot_mu_parts_advanced()
