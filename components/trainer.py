import torch.nn.functional as F


def train_model(
    model,
    X_train, y_train,
    X_val, y_val,
    task_type,
    n_outputs,
    n_epochs=300,
    lr=1e-3,
    batch_size=128,
    weight_decay=0.0,
    patience=30,
    grad_clip=5.0,
    device="cpu",
    beta=0.9,
):
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    model = model.to(device)

    # ---- tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_val_t   = torch.tensor(X_val,   dtype=torch.float32)

    if task_type == "classification":
        if n_outputs == 1:
            # binary (single logit)
            y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
            y_val_t   = torch.tensor(y_val,   dtype=torch.float32).view(-1, 1)
            loss_fn = nn.BCEWithLogitsLoss()
        else:
            # multiclass (C logits)
            y_train_t = torch.tensor(y_train, dtype=torch.long)
            y_val_t   = torch.tensor(y_val,   dtype=torch.long)
            loss_fn = nn.CrossEntropyLoss()
    elif task_type == "regression":
        y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        y_val_t   = torch.tensor(y_val,   dtype=torch.float32).view(-1, 1)
        loss_fn = nn.MSELoss()
    else:
        raise ValueError(f"Unknown task_type: {task_type}")

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    use_reconstruction = getattr(model, "uses_reconstruction", False)

    # ---- optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = float("inf")
    best_state = None
    bad_epochs = 0

    for epoch in range(1, n_epochs + 1):
        # ---- train
        model.train()
        total = 0.0
        n = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)

            if use_reconstruction:
                y_pred, x_hat = model(xb, return_recon=True)
                cls_loss = loss_fn(y_pred, yb)
                recon_loss = F.mse_loss(x_hat, xb)
                loss = beta * cls_loss + (1.0 - beta) * recon_loss
            else:
                y_pred = model(xb)
                loss = loss_fn(y_pred, yb)


            loss.backward()

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()
            total += loss.item() * xb.size(0)
            n += xb.size(0)

        train_loss = total / max(n, 1)

        # ---- validate
        model.eval()
        with torch.no_grad():
            Xv = X_val_t.to(device)
            yv = y_val_t.to(device)

            if use_reconstruction:
                outv, x_hat_v = model(Xv, return_recon=True)
                cls_val_loss = loss_fn(outv, yv)
                recon_val_loss = F.mse_loss(x_hat_v, Xv)
                val_loss = beta * cls_val_loss + (1.0 - beta) * recon_val_loss
            else:
                outv = model(Xv)
                val_loss = loss_fn(outv, yv)


            # simple metric
            metric_str = ""
            if task_type == "classification":
                if n_outputs == 1:
                    preds = (torch.sigmoid(outv) >= 0.5).float()
                    acc = (preds.eq(yv)).float().mean().item()
                else:
                    preds = outv.argmax(dim=1)
                    acc = (preds.eq(yv)).float().mean().item()
                metric_str = f", acc={acc:.4f}"
            else:
                rmse = torch.sqrt(torch.mean((outv - yv) ** 2)).item()
                metric_str = f", rmse={rmse:.4f}"

        # print every so often
        if epoch == 1 or epoch % 100 == 0 or epoch == n_epochs:
            print(f"epoch {epoch:4d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}{metric_str}")

        # ---- early stopping on val
        if val_loss < best_val - 1e-8:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping at epoch {epoch} (best val_loss={best_val:.6f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(device)
    model.eval()
    return model
