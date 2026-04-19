import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

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
    grad_clip=1.0,
    device="cpu",
    beta_start=0.1,
    beta_end=1.0,
):
    
    model = model.to(device)

    # ---- tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_val_t   = torch.tensor(X_val,   dtype=torch.float32)

    if task_type == "classification":
        if n_outputs == 1:
            y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
            y_val_t   = torch.tensor(y_val,   dtype=torch.float32).view(-1, 1)
            loss_fn = nn.BCEWithLogitsLoss()
        else:
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

    # ---- different learning rates
    mf_params = []
    other_params = []

    for name, param in model.named_parameters():
        if "mf_layer" in name:
            mf_params.append(param)
        else:
            other_params.append(param)

    optimizer = torch.optim.Adam([
        {"params": mf_params, "lr": lr * 0.3},
        {"params": other_params, "lr": lr},
    ], weight_decay=weight_decay)

    for epoch in range(1, n_epochs + 1):

        # ---- better beta schedule (quadratic)
        progress = (epoch - 1) / (n_epochs - 1)
        beta = beta_start + (beta_end - beta_start) * (progress ** 2)
        beta = min(beta, beta_end)

        # ---- train
        model.train()
        total = 0.0
        n = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)

            y_pred, phi = model(xb, return_phi=True)
            loss = loss_fn(y_pred, yb)

            entropy = -(phi * torch.log(phi + 1e-8)).sum(dim=1).mean()
            loss += 1e-3 * entropy

            # ---- gate regularization
            if hasattr(model, "mf_layer"):
                mf = model.mf_layer
                if hasattr(mf, "alpha_logits"):
                    alpha = torch.sigmoid(mf.alpha_logits)
                    beta_g = torch.sigmoid(mf.beta_logits)
                    gamma = torch.sigmoid(mf.gamma_logits)
                    delta = torch.sigmoid(mf.delta_logits)

                    reg = (
                        (alpha * (1 - alpha)).mean() +
                        (beta_g * (1 - beta_g)).mean() +
                        (gamma * (1 - gamma)).mean() +
                        (delta * (1 - delta)).mean()
                    )

                    loss = loss + 1e-3 * reg

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

            outv = model(Xv)
            val_loss = loss_fn(outv, yv)

            # ---- metrics
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

        if epoch == 1 or epoch % 100 == 0 or epoch == n_epochs:
            print(f"epoch {epoch:4d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}{metric_str}")

    model = model.to(device)
    model.eval()
    return model