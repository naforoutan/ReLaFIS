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

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_val_t   = torch.tensor(X_val,   dtype=torch.float32)

    if task_type == "classification":
        if n_outputs == 1:
            y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
            y_val_t = torch.tensor(y_val,   dtype=torch.float32).view(-1, 1)
            loss_fn = nn.BCEWithLogitsLoss()
        else:
            y_train_t = torch.tensor(y_train, dtype=torch.long)
            y_val_t = torch.tensor(y_val,   dtype=torch.long)
            loss_fn = nn.CrossEntropyLoss()
    elif task_type == "regression":
        y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        y_val_t = torch.tensor(y_val,   dtype=torch.float32).view(-1, 1)
        loss_fn = nn.MSELoss()
    else:
        raise ValueError(f"Unknown task_type: {task_type}")

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    # separate param groups — logit gates get 10x lower lr
    gate_params = []
    mf_core_params = []
    other_params = []

    for name, param in model.named_parameters():
        if any(k in name for k in ("alpha_logits", "beta_logits", "gamma_logits", "rho_logits")):
            gate_params.append(param)
        elif "mf_layer" in name:
            mf_core_params.append(param)
        else:
            other_params.append(param)

    optimizer = torch.optim.Adam([
        {"params": gate_params, "lr": lr * 0.1},   # logits saturate fast — keep small
        {"params": mf_core_params, "lr": lr * 0.3},   # centers/spreads — moderate
        {"params": other_params, "lr": lr},          # consequents — full lr
    ], weight_decay=weight_decay)

    RHO_FREEZE_EPOCHS = 100  # rho_logits don't train for first 100 epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=lr * 1e-2
    )

    for epoch in range(1, n_epochs + 1):

        progress = (epoch - 1) / max(n_epochs - 1, 1)
        beta = beta_start + (beta_end - beta_start) * (progress ** 2)
        beta = min(beta, beta_end)

        model.train()
        total = 0.0
        n = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)

            y_pred, phi = model(xb, return_phi=True)
            loss = loss_fn(y_pred, yb)

            # encourage rule specialization
            loss -= 1e-3 * phi.std(dim=1).mean()

            # gate regularization
            if hasattr(model, "mf_layer"):
                mf = model.mf_layer

                if hasattr(mf, "gamma_logits"):
                    gamma = torch.sigmoid(mf.gamma_logits)
                    rho   = torch.sigmoid(mf.rho_logits)

                    # push gamma away from 0.5
                    reg_gamma = (gamma * (1 - gamma)).mean()

                    # penalize relaxation
                    reg_rho = (rho ** 2).mean()

                    loss = loss + 1e-3 * reg_gamma + 1e-2 * reg_rho

            loss.backward()

            # gradient clipping (already present — kept)
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            if epoch <= RHO_FREEZE_EPOCHS:
                if hasattr(model, "mf_layer") and hasattr(model.mf_layer, "rho_logits"):
                    model.mf_layer.rho_logits.grad = None

            optimizer.step()

            total += loss.item() * xb.size(0)
            n += xb.size(0)

        # cont.: step scheduler every epoch
        scheduler.step()

        train_loss = total / max(n, 1)

        model.eval()
        with torch.no_grad():
            Xv = X_val_t.to(device)
            yv = y_val_t.to(device)

            outv = model(Xv)
            val_loss = loss_fn(outv, yv)

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

            # log gate health every 100 epochs
            if (epoch == 1 or epoch % 100 == 0 or epoch == n_epochs):
                if hasattr(model, "mf_layer") and hasattr(model.mf_layer, "rho_logits"):
                    mf = model.mf_layer
                    rho_mean   = torch.sigmoid(mf.rho_logits).mean().item()
                    phi_std    = phi.std(dim=1).mean().item()  # diversity of rule firing
                    phi_max    = phi.max(dim=1).values.mean().item()
                    gate_str = f" | rho={rho_mean:.3f} phi_std={phi_std:.3f} phi_max={phi_max:.3f}"
                else:
                    gate_str = ""

                print(
                    f"epoch {epoch:4d} | train_loss={train_loss:.6f} | "
                    f"val_loss={val_loss:.6f}{metric_str}{gate_str}"
                )

    model = model.to(device)
    model.eval()
    return model