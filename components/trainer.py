def train_model(model, X_train, y_train, X_test, y_test, task_type, n_epochs=1000, lr=0.001, device="cpu"):
    import torch
    import torch.nn as nn
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # convert labels to tensor
    if task_type == "binary":
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
        y_test_tensor  = torch.tensor(y_test,  dtype=torch.float32).unsqueeze(1).to(device)
        loss_fn = nn.BCEWithLogitsLoss()
    elif task_type == "multiclass":
        y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
        y_test_tensor  = torch.tensor(y_test,  dtype=torch.long).to(device)
        loss_fn = nn.CrossEntropyLoss()
    else:  # regression
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
        y_test_tensor  = torch.tensor(y_test,  dtype=torch.float32).unsqueeze(1).to(device)
        loss_fn = nn.MSELoss()

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_test_tensor  = torch.tensor(X_test,  dtype=torch.float32).to(device)

    # Training loop
    best_loss = float("inf")
    best_state = None
    for epoch in range(1, n_epochs+1):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train_tensor)
        loss = loss_fn(logits, y_train_tensor)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)
    model.eval()
    return model
