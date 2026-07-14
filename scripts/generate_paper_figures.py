#!/usr/bin/env python3
"""Generate paper-style figures and HabermanBoundary reproduction artifacts.

Uses local ``data/haberman.data`` (no UCI fetch) so generation works offline.
Training here is only for regenerating the Haberman visual demo; it does not
alter the project's evaluation protocol.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, TensorDataset

from model.giftshifter import GIFTSHIFTER
from utils.decision_boundary import plot_interpretability_2d
from utils.paper_plot_style import (
    DOUBLE_COLUMN_WIDTH,
    apply_paper_style,
    save_paper_figure,
    style_axis,
)
from utils.relaxation import plot_relaxation_heatmap


SEED = 42


def _style_smoke() -> None:
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(DOUBLE_COLUMN_WIDTH * 0.48, 2.6))
    style_axis(ax, grid=True)
    x = np.linspace(-3, 3, 400)
    ax.plot(x, np.exp(-(x ** 2) / 2), label=r"$\mu(x)$", color="#2E5A88")
    ax.set_xlabel("Input (standardized)")
    ax.set_ylabel("Membership")
    ax.set_title("(a) Membership reference proportions", fontsize=11, fontweight="bold", pad=5)
    leg = ax.legend(fontsize=8.5, framealpha=0.9, borderpad=0.3)
    from utils.paper_plot_style import make_legend_compact
    make_legend_compact(leg)
    out = ROOT / "artifacts/figures/style_smoke/style_smoke"
    paths = save_paper_figure(fig, out)
    plt.close(fig)
    print("Style smoke:", ", ".join(paths))


def _load_haberman_local():
    df = pd.read_csv(ROOT / "data/haberman.data", header=None)
    y = df.iloc[:, 3].to_numpy()
    # Map labels to {0, 1} if needed.
    classes = np.unique(y)
    if set(classes.tolist()) == {1, 2}:
        y = (y == 2).astype(np.int64)
    else:
        y = y.astype(np.int64)
    X = df.iloc[:, :3].to_numpy(dtype=np.float64)
    feature_names = ["Age", "Year", "Positive Axillary Nodes"]
    indices = np.arange(len(X))
    X_train, X_test, y_train, y_test, i_train, i_test = train_test_split(
        X, y, indices, test_size=0.3, random_state=SEED, stratify=y,
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    return (
        X_train_s, y_train, X_test_s, y_test, feature_names,
        scaler, i_train, i_test,
    )


def _train_giftshifter(X_train, y_train, device):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    binary = len(np.unique(y_train)) == 2
    model = GIFTSHIFTER(
        in_features=X_train.shape[1],
        rules=3,
        out_features=1 if binary else len(np.unique(y_train)),
        binary=binary,
        drop_out_p=0.3,
        dtype=torch.float32,
    ).to(device)
    y_tensor = torch.tensor(y_train, dtype=torch.float32 if binary else torch.long)
    loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32), y_tensor),
        batch_size=32, shuffle=True,
    )
    n_epochs = 150
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    scheduler = OneCycleLR(
        optimizer, max_lr=0.01, steps_per_epoch=len(loader), epochs=n_epochs,
    )
    l1 = nn.L1Loss()
    task_loss = nn.BCEWithLogitsLoss() if binary else nn.CrossEntropyLoss()
    model.train()
    for _ in range(n_epochs):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out, recon, _ = model(bx)
            if binary:
                loss = task_loss(out.squeeze(), by.squeeze()) + l1(recon, bx)
            else:
                loss = task_loss(out, by.long()) + l1(recon, bx)
            loss.backward()
            optimizer.step()
            try:
                scheduler.step()
            except ValueError:
                pass
    model.eval()
    return model, optimizer, n_epochs


def _haberman_figures(device) -> Path:
    (
        X_train, y_train, X_test, y_test, feature_names,
        scaler, i_train, i_test,
    ) = _load_haberman_local()
    model, optimizer, n_epochs = _train_giftshifter(X_train, y_train, device)
    idx_nodes = 2
    idx_age = 0
    class_names = [r"Survived $\geq$ 5 years", "Died within 5 years"]
    out = ROOT / "artifacts/figures/HabermanBoundary"
    fig, axes = plot_interpretability_2d(
        model, X_train, y_train,
        feature_x=idx_nodes, feature_y=idx_age,
        feature_names=feature_names, class_names=class_names,
        resolution=300,
        axis_scale="standardized",
        save_path=str(out / "HabermanBoundary"),
        scaler=scaler,
        optimizer=optimizer,
        seed=SEED,
        run_index=0,
        epoch=n_epochs,
        train_indices=i_train,
        test_indices=i_test,
        source_split="train",
        overwrite_artifacts=True,
        metadata={
            "dataset_name": "Haberman",
            "seed": SEED,
            "model_name": "GIFTSHIFTER",
            "notes": "Local CSV + StandardScaler; axes in standardized model space",
        },
    )
    assert axes[0].get_xlim() == axes[1].get_xlim()
    assert axes[0].get_ylim() == axes[1].get_ylim()
    plt.close(fig)

    # Locate the run directory that was just written.
    run_dirs = sorted(
        (ROOT / "artifacts/figures/HabermanBoundary").glob("seed_*"),
        key=lambda p: p.stat().st_mtime,
    )
    if not run_dirs:
        raise RuntimeError("No HabermanBoundary run directory found after generation")
    run_dir = run_dirs[-1]

    # Model-free reproduction.
    from scripts.reproduce_haberman_boundary import reproduce
    reproduce(run_dir)
    print(f"Haberman reproducibility bundle: {run_dir}")

    relax_out = ROOT / "artifacts/figures/HabermanRelaxation/HabermanRelaxation"
    plot_relaxation_heatmap(
        model,
        feature_names=feature_names,
        save_path=str(relax_out),
        title=None,
    )
    plt.close("all")
    return run_dir


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _style_smoke()
    run_dir = _haberman_figures(device)
    from PIL import Image
    png = run_dir / "HabermanBoundary.png"
    im = Image.open(png)
    print(f"Haberman PNG size={im.size} dpi≈{im.info.get('dpi', (None, None))[0]}")
    # Checkpoint / scaler load smoke test.
    import joblib
    ckpt = torch.load(run_dir / "HabermanBoundary_checkpoint.pt", map_location="cpu")
    assert "model_state_dict" in ckpt and len(ckpt["model_state_dict"]) > 0
    _ = joblib.load(run_dir / "HabermanBoundary_scaler.joblib")
    print("Checkpoint + scaler load OK (CPU)")
    smoke = ROOT / "artifacts/figures/style_smoke/style_smoke.png"
    if smoke.exists():
        im2 = Image.open(smoke)
        print(f"Smoke PNG size={im2.size} dpi≈{im2.info.get('dpi', (None,))[0]}")


if __name__ == "__main__":
    main()
