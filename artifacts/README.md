# Artifacts

Publication figures and reproducible exports live under `artifacts/figures/`.

## HabermanBoundary

Each run is stored under:

```text
artifacts/figures/HabermanBoundary/<run_id>/
```

for example `seed_42_run_00`, and typically contains:

- `HabermanBoundary.pdf` / `.png` — vector + 600 dpi raster
- `HabermanBoundary_points.csv` — exact plotted sample values
- `HabermanBoundary_grids.npz` — mesh / rule / class / probability grids
- `HabermanBoundary_plot_config.json` — layout / style configuration
- `HabermanBoundary_metadata.json` — seed, splits, versions, paths
- `HabermanBoundary_checkpoint.pt` — full `model.state_dict()` (+ config)
- `HabermanBoundary_scaler.joblib` — fitted scaler
- `reproduce_haberman_boundary.py` — bundled copy of the model-free script

### Model-free reproduction (no model / no training)

```bash
python scripts/reproduce_haberman_boundary.py \
  --artifact-dir artifacts/figures/HabermanBoundary/seed_42_run_00
```

### Optional checkpoint recomputation (inference only)

```bash
python scripts/recompute_haberman_boundary.py \
  --artifact-dir artifacts/figures/HabermanBoundary/seed_42_run_00
```

### Generate a fresh demo run from local Haberman CSV

```bash
python scripts/generate_paper_figures.py
```
