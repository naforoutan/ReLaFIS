import torch
import numpy as np
import matplotlib.pyplot as plt

from utils.paper_plot_style import apply_paper_style, save_paper_figure, style_axis


def plot_gift_mfs(model, feature_names=None, num_points=1000, xlim=(-7, 7), save_path=None):
    """
    Plot membership functions in a grid: rows = rules, columns = features.
    
    Parameters:
    - model: trained GIFT instance
    - feature_names: list of feature names (optional)
    - num_points: number of points for the smooth curve
    - xlim: tuple (xmin, xmax), default (-7, 7)
    - save_path: optional stem for PDF/PNG export
    """
    apply_paper_style()
    model.eval()

    # Extract parameters to CPU
    mean = model.mean.detach().cpu().numpy()                     # (in_features, rules)
    std_raw = model.std.detach().cpu().numpy()
    std = np.log(1 + np.exp(std_raw))                            # softplus
    literal_raw = model.literal.detach().cpu().numpy()
    literal = 1.0 / (1.0 + np.exp(-literal_raw))

    sigmoid_slope = model.sigmoid_slope.detach().cpu().numpy()
    temp_raw = model.temp.detach().cpu().numpy()
    temp = 1.0 / (1.0 + np.exp(-temp_raw))

    comb_weight_raw = model.comb_weight.detach().cpu().numpy()
    comb_weight = 1.0 / (1.0 + np.exp(-comb_weight_raw))

    in_features = model.in_features
    rules = model.rules_count

    if feature_names is None:
        feature_names = [f"F{i}" for i in range(in_features)]

    x_vals = np.linspace(xlim[0], xlim[1], num_points)

    # Create grid: rules rows, features columns
    fig, axes = plt.subplots(rules, in_features, figsize=(3 * in_features, 3 * rules), sharex=True, sharey=True)
    if rules == 1 and in_features == 1:
        axes = np.array([[axes]])
    elif rules == 1:
        axes = axes.reshape(1, -1)
    elif in_features == 1:
        axes = axes.reshape(-1, 1)

    for rule_idx in range(rules):
        for feat_idx in range(in_features):
            ax = axes[rule_idx, feat_idx]
            
            mu = mean[feat_idx, rule_idx]
            sigma = std[feat_idx, rule_idx]

            # Gaussian part
            g = np.exp(-((x_vals - mu) ** 2) / (2 * sigma ** 2))
            lit = literal[feat_idx, rule_idx]
            mu_pos_neg = lit * g + (1 - lit) * (1 - g)

            # Sigmoidal part
            slope = sigmoid_slope[feat_idx, rule_idx]
            s = 1.0 / (1.0 + np.exp(-(x_vals - mu) * slope))
            tmp = temp[feat_idx, rule_idx]
            mu_great_less = tmp * s + (1 - tmp) * (1 - s)

            # Combination
            w = comb_weight[feat_idx, rule_idx]
            mu_combined = w * mu_pos_neg + (1 - w) * mu_great_less
            mu_final = mu_combined

            ax.plot(x_vals, mu_final, color='b', linewidth=1.8)
            ax.set_ylim(0, 1)
            ax.set_xlim(xlim)
            style_axis(ax, grid=True)

            # Add labels only on the edges for readability
            if rule_idx == 0:
                ax.set_title(feature_names[feat_idx], fontsize=11, fontweight="bold", pad=4)
            if rule_idx == rules - 1:
                ax.set_xlabel('Input')
            if feat_idx == 0:
                ax.set_ylabel(f'Rule {rule_idx+1}')

    # Captions belong in LaTeX for paper figures.
    plt.tight_layout()
    if save_path:
        save_paper_figure(fig, save_path)
    else:
        plt.show()



def plot_gift_param_progress(history, model_type):
    """
    Plot mean values of literal, temp, weight, and relax all in one plot.
    
    Parameters:
    - history: list of dictionaries from model.get_interpretable_params()
    """
    if not history:
        print("No history data available.")
        return
    
    epochs = [entry['epoch'] for entry in history]
    
    if model_type == 'gift':
        literal_means = [entry['literal_mean'] for entry in history]
        temp_means    = [entry['temp_mean'] for entry in history]
        weight_means  = [entry['weight_mean'] for entry in history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, literal_means, 'b-', label='Literal', linewidth=2)
        plt.plot(epochs, temp_means,    'g-', label='Temp', linewidth=2)
        plt.plot(epochs, weight_means,  'r-', label='Weight', linewidth=2)
    
    elif model_type == 'litanfis':
        literal_means = [entry['literal_mean'] for entry in history]
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, literal_means, 'b-', label='Literal', linewidth=2)

    
    plt.xlabel('Epoch')
    plt.ylabel('Mean parameter value')
    plt.title('Evolution of GIFT interpretable parameters')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()



def plot_litanfis_mfs(model, feature_names=None, num_points=1000, xlim=(-7, 7)):
    """
    Plot membership functions for a LitAnfis model.
    Grid layout: rows = rules, columns = features.
    
    Parameters:
    - model: trained LitAnfis instance
    - feature_names: list of feature names (optional)
    - num_points: number of points for the smooth curve
    - xlim: tuple (xmin, xmax), default (-7, 7)
    """
    model.eval()

    # Extract parameters to CPU
    mean = model.mean.detach().cpu().numpy()          # (in_features, rules)
    std = model.std.detach().cpu().numpy()            # (in_features, rules)
    literal_raw = model.literal.detach().cpu().numpy()
    literal = 1.0 / (1.0 + np.exp(-literal_raw))      # sigmoid

    in_features = model.in_features
    rules = model.rules_count

    if feature_names is None:
        feature_names = [f"F{i}" for i in range(in_features)]

    x_vals = np.linspace(xlim[0], xlim[1], num_points)

    # Create grid: rules rows, features columns
    fig, axes = plt.subplots(rules, in_features, figsize=(3 * in_features, 3 * rules), sharex=True, sharey=True)
    if rules == 1 and in_features == 1:
        axes = np.array([[axes]])
    elif rules == 1:
        axes = axes.reshape(1, -1)
    elif in_features == 1:
        axes = axes.reshape(-1, 1)

    for rule_idx in range(rules):
        for feat_idx in range(in_features):
            ax = axes[rule_idx, feat_idx]

            mu = mean[feat_idx, rule_idx]
            sigma = std[feat_idx, rule_idx]
            lit = literal[feat_idx, rule_idx]

            # Gaussian membership
            g = np.exp(-((x_vals - mu) ** 2) / (2 * sigma ** 2))
            # Literal transformation
            mu_final = lit * g + (1 - lit) * (1 - g)

            ax.plot(x_vals, mu_final, color='b')
            ax.set_ylim(0, 1)
            ax.set_xlim(xlim)
            ax.grid(True, alpha=0.3)

            # Labels on edges for readability
            if rule_idx == 0:
                ax.set_title(feature_names[feat_idx])
            if rule_idx == rules - 1:
                ax.set_xlabel('Input')
            if feat_idx == 0:
                ax.set_ylabel(f'Rule {rule_idx+1}')

    fig.suptitle('LitANFIS Membership Functions (rows=rules, cols=features)', fontsize=14)
    plt.tight_layout()
    plt.show()



import matplotlib.pyplot as plt
import numpy as np

def plot_param_evolution_grid(history, param_name, model_type, feature_names=None):
    """
    Plot the evolution of a parameter (literal, temp, weight, relax) over epochs
    in a grid: rows = rules, columns = features.

    Parameters:
    - history: list of dicts from get_interpretable_params()
    - param_name: one of 'literal', 'temp', 'weight', 'relax'
    - model_type: 'gift' or 'litanfis'
    - feature_names: optional list of feature names (length = in_features)
    """
    if not history:
        print("No history data available.")
        return

    # Get the matrix shape from the first entry
    first_entry = history[0]
    matrix_key = f"{param_name}_matrix"
    if matrix_key not in first_entry:
        print(f"Parameter '{param_name}' not found in history (available: {list(first_entry.keys())})")
        return

    matrix = first_entry[matrix_key]          # shape (in_features, rules)
    in_features, rules = matrix.shape

    # Prepare data: for each (feat, rule) collect values over epochs
    epochs = [entry['epoch'] for entry in history]
    # Initialize list of lists: time_series[feat][rule] = list of values
    time_series = [[[] for _ in range(rules)] for _ in range(in_features)]

    for entry in history:
        mat = entry[matrix_key]               # (in_features, rules)
        for f in range(in_features):
            for r in range(rules):
                time_series[f][r].append(mat[f, r])

    if feature_names is None:
        feature_names = [f"Feature {f}" for f in range(in_features)]

    # Create grid: rows = rules, columns = features
    fig, axes = plt.subplots(rules, in_features, figsize=(3 * in_features, 3 * rules),
                             sharex=True, sharey=True)
    if rules == 1 and in_features == 1:
        axes = np.array([[axes]])
    elif rules == 1:
        axes = axes.reshape(1, -1)
    elif in_features == 1:
        axes = axes.reshape(-1, 1)

    for r in range(rules):          # rule index (row)
        for f in range(in_features): # feature index (col)
            ax = axes[r, f]
            values = time_series[f][r]   # list of values over epochs
            ax.plot(epochs, values, 'b-', linewidth=1.5)
            ax.set_ylim(0, 1)            # all parameters are in (0,1) after sigmoid
            ax.grid(True, alpha=0.3)

            # Labels only on edges
            if r == 0:
                ax.set_title(feature_names[f], fontsize=10)
            if r == rules - 1:
                ax.set_xlabel('Epoch')
            if f == 0:
                ax.set_ylabel(f'Rule {r+1}', fontsize=10)

    fig.suptitle(f'{param_name.capitalize()} parameter evolution (rows=rules, cols=features)', fontsize=14)
    plt.tight_layout()
    plt.show()