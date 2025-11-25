from sklearn.metrics import r2_score
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_hist(df: pd.DataFrame, *, columns: list[str], units: list[str]) -> None:
    """
    Plot histograms and KDE curves for specified DataFrame columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the features to plot.
    columns : list of str
        List of column names to visualize.
    units : list of str
        List of units corresponding to each feature, used for axis labeling.

    Returns
    -------
    None
        Displays a matplotlib figure with subplots for each feature.
    """
    fig, axs = plt.subplots(1, 5, figsize=(14, 4))
    fig.suptitle("Distribution of features")
    for ax, col, unit in zip(axs, columns, units):
        data = df[col]
        ax.hist(df[col], bins=20, density=True, alpha=1, rwidth=0.9, color='#0472b9')

        kde = gaussian_kde(data)
        x_vals = np.linspace(data.min(), data.max(), 200)
        y_vals = kde(x_vals)

        ax.plot(x_vals, y_vals, color='black', lw=2, label='density')
        ax.set_xlabel(f"{col} [{unit}]")
        ax.set_ylabel('Density')
        ax.legend(frameon=False, fontsize='small')

    fig.tight_layout()
    plt.show()


def plot_alpha_vs_mse(overall_avg_mse: dict[int, int]) -> None:
    """
    Plot the relationship between log-scaled regularization strength and Mean Squared Error.

    Parameters
    ----------
    overall_avg_mse : dict of int to int
        Dictionary mapping regularization parameter values (alpha) to their corresponding
        Mean Squared Error from a linear regression model.

    Returns
    -------
    None
        Displays a matplotlib plot showing generalization error vs log10(alpha),
        with the best-performing alpha highlighted.
    """
    alphas_sorted = sorted(overall_avg_mse.keys())
    log_alphas = [np.log10(a) for a in alphas_sorted]
    MSEs = [overall_avg_mse[a] for a in alphas_sorted]

    best_alpha = min(overall_avg_mse, key=overall_avg_mse.get)
    best_log_alpha = np.log10(best_alpha)
    best_mse = overall_avg_mse[best_alpha]

    plt.figure(figsize=(8, 5))
    plt.plot(log_alphas, MSEs, '--o', color='k', markerfacecolor='#0472b9', markeredgecolor='#0472b9')

    plt.scatter(best_log_alpha, best_mse, color='#d6521a', s=100, zorder=5, label=f'Best λ={best_alpha}')

    plt.xlabel('log10(λ)', fontsize=10, fontweight='bold')
    plt.ylabel('MSE', fontsize=10, fontweight='bold')
    plt.xticks(fontsize=10, weight='bold')
    plt.yticks(fontsize=10, weight='bold')
    plt.title('Generalization error vs λ', fontsize=16, fontweight='bold')
    plt.grid(ls=':', alpha=0.3)
    plt.legend(fontsize='large')
    plt.show()


def plot_actual_vs_pred(
        y_pred_dict: dict[int, np.ndarray], y_test_outer_dict: dict[int, np.ndarray],
        gen_errors_dict: dict[int, np.ndarray], /, *, polynomial_degree: int
        ) -> None:
    """
    Visualize predicted vs actual target values across outer folds of cross-validation.

    For each outer fold, this function plots:
    - A scatter plot of predicted vs actual values.
    - A fitted polynomial regression line of specified degree.
    - A reference 45° line representing perfect prediction.

    Parameters
    ----------
    y_pred_dict : dict[int, np.ndarray]
        Dictionary mapping outer fold indices to predicted target values.
    y_test_outer_dict : dict[int, np.ndarray]
        Dictionary mapping outer fold indices to actual target values.
    gen_errors_dict : dict[int, float]
        Dictionary mapping outer fold indices to generalization error (MSE).
    polynomial_degree : int
        Degree of the polynomial used to fit the trend line in each subplot.

    Returns
    -------
    None
        Displays a matplotlib figure with subplots for each outer fold.
    """
    folds = sorted(gen_errors_dict.keys())

    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(12, 6), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, fold in zip(axes, folds):
        y_true = y_test_outer_dict[fold]
        y_pred = y_pred_dict[fold]
        mse = gen_errors_dict[fold]
        r2 = r2_score(y_true, y_pred)

        coeffs = np.polyfit(y_true, y_pred, deg=polynomial_degree)
        poly_fn = np.poly1d(coeffs)
        x_rng = np.linspace(y_true.min(), y_true.max(), 200)

        ax.scatter(y_true, y_pred, alpha=0.7, color='#0472b9', label='Data')
        ax.plot(x_rng, poly_fn(x_rng), 'k--', lw=2, label=f'Poly deg={polynomial_degree}')
        ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r-', lw=1, label='45° fit')

        ax.set_title(f'Fold {fold} | MSE={mse:.2f} | R²={r2:.2f}', fontsize=8, fontweight='bold')
        ax.set_xlabel('Actual MPG', fontsize=8)
        ax.set_ylabel('Predicted MPG', fontsize=8)
        ax.legend(loc='upper left', fontsize='x-small', frameon=True)
        plt.setp(ax.get_xticklabels(), fontsize=8)
        plt.setp(ax.get_yticklabels(), fontsize=8)

    fig.suptitle('Actual vs Predicted MPG Across 10 Outer Folds',
                 fontsize=16, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
