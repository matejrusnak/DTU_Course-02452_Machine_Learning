import scipy.stats as st
from typing import Callable
from dataclasses import dataclass
import numpy as np


@dataclass
class ConfidenceInterval:
    """
    Confidence interval result for a mean loss estimate.

    Attributes
    ----------
    mean_estimated_error: float
        The sample mean of the per-sample loss values computed on the provided test set.
    confidence_int: tuple[float, float]
        Two-element tuple (ci_lower, ci_upper) representing the two-sided
        (1 - alpha) confidence interval for the true mean loss.
    p-value: float
        p-value for the null hypothesis that the true mean difference equals zero.
        Smaller p-values provide stronger evidence against the null hypothesis.
    """
    mean_estimated_error: float
    confidence_int: tuple[float, float]
    p_value: float


def loss_function() -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Return a function that computes per-sample squared errors for a single model."""
    return lambda y_true, y_pred: (np.asarray(y_true) - np.asarray(y_pred))**2


def confidence_interval_comparison(
        y_true: np.ndarray,
        y_preds_A: np.ndarray,
        y_preds_B: np.ndarray,
        loss_function: Callable[[np.ndarray, np.ndarray], np.ndarray],
        /, *,
        alpha=0.05) -> ConfidenceInterval:
    """
    Compute the mean loss and a two-sided (1 - alpha) confidence interval for the mean loss
    using the Student t-distribution.

    Parameters
    ----------
    y_true: np.ndarray
        1-D array of true target values for the test set.
    y_preds_A: np.ndarray
        1-D array of predicted values for a single model
    y_preds_B: np.ndarray
        1-D array of predicted values for a single model
    loss_function : Callable[[np.ndarray, np.ndarray], np.ndarray]
        Callable that accepts (y_true, y_pred) and returns a 1-D array of per-sample losses
        (e.g., squared errors).
    alpha : float, optional
        Significance level for the two-sided confidence interval (default 0.05).

     Returns
    -------
    ConfidenceInterval
        Mean_estimated_error: float
            The sample mean of the per-sample losses.
        Confidence_int: tuple[float, float]
            (ci_lower, ci_upper) bounds of the (1 - alpha) t-based confidence interval for the true mean loss.
    """
    estimated_error = loss_function(y_true, y_preds_A) - loss_function(y_true, y_preds_B)
    mean_estimated_error = float(np.mean(estimated_error))

    df = len(y_true) - 1

    sem = np.sqrt(sum(((estimated_error - mean_estimated_error) ** 2) / (len(y_true) * df)))
    confidence_int = st.t.interval(1 - alpha, df=df, loc=mean_estimated_error, scale=sem)

    t_stat = -np.abs(np.mean(estimated_error)) / st.sem(estimated_error)
    p_value = 2 * st.t.cdf(t_stat, df=df)

    return ConfidenceInterval(
        mean_estimated_error=mean_estimated_error,
        confidence_int=confidence_int,
        p_value=p_value)
