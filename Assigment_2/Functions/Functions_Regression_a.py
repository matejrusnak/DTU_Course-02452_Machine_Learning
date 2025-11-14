from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import os


def make_preprocessor_with_degree(
        *, continuous_cols: list, onehot_cols: list, polynomial_degree: int
        ) -> ColumnTransformer:
    """
    Create a preprocessing pipeline for continuous and one-hot encoded features.

    Applies a polynomial feature transformation followed by standard scaling to the specified
    continuous columns, while passing through the one-hot encoded columns unchanged.

    Parameters
    ----------
    continuous_cols : list of str
        Names of continuous feature columns to be transformed with polynomial features and standardized.
    onehot_cols : list of str
        Names of one-hot encoded columns to be passed through without transformation.
    polynomial_degree : int
        Degree of the polynomial transformation applied to continuous columns.

    Returns
    -------
    ColumnTransformer
        A scikit-learn ColumnTransformer that applies the specified preprocessing steps.
    """
    return ColumnTransformer([
        ('poly_scale_cont',
         make_pipeline(PolynomialFeatures(degree=polynomial_degree, include_bias=False), StandardScaler()),
         continuous_cols),
        ('onehot_passthrough', 'passthrough', onehot_cols)
    ])


def LinReg_hyperparameter_tuning(
        X, y, /, *,
        KFold_value_outer: int,
        KFold_value_inner: int,
        random_state: int,
        continuous_cols: list,
        onehot_cols: list,
        tuning_parameters: list
        ) -> tuple[dict, dict]:
    """
    Perform nested cross-validation to tune polynomial degree for linear regression.

    This function uses an outer K-fold loop to evaluate generalization error and an inner
    K-fold loop to select the best polynomial degree for feature transformation. Continuous
    features are transformed using polynomial expansion and standard scaling, while one-hot
    encoded features are passed through unchanged.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix containing both continuous and categorical columns.
    y : pd.Series
        Target variable.
    KFold_value_outer : int
        Number of splits for the outer cross-validation loop.
    KFold_value_inner : int
        Number of splits for the inner cross-validation loop.
    random_state : int
        Random seed for reproducibility.
    continuous_cols : list of str
        Names of continuous feature columns to be transformed.
    onehot_cols : list of str
        Names of one-hot encoded columns to be passed through.
    tuning_parameters : list of int
        List of polynomial degrees to evaluate during inner loop tuning.

    Returns
    -------
    gen_errors_dict : dict[int, float]
        Dictionary mapping outer fold index to generalization error (MSE).
    best_degree_dict : dict[int, int]
        Dictionary mapping outer fold index to the best polynomial degree selected.
    """

    gen_errors_dict, best_degree_dict = {}, {}

    CV_KFold_outer = KFold(n_splits=KFold_value_outer, shuffle=True, random_state=random_state)
    CV_KFold_inner = KFold(n_splits=KFold_value_inner, shuffle=True, random_state=random_state)

    # Outer loop
    for count_outer, (outer_train_idx, outer_test_idx) in enumerate(CV_KFold_outer.split(X), start=1):
        X_train_outer, X_test_outer = X.iloc[outer_train_idx], X.iloc[outer_test_idx]
        y_train_outer, y_test_outer = y.iloc[outer_train_idx], y.iloc[outer_test_idx]

        avg_mse_dict = {}

        # Looping through parameters
        for degree in tuning_parameters:
            mse_list = []
            preprocessing = make_preprocessor_with_degree(continuous_cols=continuous_cols,
                                                          onehot_cols=onehot_cols,
                                                          polynomial_degree=degree)

            # Inner loop
            for counter_inner, (inner_train_idx, inner_test_idx) in enumerate(CV_KFold_inner.split(X_train_outer),
                                                                              start=1):
                X_train_inner, X_test_inner = X_train_outer.iloc[inner_train_idx], X_train_outer.iloc[inner_test_idx]
                y_train_inner, y_test_inner = y_train_outer.iloc[inner_train_idx], y_train_outer.iloc[inner_test_idx]

                model = make_pipeline(preprocessing, LinearRegression(fit_intercept=True))
                model.fit(X_train_inner, y_train_inner)
                y_pred = model.predict(X_test_inner)
                mse_list.append(mean_squared_error(y_test_inner, y_pred))

            avg_mse_dict[degree] = np.mean(mse_list)

        # Printing progress
        best_degree, lowest_MSE = min(avg_mse_dict.items(), key=lambda kv: kv[1])
        print(
            f'Outer KFold iteration {count_outer} | '
            f'Inner KFold best polynomial degree: {best_degree} -> MSE: {lowest_MSE:.4f}')
        best_degree_dict[count_outer] = best_degree

        # Train the model with the best parameter on outer loop's X_train/y_train split
        preprocessing_best = make_preprocessor_with_degree(continuous_cols, onehot_cols, best_degree)
        final_model = make_pipeline(preprocessing_best, LinearRegression(fit_intercept=True))
        final_model.fit(X_train_outer, y_train_outer)
        y_pred = final_model.predict(X_test_outer)
        gen_errors_dict[count_outer] = mean_squared_error(y_test_outer, y_pred)

    return gen_errors_dict, best_degree_dict


def Ridge_hyperparameter_tuning(
        X, y, /, *, KFold_value_outer: int, KFold_value_inner: int, random_state: int,
        continuous_cols: list, onehot_cols: list, tuning_parameters: list, polynomial_degree: int, solver: str,
        save_models_path: str, save_models = False
        ) -> tuple[dict, dict, dict, dict, dict]:
    """
    Perform nested cross-validation to tune Ridge regression hyperparameters.

    This function uses an outer K-fold loop to evaluate generalization error and an inner
    K-fold loop to select the best regularization strength (alpha) for Ridge regression.
    Continuous features are transformed using polynomial expansion and standard scaling,
    while one-hot encoded features are passed through unchanged. Optionally, trained models
    from each outer fold can be saved to disk.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix containing both continuous and categorical columns.
    y : pd.Series
        Target variable.
    KFold_value_outer : int
        Number of splits for the outer cross-validation loop.
    KFold_value_inner : int
        Number of splits for the inner cross-validation loop.
    random_state : int
        Random seed for reproducibility.
    continuous_cols : list of str
        Names of continuous feature columns to be transformed.
    onehot_cols : list of str
        Names of one-hot encoded columns to be passed through.
    tuning_parameters : list of float
        List of alpha values to evaluate during inner loop tuning.
    polynomial_degree : int
        Degree of the polynomial transformation applied to continuous columns.
    solver : str
        Solver to use in Ridge regression (e.g., 'auto', 'svd', 'cholesky').
    save_models_path : str
        Directory path to save trained models from each outer fold.
    save_models : bool, optional
        If True, saves each trained model to disk. Defaults to False.

    Returns
    -------
    gen_errors_dict : dict[int, float]
        Dictionary mapping outer fold index to generalization error (MSE).
    best_parameters_dict : dict[int, float]
        Dictionary mapping outer fold index to the best alpha selected.
    overall_avg_mse : dict[float, float]
        Dictionary mapping each alpha to its average MSE across outer folds.
    y_pred_dict : dict[int, np.ndarray]
        Dictionary mapping outer fold index to predicted target values.
    y_test_outer_dict : dict[int, np.ndarray]
        Dictionary mapping outer fold index to actual target values.
    """
    # Return dictionaries
    gen_errors_dict, best_parameters_dict, y_pred_dict, y_test_outer_dict = {}, {}, {}, {}
    outer_mse_per_alpha = {alpha: [] for alpha in tuning_parameters}

    CV_KFold_outer = KFold(n_splits=KFold_value_outer, shuffle=True, random_state=random_state)
    CV_KFold_inner = KFold(n_splits=KFold_value_inner, shuffle=True, random_state=random_state)
    # Outer loop
    for count_outer, (outer_train_idx, outer_test_idx) in enumerate(CV_KFold_outer.split(X), start=1):
        X_train_outer, X_test_outer = X.iloc[outer_train_idx], X.iloc[outer_test_idx]
        y_train_outer, y_test_outer = y.iloc[outer_train_idx], y.iloc[outer_test_idx]

        avg_mse_dict = {}
        preprocessing = make_preprocessor_with_degree(continuous_cols=continuous_cols,
                                                      onehot_cols=onehot_cols,
                                                      polynomial_degree=polynomial_degree)
        # Looping through the parameters
        for alpha in tuning_parameters:
            mse_list = []
            for counter_inner, (inner_train_idx, inner_test_idx) in enumerate(CV_KFold_inner.split(X_train_outer),
                                                                              start=1):
                X_train_inner, X_test_inner = X_train_outer.iloc[inner_train_idx], X_train_outer.iloc[inner_test_idx]
                y_train_inner, y_test_inner = y_train_outer.iloc[inner_train_idx], y_train_outer.iloc[inner_test_idx]

                model = make_pipeline(preprocessing, Ridge(fit_intercept=True, alpha=alpha, solver=solver))
                model.fit(X_train_inner, y_train_inner)
                y_pred = model.predict(X_test_inner)
                mse_list.append(mean_squared_error(y_test_inner, y_pred))

            avg_mse_dict[alpha] = np.mean(mse_list)

        for alpha, mse in avg_mse_dict.items():
            outer_mse_per_alpha[alpha].append(mse)

        # Printing progress
        best_parameter, lowest_MSE = min(avg_mse_dict.items(), key=lambda kv: kv[1])
        print(f'Outer KFold iteration {count_outer} | '
              f'Inner KFold best alpha: {best_parameter} -> MSE: {lowest_MSE:.4f}')
        best_parameters_dict[count_outer] = best_parameter

        # Train the model with the best alpha on outer loop's X_train/y_train split
        preprocessing_best = make_preprocessor_with_degree(continuous_cols=continuous_cols,
                                                           onehot_cols=onehot_cols,
                                                           polynomial_degree=polynomial_degree)
        final_model = make_pipeline(preprocessing_best, Ridge(
                                                                    fit_intercept=True,
                                                                    alpha=best_parameter,
                                                                    solver=solver))
        final_model.fit(X_train_outer, y_train_outer)
        y_pred = final_model.predict(X_test_outer)

        # Save models for feature extraction
        if save_models is True:
            joblib.dump(final_model, f'{save_models_path}/ridge_fold_{count_outer}_lambda_{best_parameter:.4f}.pkl')
        else:
            None

        gen_errors_dict[count_outer] = mean_squared_error(y_test_outer, y_pred)
        y_pred_dict[count_outer] = y_pred
        y_test_outer_dict[count_outer] = y_test_outer

    overall_avg_mse = {
        alpha: np.mean(mse_list)
        for alpha, mse_list in outer_mse_per_alpha.items()
    }

    return gen_errors_dict, best_parameters_dict, overall_avg_mse, y_pred_dict, y_test_outer_dict


def final_model(
        X,y, /, *,
        polynomial_degree: int, alpha: float, solver: str,
        continuous_cols: list, onehot_cols: list
        ) -> list:
    """
    Train and save a Ridge regression model with polynomial preprocessing.

    This function applies polynomial feature expansion and standard scaling to the specified
    continuous columns, passes through one-hot encoded columns unchanged, fits a Ridge regression
    model using the provided alpha and solver, and saves the trained model to disk.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix containing both continuous and categorical columns.
    y : pd.Series
        Target variable.
    polynomial_degree : int
        Degree of the polynomial transformation applied to continuous columns.
    alpha : float
        Regularization strength for Ridge regression.
    solver : str
        Solver to use in Ridge regression (e.g., 'auto', 'svd', 'cholesky').
    continuous_cols : list of str
        Names of continuous feature columns to be transformed.
    onehot_cols : list of str
        Names of one-hot encoded columns to be passed through.

    Returns
    -------
    list
        A list containing the path to the saved model file.
    """
    preprocessing = make_preprocessor_with_degree(continuous_cols=continuous_cols,
                                                  onehot_cols=onehot_cols,
                                                  polynomial_degree=polynomial_degree)
    model = make_pipeline(preprocessing, Ridge(fit_intercept=True, alpha=alpha, solver=solver))
    model.fit(X, y)
    return joblib.dump(model, f'Final_model_Ridge_Poly_{polynomial_degree}_lambda_{alpha:.4f}.pkl')


# These functions were generated by Copilot - it extract features coefficients to csv for each saved model
def single_feature_extraction(
    model_path: str,
    output_csv: str = "model_coefficients.csv",
    ) -> None:
    """
    Extract feature coefficients and intercept from a saved Ridge regression pipeline.

    Loads a trained scikit-learn pipeline from disk, retrieves the feature names from the
    ColumnTransformer and the corresponding coefficients from the Ridge model, and appends
    them to a CSV file along with the model's polynomial degree and regularization strength.

    The function expects the model filename to follow the pattern:
        Final_model_Ridge_Poly_<degree>_lambda_<alpha>.pkl

    Parameters
    ----------
    model_path : str
        Path to the saved Ridge regression pipeline (.pkl file).
    output_csv : str, optional
        Path to the output CSV file where extracted coefficients will be saved.
        Defaults to "model_coefficients.csv".

    Returns
    -------
    None
        Writes the extracted model parameters to a CSV file.
    """
    path = Path(model_path)
    degree, alpha = _parse_metadata(path.stem)

    pipeline = joblib.load(path)
    ridge = pipeline.named_steps["ridge"]
    transformer = pipeline.named_steps["columntransformer"]

    feature_names = transformer.get_feature_names_out()
    coefficients = ridge.coef_
    intercept = ridge.intercept_

    # Build DataFrame
    df = pd.DataFrame({
        "feature": list(feature_names) + ["intercept"],
        "coefficient": list(coefficients) + [intercept],
    })
    df["poly_degree"] = degree
    df["lambda"] = alpha

    # Write (overwrites or creates file)
    df.to_csv(output_csv, index=False, float_format="%.4f")


def _parse_metadata(stem: str) -> tuple[int, float]:
    """
    Extract polynomial degree and regularization strength from a model filename stem.

    This function parses a filename stem formatted as:
        'Final_model_Ridge_Poly_<degree>_lambda_<alpha>'
    and returns the polynomial degree and lambda (alpha) as a tuple.

    Parameters
    ----------
    stem : str
        Filename stem containing encoded metadata about the model.

    Returns
    -------
    Tuple[int, float]
        A tuple containing:
        - degree : int
            Polynomial degree used in preprocessing.
        - alpha : float
            Regularization strength (lambda) used in Ridge regression.

    Raises
    ------
    ValueError
        If the filename stem does not match the expected format.
    """
    parts = stem.split("_")
    try:
        # find 'Poly' and grab the next part
        poly_idx = parts.index("Poly")
        degree = int(parts[poly_idx + 1])

        # find 'lambda' and grab the next part
        lambda_idx = parts.index("lambda")
        alpha = float(parts[lambda_idx + 1])

        return degree, alpha

    except (ValueError, IndexError):
        raise ValueError(f"Filename stem not in expected format: {stem}")


def multi_model_parameter_extraction(path_to_folder: str, output_csv: str = 'ridge_coefficients.csv'):
    """
    Extract coefficients and metadata from multiple saved Ridge regression models.

    Iterates through a folder of `.pkl` files containing trained Ridge regression pipelines,
    extracts feature coefficients and intercepts, parses fold index and lambda value from
    filenames, and appends the results to a CSV file.

    Expected filename format:
        ridge_folde_<fold>_lambda_<alpha>.pkl

    Parameters
    ----------
    path_to_folder : str
        Path to the folder containing saved Ridge model `.pkl` files.
    output_csv : str, optional
        Path to the output CSV file where extracted coefficients will be saved.
        Defaults to 'ridge_coefficients.csv'.

    Returns
    -------
    None
        Writes extracted model parameters to a CSV file.
    """
    first_write = True  # track whether to write header
    for fname in sorted(os.listdir(path_to_folder)):
        if not fname.endswith('.pkl'):
            continue  # skip non-model files
        # Load model
        model_path = os.path.join(path_to_folder, fname)
        model = joblib.load(model_path)
        # Extract coefficients and feature names
        coefs = model.named_steps['ridge'].coef_
        feature_names = model.named_steps['columntransformer'].get_feature_names_out()
        # Parse fold and lambda from filename
        try:
            # Example: ridge_folde_1_lambda_0.1000.pkl
            parts = fname.replace('.pkl', '').split('_')
            fold = int(parts[2])  # '1' from 'folde_1'
            lam  = float(parts[4])  # '0.1000' from 'lambda_0.1000'
        except Exception as e:
            print(f"Skipping {fname}: couldn't parse fold/lambda → {e}")
            continue
        # Create DataFrame
        df_coefs = pd.DataFrame({
            'feature': feature_names,
            'coefficient': coefs,
            'lambda': lam,
            'fold': fold
        })
        # Optional: add intercept
        intercept = model.named_steps['ridge'].intercept_
        df_intercept = pd.DataFrame({
            'feature': ['intercept'],
            'coefficient': [intercept],
            'lambda': [lam],
            'fold': [fold]
        })

        df_full = pd.concat([df_coefs, df_intercept], ignore_index=True)
        # Write to CSV
        df_full.to_csv(output_csv, index=False, mode='w' if first_write else 'a', header=first_write)
        first_write = False