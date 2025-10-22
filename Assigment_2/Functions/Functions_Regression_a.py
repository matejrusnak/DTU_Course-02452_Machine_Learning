from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from typing import Tuple
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import os


def make_preprocessor_with_degree(continuous_cols: list, onehot_cols: list, polynomial_degree: int):
    return ColumnTransformer([
        ('poly_scale_cont',
         make_pipeline(PolynomialFeatures(degree=polynomial_degree, include_bias=False), StandardScaler()),
         continuous_cols),
        ('onehot_passthrough', 'passthrough', onehot_cols)
    ])


def LinReg_hyperparameter_tuning(
        X, y, KFold_value_outer: int, KFold_value_inner: int, random_state: int,
        continuous_cols: list, onehot_cols: list, tuning_parameters: list) -> Tuple[dict, dict]:
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
            preprocessing = make_preprocessor_with_degree(continuous_cols, onehot_cols, degree)
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
        X, y, KFold_value_outer: int, KFold_value_inner: int, random_state: int,
        continuous_cols: list, onehot_cols: list, tuning_parameters: list, polynomial_degree: int, solver: str,
        save_models_path: str, save_models = False) -> Tuple[dict, dict, dict, dict, dict]:
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
        preprocessing = make_preprocessor_with_degree(continuous_cols, onehot_cols, polynomial_degree)
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
        preprocessing_best = make_preprocessor_with_degree(continuous_cols, onehot_cols, polynomial_degree)
        final_model = make_pipeline(preprocessing_best, Ridge(fit_intercept=True, alpha=best_parameter, solver=solver))
        final_model.fit(X_train_outer, y_train_outer)
        y_pred = final_model.predict(X_test_outer)

        #Save models for feature extraction
        if save_models==True:
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


def final_model(X,y, polynomial_degree: int, alpha: float, solver: str,
                continuous_cols: list, onehot_cols: list) -> list:
    preprocessing = make_preprocessor_with_degree(continuous_cols, onehot_cols, polynomial_degree)
    model = make_pipeline(preprocessing, Ridge(fit_intercept=True, alpha=alpha, solver=solver))
    model.fit(X, y)
    return joblib.dump(model, f'Final_model_Ridge_Poly_{polynomial_degree}_lambda_{alpha:.4f}.pkl')


# This function was generated by Copilot - it extract features coefficients to csv for each saved model
def single_feature_extraction(
    model_path: str,
    output_csv: str = "model_coefficients.csv",
) -> None:
    """
    Load one pipeline, extract its feature coefficients and intercept,
    and append them to a CSV.

    Expects model_path to end with:
      Final_model_Ridge_Poly_<degree>_lambda_<alpha>.pkl
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


def _parse_metadata(stem: str) -> Tuple[int, float]:
    """
    Parse polynomial degree and lambda from stem like:
      'Final_model_Ridge_Poly_2_lambda_0.1000'
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