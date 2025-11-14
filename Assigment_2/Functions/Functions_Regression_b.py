from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from dataclasses import dataclass
from sklearn.base import clone
from typing import Any
import numpy as np


@dataclass
class ModelComparisonResult:
    """
    Container for results produced by model_comparison.

    Attributes
    ----------
    baseline_final_dict : dict[int, float]
        Mapping from outer CV fold index to final baseline MSE on the outer test split.
    ridge_final_dict : dict[int, float]
        Mapping from outer CV fold index to final Ridge MSE on the outer test split.
    ann_final_dict : dict[int, float]
        Mapping from outer CV fold index to final ANN MSE on the outer test split.
    ridge_best_alpha_dict : dict[int, Any]
        Mapping from outer CV fold index to the best Ridge hyperparameter (alpha) found in the inner CV.
    ann_best_layer_dict : dict[int, Any]
        Mapping from outer CV fold index to the best ANN hidden-layer-size tuple found in the inner CV.
    y_true : numpy.ndarray
        Concatenated ground-truth target values from all outer test splits.
    y_preds : dict[str, numpy.ndarray]
        Mapping model-name -> concatenated predictions across outer test splits (keys: 'Ridge', 'ANN', 'Baseline').
    """
    baseline_final_dict: dict[int, float]
    ridge_final_dict: dict[int, float]
    ann_final_dict: dict[int, float]
    ridge_best_alpha_dict: dict[int, Any]
    ann_best_layer_dict: dict[int, Any]
    y_true: np.ndarray
    y_preds: dict[str, np.ndarray]


def make_baseline_preprocessor(
        *,
        continuous_cols: list,
        onehot_cols: list
        ) -> ColumnTransformer:
    """
    Create a baseline preprocessing pipeline for tabular data.

    This transformer standardizes continuous features using
    `StandardScaler` and passes categorical one-hot encoded
    features through without modification.

    Parameters
    ----------
    continuous_cols : list
        List of column names corresponding to continuous (numeric) features.
    onehot_cols : list
        List of column names corresponding to categorical features
        that are already one-hot encoded.

    Returns
    -------
    ColumnTransformer
        A scikit-learn `ColumnTransformer` that applies scaling
        to continuous features and passthrough to one-hot features.
    """
    return ColumnTransformer([
        ('scale_cont', StandardScaler(), continuous_cols),
        ('onehot_passthrough', 'passthrough', onehot_cols)
    ])


def make_ridge_preprocessor_with_degree(
        continuous_cols: list,
        onehot_cols: list,
        polynomial_degree: int
        ) -> ColumnTransformer:
    """
    Create a preprocessing pipeline for ridge regression with polynomial features.

    This transformer expands continuous features into polynomial terms
    of a specified degree, scales them with `StandardScaler`, and passes
    one-hot encoded categorical features through unchanged.

    Parameters
    ----------
    continuous_cols : list
        List of column names corresponding to continuous (numeric) features.
    onehot_cols : list
        List of column names corresponding to categorical features
        that are already one-hot encoded.
    polynomial_degree : int
        Degree of polynomial expansion to apply to continuous features.

    Returns
    -------
    ColumnTransformer
        A scikit-learn `ColumnTransformer` that applies polynomial
        feature expansion and scaling to continuous features, and
        passthrough to one-hot features.
    """
    return ColumnTransformer([
        ('poly_scale_cont',
         make_pipeline(PolynomialFeatures(degree=polynomial_degree, include_bias=False), StandardScaler()),
         continuous_cols),
        ('onehot_passthrough', 'passthrough', onehot_cols)
    ])


def make_ann_preprocessor(
        *,
        continuous_cols: list,
        onehot_cols: list
        ) -> ColumnTransformer:
    """
    Create a preprocessing pipeline for artificial neural networks (ANNs).

    This transformer standardizes continuous features using
    `StandardScaler` and passes one-hot encoded categorical
    features through unchanged. This setup is commonly used
    before feeding data into neural networks.

    Parameters
    ----------
    continuous_cols : list
        List of column names corresponding to continuous (numeric) features.
    onehot_cols : list
        List of column names corresponding to categorical features
        that are already one-hot encoded.

    Returns
    -------
    ColumnTransformer
        A scikit-learn `ColumnTransformer` that applies scaling
        to continuous features and passthrough to one-hot features.
    """
    return ColumnTransformer([
        ('scale_cont', StandardScaler(), continuous_cols),
        ('onehot_passthrough', 'passthrough', onehot_cols)
    ])


def ann_hyperparameter_tuning_gridsearch(
        X, y, /, *,
        continuous_cols: list,
        onehot_cols: list,
        random_state: int = 42,
        cv_splits: int = 10,
        n_jobs: int = -1
    ):
    """
    Perform hyperparameter tuning for an Artificial Neural Network (ANN) regressor using GridSearchCV.

    This function builds a preprocessing pipeline that standardizes continuous features,
    passes one-hot encoded categorical features through unchanged, and fits an
    `MLPRegressor` wrapped in a `TransformedTargetRegressor` (with target scaling).
    A grid search is performed over key ANN hyperparameters to minimize mean squared error (MSE).

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix containing both continuous and one-hot encoded categorical features.
    y : array-like of shape (n_samples,)
        Target values for regression.
    continuous_cols : list
        List of column names corresponding to continuous (numeric) features.
    onehot_cols : list
        List of column names corresponding to categorical features that are already one-hot encoded.
    random_state : int, default=42
        Random seed for reproducibility in cross-validation and ANN initialization.
    cv_splits : int, default=10
        Number of cross-validation folds for GridSearchCV.
    n_jobs : int, default=-1
        Number of parallel jobs to run for GridSearchCV.
        `-1` means using all available processors.

    Returns
    -------
    GridSearchCV
        A fitted `GridSearchCV` object containing the best estimator,
        cross-validation results, and best hyperparameters.

    Side Effects
    ------------
    Prints the best hyperparameters and the corresponding cross-validated MSE.

    Notes
    -----
    - The ANN is trained with the Adam optimizer, early stopping, and a maximum of 2000 iterations.
    - The parameter grid includes variations in hidden layer sizes, activation functions,
      L2 regularization strength (`alpha`), learning rate, and batch size.
    - The target variable is scaled using `StandardScaler` inside `TransformedTargetRegressor`.
    """
    preprocessor = make_ann_preprocessor(continuous_cols=continuous_cols, onehot_cols=onehot_cols)
    ann = MLPRegressor(random_state=random_state,
                       solver='adam',
                       max_iter=2000,
                       early_stopping=True,
                       tol=1e-5,
                       n_iter_no_change=20)
    model_pipeline = make_pipeline(preprocessor,
                                   TransformedTargetRegressor(regressor=ann, transformer=StandardScaler()))

    param_grid = {
        'transformedtargetregressor__regressor__hidden_layer_sizes': [
            (1,), (3,), (10,), (50,), (100,), (200,)
        ],
        'transformedtargetregressor__regressor__activation': ['relu', 'tanh'],
        'transformedtargetregressor__regressor__alpha': [1e-4, 1e-3, 1e-2, 1e-1],
        'transformedtargetregressor__regressor__learning_rate_init': [1e-4, 1e-3, 1e-2, 1e-1],
        'transformedtargetregressor__regressor__batch_size': [3, 16, 32]
    }

    cross_fold = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    best_parameters = GridSearchCV(
        estimator=model_pipeline,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=cross_fold,
        n_jobs=n_jobs,
        verbose=2,
        return_train_score=True
    )

    best_parameters.fit(X, y)

    best_params = best_parameters.best_params_
    best_mse = -best_parameters.best_score_

    print(f"Best parameters: {best_params}")
    print(f"Best CV MSE: {best_mse:.4f}")

    return best_parameters


def model_comparison(
        X, y, /, *,
        KFold_value_outer: int, KFold_value_inner: int, random_state: int,
        continuous_cols: list, onehot_cols: list, polynomial_degree: int,
        ridge_alpha: list, ridge_solver: str, ann_hidden_layer_sizes: list[tuple[int]]
        ) -> ModelComparisonResult:
    """
    Perform nested cross-validated model comparison for a baseline regressor, Ridge, and an ANN.

    The function runs an outer K-fold loop for performance estimation. For each outer fold it
    runs an inner K-fold search over pairs of hyperparameters (ridge_alpha, ann_hidden_layer_sizes)
    to select the best Ridge alpha and ANN hidden-layer size. Final models (baseline, Ridge with
    the best alpha, ANN with best hidden-layer-size) are trained on the outer training split and
    evaluated on the outer test split. Results and concatenated predictions are returned in a
    ModelComparisonResult.

    Parameters
    ----------
    X : pandas.DataFrame
        Feature dataframe.
    y : pandas.Series or array-like
        Target vector.
    KFold_value_outer : int
        Number of splits for outer cross-validation.
    KFold_value_inner : int
        Number of splits for inner cross-validation (hyperparameter search).
    random_state : int
        Random seed used for KFold shuffling and model stochasticity.
    continuous_cols : list[str]
        Names of continuous columns to be scaled (and possibly used with PolynomialFeatures).
    onehot_cols : list[str]
        Names of categorical columns to pass through or one-hot encode.
    polynomial_degree : int
        Degree of polynomial features used in the Ridge preprocessing pipeline.
    ridge_alpha : list[float]
        List of Ridge alpha values to evaluate in the inner loop. Must have same length as ann_hidden_layer_sizes.
    ridge_solver : str
        Solver name passed to sklearn.linear_model.Ridge.
    ann_hidden_layer_sizes : list[tuple[int]]
        List of hidden-layer-size tuples for MLPRegressor to evaluate in the inner loop.
        Must have same length as ridge_alpha; zipped one-to-one for inner-loop evaluations.

    Returns
    -------
    ModelComparisonResult
        Dataclass containing per-fold final metrics, best hyperparameters per outer fold,
        concatenated ground-truth values, and concatenated predictions for each model.

    Raises
    ------
    ValueError
        If len(ridge_alpha) != len(ann_hidden_layer_sizes).

    Notes
    -----
    - The function currently prints progress messages for each outer fold; consider replacing
      prints with logging for production use.
    - The returned y_true and y_preds arrays are formed by concatenating the outer test splits
      in the order of the outer KFold iteration.
    - The baseline model uses DummyRegressor(strategy='mean'); modify if a different baseline is desired.
    - Hyperparameter lists (ridge_alpha and ann_hidden_layer_sizes) are zipped and therefore must be
      provided in corresponding order if you intend to test specific alpha/architecture pairs.
    """
    # Raise exception when ridge_alpha and ann_hidden_layer_sizes are not the same length
    if len(ridge_alpha) != len(ann_hidden_layer_sizes):
        raise ValueError(f'Length of ridge_alpha must equal to length of ann_hidden_layer_sizes')
    else:
        pass

    # Returns for Statistical comparison
    y_true = []
    y_preds = {'Ridge': [], 'ANN': [], 'Baseline': []}

    # Return dictionaries for model comparison
    baseline_final_dict, ridge_final_dict, ann_final_dict, ridge_best_alpha_dict, ann_best_layer_dict = {}, {}, {}, {}, {}

    CV_KFold_outer = KFold(n_splits=KFold_value_outer, shuffle=True, random_state=random_state)
    CV_KFold_inner = KFold(n_splits=KFold_value_inner, shuffle=True, random_state=random_state)

    # Outer loop
    for count_outer, (outer_train_idx, outer_test_idx) in enumerate(CV_KFold_outer.split(X), start=1):
        X_train_outer, X_test_outer = X.iloc[outer_train_idx], X.iloc[outer_test_idx]
        y_train_outer, y_test_outer = y.iloc[outer_train_idx], y.iloc[outer_test_idx]

        # Dictionaries {parameter: lowest_avg_mse} from inner loops
        avg_mse_ridge_dict, avg_mse_ann_dict = {}, {}

        # Preprocessing pipelines
        preprocessing_ridge = make_ridge_preprocessor_with_degree(continuous_cols, onehot_cols, polynomial_degree)
        preprocessing_ann = make_ann_preprocessor(continuous_cols=continuous_cols, onehot_cols=onehot_cols)

        # Looping through the parameters
        for a, h in zip(ridge_alpha, ann_hidden_layer_sizes):
            mse_ridge, mse_ann = [], []

            # Inner loop
            for counter_inner, (inner_train_idx, inner_test_idx) in enumerate(CV_KFold_inner.split(X_train_outer), start=1):
                X_train_inner, X_test_inner = X_train_outer.iloc[inner_train_idx], X_train_outer.iloc[inner_test_idx]
                y_train_inner, y_test_inner = y_train_outer.iloc[inner_train_idx], y_train_outer.iloc[inner_test_idx]

                # Ridge regression model
                preprocessing_ridge_clone = clone(preprocessing_ridge)
                model_ridge = make_pipeline(preprocessing_ridge_clone,
                                            Ridge(
                                                fit_intercept=True,
                                                alpha=a,
                                                solver=ridge_solver)
                                            )
                model_ridge.fit(X_train_inner, y_train_inner)
                y_pred_ridge = model_ridge.predict(X_test_inner)
                mse_ridge.append(mean_squared_error(y_test_inner, y_pred_ridge))

                # ANN model
                preprocessing_ann_clone = clone(preprocessing_ann)
                ann_regressor = TransformedTargetRegressor(
                                                        regressor=MLPRegressor(
                                                            hidden_layer_sizes=h,
                                                            activation='relu',
                                                            solver='adam',
                                                            alpha=0.001,
                                                            learning_rate_init=0.001,
                                                            max_iter=2000,
                                                            batch_size=32,
                                                            early_stopping=True,
                                                            tol=1e-5,
                                                            n_iter_no_change=20,
                                                            random_state=random_state
                                                            ),
                                                        transformer=StandardScaler()
                                                        )

                model_ann = make_pipeline(preprocessing_ann_clone, ann_regressor)
                model_ann.fit(X_train_inner, y_train_inner)
                y_pred_ann = model_ann.predict(X_test_inner)
                mse_ann.append(mean_squared_error(y_test_inner, y_pred_ann))

            avg_mse_ridge_dict[a] = np.mean(mse_ridge)
            avg_mse_ann_dict[h] = np.mean(mse_ann)

        # Pick best from Ridge dictionary
        best_ridge_alpha, lowest_ridge_mse = min(avg_mse_ridge_dict.items(), key=lambda kv: kv[1])
        print(f'Outer KFold iteration {count_outer} | '
              f'Inner KFold RIDGE best alpha: {best_ridge_alpha} -> MSE: {lowest_ridge_mse:.4f}')
        ridge_best_alpha_dict[count_outer] = best_ridge_alpha

        # Pick best from ANN dictionary
        best_ann_layer, lowest_ann_mse = min(avg_mse_ann_dict.items(), key=lambda kv: kv[1])
        print(f'Outer KFold iteration {count_outer} | '
              f'Inner KFold ANN best hidden layer count: {best_ann_layer} -> MSE: {lowest_ann_mse:.4f}')
        ann_best_layer_dict[count_outer] = best_ann_layer

        # BASELINE: train on outer loop's X_train/y_train split
        baseline_preprocessing = make_baseline_preprocessor(continuous_cols=continuous_cols, onehot_cols=onehot_cols)
        model_baseline = make_pipeline(baseline_preprocessing, DummyRegressor(strategy='mean'))
        model_baseline.fit(X_train_outer, y_train_outer)
        y_pred_baseline = model_baseline.predict(X_test_outer)

        # RIDGE: model with the best alpha trained on outer loop's X_train/y_train split
        ridge_preprocessing_best = clone(preprocessing_ridge)
        final_ridge_model = make_pipeline(ridge_preprocessing_best,
                                          Ridge(
                                                fit_intercept=True,
                                                alpha=best_ridge_alpha,
                                                solver=ridge_solver)
                                          )
        final_ridge_model.fit(X_train_outer, y_train_outer)
        y_pred_ridge_final = final_ridge_model.predict(X_test_outer)

        # ANN: model with the best hidden layers trained on outer loop's X_train/y_train split
        ann_preprocessing_best = clone(preprocessing_ann)
        final_ann_regressor = TransformedTargetRegressor(
                                                        regressor=MLPRegressor(
                                                        hidden_layer_sizes=best_ann_layer,
                                                        activation='relu',
                                                        solver='adam',
                                                        alpha=0.001,
                                                        learning_rate_init=0.001,
                                                        max_iter=2000,
                                                        batch_size=32,
                                                        early_stopping=True,
                                                        tol=1e-5,
                                                        n_iter_no_change=20,
                                                        random_state=random_state
                                                        ),
                                                        transformer=StandardScaler()
                                                        )
        final_ann_model = make_pipeline(ann_preprocessing_best, final_ann_regressor)
        final_ann_model.fit(X_train_outer, y_train_outer)
        y_pred_ann_final = final_ann_model.predict(X_test_outer)

        # Create final dictionaries for best models
        baseline_final_dict[count_outer] = mean_squared_error(y_test_outer, y_pred_baseline)
        ridge_final_dict[count_outer] = mean_squared_error(y_test_outer, y_pred_ridge_final)
        ann_final_dict[count_outer] = mean_squared_error(y_test_outer, y_pred_ann_final)

        # Create returns for statistical testing
        y_preds['Ridge'].append(y_pred_ridge_final)
        y_preds['ANN'].append(y_pred_ann_final)
        y_preds['Baseline'].append(y_pred_baseline)
        y_true.append(y_test_outer)

    y_true = np.concatenate(y_true)
    y_preds = {model: np.concatenate(model_preds) for model, model_preds in y_preds.items()}

    return ModelComparisonResult(
        baseline_final_dict=baseline_final_dict,
        ridge_final_dict=ridge_final_dict,
        ann_final_dict=ann_final_dict,
        ridge_best_alpha_dict=ridge_best_alpha_dict,
        ann_best_layer_dict=ann_best_layer_dict,
        y_true=y_true,
        y_preds=y_preds
    )

