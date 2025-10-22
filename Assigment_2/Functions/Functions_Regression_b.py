from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.base import clone
from typing import Tuple
import numpy as np


def make_baseline_preprocessor(continuous_cols: list, onehot_cols: list):
    return ColumnTransformer([
        ('scale_cont', StandardScaler(), continuous_cols),
        ('onehot_passthrough', 'passthrough', onehot_cols)
    ])


def make_ridge_preprocessor_with_degree(continuous_cols: list, onehot_cols: list, polynomial_degree: int):
    return ColumnTransformer([
        ('poly_scale_cont',
         make_pipeline(PolynomialFeatures(degree=polynomial_degree, include_bias=False), StandardScaler()),
         continuous_cols),
        ('onehot_passthrough', 'passthrough', onehot_cols)
    ])


def make_ann_preprocessor(continuous_cols: list, onehot_cols: list):
    return ColumnTransformer([
        ('scale_cont', StandardScaler(), continuous_cols),
        ('onehot_passthrough', 'passthrough', onehot_cols)
    ])


def ann_hyperparameter_tuning_gridsearch(
        X, y,
        continuous_cols, onehot_cols,
        random_state: int = 42,
        cv_splits: int = 10,
        n_jobs: int = -1
    ):
    '''
    Returns fitted GridSearchCV and prints best params and MSE.
    '''

    preprocessor = make_ann_preprocessor(continuous_cols, onehot_cols)
    ann = MLPRegressor(random_state=random_state,
                       solver = 'adam',
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


def model_comaparison(
        X, y, KFold_value_outer: int, KFold_value_inner: int, random_state: int,
        continuous_cols: list, onehot_cols: list, polynomial_degree: int,
        ridge_alpha: list, ridge_solver: str, ann_hidden_layer_sizes:list
        ) -> Tuple[dict, dict, dict, dict, dict]:
    '''
    Returns three dictionaries with MSE for Baseline model, Linear Regression model, and ANN model.
    Also returns a dictionary with lists of the best alpha values for LR and
    hidden layers number for ANN for each outer fold.
    '''

    # Return dictionaries
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
        preprocessing_ann = make_ann_preprocessor(continuous_cols, onehot_cols)

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
        baseline_preprocessing = make_baseline_preprocessor(continuous_cols, onehot_cols)
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

    return baseline_final_dict, ridge_final_dict, ann_final_dict, ridge_best_alpha_dict, ann_best_layer_dict
