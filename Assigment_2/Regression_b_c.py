from Functions.Functions_DataProcessing import load_dataset
from Functions.Functions_Regression_b import *
from Functions.Functions_Regression_c import *
import pandas as pd


pd.set_option('display.max_columns', None)
df = load_dataset(r'C:\Users\mrusn\PycharmProjects\Temp\Data\Data_processed.csv')
X = df.drop(columns=['mpg'])
y = df['mpg']

# Define columns for standardization
continuous_cols = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year']
onehot_cols = ['origin_europe', 'origin_japan', 'origin_usa']
'''
# Grid search for best ANN parameters
ann_hyperparameter_tuning_gridsearch(X, y, continuous_cols, onehot_cols,
        random_state=42,
        cv_splits=10,
        n_jobs=-1
    )
'''
alphas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1e3, 1e4]
ann_hidden_layers = [(1,), (3,), (25,), (50,), (75,), (100,), (125,), (150,), (175,), (200,)]

results = model_comparison(
        X, y, KFold_value_outer=10, KFold_value_inner=10, random_state=42,
        continuous_cols=continuous_cols, onehot_cols=onehot_cols, polynomial_degree=2,
        ridge_alpha=alphas, ridge_solver='auto', ann_hidden_layer_sizes=ann_hidden_layers
        )

baseline_final_dict = results.baseline_final_dict
ridge_final_dict = results.ridge_final_dict
ann_final_dict = results.ann_final_dict
ridge_best_alpha_dict = results.ridge_best_alpha_dict
ann_best_layer_dict = results.ann_best_layer_dict
y_true = results.y_true
y_preds = results.y_preds

folds = sorted(baseline_final_dict.keys())
rows = []

for fold in folds:
    baseline_MSE = baseline_final_dict[fold]
    ridge_MSE = ridge_final_dict[fold]
    ann_MSE = ann_final_dict[fold]
    ridge_best_alpha = ridge_best_alpha_dict[fold]
    ann_best_layers = ann_best_layer_dict[fold]

    print(f'KFold: {fold} | Baseline model | Generalization error: {baseline_MSE:.3f}')
    print(f'KFold: {fold} | Ridge model | Alpha: {ridge_best_alpha} | Generalization error: {ridge_MSE:.3f}')
    print(f'KFold: {fold} | ANN model | Layers: {ann_best_layers} | Generalization error: {ann_MSE:.3f}')

    rows.append([fold, str(ann_best_layers), ann_MSE, ridge_best_alpha, ridge_MSE, baseline_MSE])

top = ["Outer fold", "ANN", "ANN", "Linear regression", "Linear regression", "Baseline"]
second = ["", "i h*", "i E_test", "i λ*", "i E_test", "i E_test"]
multi_cols = pd.MultiIndex.from_arrays([top, second])

df = pd.DataFrame(rows, columns=multi_cols)

# Statistical comparison of the models
loss_funct = loss_function()

model_names = []
for model_name in y_preds.keys():
    model_names.append(model_name)

# Statistical comparison of model A and B
stat_results = confidence_interval_comparison(
                                        y_true,
                                        y_preds[model_names[0]],
                                        y_preds[model_names[1]],
                                        loss_funct, alpha=0.05
                                        )

mean_estimated_error = stat_results.mean_estimated_error
confidence_int = stat_results.confidence_int
p_value = stat_results.p_value

print(f"Difference in loss between {model_names[0]} and {model_names[1]}:"
      f"\n    Mean Estimated Error: {mean_estimated_error:.4f}, "
      f"\n    Confidence interval: [{confidence_int[0]:.4f}, {confidence_int[1]:.4f}], "
      f"\n    p-value: {p_value:.4f}")

# Statistical comparison of model A and C
stat_results = confidence_interval_comparison(
                                        y_true,
                                        y_preds[model_names[0]],
                                        y_preds[model_names[2]],
                                        loss_funct, alpha=0.05
                                        )

mean_estimated_error = stat_results.mean_estimated_error
confidence_int = stat_results.confidence_int
p_value = stat_results.p_value

print(f"Difference in loss between {model_names[0]} and {model_names[2]}:"
      f"\n    Mean Estimated Error: {mean_estimated_error:.4f}, "
      f"\n    Confidence interval: [{confidence_int[0]:.4f}, {confidence_int[1]:.4f}], "
      f"\n    p-value: {p_value:.4f}")

# Statistical comparison of model B and C
stat_results = confidence_interval_comparison(
                                        y_true,
                                        y_preds[model_names[1]],
                                        y_preds[model_names[2]],
                                        loss_funct, alpha=0.05
                                        )

mean_estimated_error = stat_results.mean_estimated_error
confidence_int = stat_results.confidence_int
p_value = stat_results.p_value

print(f"Difference in loss between {model_names[1]} and {model_names[2]}:"
      f"\n    Mean Estimated Error: {mean_estimated_error:.4f}, "
      f"\n    Confidence interval: [{confidence_int[0]:.4f}, {confidence_int[1]:.4f}], "
      f"\n    p-value: {p_value:.4f}")


'''
Interpretation
- Ridge vs ANN
    - Mean difference = 0.9053 means Ridge’s loss is on average 0.9053 higher than ANN (Ridge − ANN = +0.9053).
    - 95% CI [0.3162, 1.4944] indicates the true mean difference is plausibly between 0.316 and 1.494 (always positive).
    - p = 0.0027 strongly rejects the null of zero difference under the test assumptions; the positive difference is statistically significant.
    - Practical take: ANN performs better (lower loss) than Ridge by about 0.9 units on average; the effect is small-to-moderate but reliably nonzero.
- Ridge vs Baseline
    - Mean difference = −53.0548 means Ridge’s loss is on average 53.05 lower than Baseline (Ridge − Baseline = −53.05).
    - 95% CI [−59.9082, −46.2015] is well below zero, so the reduction is large and precisely estimated.
    - p ≈ 0.0000 gives extremely strong evidence that Ridge outperforms Baseline.
    - Practical take: Ridge delivers a very large improvement over the Baseline model.
- ANN vs Baseline
    - Mean difference = −53.9602 means ANN’s loss is on average 53.96 lower than Baseline (ANN − Baseline = −53.96).
    - 95% CI [−60.8136, −47.1067] again shows a large, precise improvement.
    - p ≈ 0.0000 strongly rejects no-difference.
    - Practical take: ANN also delivers a very large improvement over Baseline, slightly larger than Ridge’s improvement.
'''
