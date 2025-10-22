from Functions.Functions_DataProcessing import load_dataset
from Functions.Functions_Regression_b import *
import pandas as pd


pd.set_option('display.max_columns', None)
df = load_dataset(r'C:\Users\mrusn\PycharmProjects\Temp\Data\Data_processed.csv')
X = df.drop(columns = ['mpg'])
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
# Lists must be the same length!
alphas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1e3, 1e4]
ann_hidden_layers = [(1,), (3,), (25,), (50,), (75,), (100,), (125,), (150,), (175,), (200,)]

baseline_final_dict, ridge_final_dict, ann_final_dict, ridge_best_alpha_dict, ann_best_layer_dict = model_comaparison(
        X, y, KFold_value_outer=10, KFold_value_inner=10, random_state=42,
        continuous_cols=continuous_cols, onehot_cols=onehot_cols, polynomial_degree=2,
        ridge_alpha=alphas, ridge_solver='auto', ann_hidden_layer_sizes=ann_hidden_layers
        )

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

print(df.round(3))
df.to_csv('Regression-b_Final_Table.csv', float_format='%.3f', index=False)