from Functions.Functions_DataProcessing import load_dataset
from Functions.Functions_Regression_a import Ridge_hyperparameter_tuning
from Functions.Functions_Regression_a import final_model
from Functions.Functions_Regression_a import single_feature_extraction
from Functions.Functions_Visualization import plot_alpha_vs_mse, plot_actual_vs_pred
from Functions.Functions_Regression_a import multi_model_parameter_extraction
from Functions.Functions_Regression_a import LinReg_hyperparameter_tuning
import pandas as pd


pd.set_option('display.max_columns', None)
df = load_dataset(r'Data\Data_processed.csv')
X = df.drop(columns=['mpg'])
y = df['mpg']

# Define columns for standardization
continuous_cols = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year']
onehot_cols = ['origin_europe', 'origin_japan', 'origin_usa']

'''
# Picking the best polynomial for Linear Regression model
degrees = [1, 2, 3, 4, 5, 6]
model_generalization_error, model_best_parameter = LinReg_hyperparameter_tuning(X, y,
                                    KFold_value_outer = 10, KFold_value_inner = 10, random_state = 42,
                                    continuous_cols=continuous_cols, onehot_cols=onehot_cols,
                                    tuning_parameters = degrees)

for (KFold_loop_model, gen_error), (KFold_loop_parameter, best_parameter) in zip(model_generalization_error.items(), model_best_parameter.items()):
    print(f'KFold: {KFold_loop_model} | Generalization error: {gen_error:.3f} with best polynomial degree {best_parameter}')
'''

# Picking the best regularization alpha for Linear Regression model
save_models_path = 'C:/Users/mrusn/PycharmProjects/Temp/Models'
alphas = [1e-3, 1e-2, 1e-1, 7, 10, 100, 1e3]
model_generalization_error, model_best_parameter, overall_avg_mse, y_pred_dict, y_test_outer_dict = Ridge_hyperparameter_tuning(
                            X, y,
                            KFold_value_outer = 10, KFold_value_inner = 10, random_state = 42,
                            continuous_cols=continuous_cols, onehot_cols=onehot_cols,
                            tuning_parameters = alphas, polynomial_degree = 2, solver='auto',
                            save_models_path=save_models_path, save_models=False)
for (KFold_loop_model, gen_error), (KFold_loop_parameter, best_parameter) in zip(model_generalization_error.items(), model_best_parameter.items()):
    print(f'KFold: {KFold_loop_model} | Generalization error: {gen_error:.3f} with alpha {best_parameter}')

# Plotting the data
plot_alpha_vs_mse(overall_avg_mse)
plot_actual_vs_pred(y_pred_dict, y_test_outer_dict, model_generalization_error, polynomial_degree=2)


# Feature extraction for all models from 2-layer CV outer loop
#feature_extraction_path = 'C:/Users/mrusn/PycharmProjects/Temp/Models'
#multi_model_parameter_extraction(feature_extraction_path)

# Training the best model on ALL training data and extract features
final_model(X, y,
    polynomial_degree=2, alpha=0.1, solver='auto',
    continuous_cols=continuous_cols, onehot_cols=onehot_cols)

# Models' parameters extraction
final_model_name = r'\Regression-a_Final_model_Ridge_Poly_2_lambda_0.1.pkl'
extract_features_path = r'C:\Users\mrusn\PycharmProjects\Temp\Data\Final_Results' + final_model_name
#single_feature_extraction(extract_features_path, 'Regression-a_Final_Model_Feature_Ext.csv')

