from Functions.Functions_DataProcessing import *
from Functions.Functions_Visualization import plot_hist
import os


df = load_dataset(r'../Assigment_2/Data/raw_mpg_dataset.csv')
df = column_drop(df, column_name=['Unnamed: 0', 'name'])
df = replace_with_mean(df)

assert df.isna().sum().sum() == 0, f'There are {df.isna().sum().sum()} NaN values.'
assert df.duplicated().sum() == 0, f'There are {df.duplicated().sum()} duplicates.'

pd.set_option('display.max_columns', None)
print(df.head())
print(df.shape)

columns = ['mpg', 'displacement', 'horsepower', 'weight', 'acceleration']
units = ['miles/gallon', r'$\mathrm{in}^2$', 'hp', 'lb', 's']
label_columns = ['origin']
df = one_hot_encoder(df, label_columns)
print(df.head())
print(df.describe())

# transformation_columns = ['mpg', 'displacement', 'horsepower', 'weight']
# df = log_transform(df, cols=transformation_columns, base=np.e)
# df = power(df, columns, exponent = 2)

plot_hist(df, columns=columns, units=units)
target_dir = os.path.join('.', "Data")
df.to_csv(os.path.join(target_dir, 'Data_processed.csv'), index = False)
