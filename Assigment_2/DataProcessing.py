from Functions.Functions_DataProcessing import *
from Functions.Functions_Visualization import plot_hist


df = load_dataset(r'C:\Users\mrusn\PycharmProjects\Temp\Data\raw_mpg_dataset.csv')
df = column_drop(df, 'Unnamed: 0', 'name')
df = replace_with_mean(df)

assert df.isna().sum().sum() == 0, (f'There are {df.isna().sum().sum()} NaN values.')
assert df.duplicated().sum() == 0, (f'There are {df.duplicated().sum()} duplicates.')

pd.set_option('display.max_columns', None)
print(df.head())
print(df.shape)

columns = ['mpg', 'displacement', 'horsepower', 'weight', 'acceleration']
units = ['miles/gallon', r'$\mathrm{in}^2$', 'hp', 'lb', 's']
label_columns = ['origin']
df = one_hot_encoder(df, label_columns)
print(df.head())
print(df.describe())

#transformation_columns = []
#df = log_transform(df, transformation_columns, base = np.e)
#df = power(df, columns, exponent = 2)

plot_hist(df, columns, units)
#df.to_csv('Data_processed.csv', index = False)