import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


path = 'C:/Users/mrusn/PycharmProjects/02452_Machine_Learning/Lecture_2/data/wine.csv'

# Load the dataset splits and construct the datamatrices X_train, X_test and target vectors y_train, y_test
df_train = pd.read_csv(path)
# Split into features and labels
X = df_train.drop(columns=['Color'])
# X = pd.get_dummies(X) # one-hot-encoding
y = df_train['Color']

assert X.shape == (6497, 12), "There should be 6497 samples and 12 features in the wine dataset"
assert y.shape == (6497,), "There should be 6497 labels in the wine dataset"

X.plot(kind = 'box', subplots=True, layout=(2, 6), figsize=(16,5), sharex=False, sharey=False)
plt.show()

X.describe()

# Get the boolean mask for outlier detection
mask = (X['Volatile acidity'] > 2) | (X['Density'] > 1) | (X['Alcohol'] > 20)
# Remove outliers with conditional filtering
X = X[~mask]
y = y[~mask]

# Plot a boxplot of the data after removing these outliers
X.plot(kind='box', subplots=True, layout=(2, 6), figsize=(16,5), sharex=False, sharey=False)
plt.show()

assert X.shape == (6304, 12), "There should be 6304 samples and 12 features in the wine dataset after outlier removal"
assert y.shape == (6304,), "There should be 6304 labels in the wine dataset after outlier removal"

# Number of attributes
M = X.shape[1]

# Plot a matrix scatter plot of the wine attributes, colored by the wine color
fig, axs = plt.subplots(M, M, figsize=(20, 20), sharex='col', sharey='row')
for i in range(M):
    for j in range(M):
        for color in y.unique(): # loop through each label
            # Construct a mask based on the label
            mask = (y == color)
            # Plot the scatter plot for attribute pair (if not on the diagonal)
            axs[i, j].scatter(
                x=X[mask].iloc[:, j],        # x-values for the $j$'th attribute
                y=X[mask].iloc[:, i],        # y-values for the $i$'th attribute
                label=color, alpha=0.3,
                color='r' if color == 'Red' else 'y'
            )

# Update titles
for col in range(M):
    axs[0, col].set_title(X.columns[col])
    axs[col, 0].set_ylabel(X.columns[col])

# Add the legend to the last subplot only
axs[0,0].legend(loc='upper left')
plt.tight_layout(pad=1.)
plt.show()

# Transform the target variable into a numerical format
y_numerical = y.astype("category").cat.codes
# Construct the modified dataframe
df_tilde = pd.concat([X, y_numerical], axis=1)
# Compute the correlation matrix
correlation_matrix = df_tilde.corr()

# Plot the correlation matrix
fig = plt.figure(figsize=(8, 6))
fig.suptitle('Correlation matrix of the standardized data', fontsize=16)
# Plot correlation matrix and set colorbar min and max to -1 and 1 since thats the range of the correlation coefficients
plt.imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
plt.colorbar()
plt.xticks(ticks=np.arange(correlation_matrix.shape[1]), labels=list(X.columns) + ["Wine color"], rotation=90)
plt.yticks(ticks=np.arange(correlation_matrix.shape[1]), labels=list(X.columns) + ["Wine color"])
plt.grid(False)
plt.show()

# Split the data into training and test sets of proportion 80/20
X_train, X_test, y_train, y_test = train_test_split(X.drop(columns=["Quality score (0-10)"]), y, test_size=0.2, random_state=42)

# Compute mean and standard deviation of each attribute of the training dataset
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)

# Standardize the training and test data. This ensures that the models are not biased towards
# any particular feature due to differences in scale, which could impact model performance, especially for distance-based algorithms.
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

# Fit the models to the training data 0and predict on the test set
distance_metric = 'euclidean'
neigh = KNeighborsClassifier(n_neighbors=5, metric=distance_metric)
neigh.fit(X_train, y_train)
y_pred_neigh = neigh.predict(X_test)
accuracy_neigh = accuracy_score(y_test, y_pred_neigh)
print(accuracy_neigh)

criterion = "entropy"
dct = DecisionTreeClassifier(max_depth = None, criterion = criterion, min_samples_split=100, random_state = 42)
dct.fit(X_test, y_test)
y_pred_dct = dct.predict(X_test)
accuracy_dct = accuracy_score(y_test, y_pred_dct)
print(accuracy_dct)


# Compute the error rates
error_rate_knn = 1 - accuracy_neigh
error_rate_dct = 1 - accuracy_dct
# Confusion matrices
confusion_matrix_knn = confusion_matrix(y_test, y_pred_neigh)
confusion_matrix_dct = confusion_matrix(y_test, y_pred_dct)
# Plot the confusion matrices side-by-side
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
sns.heatmap(confusion_matrix_knn, annot=True, fmt="d", ax=axs[0], cmap="Blues", cbar=False)
axs[0].set_title(f"kNN - Confusion Matrix \n (Accuracy: {accuracy_neigh * 100:.2f}%, Error Rate: {error_rate_knn * 100:.2f}%)")
sns.heatmap(confusion_matrix_dct, annot=True, fmt="d", ax=axs[1], cmap="Blues", cbar=False)
axs[1].set_title(f"Decision Tree - Confusion Matrix \n (Accuracy: {accuracy_dct * 100:.2f}%, Error Rate: {error_rate_dct * 100:.2f}%)")

for ax in axs:
    ax.set_xticks([0.5, 1.5], np.unique(y_test))
    ax.set_yticks([0.5, 1.5], np.unique(y_test))
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    ax.set_aspect('equal')
plt.show()
