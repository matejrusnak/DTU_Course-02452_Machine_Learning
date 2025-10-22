import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split


#Load the dataset
dataset_no = 4
dataset_name = 'synth'+ str(dataset_no)

path = 'C:/Users/mrusn/PycharmProjects/02452_Machine_Learning/Lecture_2/data/synth/'

# Load the dataset splits and construct the datamatrices X_train, X_test and target vectors y_train, y_test
df_train = pd.read_csv(path + 'synth' + str(dataset_no) + '_train.csv')
X_train = df_train[['x0', 'x1']].values
y_train = df_train['y'].values

df_test = pd.read_csv(path + 'synth' + str(dataset_no) + '_test.csv')
X_test = df_test[['x0', 'x1']].values
y_test = df_test['y'].values

# Check that the input shapes are correct
assert X_train.shape[1] == 2, f"Expected 2 features, but got {X_train.shape[1]}"
assert X_test.shape[1] == 2, f"Expected 2 features, but got {X_test.shape[1]}"

# Visualize the training data, colored by the target variable
df_train.plot(kind='scatter', x='x0', y='x1', c='y', cmap='viridis', title=f'Training set: {dataset_name}', figsize=(8, 6), edgecolor='gray')
plt.show()

# Create a model
criterion = 'gini' # Alternative 'entropy'
model = DecisionTreeClassifier(max_depth = None, criterion = criterion, random_state = 42)
model.fit(X_train, y_train)
# Predict test data
y_pred = model.predict(X_test)


# Compute the accuracy and error rate
accuracy = np.mean(y_pred == y_test)
error_rate = 1 - accuracy
print(accuracy)
# Compute the elements of the confusion matrix
cm = confusion_matrix(y_test, y_pred)
# Plot the confusion matrix
plt.figure(2)
plt.imshow(cm, cmap="binary", interpolation="None")
plt.colorbar()
plt.xticks(np.unique(y_test))
plt.yticks(np.unique(y_test))
plt.xlabel("Predicted class")
plt.ylabel("Actual class")
plt.title("Confusion matrix (Accuracy: {0}%, Error Rate: {1}%)".format(accuracy, error_rate))
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha='center', va='center', color='white')
plt.grid(False)
plt.show()

# Define resolution of the grid, i.e. how many points per axis
resolution = 500
# Construct the grid
min_x0, max_x0 = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
min_x1, max_x1 = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5
X_pred = np.meshgrid(
    np.linspace(min_x0, max_x0, resolution),
    np.linspace(min_x1, max_x1, resolution)
)
# Stack the grid points into the format of X, i.e. shape N x M
X_pred = np.stack([X_pred[0].ravel(), X_pred[1].ravel()]).T

# Get the decision boundary
y_pred = model.predict(X_pred).reshape(resolution, resolution)
# Plot the decision boundary
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)
ax.set_title('Decision boundary of kNN')
ax.imshow(y_pred, extent=(min_x0, max_x0, min_x1, max_x1), origin='lower', alpha=0.5, cmap='viridis')
# Plot the training points
df_train.plot(kind='scatter', x='x0', y='x1', c='y', cmap='viridis', ax=ax, edgecolors='gray', colorbar=False)
ax.set_aspect('auto')
ax.grid(False)
plt.show()