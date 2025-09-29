from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Dataset with different data layouts: synth1, synth2, synth3, synth4
dataset_name = 'synth4'

# Load the dataset splits and construct the datamatrices X_train, X_test and target vectors y_train, y_test
### BEGIN SOLUTION

# Load the training data and construct the datamatrices
df_train = pd.read_csv(f'C:/Users/mrusn/PycharmProjects/02452_Machine_Learning/Lecture_2/data/synth/{dataset_name}_train.csv')
X_train = df_train.drop(columns=['y']).values
y_train = df_train['y'].astype(int).values

# Load the test data and construct the datamatrices
df_test = pd.read_csv(f'C:/Users/mrusn/PycharmProjects/02452_Machine_Learning/Lecture_2/data/synth/{dataset_name}_test.csv')
X_test = df_test.drop(columns=['y']).values
y_test = df_test['y'].astype(int).values

### END SOLUTION

# Check that the input shapes are correct
assert X_train.shape[1] == 2, f"Expected 2 features, but got {X_train.shape[1]}"
assert X_test.shape[1] == 2, f"Expected 2 features, but got {X_test.shape[1]}"

# Visualize the training data, colored by the target variable
df_train.plot(kind='scatter', x='x0', y='x1', c='y', cmap='viridis', title=f'Training set: {dataset_name}', figsize=(8, 6), edgecolor='gray')
plt.show()

k = 5
x_new = np.array([[1, -1]])

### BEGIN SOLUTION
metrics = ['euclidean', 'manhattan', 'cosine']
accuracy_list = []
for distance_metric in metrics:
    # Create the model object with the specific hyperparameters
    model = KNeighborsClassifier(n_neighbors=k, metric = distance_metric)
    # "Train" the model by giving it the training data
    model.fit(X_train, y_train)
    # Make predictions on the test data
    y_pred = model.predict(x_new)

    # Get the indices of the closest neighbors from the training set
    closest_neighbors_indices = model.kneighbors(x_new, n_neighbors=k, return_distance=False).flatten()
    # Get the closest neighbors and their labels based on the indices
    closest_neighbors = X_train[closest_neighbors_indices, :]
    closest_labels = y_train[closest_neighbors_indices]
    ### END SOLUTION

    # Visualize the training data, colored by the target variable
    df_train.plot(kind='scatter', x='x0', y='x1', c='y', cmap='viridis', title=f'Training set: {dataset_name}', figsize=(8, 6), edgecolor='gray')

    # Plot the new point and its closest neighbors
    ### BEGIN SOLUTION
    plt.scatter(closest_neighbors[:, 0], closest_neighbors[:, 1], color='k', marker='x', s=50, label='Closest Neighbors')
    plt.scatter(x_new[0, 0], x_new[0, 1], color='red', marker='o', s=50, label='New Point')
    plt.legend()

    ### END SOLUTION
    plt.show()

    ### BEGIN SOLUTION

    # Predict using the model
    y_pred = model.predict(X_test)
    ## Or alternatively with cosine distance
    # model = KNeighborsClassifier(n_neighbors=k, metric='cosine')
    # model.fit(X_train, y_train)
    # y_pred = model.predict(X_test)


    # Compute the accuracy and error rate
    accuracy = np.mean(y_pred == y_test)
    error_rate = 1 - accuracy

    # Compute the elements of the confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Plot the confusion matrix
    fig = plt.figure(figsize=(8, 6))
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

    # Or alternatively with seaborn
    # fig = plt.figure(figsize=(8, 6))
    # plt.title("Confusion matrix (Accuracy: {0}%, Error Rate: {1}%)".format(accuracy * 100, error_rate * 100))
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    # plt.xticks(np.unique(y_test))
    # plt.yticks(np.unique(y_test))
    # plt.xlabel("Predicted class")
    # plt.ylabel("Actual class")
    # plt.show()

    ### END SOLUTION

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

    ### BEGIN SOLUTION

    # Predict using the model
    y_pred = model.predict(X_pred).reshape(resolution, resolution)
    ## Or alternatively with cosine distance
    # model = KNeighborsClassifier(n_neighbors=k, metric='cosine')
    # model.fit(X_train, y_train)
    # y_pred = model.predict(X_test)


    # Plot the decision boundary
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.set_title('Decision boundary of kNN')
    ax.imshow(y_pred, extent=(min_x0, max_x0, min_x1, max_x1), origin='lower', alpha=0.5, cmap='viridis')
    # Plot the training points
    df_train.plot(kind='scatter', x='x0', y='x1', c='y', cmap='viridis', ax=ax, edgecolors='gray', colorbar=False, label='Training data')
    df_test.plot(kind='scatter', x='x0', y='x1', c='y', cmap='viridis', ax=ax, edgecolors='gray', colorbar=False, marker='s', label='Test data')
    ax.set_aspect('auto')
    ax.grid(False)
    plt.show()
    accuracy_list.append(accuracy)
    print('Dataset {0} | K-neighbours: {1} | Distance metric: {2} | Accuracy: {3}'.format(dataset_name, k,
                                                                                          distance_metric, accuracy))


    ### END SOLUTION
