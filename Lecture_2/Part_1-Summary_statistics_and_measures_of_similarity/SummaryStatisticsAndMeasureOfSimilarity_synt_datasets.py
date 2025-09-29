import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split


def smc(q, X):
    M = X.shape[1]
    return (q == X).sum(axis=1) / M  # or preferably: (q == X).mean(axis=1)


def jaccard(q, X):  # or alternatively using sklearn.metrics.jaccard_score
    M = X.shape[1]
    positive_matches = np.sum((q == 1) & (X == 1), axis=1)
    negative_matches = np.sum((q == 0) & (X == 0), axis=1)
    return positive_matches / (M - negative_matches)


def cosine(q, X):
    q_norm = np.linalg.norm(q)
    X_norm = np.linalg.norm(X, axis=1)
    dot_product = np.dot(q, X.T).flatten()
    return dot_product / (q_norm * X_norm)


def extended_jaccard(q, X):
    q_norm = np.linalg.norm(q)
    X_norm = np.linalg.norm(X, axis=1)
    dot_product = np.dot(q, X.T).flatten()
    return dot_product / (np.power(q_norm, 2) + np.power(X_norm, 2))


def correlation(q, X):
    # Center the vectors
    q_centered = q - np.mean(q)
    X_centered = X - np.mean(X, axis=1, keepdims=True)
    # Compute covariance term
    covariance = np.dot(q_centered, X_centered.T).flatten()
    # Compute standard deviations
    std_q = np.linalg.norm(q_centered)
    std_X = np.linalg.norm(X_centered, axis=1)
    return covariance / (std_q * std_X)


path = 'C:/Users/mrusn/PycharmProjects/02452_Machine_Learning/Lecture_2/data/digits.npy'
data = np.load(path)

reshaped_data = data.reshape(-1, 16, 16)
q = reshaped_data[1]
X = np.vstack((reshaped_data[0:1], reshaped_data[2:]))

fig, axs = plt.subplots(1, 6, figsize=(20, 20))
axs[0].imshow(X[0], cmap='gray')
axs[0].set_title('$x_{1}$')
axs[1].imshow(X[1], cmap='gray')
axs[1].set_title('$x_{2}$')
axs[2].imshow(X[2], cmap='gray')
axs[2].set_title('$x_{3}$')
axs[3].imshow(X[3], cmap='gray')
axs[3].set_title('$x_{4}$')
axs[4].imshow(X[4], cmap='gray')
axs[4].set_title('$x_{5}$')
axs[5].imshow(q, cmap='gray')
axs[5].set_title('q')

for ax in axs:
    ax.set_xticks([])  # Hides x-axis tick labels
    ax.set_yticks([])  # Hides y-axis tick labels
    ax.grid(False)

plt.show()

X = X.reshape(-1, 256)
q = q.reshape(-1, 256)

median_X = np.median(X, axis = 1)          # shape (9297,1)
median_q = np.median(q, axis = 1)          #shape (9297, 1)
print(median_X[:, np.newaxis].shape)
print(median_q[:, np.newaxis].shape)
binarized_X = (X > median_X[:, np.newaxis]).astype(int)  # shape (9297, 256)
binarized_q = (q > median_q[:, np.newaxis]).astype(int)  # shape (1, 256)

fig, axs = plt.subplots(1, 6, figsize=(20, 20))
axs[0].imshow(binarized_X[0].reshape(16, 16), cmap='gray')
axs[0].set_title('$x_{1}$')
axs[1].imshow(binarized_X[1].reshape(16, 16), cmap='gray')
axs[1].set_title('$x_{2}$')
axs[2].imshow(binarized_X[2].reshape(16, 16), cmap='gray')
axs[2].set_title('$x_{3}$')
axs[3].imshow(binarized_X[3].reshape(16, 16), cmap='gray')
axs[3].set_title('$x_{4}$')
axs[4].imshow(binarized_X[4].reshape(16, 16), cmap='gray')
axs[4].set_title('$x_{5}$')
axs[5].imshow(binarized_q.reshape(16, 16), cmap='gray')
axs[5].set_title('q')

for ax in axs:
    ax.set_xticks([])  # Hides x-axis tick labels
    ax.set_yticks([])  # Hides y-axis tick labels
    ax.grid(False)

assert all(np.unique(binarized_X) == [0, 1]), "X_binarized should be binary"
assert all(np.unique(binarized_q) == [0, 1]), "q_binarized should be binary"

smc_q_X = smc(binarized_q, binarized_X)
print(type(smc_q_X[0]))

jaccard_q_X = jaccard(binarized_q, binarized_X)
print(type(jaccard_q_X[0]))

cosine_q_X = cosine(binarized_q, binarized_X)
print(type(cosine_q_X[0]))

extended_jaccard_q_X = extended_jaccard(binarized_q, binarized_X)
print(type(extended_jaccard_q_X[0]))

correlation_q_X = correlation(binarized_q, binarized_X)
print(type(correlation_q_X[0]))

#Define the number of top results to retrieve
top_k = 5

# Define similarity functions to run in a list
similarity_functions = [smc, jaccard, cosine, extended_jaccard, correlation]

# Initialize figure
fig = plt.figure(figsize=(3*top_k, 3*(len(similarity_functions)+1)))
# Plot the query image
ax = fig.add_subplot(len(similarity_functions)+1, 1, 1)
ax.imshow(binarized_q.reshape(16, 16), cmap='gray')
ax.set_title(r'$\boldsymbol{q}$')
ax.grid(False)
plot_idx = top_k + 1 # update plot index

# Iterate over the similarity functions and compute similarities
for sim_func in similarity_functions:
    # Compute similarities using the binarized images and query
    similarities = sim_func(binarized_q, binarized_X)

    # Sort indices by similarity in descending order.
    most_similar_order = np.argsort(similarities)[::-1] #most_similar_order = np.argsort(similarities)[::1] for least simmilar

    # Sort the images in the database
    sorted_images = X[most_similar_order]

    # Plot the top-k most similar images under the corresponding similarity function
    for k in range(top_k):
        # Create subplot for each image
        ax = fig.add_subplot(len(similarity_functions)+1, top_k, plot_idx)
        # Plot the image
        ax.imshow(sorted_images[k].reshape(16, 16), cmap='gray')
        # Set a title
        ax.set_title(f'image #{most_similar_order[k]}\nSimilarity: {similarities[most_similar_order[k]]:.2f}')
        ax.grid(False)

        # Add y-label if it's the first image in the row
        if k == 0:
            ax.set_ylabel(f'{sim_func.__name__.capitalize().replace("_", " ")}')

        # Update plot index
        plot_idx += 1

fig.tight_layout()
plt.show()

np.random.seed(42)

# Generate two data objects with M random attributes
M = 5
x = np.random.rand(1, M)
y = np.random.rand(1, M)

# Two constants
a = 1.5
b = 1.5

print(f'Cosine(x, y) == Cosine(a*x, y): ', cosine(x, y) == cosine(a*x, y))
print(f'Extended Jaccard(x, y) == Extended Jaccard(a*x, y): ', extended_jaccard(x, y) == extended_jaccard(a*x, y))
print(f'Correlation(x, y) == correlation(a*x, y): ', correlation(x, y) == correlation(a*x, y))
print()
print(f'Cosine(x, y) == Cosine(b+x, y): ', cosine(x, y) == cosine(b+x, y))
print(f'Extended Jaccard(x, y) == Extended Jaccard(b+x, y): ', extended_jaccard(x, y) == extended_jaccard(b+x, y))
print(f'Correlation(x, y) == correlation(b+x, y): ', correlation(x, y) == correlation(b+x, y))

print('For some comparisons it depends on the number value!')


