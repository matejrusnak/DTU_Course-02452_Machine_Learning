import numpy as np
import matplotlib.pyplot as plt


N = 5000
M = 2

rng = np.random.default_rng(seed = 42)
X = rng.uniform(-2, 2, size=(N, M)) # np.ndarray(5000,2)
# Define GENERAL decision rules for the two classes
center_A = np.array([0.5, 0.5])
decision_rule_A = lambda x: np.linalg.norm(x - center_A, ord = 2, axis = 1) < 1
# APPLY the decision rule to the generated points
A = decision_rule_A(X) # np.ndarray(), dtype = bool, shape = (5000,)

center_B = np.array([-1., -1.])
decision_rule_B = lambda x: np.linalg.norm(x - center_B, ord = 1, axis = 1) < 1
# Apply the decision rule to the generated points
B = decision_rule_B(X[~A]) # np.ndarray(), dtype = bool, shape = (5000,)
print(X[~A].shape)
'''
Switchng ord=2 to ord=1
- Non-overlapping regions: A circle and a diamond around different centers often don’t intersect, making it easy to split your dataset into two classes with minimal conflict.
- Illustrate metric effects: In higher-dimensional tasks, the choice of norm drastically changes neighborhoods. This toy example shows that plainly.
- Flexibility: You could swap in other norms—ord=np.inf for an axis-aligned square, ord=0.5 for a star-shaped region, or even Mahalanobis distance for elliptical contours.

'''

# Generate the predicted labels based on the decision rules
point_idxs = np.arange(N)
predicted_labels = np.zeros(N, dtype=int)
predicted_labels[point_idxs[A]] = 1         # Add points respecting rule A to class 1
predicted_labels[point_idxs[~A][B]] = 1     # Add points not respecting rule A but rule B to class 1
predicted_labels[point_idxs[~A][~B]] = 0    # Add all other points to class 0

# Plot the randomly generated points
fig, axs = plt.subplots(1, 3, figsize=(15,5), sharex=True, sharey=True)
axs[0].set_title('Randomly generated points')
axs[0].scatter(X[:, 0], X[:, 1], c='C2', marker='.', alpha=0.5)
axs[0].set_xlabel(r'$x_1$')
axs[0].set_ylabel(r'$x_2$')
axs[0].set_aspect('equal')

# Plot the points based on the decision rule A
axs[1].set_title('Visualization of decision rule A')
axs[1].scatter(X[:, 0][A], X[:, 1][A], c='C0', label=f'$A$ = True', marker='.')
axs[1].scatter(X[:, 0][~A], X[:, 1][~A], c='C1', label=f'$A$ = False', marker='.')
axs[1].set_xlabel(r'$x_1$')
axs[1].set_aspect('equal')
axs[1].legend(loc='upper left')

# Plot the points based on the predicted labels
axs[2].set_title('Visualization of decision tree output')
axs[2].scatter(X[:, 0][predicted_labels == 1], X[:, 1][predicted_labels == 1], c='C0', label='Predicted class 1', marker='.')
axs[2].scatter(X[:, 0][predicted_labels == 0], X[:, 1][predicted_labels == 0], c='C1', label='Predicted class 0', marker='.')
axs[2].set_xlabel(r'$x_1$')
axs[2].set_aspect('equal')
axs[2].legend(loc='upper left')
plt.show()