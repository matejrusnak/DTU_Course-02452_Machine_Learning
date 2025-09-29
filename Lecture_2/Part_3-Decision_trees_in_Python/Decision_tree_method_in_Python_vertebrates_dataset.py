from sklearn import tree
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


path = 'C:/Users/mrusn/PycharmProjects/02452_Machine_Learning/Lecture_2/data/vertebrate.csv'

# Load the dataset splits and construct the datamatrices X_train, X_test and target vectors y_train, y_test
df_train = pd.read_csv(path).drop(['Species'], axis = 1)
# Split into features and labels
X = df_train.drop(columns=['Class label'])
# X = pd.get_dummies(X) # one-hot-encoding
y = df_train['Class label']

# Construct a one-hot-encoder for the data matrix X
one_hot_encoder = OneHotEncoder(sparse_output=False)
# Construct a label encoder for the target attribute y
label_encoder = LabelEncoder()

# Cast the data to numeric using the encoders
X = one_hot_encoder.fit_transform(X)
y = label_encoder.fit_transform(y)

# Check the shapes of the resulting arrays
assert X.shape == (15, 19), "There should be 15 samples and 19 features after one-hot encoding"
assert y.shape == (15,), "There should be 15 labels after encoding"

criterion = "gini"

# Create a model
model = DecisionTreeClassifier(max_depth = None, criterion = criterion, random_state = 42)
model.fit(X, y)

# Visualize the graph. Hint: Try to maximize the figure after it displays.
fig = plt.figure(figsize=(4, 4), dpi=300)
_ = plot_tree(model, filled=False, feature_names=[col.replace("_", ": ") for col in one_hot_encoder.get_feature_names_out()])
plt.show()

x_new = {
    'Body temperature': 'Cold-blooded',
    'Skin cover': 'Scales',
    'Gives birth': 'Yes',
    'Aquatic creature': 'Semi',
    'Aerial creature': 'Yes',
    'Has legs': 'Yes',
    'Hibernates': 'Yes',
}
print(f"New data point: {x_new}")

# Create a new data point
x_new = pd.DataFrame.from_records([x_new])

# Transform the new data point using the one-hot encoder
x_new = one_hot_encoder.transform(x_new)
# Predict the class label for the new data point
y_pred = model.predict(x_new)
# Get the string representation of the predicted class label
y_pred_str = label_encoder.inverse_transform(y_pred)

# Print the predicted class label
print(f"Predicted class label: {y_pred_str}")