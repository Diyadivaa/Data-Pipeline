import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

# Preprocess dataset
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'label']
df['label'] = df['label'].map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})

# Split into train and test
X = df.drop('label', axis=1)
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Build model

model = LogisticRegression()

# Train model
model.fit(X_train, y_train)

# Test model
predictions = model.predict(X_test)

# Evaluate performance

print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

# save model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# load model
with open('model.pkl', 'rb') as loaded_model:
    model = pickle.load(loaded_model)

# make predictions
new_data = np.array([[5.1, 3.5, 1.4, 0.2]])
prediction = model.predict(new_data)
print(prediction)

# visualize data
X_train.plot(kind='scatter', x='sepal_length', y='sepal_width', c='blue', colormap='viridis')
plt.show()

# inspect features
pd.plotting.scatter_matrix(X_train, c=y_train, figsize=[8, 8], s=150, marker='D')
plt.show()


# clean data
X_train = X_train[X_train['sepal_length'] > 0]

# transform data
X_train_log = np.log(X_train)

# merge data
X_train_log['label'] = y_train

# aggregate data
grouped_data = X_train_log.groupby('sepal_length')['sepal_width'].mean()

# normalize data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# impute missing values
imputer = SimpleImputer()
X_train_imputed = imputer.fit_transform(X_train)

# encode categorical variables
onehot_encoder = OneHotEncoder()
X_train_encoded = onehot_encoder.fit_transform(X_train)

# feature selection
selector = SelectKBest(chi2, k=2)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)


#Importing RF
from sklearn.ensemble import RandomForestClassifier

# Initialize the model
model = RandomForestClassifier()

# Fit the model to the training data
model.fit(X_train_selected, y_train)

# Make predictions on the test data
predictions = model.predict(X_test_selected)

# Evaluate the model performance
accuracy = model.score(X_test_selected, y_test)
print(f'Test accuracy: {accuracy:.2f}')


#It is generally acceptable to have a high accuracy on the Iris dataset using a machine learning model, particularly if the model is able to achieve a high accuracy on the test set as well as the training set. This is because the Iris dataset is a relatively small and simple dataset, with only 150 samples and three classes. In this case, it is possible for a machine learning model to achieve a high accuracy by learning the patterns in the training data and generalizing well to new, unseen data.
#However, it's important to note that the Iris dataset is often used as a benchmark dataset, and it may not be representative of more complex, real-world datasets. In these cases, a model that performs well on the Iris dataset may not necessarily perform well on other datasets, and it's important to evaluate the model on a variety of data to ensure that it is generalizing well.
#Additionally, it's important to consider other metrics in addition to accuracy when evaluating the performance of a machine learning model, such as precision, recall, and F1 score. These metrics can provide a more detailed understanding of how the model is performing, particularly if the classes in the dataset are imbalanced or there are multiple classes. However, as visualized below the model is likely overfitting or the patterns are too clear in the dataset. We may want to consider removing the feature selection until we gain more data.

report = classification_report(y_test, predictions)
print(report)
