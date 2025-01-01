import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load and validate the dataset
try:
    data = pd.read_csv("Y:/VS code/CODSOFT/Task-3/archive/IRIS.csv")
    print("Dataset loaded successfully.")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# Ensure correct column names
expected_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
if list(data.columns) != expected_columns:
    print("Dataset columns do not match the expected format.")
    print(f"Expected: {expected_columns}")
    print(f"Found: {list(data.columns)}")
    exit()

# Features (X) and target (y)
X = data.drop(columns=['species'])
y = data['species']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter tuning for Random Forest
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
try:
    grid_search.fit(X_train, y_train)
except Exception as e:
    print(f"Error during model training: {e}")
    exit()

# Best model from Grid Search
best_model = grid_search.best_estimator_

# Train the best model on the training set
best_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = best_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Optimized Accuracy: {accuracy:.2f}\n")
print("Classification Report:\n", classification_report(y_test, y_pred))






