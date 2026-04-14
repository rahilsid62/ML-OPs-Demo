from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier

# Load dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Train model
model = RandomForestClassifier()
model.fit(X, y)

print("Model training completed successfully")