import os
import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Get project root directory
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

# Create models folder in root
model_dir = os.path.join(BASE_DIR, "models")
os.makedirs(model_dir, exist_ok=True)

# Load dataset
data = load_iris()
X, y = data.data, data.target

# Model A
model_a = RandomForestClassifier(max_depth=5)
model_a.fit(X, y)
joblib.dump(model_a, os.path.join(model_dir, "model_a.pkl"))

# Model B
model_b = RandomForestClassifier(max_depth=2)
model_b.fit(X, y)
joblib.dump(model_b, os.path.join(model_dir, "model_b.pkl"))

print("Model A and Model B saved successfully")