import mlflow
import mlflow.sklearn
from pathlib import Path
import math
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error

# Use a fixed tracking location so MLflow UI always reads the same run data.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRACKING_DIR = PROJECT_ROOT / "mlruns"
mlflow.set_tracking_uri(TRACKING_DIR.as_uri())
mlflow.set_experiment("Default")

data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    data.data,
    data.target,
    test_size=0.2,
    random_state=42,
    stratify=data.target,
)

with mlflow.start_run(run_name="rf_iris_baseline"):
    print("MLflow run started")

    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = model.score(X_test, y_test)
    rmse = math.sqrt(mean_squared_error(y_test, preds))

    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 5)
    mlflow.log_param("random_state", 42)
    mlflow.log_param("model", "tree")
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("rmse", float(rmse))

    mlflow.sklearn.log_model(model, "model")

    print("Logged to MLflow")

print("Accuracy:", acc)
print("RMSE:", rmse)
print("Tracking URI:", mlflow.get_tracking_uri())