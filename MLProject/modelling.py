import argparse
import pandas as pd
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="breast_cancer_preprocessing/breast_cancer_preprocessed.csv")
args = parser.parse_args()

mlflow.set_experiment("CI Breast Cancer")
mlflow.autolog()

data = pd.read_csv(args.data_path)
X = data.drop(columns=["target"])
y = data["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# MLflow run already started by mlflow run (MLProject command)
# so we avoid using with mlflow.start_run() unless there is no active run
if mlflow.active_run():
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
else:
    with mlflow.start_run(run_name="ci_random_forest"):
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

print("Training via MLflow Project selesai.")
