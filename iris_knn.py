"""
Iris Flower Classification using K-Nearest Neighbors (KNN)

Reproduces the pipeline from notebooks/Iris_Classification_KNN.ipynb as a
standalone script, and adds a 5-fold cross-validation check for a more
reliable performance estimate.

Usage:
    python src/iris_knn.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

RANDOM_STATE = 42
TEST_SIZE = 0.2
K_DEFAULT = 5


def load_data():
    """Load the Iris dataset as a DataFrame with a numeric target column."""
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["target"] = iris.target
    return df, iris.target_names


def main():
    df, class_names = load_data()
    print("First 5 rows:\n", df.head(), "\n")

    X = df.drop("target", axis=1)
    y = df["target"]

    # 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"Training data size: {X_train.shape}")
    print(f"Testing data size:  {X_test.shape}\n")

    # Scale features (fit on train only to avoid data leakage)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train KNN (K=5)
    model = KNeighborsClassifier(n_neighbors=K_DEFAULT)
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)

    # Evaluation
    print(f"Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%\n")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions), "\n")
    print("Classification Report:")
    print(classification_report(y_test, predictions, target_names=class_names))

    # Error rate vs K (elbow method)
    error_rate = []
    for k in range(1, 21):
        knn = KNeighborsClassifier(n_neighbors=k).fit(X_train_scaled, y_train)
        error_rate.append(np.mean(knn.predict(X_test_scaled) != y_test))

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 21), error_rate, color="blue", linestyle="dashed",
             marker="o", markerfacecolor="red", markersize=10)
    plt.title("Error Rate vs. K Value (Elbow Method)")
    plt.xlabel("K Value")
    plt.ylabel("Error Rate")
    plt.savefig("elbow_plot_generated.png", dpi=150, bbox_inches="tight")
    print("\nSaved elbow plot to elbow_plot_generated.png")

    # 5-fold cross-validation (scaler inside the pipeline -> no leakage)
    pipe = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=K_DEFAULT))
    scores = cross_val_score(pipe, X, y, cv=5)
    print(f"\n5-fold CV accuracy: {scores.mean() * 100:.2f}% (+/- {scores.std() * 100:.2f}%)")
    print("Per-fold:", np.round(scores, 4))


if __name__ == "__main__":
    main()
