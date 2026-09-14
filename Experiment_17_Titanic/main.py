import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# LOAD AND PREPROCESS DATA
# ============================================================

def load_and_preprocess_data(filepath):
    """
    Load the Titanic dataset and preprocess the data.
    """

    df = pd.read_csv(filepath)

    print("=" * 55)
    print("        TITANIC SURVIVAL PREDICTION SYSTEM")
    print("             DECISION TREE CLASSIFIER")
    print("=" * 55)

    print("\nDataset Information")
    print("-" * 55)
    print(f"Rows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")

    print("\nFirst 5 Records:")
    print(df.head())

    # Standardize column name if Gender is used instead of Sex
    if "Gender" in df.columns and "Sex" not in df.columns:
        df = df.rename(columns={"Gender": "Sex"})

    # Drop columns that are not useful for prediction
    columns_to_drop = [
        "PassengerId",
        "Name",
        "Ticket",
        "Cabin"
    ]

    df = df.drop(
        columns=[col for col in columns_to_drop if col in df.columns]
    )

    # Handle missing values
    if "Age" in df.columns:
        df["Age"] = df["Age"].fillna(df["Age"].median())

    if "Fare" in df.columns:
        df["Fare"] = df["Fare"].fillna(df["Fare"].median())

    if "Embarked" in df.columns:
        df["Embarked"] = df["Embarked"].fillna(
            df["Embarked"].mode()[0]
        )

    # Convert categorical columns into numerical columns
    categorical_columns = [
        col for col in ["Sex", "Embarked"]
        if col in df.columns
    ]

    if categorical_columns:
        df = pd.get_dummies(
            df,
            columns=categorical_columns,
            drop_first=True
        )

    # Separate features and target
    X = df.drop(columns=["Survived"])
    y = df["Survived"]

    return X, y


# ============================================================
# SAVE DECISION TREE VISUALIZATION
# ============================================================

def save_decision_tree(model, feature_names, output_path):

    plt.figure(figsize=(18, 10))

    plot_tree(
        model,
        feature_names=feature_names,
        class_names=["Did Not Survive", "Survived"],
        filled=True,
        rounded=True,
        fontsize=9
    )

    plt.title("Titanic Survival Prediction - Decision Tree")

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# SAVE FEATURE IMPORTANCE GRAPH
# ============================================================

def save_feature_importance(model, feature_names, output_path):

    feature_importances = pd.Series(
        model.feature_importances_,
        index=feature_names
    ).sort_values(ascending=True)

    plt.figure(figsize=(10, 7))

    feature_importances.plot(
        kind="barh"
    )

    plt.title("Feature Importance - Decision Tree")
    plt.xlabel("Importance")
    plt.ylabel("Feature")

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    return feature_importances


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

def save_confusion_matrix(cm, output_path):

    plt.figure(figsize=(6, 5))

    plt.imshow(cm)

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.xticks(
        [0, 1],
        ["Did Not Survive", "Survived"]
    )

    plt.yticks(
        [0, 1],
        ["Did Not Survive", "Survived"]
    )

    # Display values inside the matrix
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center"
            )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    # --------------------------------------------------------
    # 1. Dataset path
    # --------------------------------------------------------

    dataset_path = r"C:\Users\Lenovo\Downloads\Titanic-Dataset.csv"

    # Folder where main.py is located
    project_folder = Path(__file__).resolve().parent

    # Output files
    decision_tree_path = project_folder / "decision_tree.png"
    feature_importance_path = project_folder / "feature_importance.png"
    confusion_matrix_path = project_folder / "confusion_matrix.png"
    feature_importance_csv = project_folder / "feature_importances.csv"

    # Check dataset
    if not Path(dataset_path).exists():

        print("\nERROR: Titanic-Dataset.csv was not found.")
        print("Expected location:")
        print(dataset_path)

        return

    # --------------------------------------------------------
    # 2. Load and preprocess data
    # --------------------------------------------------------

    X, y = load_and_preprocess_data(dataset_path)

    # --------------------------------------------------------
    # 3. Train-Test Split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nData Split")
    print("-" * 55)
    print(f"Training records : {len(X_train)}")
    print(f"Testing records  : {len(X_test)}")

    # --------------------------------------------------------
    # 4. Create Decision Tree Classifier
    # --------------------------------------------------------

    model = DecisionTreeClassifier(
        criterion="gini",
        max_depth=4,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )

    # --------------------------------------------------------
    # 5. Train Model
    # --------------------------------------------------------

    model.fit(X_train, y_train)

    print("\nModel training completed successfully.")

    # --------------------------------------------------------
    # 6. Make Predictions
    # --------------------------------------------------------

    y_pred = model.predict(X_test)

    # --------------------------------------------------------
    # 7. Model Evaluation
    # --------------------------------------------------------

    accuracy = accuracy_score(y_test, y_pred)

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    print("\n" + "=" * 55)
    print("                 MODEL PERFORMANCE")
    print("=" * 55)

    print(f"\nAccuracy: {accuracy:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=[
                "Did Not Survive",
                "Survived"
            ]
        )
    )

    # --------------------------------------------------------
    # 8. Feature Importance
    # --------------------------------------------------------

    feature_importances = save_feature_importance(
        model,
        X.columns,
        feature_importance_path
    )

    feature_importances.to_csv(
        feature_importance_csv,
        header=["Importance"]
    )

    print("\nFeature Importances:")
    print(
        feature_importances.sort_values(
            ascending=False
        )
    )

    # --------------------------------------------------------
    # 9. Save Decision Tree
    # --------------------------------------------------------

    save_decision_tree(
        model,
        X.columns,
        decision_tree_path
    )

    # --------------------------------------------------------
    # 10. Save Confusion Matrix
    # --------------------------------------------------------

    save_confusion_matrix(
        cm,
        confusion_matrix_path
    )

    # --------------------------------------------------------
    # 11. Output Summary
    # --------------------------------------------------------

    print("\n" + "=" * 55)
    print("                  OUTPUT FILES")
    print("=" * 55)

    print("\nDecision tree:")
    print(decision_tree_path)

    print("\nFeature importance graph:")
    print(feature_importance_path)

    print("\nConfusion matrix:")
    print(confusion_matrix_path)

    print("\nFeature importance data:")
    print(feature_importance_csv)

    print("\n" + "=" * 55)
    print("       EXPERIMENT COMPLETED SUCCESSFULLY")
    print("=" * 55)


# ============================================================
# PROGRAM START
# ============================================================

if __name__ == "__main__":
    main()