from pathlib import Path
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


def load_data(directory):
    X_train = pd.read_csv(f"{directory}/X_train.csv")
    X_test = pd.read_csv(f"{directory}/X_test.csv")
    y_train = pd.read_csv(f"{directory}/y_train.csv").squeeze()
    y_test = pd.read_csv(f"{directory}/y_test.csv").squeeze()
    hela_X = pd.read_csv(f"{directory}/hela_all.csv")
    hela_y = pd.read_csv(f"{directory}/hela_all_y_labels.csv").squeeze()
    all_df = pd.read_csv(f"{directory}/all_df_with_times.csv")
    return X_train, X_test, y_train, y_test, hela_X, hela_y, all_df


def evaluate_model(model, X_train_scaled, y_train, X_test_scaled, y_test, hela_X_scaled, hela_y):
    model.fit(X_train_scaled, y_train)

    # Test dataset
    y_pred_test = model.predict(X_test_scaled)
    accuracy_test = accuracy_score(y_test, y_pred_test)
    metrics_test = precision_recall_fscore_support(y_test, y_pred_test, average="weighted")

    # Hela dataset
    y_pred_hela = model.predict(hela_X_scaled)
    accuracy_hela = accuracy_score(hela_y, y_pred_hela)
    metrics_hela = precision_recall_fscore_support(hela_y, y_pred_hela, average="weighted")

    return accuracy_test, metrics_test, accuracy_hela, metrics_hela


def save_results(directory, results, start_time=None, end_time=None):
    results_df = pd.DataFrame(results)
    subdir = f"start-{start_time}_end-{end_time}" if start_time is not None or end_time is not None else "default"
    out_path = Path(f"{directory}/{subdir}")
    out_path.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_path / "classifier_report.csv", index=False)


def main(args):
    directory = args.directory
    X_train, X_test, y_train, y_test, hela_X, hela_y, all_df = load_data(directory)
    if args.start_time is not None or args.end_time is not None:
        args.start_time = args.start_time if args.start_time is not None else float("-inf")
        args.end_time = args.end_time if args.end_time is not None else float("inf")

        print(">>> Filtering data and re-creating X_train, X_test, y_train, y_test")
        mitosis_df = all_df[all_df["label"] == 0]
        mitosis_df = all_df[
            (all_df["mitosis_relative_time"] >= args.start_time) & (all_df["mitosis_relative_time"] <= args.end_time)
        ]
        non_mitosis_df = all_df[all_df["label"] == 1]
        filtered_all_df = pd.concat([mitosis_df, non_mitosis_df])
        # Drop all the columns not start tieh skimage_
        y = filtered_all_df["label"]
        filtered_all_df = filtered_all_df.filter(regex="^skimage_")
        print(">>> Columns in filtered_all_df:", filtered_all_df.columns)
        X = filtered_all_df
        # Resplit X and y into train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(">>> Original shape of all_df:", all_df.shape)
        print(">>> New shape of all_df:", filtered_all_df.shape)
        print(
            ">>> Done filtering data and re-creating X_train, X_test, y_train, y_test, shape of X_train, X_test, y_train, y_test",
            X_train.shape,
            X_test.shape,
            y_train.shape,
            y_test.shape,
        )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    hela_X_scaled = scaler.transform(hela_X)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "SVM": SVC(),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
    }

    results = {
        "Model": [],
        "Dataset": [],
        "Accuracy": [],
        "Precision": [],
        "Recall": [],
        "F1-Score": [],
        "Support": [],  # Note: Support is not directly meaningful in this context as it's averaged
    }

    for name, model in models.items():
        print(">>> Evaluating", name)
        accuracy_test, metrics_test, accuracy_hela, metrics_hela = evaluate_model(
            model, X_train_scaled, y_train, X_test_scaled, y_test, hela_X_scaled, hela_y
        )

        # Test dataset metrics
        results["Model"].append(name)
        results["Dataset"].append("Test")
        results["Accuracy"].append(accuracy_test)
        results["Precision"].append(metrics_test[0])
        results["Recall"].append(metrics_test[1])
        results["F1-Score"].append(metrics_test[2])
        results["Support"].append(metrics_test[3])  # This is a placeholder

        # Hela dataset metrics
        results["Model"].append(name)
        results["Dataset"].append("Hela")
        results["Accuracy"].append(accuracy_hela)
        results["Precision"].append(metrics_hela[0])
        results["Recall"].append(metrics_hela[1])
        results["F1-Score"].append(metrics_hela[2])
        results["Support"].append(metrics_hela[3])  # This is a placeholder

    save_results(directory, results, args.start_time, args.end_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate models on given datasets and save the results.")
    parser.add_argument("--directory", type=str, help="Directory containing the datasets")
    # Add start time
    parser.add_argument("--start_time", type=int, help="Start time of the dataset", default=None)
    # Add end time
    parser.add_argument("--end_time", type=int, help="End time of the dataset", default=None)

    args = parser.parse_args()

    main(args)
