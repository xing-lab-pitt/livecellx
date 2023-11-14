import os
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import zipfile

# Function to update labels in the DataFrame
def update_labels(df):
    df["true_label"] = df["true_label"].replace(2, 1)
    df["predicted_label"] = df["predicted_label"].replace(2, 1)
    return df


# Function to calculate precision, recall, F1 score, and accuracy
def calculate_metrics(df):
    true_labels = df["true_label"]
    predicted_labels = df["predicted_label"]
    precision = precision_score(true_labels, predicted_labels, average="weighted", zero_division=0)
    recall = recall_score(true_labels, predicted_labels, average="weighted", zero_division=0)
    f1 = f1_score(true_labels, predicted_labels, average="weighted", zero_division=0)
    accuracy = accuracy_score(true_labels, predicted_labels)
    return precision, recall, f1, accuracy


# Function to extract mitosis_traj_type and frame_type from subfolder name
def extract_info_from_subfolder_name(subfolder_name):
    mitosis_traj_types = ["inclusive", "drop-div", "-st"]
    frame_type = subfolder_name.split("-")[-1]
    for traj_type in mitosis_traj_types:
        if traj_type in subfolder_name:
            return traj_type, frame_type
    return "unknown", frame_type


# Paths
# zip_file_path = '/path/to/your/zip/file.zip'
# extracted_dir_path = '/path/to/extracted/content'
# test_results_dir_path = "./work_dirs/eval_results/hela-all"
# results_csv_path = './tmp_eval-results-hela-all.csv'

test_results_dir_path = "./work_dirs/eval_results/hela-all"
results_csv_path = "./tmp_eval-results-hela-all.csv"

# Unzipping the file
# with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
#     zip_ref.extractall(extracted_dir_path)

# Getting list of subfolders
subfolders = os.listdir(test_results_dir_path)

# Processing each subfolder
results = []
for subfolder in subfolders:
    subfolder_path = os.path.join(test_results_dir_path, subfolder)
    if os.path.exists(os.path.join(subfolder_path, "all_predictions.csv")) == False:
        print("Skipping", subfolder, ", as all_predictions.csv does not exist in the subfolder")
        continue
    all_predictions_csv_path = os.path.join(subfolder_path, "all_predictions.csv")

    # Reading data
    df = pd.read_csv(all_predictions_csv_path)

    # Update labels
    df = update_labels(df)

    frame_types = df["frame_type"].unique()
    # Extract mitosis_traj_type and frame_type
    mitosis_traj_type, _ = extract_info_from_subfolder_name(subfolder)

    # Calculate metrics
    precision, recall, f1, accuracy = calculate_metrics(df)

    # Store results
    subfolder_results = {
        "subfolder": subfolder,
        "mitosis_traj_type": mitosis_traj_type,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }

    for frame_type in frame_types:
        subset_df = df[df["frame_type"] == frame_type]
        assert len(subset_df) != 0
        precision, recall, f1, accuracy = calculate_metrics(subset_df)
        subfolder_results[f"precision_{frame_type}"] = precision
        subfolder_results[f"recall_{frame_type}"] = recall
        subfolder_results[f"f1_{frame_type}"] = f1
        subfolder_results[f"accuracy_{frame_type}"] = accuracy

    results.append(subfolder_results)
# Convert results to a DataFrame
results_df = pd.DataFrame(results)

# Save results to a CSV file
results_df.to_csv(results_csv_path, index=False)
