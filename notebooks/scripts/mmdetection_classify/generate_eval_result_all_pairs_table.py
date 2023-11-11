import zipfile
import os
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


def process_subfolder(subfolder_path):
    """Process a single subfolder and calculate the required metrics."""
    # Paths to the CSV files
    all_predictions_path = os.path.join(subfolder_path, "all_predictions.csv")

    # Read the CSV file
    try:
        data = pd.read_csv(all_predictions_path)
    except Exception as e:
        return {"error": str(e), "subfolder": subfolder_path.split("/")[-1]}

    # Initialize results dictionary
    results = {
        "subfolder": subfolder_path.split("/")[-1],  # Store only the subfolder name
        "precision_all": None,
        "recall_all": None,
        "f1_all": None,
        "accuracy_all": None,
    }

    # Calculate metrics for all rows
    results["precision_all"], results["recall_all"], results["f1_all"], results["accuracy_all"] = calculate_metrics(
        data["true_label"], data["predicted_label"]
    )

    # Check if 'mitosis_traj_type' column is present
    if "mitosis_traj_type" in data.columns:
        # Calculate metrics for all traj_type
        for traj_type in data["mitosis_traj_type"].unique():
            subset = data[data["mitosis_traj_type"] == traj_type]
            precision, recall, f1, accuracy = calculate_metrics(subset["true_label"], subset["predicted_label"])
            results[f"precision_{traj_type}"] = precision
            results[f"recall_{traj_type}"] = recall
            results[f"f1_{traj_type}"] = f1
            results[f"accuracy_{traj_type}"] = accuracy

    # Calculate metrics for all frame_type
    for frame_type in data["frame_type"].unique():
        subset = data[data["frame_type"] == frame_type]
        precision, recall, f1, accuracy = calculate_metrics(subset["true_label"], subset["predicted_label"])
        results[f"precision_{frame_type}"] = precision
        results[f"recall_{frame_type}"] = recall
        results[f"f1_{frame_type}"] = f1
        results[f"accuracy_{frame_type}"] = accuracy

    # Calculate metrics for every pair of traj_type x frame_type
    if "mitosis_traj_type" in data.columns:
        for traj_type in data["mitosis_traj_type"].unique():
            for frame_type in data["frame_type"].unique():
                subset = data[(data["mitosis_traj_type"] == traj_type) & (data["frame_type"] == frame_type)]
                if not subset.empty:
                    precision, recall, f1, accuracy = calculate_metrics(subset["true_label"], subset["predicted_label"])
                    results[f"precision_{traj_type}_{frame_type}"] = precision
                    results[f"recall_{traj_type}_{frame_type}"] = recall
                    results[f"f1_{traj_type}_{frame_type}"] = f1
                    results[f"accuracy_{traj_type}_{frame_type}"] = accuracy
                else:
                    results[f"precision_{traj_type}_{frame_type}"] = "NA"
                    results[f"recall_{traj_type}_{frame_type}"] = "NA"
                    results[f"f1_{traj_type}_{frame_type}"] = "NA"
                    results[f"accuracy_{traj_type}_{frame_type}"] = "NA"

    return results


# Apply the function to all subfolders and collect the results
results_list = []
for subfolder in inner_folder_contents:
    subfolder_path = os.path.join(inner_folder_path, subfolder)
    result = process_subfolder(subfolder_path)
    results_list.append(result)

# Create a DataFrame from the results
results_df = pd.DataFrame(results_list)

# Filter out rows with errors
valid_results_df = results_df[results_df["error"].isna()].drop(columns="error")

# Save the results to a CSV file
output_csv_path = "/mnt/data/eval_results_metrics_updated.csv"
valid_results_df.to_csv(output_csv_path, index=False)

output_csv_path
