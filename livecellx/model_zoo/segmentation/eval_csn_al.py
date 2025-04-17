import os
import argparse
import pandas as pd
from datetime import datetime
from livecellx.model_zoo.segmentation.eval_csn import compute_metrics, compute_metrics_batch, assemble_dataset_model
from livecellx.model_zoo.segmentation.sc_correction_aux import CorrectSegNetAux
from livecellx.model_zoo.segmentation import csn_configs


def get_eval_dataset(data_csv, model):
    df = pd.read_csv(data_csv)
    eval_transform = csn_configs.CustomTransformEdtV9(use_gaussian_blur=True, gaussian_blur_sigma=30)
    dataset = assemble_dataset_model(df, model)
    dataset.transform = eval_transform
    return dataset


def find_iteration_dirs(model_root):
    return sorted(
        [d for d in os.listdir(model_root) if d.startswith("iter_") and os.path.isdir(os.path.join(model_root, d))],
        key=lambda x: int(x.split("_")[1]),
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate all iterations and save metrics table.")
    parser.add_argument("--model_root", type=str, required=True, help="Root directory containing iter_* folders")
    parser.add_argument("--data_csv", type=str, required=True, help="CSV file for evaluation dataset")
    parser.add_argument("--save_csv", type=str, default=None, help="Optional: output CSV file path")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for evaluation")

    args = parser.parse_args()

    # Prepare save directory and file
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_dir = os.path.join(args.model_root, f"eval_results_{now_str}")
    os.makedirs(eval_dir, exist_ok=True)
    save_csv = args.save_csv or os.path.join(eval_dir, "metrics.csv")

    results = []
    iter_dirs = find_iteration_dirs(args.model_root)
    for iter_dir in iter_dirs:
        ckpt_path = os.path.join(args.model_root, iter_dir, "checkpoints", "last.ckpt")
        if not os.path.exists(ckpt_path):
            print(f"Checkpoint not found for {iter_dir}, skipping.")
            continue
        print(f"Evaluating {ckpt_path} ...")
        model = CorrectSegNetAux.load_from_checkpoint(ckpt_path)
        model.eval()
        model.cuda()
        dataset = get_eval_dataset(args.data_csv, model)
        metrics = compute_metrics_batch(dataset, model, out_threshold=0.6, return_mean=True, batch_size=args.batch_size)
        metrics["iteration"] = iter_dir
        results.append(metrics)
    if not results:
        print("No results to save.")
        return
    df = pd.DataFrame(results)
    df.set_index("iteration", inplace=True)
    df.to_csv(save_csv)
    print(f"Saved metrics table to {save_csv}")
    print(df)


if __name__ == "__main__":
    main()
