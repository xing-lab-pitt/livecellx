from livecellx.model_zoo.harmony_vae_2d.sc_dataloader import scs_train_test_dataloader


if __name__ == "__main__":
    pixel = 64
    padding = 30
    train_out = f"/data/Ke_data/livecellx_linked_data/MCF10A_a549_harmony_train_pixel_{pixel}_padding_{padding}"
    test_out = f"/data/Ke_data/livecellx_linked_data/MCF10A_a549_harmony_test_pixel_{pixel}_padding_{padding}"
    print("[INFO] Creating MCF10A_and_A549 dataset with pixel size: {}".format(pixel))
    print("[INFO] Padding size: {}".format(padding))
    print("[INFO] Train output directory: {}".format(train_out))
    print("[INFO] Test output directory: {}".format(test_out))

    # Create folders
    from pathlib import Path

    Path(train_out).mkdir(parents=True, exist_ok=True)
    Path(test_out).mkdir(parents=True, exist_ok=True)

    print("[INFO] Using MCF10A_and_A549 dataset")
    train_loader, test_loader = scs_train_test_dataloader(
        padding=padding, img_shape=(pixel, pixel), batch_size=1, img_only=False
    )
    train_table_csv_path = f"{train_out}/train_data.csv"
    # Create df with columns: idx, image_path, sc_id
    import pandas as pd

    train_df = pd.DataFrame(columns=["idx", "image_path", "sc_id"])
    test_table_csv_path = f"{test_out}/test_data.csv"
    test_df = pd.DataFrame(columns=["idx", "image_path", "sc_id"])
    for batch_idx, input_dict in enumerate(train_loader):
        images = input_dict["image"]
        idx = input_dict["idx"].item()
        print(f"Batch {batch_idx + 1}: {images.shape}")
        # Save the batch to disk
        image_path = f"{train_out}/image_{idx:05d}.png"
