import unittest
import numpy as np
import pandas as pd
import torch
import os
from livecellx.model_zoo.segmentation import csn_configs
from livecellx.model_zoo.segmentation.eval_csn import (
    assemble_dataset_model,
    compute_metrics,
    compute_metrics_batch,
    assemble_dataset,
)
from livecellx.model_zoo.segmentation.sc_correction import CorrectSegNet
from livecellx.model_zoo.segmentation.sc_correction_aux import CorrectSegNetAux
from livecellx.model_zoo.segmentation.train_csn_aux_al import df2dataset


def get_sample_dataset(model):
    df_path = "./notebooks/notebook_results/a549_ccp_vim/test_data_v18/train_data_aux.csv"
    df = pd.read_csv(df_path)
    df = df[:128]
    eval_transform = csn_configs.CustomTransformEdtV9(use_gaussian_blur=True, gaussian_blur_sigma=30)
    dataset = assemble_dataset_model(df, model)
    dataset.transform = eval_transform
    return dataset


def get_real_model():
    ckpt_path = "./notebooks/lightning_logs/version_v18_02-inEDTv1-augEdtV9-scaleV2-lr-0.0001-aux-seed-404/checkpoints/epoch=453-global_step=0.ckpt"
    model = CorrectSegNetAux.load_from_checkpoint(ckpt_path)
    model.eval()
    model.cuda()
    return model


class TestEvalCsnBatch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.model = get_real_model()
            cls.dataset = get_sample_dataset(cls.model)
        except Exception as e:
            raise unittest.SkipTest(f"Skipping TestEvalCsnBatch: CSN eval setup failed.")

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA GPU is required for this test.")
    def test_compute_metrics_vs_batch(self):
        # Require export CUBLAS_WORKSPACE_CONFIG=:4096:8
        # torch.use_deterministic_algorithms(True)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        import os

        old_cwd = os.getcwd()
        try:
            os.chdir("./notebooks")
            subset = self.dataset
            metrics = compute_metrics(subset, self.model, out_threshold=0.6, return_mean=False)
            metrics_batch = compute_metrics_batch(
                subset, self.model, out_threshold=0.6, return_mean=False, batch_size=4
            )
            for key in metrics:
                self.assertIn(key, metrics_batch)
                np.testing.assert_allclose(
                    metrics[key],
                    metrics_batch[key],
                    rtol=0,
                    atol=1e-3,
                    err_msg="{}: {} vs {}".format(key, metrics[key], metrics_batch[key]),
                )

            # Check mean metrics results
            metrics_mean = compute_metrics(subset, self.model, out_threshold=0.6, return_mean=True)
            metrics_batch_mean = compute_metrics_batch(
                subset, self.model, out_threshold=0.6, return_mean=True, batch_size=2
            )
            for key in metrics_mean:
                self.assertIn(key, metrics_batch_mean)
                np.testing.assert_allclose(
                    metrics_mean[key],
                    metrics_batch_mean[key],
                    rtol=0,
                    atol=1e-3,
                    err_msg="{}: {} vs {}".format(key, metrics_mean[key], metrics_batch_mean[key]),
                )
        finally:
            os.chdir(old_cwd)


if __name__ == "__main__":
    unittest.main()
