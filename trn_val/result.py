import os
import time
from glob import glob
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import sastvd as svd
import sastvd.linevd as lvd
from ray.tune import ExperimentAnalysis



main_savedir = svd.get_dir(svd.outputs_dir() / "rq_results_new")
chkpt = svd.processed_dir() / "clean_codebert/202307141026_1fca96a_update_get_data.sh/epoch=9-step=30.ckpt"
chkpt_info = Path(chkpt).parent.name
chkpt_res_path = main_savedir / f"1_{chkpt_info}.csv"
data = lvd.BigVulDatasetLineVDDataModule(
    batch_size=1024,
    nsampling_hops=2,
    gtype="pdg+raw",
    splits="default",
    feat="codebert",
)
# Load model and test
model = lvd.LitGNN()
model = lvd.LitGNN.load_from_checkpoint(chkpt, strict=False)
trainer = pl.Trainer(gpus=1, default_root_dir="/tmp/")
trainer.test(model, data)
res = [
    1,
    chkpt_info,
    model.res1vo,
    model.res2mt,
    model.res2f,
    model.res3vo,
    model.res2,
    model.lr,
]
# Save DF
mets = lvd.get_relevant_metrics(res)
res_df = pd.DataFrame.from_records([{**mets}])
res_df.to_csv(chkpt_res_path, index=0)