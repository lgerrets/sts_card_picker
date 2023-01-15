import re
import pandas as pd
import numpy as np
from glob import glob
import os, os.path
import json
import torch

from sts_ml.train import Model, pad_samples, TRAINING_DIR, PARAMS_FILENAME, TOKENS_FILENAME

def main():
    dataset = json.load(open("./draft_dataset.data", "r"))
    dataset = pad_samples(dataset)

    training_dirname = "2022-12-23-21-39-39_november_blocks8"
    ckpt = 3486
    training_dir = os.path.join(".", TRAINING_DIR, training_dirname)

    model = Model.load_model(training_dir, ckpt)
    model.eval()

    train_val_split = int(0.8*len(dataset))
    idx = 0
    while 1:
        idx = (idx + 1001) % len(dataset)
        model.predict(dataset[idx])

def count_parameters(parameters):
    count = 0
    for p in parameters:
        count += np.prod(p.shape)
    return count

def generate_metrics(training_dirname):

    training_dir = os.path.join(".", TRAINING_DIR, training_dirname)
    model = Model.load_model(training_dir, ckpt=None)
    params = model.params

    # dataset = json.load(open("./SlayTheData_win_a20_ic_21400.data", "r"))
    dataset = json.load(open(params["train"]["dataset"], "r"))
    dataset = pad_samples(dataset)
    
    split = params["train"]["split"]
    train_val_split = int(split * len(dataset))
    train_dataset = dataset[:train_val_split]
    val_dataset = dataset[train_val_split:]

    metrics_df = pd.DataFrame()
    batch_size = int(2**8)
    ckpt_filenames = glob(os.path.join(training_dir, "*.ckpt"))
    ckpts = [int(re.match(".*[^\d](\d+)\.ckpt", filename).groups()[0]) for filename in ckpt_filenames]
    order = np.argsort(ckpts)
    ckpts = np.array(ckpts)[order]
    ckpt_filenames = np.array(ckpt_filenames)[order]

    for ckpt_filename, ckpt in zip(ckpt_filenames, ckpts):
        state_dict = torch.load(ckpt_filename)
        assert count_parameters(list(model.parameters())) == count_parameters(list(state_dict.values()))
        model.load_state_dict(state_dict)
        model.eval()

        logits, cross_ent_loss, l_inf, l_1 = model.predict_samples(np.random.choice(train_dataset, size=batch_size))
        cross_ent_loss = cross_ent_loss.item()
        l_inf = l_inf.item()
        l_1 = l_1.item()
        metrics_row = {
            "epoch": ckpt,
            "training_loss": cross_ent_loss,
            "training_L_inf": l_inf,
            "training_L_1": l_1,
        }

        cross_ent_loss, l_inf, l_1 = model.predict_batched(np.random.choice(val_dataset, size=batch_size*8), batch_size)
        metrics_row.update({
            "val_loss": cross_ent_loss,
            "val_L_inf": l_inf,
            "val_L_1": l_1,
        })

        metrics_df = metrics_df.append(metrics_row, ignore_index=True)
        metrics_df.to_csv(f"{training_dir}/metrics.csv")

if __name__ == "__main__":
    # main()

    training_dirname = ""
    # training_dirname = "2023-01-06-23-41-35_21400_blocks4-256_split0.8"
    training_dirname = "2023-01-06-20-25-13_21400_blocks4-256_split0.8"
    generate_metrics(training_dirname)