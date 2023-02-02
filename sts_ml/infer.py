import math
import matplotlib.pyplot as plt
import re
import pandas as pd
import numpy as np
from glob import glob
import os, os.path
import json
import torch

from sts_ml.train import Model, pad_samples, TRAINING_DIR, PARAMS_FILENAME, TOKENS_FILENAME

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

        logits, losses = model.predict_samples(np.random.choice(train_dataset, size=batch_size))
        losses = {f"training_{key}": value.item() for key, value in losses.items()}
        metrics_row = {
            "epoch": ckpt,
        }
        metrics_row.update(losses)

        losses = model.predict_batched(np.random.choice(val_dataset, size=batch_size*8), batch_size)
        losses = {f"val_{key}": value.item() for key, value in losses.items()}
        metrics_row.update(losses)

        metrics_df = metrics_df.append(metrics_row, ignore_index=True)
        metrics_df.to_csv(f"{training_dir}/post_generated_metrics.csv")

def plot_training_metrics(training_dirname):
    plt.figure(figsize=(20,15))

    df = pd.read_csv(f"trainings/{training_dirname}/metrics.csv")
    epochs = df.epoch

    plt.grid()

    def plot_holed_values(epochs, values, linestyle, color):
        xs = [x for x,val in zip(epochs, values) if not math.isnan(val)]
        ys = [val for x,val in zip(epochs, values) if not math.isnan(val)]
        plt.plot(xs, ys, linestyle=linestyle, color=color)

    themes = [
        ("blue", "cyan"),
        ("orange", "yellow"),
        ("green", "lime"),
        ("black", "grey"),
    ]
    loss_name_to_color_theme = {}

    for key in df.columns:
        
        loss_name = None
        if key.startswith("training"):
            loss_name = key.split("training_")[-1]
        elif key.startswith("val"):
            loss_name = key.split("val_")[-1]
        if loss_name is not None:
            if loss_name not in loss_name_to_color_theme:
                loss_name_to_color_theme[loss_name] = themes.pop(0)
            theme = loss_name_to_color_theme[loss_name]
        
        if key.startswith("training"):
            plt.plot(epochs, list(df[key]), label=loss_name, color=theme[0])
        elif key.startswith("val"):
            plot_holed_values(epochs, list(df[key]), linestyle='dashed', color=theme[1])
    plt.legend()

    return df.head(len(df))


if __name__ == "__main__":
    training_dirname = ""
    # training_dirname = "2023-01-06-23-41-35_21400_blocks4-256_split0.8"
    training_dirname = "2023-01-06-20-25-13_21400_blocks4-256_split0.8"
    generate_metrics(training_dirname)