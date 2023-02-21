import copy
import subprocess
from typing import Callable, Union
import torch
import shutil
import os
import pandas as pd
import json
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sts_ml.deck_history import PAD_TOKEN
from sts_ml.model_abc import PARAMS_FILENAME, TOKENS_FILENAME
from sts_ml.model import CardModel
from sts_ml.win_model import WinModel
from sts_ml.helper import count_parameters, torch_to_numpy, numpy_to_torch, save_df, get_available_device

TRAINING_DIR = "trainings"

def gen_ckpt_idxes(asymptot):
    """
    Used to progressively space out checkpoints

    Example: If asymptot == 10, generates the sequence:
    0**3, 1**3, 2**3, ... 10**3, then 2*10**3, 3*10**3, etc...
    """

    idxes = np.arange(asymptot+1)
    idxes = idxes ** 3
    for idx in idxes: 
        yield idx
    delta = idxes[-1] - idxes[-2]
    idx = idxes[-1]
    while 1:
        idx += delta
        yield idx

def train(model_cls: Union[CardModel, WinModel], state_dict: dict = None):
    model, train_dataloader, val_dataloader = model_cls.create_model()

    if state_dict is not None:
        model.load_state_dict(state_dict)
    params = model.params
    model.train()

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    exp_name = f"{timestamp}_{model.__class__.__name__}_blocks{params['model']['blocks']}x{params['model']['dim']}_split{params['train']['split']}"
    if params["model"].get("input_relics", False):
        exp_name += "_relics"
    exp_dir = os.path.join(".", TRAINING_DIR, exp_name)
    os.makedirs(exp_dir)
    params_cp = copy.deepcopy(params)
    params_cp["runtime"] = {
        "git_hash": subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip(),
    }
    json.dump(params, open(os.path.join(exp_dir, PARAMS_FILENAME), "w"), indent=4)
    json.dump(model.tokens, open(os.path.join(exp_dir, TOKENS_FILENAME), "w"), indent=4)
    metrics_df = pd.DataFrame()
    n_epochs = 30000
    ckpt_idxes = gen_ckpt_idxes(asymptot=10)
    next_epock_save_idx = next(ckpt_idxes)
    epoch = 0
    metrics_filepath = os.path.join(exp_dir, "metrics.csv")
    while epoch < n_epochs:
        model.train()

        losses = model.learn(next(train_dataloader))
        losses = {f"training_{key}": value.item() for key, value in losses.items()}
        
        metrics_row = {
            "epoch": epoch,
        }
        metrics_row.update(losses)

        if epoch == next_epock_save_idx:
            model.eval()
            logits, losses = model.predict_batch(next(val_dataloader))
            losses = {f"val_{key}": value.item() for key, value in losses.items()}
            metrics_row.update(losses)

            next_epock_save_idx = next(ckpt_idxes)
            torch.save(model.state_dict(), os.path.join(exp_dir, f"{epoch}.ckpt"))

        print(metrics_row)
        row_df = pd.DataFrame(metrics_row, index=[0])
        metrics_df = pd.concat([metrics_df, row_df], ignore_index=True)
        save_df(metrics_df, metrics_filepath)
        
        epoch += 1

    torch.save(model.state_dict(), os.path.join(exp_dir, f"{epoch}.ckpt"))
    save_df(metrics_df, metrics_filepath)

    model.eval()
    
def pursue_training(training_dirname, ckpt):
    params = json.load(open(os.path.join(training_dirname, PARAMS_FILENAME), "r"))
    state_dict = torch.load(os.path.join(training_dirname, f"{ckpt}.ckpt"))
    train(params, state_dict=state_dict)

if __name__ == "__main__":
    train(model_cls=CardModel)
    # train(model_cls=WinModel)
    # pursue_training()
