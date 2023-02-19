import numpy as np
import pandas as pd
import seaborn as sns
import shutil
import os

import torch, torch.nn as nn

cm = sns.light_palette("green", as_cmap=True)

def count_parameters(parameters):
    count = 0
    for p in parameters:
        count += np.prod(p.shape)
    return count

def torch_to_numpy(tensor):
    return tensor.detach().cpu().numpy()

def numpy_to_torch(arr, device):
    return torch.tensor(arr).to(device)

def save_df(df, filepath):
    df.to_csv(filepath)
    words : list = filepath.split(".")
    words.insert(len(words)-1, "copy")
    filepath_copy = ".".join(words)
    if os.path.exists(filepath_copy):
        os.remove(filepath_copy)
    shutil.copy(filepath, filepath_copy)

def get_available_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return device
