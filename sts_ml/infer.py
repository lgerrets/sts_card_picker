import os, os.path
import json
import torch

from sts_ml.train import Model, pad_samples, TRAINING_DIR, PARAMS_FILENAME

def main():
    dataset = json.load(open("./draft_dataset.data", "r"))
    dataset = pad_samples(dataset)

    training_dirname = "2022-12-23-21-39-39_november_blocks8"
    ckpt = 3486
    training_dir = os.path.join(".", TRAINING_DIR, training_dirname)

    params = json.load(open(os.path.join(training_dir, PARAMS_FILENAME), "r"))
    model = Model(params)
    model.load_state_dict(torch.load(os.path.join(training_dir, f"{ckpt}.ckpt")))
    model.eval()

    train_val_split = int(0.8*len(dataset))
    idx = 0
    while 1:
        idx = (idx + 1001) % len(dataset)
        model.predict(dataset[idx])

if __name__ == "__main__":
    main()