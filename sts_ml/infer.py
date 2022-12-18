import json
import torch

from sts_ml.train import Model, pad_samples

def main():
    dataset = json.load(open("./draft_dataset.data", "r"))
    dataset = pad_samples(dataset)

    model = Model()
    model.load_state_dict(torch.load("2022-12-18-15-12-31_padDataset_999.ckpt"))
    model.eval()

    idx = 0
    while 1:
        idx = (idx + 1001) % len(dataset)
        model.predict(dataset[idx])

if __name__ == "__main__":
    main()