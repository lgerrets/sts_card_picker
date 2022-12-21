import os
import pandas as pd
import copy
from typing import List
import json
import numpy as np
import torch, torch.nn as nn
from torch.distributions import Categorical
from datetime import datetime
import matplotlib.pyplot as plt

from sts_ml.deck_history import ALL_CARDS_FORMATTED, card_to_name, card_to_n_upgrades

ALL_TOKENS = ["PAD_TOKEN"] + ALL_CARDS_FORMATTED
TRAINING_DIR = "./trainings"

def token_to_index(token : str):
    name = card_to_name(token)
    assert name in ALL_TOKENS, name
    index = ALL_TOKENS.index(name)
    return index

def pad_samples(samples : List[dict]):
    seq_max_size = 0
    for sample in samples:
        deck_n = len(sample["deck"])
        picked_n = len(sample["cards_picked"])
        skipped_n = len(sample["cards_skipped"])
        seq_max_size = max(seq_max_size, deck_n + picked_n + skipped_n)
    ret = []
    for sample in samples:
        deck_n = len(sample["deck"])
        picked_n = len(sample["cards_picked"])
        skipped_n = len(sample["cards_skipped"])
        pad_left = seq_max_size - deck_n - picked_n - skipped_n
        padded_sample = pad_sample(sample, pad_left)
        ret.append(padded_sample)
    return ret

def pad_sample(sample : dict, pad_left : int):
    sample = copy.deepcopy(sample)
    sample["deck"] = ["PAD_TOKEN"]*pad_left + sample["deck"]
    return sample

class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        self.dim = dim = 256
        self.embedding = nn.Embedding(len(ALL_TOKENS), dim)
        blocks = [Block(dim, 2*dim, 4) for _ in range(4)]
        self.blocks = nn.Sequential(*blocks)
        self.projection = nn.Linear(dim, 2)

        self.device = device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(device)

        self.opt = torch.optim.Adam(self.parameters(), lr=1e-4)

    def forward(self, idxes : torch.tensor, deck_sizes : np.ndarray, n_upgrades : torch.tensor):
        feat = self.embedding(idxes)
        n_upgrades = torch.clip(n_upgrades, 0, 1)
        for idx, deck_size in enumerate(deck_sizes):
            feat[idx,:deck_size,-1] = 0
            feat[idx,deck_size:,-1] = 1
            feat[idx,:,-2] = n_upgrades[idx]
        feat = self.blocks(feat)
        feat = self.projection(feat)
        logits = torch.softmax(feat, dim=2)
        return logits
    
    def predict_samples(self, samples : List[dict]):
        logits = self.infer(samples)
        cross_ent_loss = 0.
        l_inf = 0.
        l_1 = 0.
        for sample_idx, sample in enumerate(samples):
            deck_n = len(sample["deck"])
            picked_n = len(sample["cards_picked"])
            skipped_n = len(sample["cards_skipped"])
            cross_ent_loss += - torch.sum(torch.log(logits[sample_idx, deck_n:deck_n+picked_n, 1])) - torch.sum(torch.log(logits[sample_idx, deck_n+picked_n:deck_n+picked_n+skipped_n, 0]))
            modes = logits > 0.5
            pred_modes = torch.concatenate([modes[sample_idx:sample_idx+1, deck_n:deck_n+picked_n, 1], modes[sample_idx:sample_idx+1, deck_n+picked_n:deck_n+picked_n+skipped_n, 0]], axis=1)
            l_inf += 1 - torch.min(pred_modes.float())
            l_1 += 1 - torch.mean(pred_modes.float())
        return logits, cross_ent_loss, l_inf, l_1
    
    def learn(self, dataset, batch_size):
        self.opt.zero_grad()
        samples = []
        for sample_idx in range(batch_size):
            dataset_idx = np.random.randint(len(dataset))
            sample = dataset[dataset_idx]
            samples.append(sample)
        logits, cross_ent_loss, l_inf, l_1 = self.predict_samples(samples)
        cross_ent_loss.backward()
        self.opt.step()
        return cross_ent_loss, l_inf, l_1
    
    def infer(self, samples : dict):
        padded_samples = pad_samples(samples)
        batched_idxes = []
        deck_sizes = []
        n_upgrades = []
        for sample in padded_samples:
            tokens = sample["deck"] + sample["cards_picked"] + sample["cards_skipped"]
            idxes = [token_to_index(token) for token in tokens]
            batched_idxes.append(idxes)
            deck_sizes.append(len(sample["deck"]))
            n_upgrades.append([card_to_n_upgrades(token) for token in tokens])
        batched_idxes = np.array(batched_idxes)
        batched_idxes = torch.from_numpy(batched_idxes).to(self.device)
        deck_sizes = np.array(deck_sizes)
        n_upgrades = np.array(n_upgrades)
        n_upgrades = torch.from_numpy(n_upgrades).to(self.device)
        logits = self.forward(batched_idxes, deck_sizes=deck_sizes, n_upgrades=n_upgrades)
        return logits
    
    def predict(self, sample : dict):
        sample_idx = 0
        samples = [sample]

        logits, cross_ent_loss, l_inf, l_1 = self.predict_samples(samples)

        deck_n = len(sample["deck"])
        picked_n = len(sample["cards_picked"])
        skipped_n = len(sample["cards_skipped"])
        pick_logits_np = logits[sample_idx].detach().cpu().numpy()[- picked_n - skipped_n:,1]
        offered_cards = sample["cards_picked"] + sample["cards_skipped"]
        card_was_picked = [1]*picked_n + [0]*skipped_n
        
        deck = [card for card in sample["deck"] if card != "PAD_TOKEN"]
        preferences = np.argsort(- pick_logits_np)
        df = {
            "cards": deck + list(np.array(offered_cards)[preferences]),
            "expert_scores": ["deck"]*len(deck) + list(np.array(card_was_picked)[preferences]),
            "predicted_scores": ["deck"]*len(deck) + list(np.array(pick_logits_np)[preferences]),
        }
        df = pd.DataFrame(df)
        print(df)

        print(f"L_inf = {l_inf.item()} ; L_1 = {l_1.item()}")

class Block(nn.Module):
    def __init__(self, dim, ffdim, nheads=4) -> None:
        super().__init__()
        self.dim = dim
        self.att = nn.MultiheadAttention(dim, nheads, batch_first=True)
        self.ln1 = nn.LayerNorm(dim)
        self.ff1 = nn.Linear(dim, ffdim)
        self.relu = nn.ReLU()
        self.ff2 = nn.Linear(ffdim, dim)
        self.ln2 = nn.LayerNorm(dim)
    
    def forward(self, feat):
        mha_feat = self.att(feat, feat, feat)[0]
        feat = feat + mha_feat
        feat = self.ln1(feat)
        ff_feat = self.ff2(self.relu(self.ff1(feat)))
        feat = feat + ff_feat
        feat = self.ln2(feat)
        return feat

def gen_ckpt_idxes():
    idx = 0
    delta = 1
    while 1:
        yield idx
        idx += delta
        delta += 1

def main(model=None):
    if model is None:
        model = Model()
        model.train()

    dataset = json.load(open("./november_dataset.data", "r"))
    dataset = pad_samples(dataset)
    train_val_split = int(0.8*len(dataset))
    train_dataset = dataset[:train_val_split]
    val_dataset = dataset[train_val_split:]

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    exp_dir = f"{TRAINING_DIR}/{timestamp}_november"
    os.makedirs(exp_dir)
    metrics_df = pd.DataFrame()
    n_epochs = 10000
    ckpt_idxes = gen_ckpt_idxes()
    next_epock_save_idx = next(ckpt_idxes)
    for epoch in range(n_epochs):
        cross_ent_loss, l_inf, l_1 = model.learn(train_dataset, 256)
        
        cross_ent_loss = cross_ent_loss.item()
        l_inf = l_inf.item()
        l_1 = l_1.item()
        metrics_row = {
            "training_loss": cross_ent_loss,
            "training_L_inf": l_inf,
            "training_L_1": l_1,
        }

        if epoch == next_epock_save_idx:
            logits, cross_ent_loss, l_inf, l_1 = model.predict_samples(np.random.choice(val_dataset, size=256))
            cross_ent_loss = cross_ent_loss.item()
            l_inf = l_inf.item()
            l_1 = l_1.item()
            metrics_row.update({
                "val_loss": cross_ent_loss,
                "val_L_inf": l_inf,
                "val_L_1": l_1,
            })

        metrics_df = metrics_df.append(metrics_row, ignore_index=True)

        print(epoch, metrics_row)
        if epoch == next_epock_save_idx:
            next_epock_save_idx = next(ckpt_idxes)
            torch.save(model.state_dict(), f"{exp_dir}/{epoch}.ckpt")

    torch.save(model.state_dict(), f"{exp_dir}/{epoch}.ckpt")

    metrics_df.to_csv(f"{exp_dir}/metrics.csv")

    model.eval()
    
    idx = 0
    b = 1
    while 1:
        if b:
            idx = (idx + 1001) % len(train_dataset)
            model.predict(train_dataset[idx])
        else:
            idx = (idx + 1001) % len(val_dataset)
            model.predict(val_dataset[idx])

def pursue_training():
    model = Model()
    model.train()

    ckpt = f"{TRAINING_DIR}/"
    model.load_state_dict(torch.load(ckpt))

    main(model)

if __name__ == "__main__":
    main()
    # pursue_training()
