import random
import shutil
import os
import pandas as pd
import copy
from typing import List
import json
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import sys

import torch, torch.nn as nn
from torch.distributions import Categorical
from torch.utils.data import IterableDataset, DataLoader
from torch.utils.data._utils import collate

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sts_ml.deck_history import ALL_CARDS_FORMATTED, ALL_RELICS_FORMATTED, card_to_name, card_to_n_upgrades

PAD_TOKEN = "PAD_TOKEN"
CARD_TOKENS = [PAD_TOKEN] + list(ALL_CARDS_FORMATTED)
CARD_AND_RELIC_TOKENS = CARD_TOKENS + list(ALL_RELICS_FORMATTED)
TRAINING_DIR = "trainings"
PARAMS_FILENAME = "params.json"
TOKENS_FILENAME = "tokens.json"

def pad_samples(samples : List[dict]):
    seq_max_size = 0
    for sample in samples:
        items_n = len(sample["deck"])
        picked_n = len(sample["cards_picked"])
        skipped_n = len(sample["cards_skipped"])
        seq_max_size = max(seq_max_size, items_n + picked_n + skipped_n)
    ret = []
    for sample in samples:
        items_n = len(sample["deck"])
        picked_n = len(sample["cards_picked"])
        skipped_n = len(sample["cards_skipped"])
        pad_left = seq_max_size - items_n - picked_n - skipped_n
        padded_sample = pad_sample(sample, pad_left)
        ret.append(padded_sample)
    return ret

def pad_sample(sample : dict, pad_left : int):
    sample = copy.deepcopy(sample)
    sample["deck"] = [PAD_TOKEN]*pad_left + sample["deck"]
    return sample

def unpad(sample):
    sample = copy.deepcopy(sample)
    sample["deck"] = [card for card in sample["deck"] if card != PAD_TOKEN]
    return sample

def torch_to_numpy(tensor):
    return tensor.detach().cpu().numpy()

class DeckDataset(IterableDataset):
    PAD_INDEX = 0

    def __init__(self, samples: list, tokens: list, do_relics: bool):
        self.samples = samples
        self.tokens = tokens
        self.do_relics = do_relics
        assert tokens.index(PAD_TOKEN) == DeckDataset.PAD_INDEX

    def token_to_index(self, token : str):
        name = card_to_name(token)
        assert name in self.tokens, name
        index = self.tokens.index(name)
        return index
    
    def generator(self):
        while 1:
            sample = random.choice(self.samples)
            ret = self.preprocess(sample)
            yield ret
    
    def preprocess(self, sample: dict):
        tokens = sample["deck"] + sample["cards_picked"] + sample["cards_skipped"]
        items_n = len(sample["deck"])
        if self.do_relics:
            tokens = sample["relics"] + tokens
            items_n += len(sample["relics"])
        idxes = np.array([self.token_to_index(token) for token in tokens], dtype=int)
        cards_picked_n = len(sample["cards_picked"])
        cards_skipped_n = len(sample["cards_skipped"])
        n_upgrades = np.array([card_to_n_upgrades(token) for token in tokens], dtype=int)

        ret = {
            "token_idxes": idxes,
            "items_n": items_n,
            "cards_picked_n": cards_picked_n,
            "cards_skipped_n": cards_skipped_n,
            "n_upgrades": n_upgrades,
        }
        return ret
    
    def sample_unpreprocessed(self):
        sample = random.choice(self.samples)
        return sample

    def __iter__(self):
        return self.generator()

    @staticmethod
    def collate_fn(samples):
        max_seq_len = -1
        for sample in samples:
            seq_len = sample["items_n"] + sample["cards_picked_n"] + sample["cards_skipped_n"]
            max_seq_len = max(max_seq_len, seq_len)
        assert max_seq_len > -1, max_seq_len

        for sample in samples:
            seq_len = sample["items_n"] + sample["cards_picked_n"] + sample["cards_skipped_n"]
            n_pad_left = max_seq_len - seq_len
            padding = DeckDataset.PAD_INDEX * np.ones(n_pad_left, dtype=int)
            sample["token_idxes"] = np.hstack([padding, sample["token_idxes"]])
            sample["n_upgrades"] = np.hstack([padding, sample["n_upgrades"]])
            sample["n_pad_lefts"] = n_pad_left
        
        ret = {}
        for key in ["token_idxes", "n_upgrades"]:
            ret[key] = collate.default_collate([sample[key] for sample in samples])
        
        for key in ["items_n", "cards_picked_n", "cards_skipped_n", "n_pad_lefts"]:
            ret[key] = np.array([sample[key] for sample in samples])

        return ret

class Model(nn.Module):
    def load_model(training_dir : str, ckpt : int) -> "Model":
        params = json.load(open(os.path.join(training_dir, PARAMS_FILENAME), "r"))
        tokens = json.load(open(os.path.join(training_dir, TOKENS_FILENAME), "r"))
        model = Model(params=params, tokens=tokens)
        if ckpt is not None:
            model.load_state_dict(torch.load(os.path.join(training_dir, f"{ckpt}.ckpt")))
        return model

    def __init__(self, params, tokens=None) -> None:
        super().__init__()
        
        self.params = params
        self.dim = dim = params["model"]["dim"]
        if tokens is None:
            if params["model"].get("input_relics", False):
                tokens = CARD_AND_RELIC_TOKENS
                raise NotImplementedError("Wip: implement the forward pass")
            else:
                tokens = CARD_TOKENS
        if PAD_TOKEN not in tokens:
            tokens = [PAD_TOKEN] + tokens
        self.tokens = tokens
        self.pad_idx = tokens.index(PAD_TOKEN)
        self.embedding = nn.Embedding(len(tokens), dim)
        if params["model"]["block_class"] == "MHALayer":
            blocks = [MHALayer(
                dim=dim,
                ffdim=params["model"]["ffdim"],
                nheads=params["model"]["nheads"],
                mask_inter_choices=params["model"].get("mask_inter_choices")) for _ in range(params["model"]["blocks"]
            )]
        elif params["model"]["block_class"] == "Linear":
            blocks = [nn.Linear(dim, dim) for _ in range(params["model"]["blocks"])]
        else:
            assert False
        self.blocks = nn.ModuleList(blocks)
        self.projection = nn.Linear(dim, 2)

        self.device = device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(device)

        self.opt = torch.optim.Adam(self.parameters(), lr=params["train"]["lr"])
    
    def token_to_index(self, token : str):
        name = card_to_name(token)
        assert name in self.tokens, name
        index = self.tokens.index(name)
        return index
    
    def forward(self, batch):
        """
        tensors, arrays -> logits
        """
        batch = self._to_device(batch)

        idxes = batch["token_idxes"]
        n_pad_lefts = batch["n_pad_lefts"]
        items_n = batch["items_n"]
        cards_picked_n = batch["cards_picked_n"]
        cards_skipped_n = batch["cards_skipped_n"]
        n_upgrades = batch["n_upgrades"]

        feat = self.embedding(idxes)
        n_upgrades = torch.clip(n_upgrades, 0, 1)
        for idx, n_choice in enumerate(cards_picked_n + cards_skipped_n):
            # scalar #0 of embedding encodes whether this is a choice token
            feat[idx,:-n_choice,0] = 0
            feat[idx,n_choice:,0] = 1
            # scalar #1 of embedding encodes whether this is an upgraded card
            feat[idx,:,1] = n_upgrades[idx]
        
        kwargs = {}
        if self.params["model"]["block_class"] == "MHALayer":
            kwargs.update({
                "n_pad_lefts": n_pad_lefts,
                "n_choices": cards_picked_n + cards_skipped_n,
            })
        for block in self.blocks:
            feat = block.forward(feat, **kwargs)

        feat = self.projection(feat)
        logits = torch.softmax(feat, dim=2)

        return logits
    
    def predict_batch(self, batch):
        """
        samples -> logits, metrics
        """
        bs = batch["token_idxes"].shape[0]
        logits = self.forward(batch)
        cross_ent_loss = 0.
        top_inf_acc = 0.
        top_true_acc = 0.
        top_1_acc = 0.
        top_1_acc_n_items = 0
        for sample_idx in range(bs):
            n_pad_lefts = batch["n_pad_lefts"][sample_idx]
            picked_n = batch["cards_picked_n"][sample_idx]
            skipped_n = batch["cards_skipped_n"][sample_idx]
            pred_logits = logits[sample_idx, -picked_n-skipped_n:]
            
            cross_ent_loss += - torch.sum(torch.log(pred_logits[:picked_n, 1])) - torch.sum(torch.log(pred_logits[picked_n:, 0]))
            modes = pred_logits > 0.5
            pred_modes = torch.concatenate([modes[:picked_n, 1], modes[picked_n:, 0]], axis=0).float()
            top_inf_acc += 1 - torch.mean(pred_modes) # 0 if all modes are true, 1 if all modes are false; NOTE: it may be far from the target metric, if each card is predicted independently of others; NOTE: this was renamed from L_1
            if picked_n == 0: # it's true to skip
                top_true_acc += torch.max(modes[:, 1]) # 1 if picked at least one
                top_1_acc += torch.max(modes[:, 1])
                top_1_acc_n_items += 1
            else:
                top_picked_target = torch.arange(0, picked_n, device=self.device)
                predict_picked_n = torch.sum(pred_logits[:,1] > 0.5).item()
                sorted_picks_pred = torch.argsort(- pred_logits[:,1])
                top_picked_preds = sorted_picks_pred[:min(picked_n, predict_picked_n)]
                diff = top_picked_preds[None, :] == top_picked_target[:, None]
                score = torch.sum(diff.float()) / picked_n
                top_true_acc += 1 - score
                if picked_n == 1:
                    top_1_acc += 1 - score
                    top_1_acc_n_items += 1

        cross_ent_loss /= bs
        top_inf_acc /= bs
        top_true_acc /= bs
        top_1_acc /= top_1_acc_n_items
        losses = {
            "cross_ent_loss": cross_ent_loss,
            "top_inf_acc": top_inf_acc,
            "top_true_acc": top_true_acc,
            "top_1_acc": top_1_acc,
        }
        return logits, losses
    
    def learn(self, batch):
        self.opt.zero_grad()
        logits, losses = self.predict_batch(batch)
        cross_ent_loss = losses["cross_ent_loss"]
        cross_ent_loss.backward()
        self.opt.step()
        return losses
    
    def preprocess_one(self, sample: dict):
        preprocessed_sample = self.dataset.preprocess(sample)
        batch = DeckDataset.collate_fn([preprocessed_sample])
        # batch = self._to_device(batch)
        return batch

    def _to_device(self, batch: dict):
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(self.device)
        return batch
    
    def predict_one(self, sample : dict):
        """
        batch of 1 sample -> print df and metrics
        """
        batch = self.preprocess_one(sample)
        sample_idx = 0
        bs = 1
        logits, losses = self.predict_batch(batch)
        losses = {key: value.item() for key, value in losses.items()}

        token_idxes = batch["token_idxes"][sample_idx]
        n_pad_lefts = batch["n_pad_lefts"][sample_idx]
        assert n_pad_lefts == 0
        items_n = batch["items_n"][sample_idx]
        picked_n = batch["cards_picked_n"][sample_idx]
        skipped_n = batch["cards_skipped_n"][sample_idx]
        
        pick_logits_np = torch_to_numpy(logits[sample_idx, - picked_n - skipped_n:, 1])

        deck = sample["deck"]
        offered_cards = sample["cards_picked"] + sample["cards_skipped"]
        card_was_picked = [1]*picked_n + [0]*skipped_n
        
        preferences = np.argsort(- pick_logits_np)
        df = {
            "cards": deck + list(np.array(offered_cards)[preferences]),
            "expert_scores": ["deck"]*len(deck) + list(np.array(card_was_picked)[preferences]),
            "predicted_scores": ["deck"]*len(deck) + list(np.array(pick_logits_np)[preferences]),
        }
        df = pd.DataFrame(df)
        print(df)

        print(losses)

class MHALayer(nn.Module):
    def __init__(self, dim, ffdim, nheads=4, mask_inter_choices=False) -> None:
        super().__init__()
        self.dim = dim
        self.nheads = nheads
        self.att = nn.MultiheadAttention(dim, nheads, batch_first=True)
        self.ln1 = nn.LayerNorm(dim)
        self.ff1 = nn.Linear(dim, ffdim)
        self.relu = nn.ReLU()
        self.ff2 = nn.Linear(ffdim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.attn_weights = None
        self.mask_inter_choices = mask_inter_choices
    
    def forward(self, feat : torch.tensor, n_pad_lefts: np.ndarray, n_choices: np.ndarray):
        batch_size, seq_len, dim = feat.shape

        bs, seq, dim = feat.shape

        # NOTE: mask[i,j,k]==0 means embedding j may depend on embedding k
        mask = torch.ones((batch_size, seq_len, seq_len), dtype=torch.bool) # disable all attentions
        for idx in range(batch_size):
            n_pad = n_pad_lefts[idx]
            n_choice = n_choices[idx]
            for idx_pad in range(n_pad):
                mask[idx, idx_pad, idx_pad] = 0 # pad token may attend to itself (just works around nan values)
            mask[idx, n_pad:-n_choice, n_pad:-n_choice] = 0 # deck may attend to deck
            for idx_choice in range(n_choice):
                mask[idx, -n_choice+idx_choice, n_pad:-n_choice] = 0 # choice may attend to deck
                if self.mask_inter_choices:
                    mask[idx, -n_choice+idx_choice, -n_choice+idx_choice] = 0 # choice may attend to itself
            if not self.mask_inter_choices:
                mask[idx, -n_choice:, -n_choice:] = 0 # choices may attend to each other

        mask = mask.reshape((batch_size, 1, seq_len, seq_len))
        mask = torch.repeat_interleave(mask, self.nheads, dim=1)
        mask = mask.reshape((batch_size*self.nheads, seq_len, seq_len))
        mask = mask.to(feat.device)
        
        mha_feat, attn_weights = self.att(feat, feat, feat, need_weights=True, average_attn_weights=True, attn_mask=mask)
        self.attn_weights = attn_weights

        feat = feat + mha_feat
        feat = self.ln1(feat)
        ff_feat = self.ff2(self.relu(self.ff1(feat)))
        feat = feat + ff_feat
        feat = self.ln2(feat)
        return feat

def gen_ckpt_idxes():
    idxes = np.arange(100)
    idxes = idxes ** 3
    for idx in idxes: 
        yield idx
    delta = idxes[-1] - idxes[-2]
    idx = idxes[-1]
    while 1:
        idx += delta
        yield idx

def save_df(df, filepath):
    df.to_csv(filepath)
    words : list = filepath.split(".")
    words.insert(len(words)-1, "copy")
    filepath_copy = ".".join(words)
    if os.path.exists(filepath_copy):
        os.remove(filepath_copy)
    shutil.copy(filepath, filepath_copy)

def load_datasets(params):
    # dataset = json.load(open("./november_dataset.data", "r"))
    data = json.load(open(params["train"]["dataset"], "r"))
    if "dataset" in data:
        data_tokens = data["items"]
        full_dataset = data["dataset"]
    else: # backward compat
        data_tokens = CARD_TOKENS
        full_dataset = data
    
    if PAD_TOKEN not in data_tokens:
        data_tokens = [PAD_TOKEN] + data_tokens

    assert set(data_tokens).issubset(set(data_tokens) | set([PAD_TOKEN])), set.symmetric_difference(set(data_tokens), set(data_tokens) | set([PAD_TOKEN]))

    split = params["train"]["split"]
    do_relics = params["model"].get("input_relics", False)
    train_val_split = int(split * len(full_dataset))

    train_dataset = DeckDataset(
        samples=full_dataset[:train_val_split],
        tokens=data_tokens,
        do_relics=do_relics,
    )
    val_dataset = DeckDataset(
        samples=full_dataset[train_val_split:],
        tokens=data_tokens,
        do_relics=do_relics,
    )
    print(f"Train ({train_val_split}) / validation ({len(full_dataset) - train_val_split}) ratio = {split}")

    return data_tokens, train_dataset, val_dataset

def load_dataloaders(params):

    data_tokens, train_dataset, val_dataset = load_datasets(params)

    batch_size = params["train"]["batch_size"]
    device = get_available_device()
    train_dataloader = iter(DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        collate_fn=DeckDataset.collate_fn,
        pin_memory=True,
        pin_memory_device=device,
    ))
    val_dataloader = iter(DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        collate_fn=DeckDataset.collate_fn,
        pin_memory=True,
        pin_memory_device=device,
    ))

    return data_tokens, train_dataloader, val_dataloader

def get_available_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return device

def train(params : dict = None, state_dict: dict = None):
    if params is None:
        from sts_ml.params import params

    data_tokens, train_dataloader, val_dataloader = load_dataloaders(params)

    model = Model(params, tokens=data_tokens)
    if state_dict is not None:
        model.load_state_dict(state_dict)
    model.train()

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    exp_name = f"{timestamp}_blocks{params['model']['blocks']}x{params['model']['dim']}_split{params['train']['split']}"
    if params["model"].get("input_relics", False):
        exp_name += "_relics"
    exp_dir = os.path.join(".", TRAINING_DIR, exp_name)
    os.makedirs(exp_dir)
    json.dump(params, open(os.path.join(exp_dir, PARAMS_FILENAME), "w"), indent=4)
    json.dump(model.tokens, open(os.path.join(exp_dir, TOKENS_FILENAME), "w"), indent=4)
    metrics_df = pd.DataFrame()
    n_epochs = 30000
    ckpt_idxes = gen_ckpt_idxes()
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
    train()
    # pursue_training()
