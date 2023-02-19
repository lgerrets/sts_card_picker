import pandas as pd
import numpy as np
import copy
import json
import os, os.path
import random
import seaborn as sns

import torch, torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader
from torch.utils.data._utils import collate

from sts_ml.deck_history import ALL_CARDS_FORMATTED, ALL_RELICS_FORMATTED, card_to_name, card_to_n_upgrades
from sts_ml.model import PARAMS_FILENAME, TOKENS_FILENAME, CARD_AND_RELIC_TOKENS, PAD_TOKEN, CARD_TOKENS, PoolTimeDimension, MHALayer, torch_to_numpy, numpy_to_torch
from sts_ml.helper import count_parameters, torch_to_numpy, numpy_to_torch, save_df, get_available_device

cm = sns.light_palette("green", as_cmap=True)

class WinDataset(IterableDataset):
    PAD_INDEX = 0

    def __init__(self, params: dict, samples: list = None, tokens: list = None):
        self.is_empty = tokens is None
        self.samples = samples
        self.tokens = tokens
        self.params = params
        self.do_relics = params["model"]["input_relics"]
        self.gamma = params["model"]["gamma"]
        assert tokens.index(PAD_TOKEN) == WinDataset.PAD_INDEX

    def token_to_index(self, token : str):
        name = card_to_name(token)
        assert name in self.tokens, name
        index = self.tokens.index(name)
        return index
    
    def generator(self):
        assert not self.is_empty
        while 1:
            sample = self.sample_unpreprocessed()
            ret = self.preprocess(sample)
            yield ret
    
    def preprocess(self, sample: dict):
        """
        NOTE: typical wins have floor_reached == 51 (56 if heart); A20 is 52 (57 if heart)
        """

        tokens = sample["deck"] + sample["cards_picked"]
        if self.do_relics:
            tokens = sample["relics"] + tokens
        items_n = len(tokens)
        idxes = np.array([self.token_to_index(token) for token in tokens], dtype=int)
        n_upgrades = np.array([card_to_n_upgrades(token) for token in tokens], dtype=int)

        current_floor = sample["floor"]
        current_act = sample["act"]
        floor_reached = sample["floor_reached"]
        act_reached = sample["act_reached"]
        delta_floors_between_victory_and_heart_victory = 5

        if act_reached == 4:
            heart_win_score_is_known = True
            if sample["victory"]: # slayed the heart
                heart_win_score = self.gamma ** (floor_reached - current_floor)
                if current_act <= 3:
                    victory_score = self.gamma ** (floor_reached - delta_floors_between_victory_and_heart_victory - current_floor)
                else:
                    victory_score = 1.
            else: # lost at act 4
                heart_win_score = - self.gamma ** (floor_reached - current_floor)
                if current_act <= 3:
                    victory_score = self.gamma ** (floor_reached - delta_floors_between_victory_and_heart_victory + 2 - current_floor) # TODO: this + 2 is an approximation!
                else:
                    victory_score = 1.
        else:
            if sample["victory"]: # completed act 3
                victory_score = self.gamma ** (floor_reached - current_floor)
                heart_win_score = 0. # we don't know
                heart_win_score_is_known = False
            else: # lost before act 3
                victory_score = - self.gamma ** (floor_reached - current_floor)
                heart_win_score = - self.gamma ** (floor_reached - current_floor)
                heart_win_score_is_known = True

        ret = {
            "token_idxes": idxes,
            "items_n": items_n,
            "n_upgrades": n_upgrades,
            "win_score": victory_score,
            "heart_win_score": heart_win_score,
            "heart_win_score_is_known": heart_win_score_is_known,
        }
        return ret
    
    def sample_unpreprocessed(self):
        assert not self.is_empty
        sample = random.choice(self.samples)
        sample["deck"] += sample["cards_picked"]
        return sample

    def __iter__(self):
        assert not self.is_empty
        return self.generator()

    @staticmethod
    def collate_fn(samples):
        max_seq_len = -1
        for sample in samples:
            seq_len = sample["items_n"]
            max_seq_len = max(max_seq_len, seq_len)
        assert max_seq_len > -1, max_seq_len

        for sample in samples:
            seq_len = sample["items_n"]
            n_pad_left = max_seq_len - seq_len
            padding = WinDataset.PAD_INDEX * np.ones(n_pad_left, dtype=int)
            sample["token_idxes"] = np.hstack([padding, sample["token_idxes"]])
            sample["n_upgrades"] = np.hstack([padding, sample["n_upgrades"]])
            sample["n_pad_lefts"] = n_pad_left
        
        ret = {}
        for key in [
            "token_idxes",
            "n_upgrades",
            "win_score",
            "heart_win_score",
            "heart_win_score_is_known",
        ]:
            ret[key] = collate.default_collate([sample[key] for sample in samples])
        
        for key in ["items_n", "n_pad_lefts"]:
            ret[key] = np.array([sample[key] for sample in samples])

        return ret

class WinModel(nn.Module):
    def load_model(training_dir : str, ckpt : int) -> "Model":
        params = json.load(open(os.path.join(training_dir, PARAMS_FILENAME), "r"))
        tokens = json.load(open(os.path.join(training_dir, TOKENS_FILENAME), "r"))
        model = WinModel(params=params, tokens=tokens)
        model.dataset = WinDataset(params=params, samples=None, tokens=tokens)
        if ckpt is not None:
            model.load_state_dict(torch.load(os.path.join(training_dir, f"{ckpt}.ckpt")))
        return model

    @staticmethod
    def create_model():
        from sts_ml.params import win_predictor_params
        data_tokens, train_dataloader, val_dataloader = WinModel.load_dataloaders(win_predictor_params)
        model = WinModel(win_predictor_params, tokens=data_tokens)
        return model, train_dataloader, val_dataloader

    @staticmethod
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
        train_val_split = int(split * len(full_dataset))

        train_dataset = WinDataset(
            params=params,
            samples=full_dataset[:train_val_split],
            tokens=data_tokens,
        )
        val_dataset = WinDataset(
            params=params,
            samples=full_dataset[train_val_split:],
            tokens=data_tokens,
        )
        print(f"Train ({train_val_split}) / validation ({len(full_dataset) - train_val_split}) ratio = {split}")

        return data_tokens, train_dataset, val_dataset

    @staticmethod
    def dataset_to_dataloader(params, dataset):
        batch_size = params["train"]["batch_size"]
        device = get_available_device()
        dataloader = iter(DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            collate_fn=WinDataset.collate_fn,
            pin_memory=True,
            pin_memory_device=device,
        ))
        return dataloader

    @staticmethod
    def load_dataloaders(params):

        data_tokens, train_dataset, val_dataset = WinModel.load_datasets(params)

        train_dataloader = WinModel.dataset_to_dataloader(params, train_dataset)
        val_dataloader = WinModel.dataset_to_dataloader(params, val_dataset)

        return data_tokens, train_dataloader, val_dataloader

    def __init__(self, params, tokens=None) -> None:
        super().__init__()
        
        self.params = params
        self.input_relics = params["model"].get("input_relics", False)
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
                mask_inter_choices=False,
            ) for _ in range(params["model"]["blocks"])]
        elif params["model"]["block_class"] == "Linear":
            blocks = [nn.Linear(dim, dim) for _ in range(params["model"]["blocks"])]
        else:
            assert False
        self.blocks = nn.ModuleList(blocks)
        self.pool = PoolTimeDimension(dim)
        self.projection = nn.Linear(dim, 2)

        self.device = device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(device)

        self.opt = torch.optim.Adam(self.parameters(), lr=params["train"]["lr"])

        self.just_predicted_one = False
    
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
        assert "cards_picked_n" not in batch

        idxes = batch["token_idxes"]
        bs, seqlen = idxes.shape
        n_pad_lefts = batch["n_pad_lefts"]
        items_n = batch["items_n"]
        n_upgrades = batch["n_upgrades"]

        feat = self.embedding(idxes)
        n_upgrades = torch.clip(n_upgrades, 0, 1) # TODO: we lose the nb of upgrades of searing blow
        for idx, n_upgrade in enumerate(n_upgrades):
            # scalar #1 of embedding encodes whether this is an upgraded card
            feat[idx,:,1] = n_upgrade
        
        kwargs = {}
        if self.params["model"]["block_class"] == "MHALayer":
            kwargs.update({
                "n_pad_lefts": n_pad_lefts,
                "n_choices": [0]*bs,
            })
        for block in self.blocks:
            feat = block.forward(feat, **kwargs)
        
        feat = self.pool(feat)

        feat = self.projection(feat)
        logits = torch.tanh(feat)

        return logits
    
    def predict_batch(self, batch):
        """
        samples -> logits, metrics
        """
        self.just_predicted_one = False

        logits = self.forward(batch)

        bs = batch["token_idxes"].shape[0]
        win_score_targets = torch.reshape(batch["win_score"], (bs, 1))
        heart_win_score_targets = torch.reshape(batch["heart_win_score"], (bs, 1))
        heart_win_score_is_known = torch.reshape(batch["heart_win_score_is_known"], (bs,)).float()
        targets = torch.concatenate([win_score_targets, heart_win_score_targets], dim=1)

        # assert (targets <= 1).all()
        # assert (-1 <= targets).all()
        
        delta = torch.abs(targets - logits)
        delta[:,1] *= heart_win_score_is_known

        losses = {}
        names = ["win", "heart_win"]
        for idx in range(2):
            l1_loss = delta[:,idx].mean()

            l2_loss = (delta[:,idx]**2).mean()

            pred_wins = logits[:,idx] > 0
            target_wins = targets[:,idx] > 0
            l0_loss = torch.abs(pred_wins != target_wins).float().mean()

            name = names[idx]
            losses.update({
                f"{name}_l0": l0_loss,
                f"{name}_l1": l1_loss,
                f"{name}_l2": l2_loss,
            })
        return logits, losses
    
    def learn(self, batch):
        self.just_predicted_one = False

        self.opt.zero_grad()
        logits, losses = self.predict_batch(batch)
        loss = losses["win_l2"] + losses["heart_win_l2"]
        loss.backward()
        self.opt.step()
        return losses
    
    def preprocess_one(self, sample: dict):
        preprocessed_sample = self.dataset.preprocess(sample)
        batch = WinDataset.collate_fn([preprocessed_sample])
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

        Usage:
        >>> sample = {"relics": ["deadbranch"], "deck": ["strike_r", "ascendersbane"]}
        
        Alternatively, to get winrate prediction for each offered card:
        >>> sample = {"relics": ["deadbranch"], "deck": ["strike_r", "ascendersbane"], "offered": ["boomerang", "twinstrike"]}
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
        true_win_score = batch.get("win_score", [None])[0]
        true_heart_win_score = batch.get("heart_win_score", [None])[0]

        logits = (logits[0] + 1) * 0.5 # range [-1,1] -> [0,1]
        pred_win_score = logits[0].item()
        pred_heart_win_score = logits[1].item()
        print(f"Win score: {pred_win_score} (true={true_win_score})")
        print(f"Heart win score: {pred_heart_win_score} (true={true_heart_win_score})")

        self.just_predicted_one = True
        self.last_sample = copy.deepcopy(sample)
    
    def get_attn_weights(self, block_idx: int, head_idx: int):
        assert self.just_predicted_one, "This function should be called right after a call to predict_one"
        assert self.params["model"]["block_class"] == "MHALayer", "This function is intended to investigate MHALayer layer activations"
        assert block_idx < len(self.blocks)
        nheads = self.blocks[0].nheads
        assert head_idx < nheads
        print(f"Attention weights for layer #{block_idx}/{len(self.blocks)-1}, head #{head_idx}/{nheads-1}.")
        print("This shows eg 'the output embedding (at row R) attends this much to input embeddings (at columns C)'")
        print("NOTE: our attention weights are such that an output embedding cannot attend to padding embeddings (which we effectively do not show in the table). cf our MHALayer implementation")

        sample = self.last_sample
        sample_idx = 0

        attn_weights_all = torch_to_numpy(self.blocks[block_idx].attn_weights)
        attn_weights = attn_weights_all[sample_idx, head_idx]

        tokens = sample["deck"]
        if self.input_relics:
            tokens = sample["relics"] + tokens
        n_tokens = len(tokens)
        assert len(attn_weights.shape) == 2, attn_weights_all.shape
        attn_weights = attn_weights[-n_tokens:, -n_tokens:]
        
        df = pd.DataFrame(attn_weights)
        df = df.rename(columns=lambda idx: f"{idx} {tokens[idx]}", index=lambda idx: f"{idx} {tokens[idx]}")
        
        styler = df
        styler = styler.style.background_gradient(cmap=cm)
        styler = styler.format(precision=3)

        return styler
