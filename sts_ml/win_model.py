import pandas as pd
import numpy as np
import copy
import json
import os, os.path
import random

import torch, torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader
from torch.utils.data._utils import collate

from sts_ml.deck_history import ALL_CARDS_FORMATTED, ALL_RELICS_FORMATTED, card_to_name, card_to_n_upgrades
from sts_ml.model import MHALayer
from sts_ml.helper import count_parameters, torch_to_numpy, numpy_to_torch, save_df, get_available_device
from sts_ml.model_abc import StsDataset, ModelAbc, MHALayer, PARAMS_FILENAME, TOKENS_FILENAME, PAD_TOKEN, PoolTimeDimension, PositionalEncoding

class WinDataset(StsDataset):

    def __init__(self, params: dict, is_train_set: bool = None, tokens: list = None):
        super().__init__(params, is_train_set, tokens)
        self.gamma = params["model"]["gamma"]
        
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
            "floor": current_floor,
            "win_score": victory_score,
            "heart_win_score": heart_win_score,
            "heart_win_score_is_known": heart_win_score_is_known,
        }
        return ret
    
    def sample_unpreprocessed(self):
        sample = super().sample_unpreprocessed()
        sample["deck"] += sample["cards_picked"]
        return sample

    @classmethod
    def collate_fn(cls, samples):
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
        
        non_tensor_keys = ["items_n", "n_pad_lefts"]
        for key in sample:
            if key in non_tensor_keys:
                ret[key] = np.array([sample[key] for sample in samples])
            else:
                ret[key] = collate.default_collate([sample[key] for sample in samples])

        return ret

class WinModel(ModelAbc):
    DatasetCls = WinDataset

    @classmethod
    def create_model(cls):
        from sts_ml.params import win_predictor_params
        ret = super().create_model(win_predictor_params)
        return ret

    def __init__(self, params, tokens=None) -> None:
        super().__init__(params, tokens)        

        dim = self.dim
        
        n_custom_embeddings = 1
        self.embedding = nn.Embedding(len(tokens), dim-n_custom_embeddings)
        
        self.floor_encoding = PositionalEncoding(self.dim, do_phases=False) # contrarily to how PE is used in transformers (eg encode the time dimension), we use it to encode the floor, and we use it in a different way; moreover, this could be argued, but here we disable phases because the original authors did it for a specific reason that we are not really concerned with (eg help transformer models understand that a delta of positions of 1 is very different than 2 in text sentences; in our case, deck winrates at floors x and x+1 are pretty similar)

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

        # scalar #0 of embedding encodes whether this is an upgraded card
        n_upgrades = torch.clip(n_upgrades, 0, 1) # TODO: we lose the nb of upgrades of searing blow
        assert n_upgrades.shape == (bs, seqlen)
        n_upgrades = torch.reshape(n_upgrades, (bs, seqlen, 1))
        feat = torch.concat([n_upgrades, feat], dim=2)

        floor_encodings = self.floor_encoding.forward(batch["floor"])
        floor_encodings = torch.reshape(floor_encodings, (bs, 1, self.dim))
        feat += floor_encodings # add rather than concatenate on the feature dimension, as they do in transformers; also we add the same floor_embedding to every embedding accross seqlen (I argue that it would be semantically wrong to make it its own token ie concatenate on the seqlen dimension, but TODO experiment)
        
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
    
    def sample_to_tokens(self, sample):
        tokens = sample["deck"]
        return tokens
