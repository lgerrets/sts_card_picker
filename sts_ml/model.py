import random
import json
import shutil
import os
import pandas as pd
import numpy as np
from typing import List
import copy

import torch, torch.nn as nn
from torch.distributions import Categorical
from torch.utils.data import IterableDataset, DataLoader
from torch.utils.data._utils import collate

from sts_ml.deck_history import ALL_CARDS_FORMATTED, ALL_RELICS_FORMATTED, card_to_name, card_to_n_upgrades
from sts_ml.helper import count_parameters, torch_to_numpy, numpy_to_torch, save_df, get_available_device
from sts_ml.model_abc import StsDataset, ModelAbc, MHALayer

class DeckDataset(StsDataset):

    def __init__(self, params: dict, is_train_set: bool = None, tokens: list = None):
        super().__init__(params, is_train_set, tokens)
        
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

    @classmethod
    def collate_fn(cls, samples):
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

class CardModel(ModelAbc):
    DatasetCls = DeckDataset
    
    @classmethod
    def create_model(cls):
        from sts_ml.params import card_predictor_params
        ret = super().create_model(card_predictor_params)
        return ret
    
    def __init__(self, params, tokens=None) -> None:
        super().__init__(params, tokens)
        
        dim = self.dim
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
        n_upgrades = torch.clip(n_upgrades, 0, 1) # TODO: we lose the nb of upgrades of searing blow
        assert isinstance(cards_picked_n, np.ndarray)
        for idx, (n_choice, n_upgrade) in enumerate(zip(cards_picked_n + cards_skipped_n, n_upgrades)):
            # scalar #0 of embedding encodes whether this is a choice token
            feat[idx,:-n_choice,0] = 0
            feat[idx,n_choice:,0] = 1
            # scalar #1 of embedding encodes whether this is an upgraded card
            feat[idx,:,1] = n_upgrade
        
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
        self.just_predicted_one = False

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
        self.just_predicted_one = False

        self.opt.zero_grad()
        logits, losses = self.predict_batch(batch)
        cross_ent_loss = losses["cross_ent_loss"]
        cross_ent_loss.backward()
        self.opt.step()
        return losses
    
    def predict_one(self, sample : dict):
        """
        batch of 1 sample -> print df and metrics

        Usage:
        >>> sample = {"relics": ["deadbranch"], "deck": ["strike_r", "ascendersbane"], "cards_picked": ["armaments+1"], "cards_skipped": ["impervious", "immmolate"]}
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
        self.just_predicted_one = True
        self.last_sample = copy.deepcopy(sample)
    
    def sample_to_tokens(self, sample):
        tokens = sample["deck"] + sample["cards_picked"] + sample["cards_skipped"]

        if self.input_relics:
            tokens = sample["relics"] + tokens
        
        return tokens

