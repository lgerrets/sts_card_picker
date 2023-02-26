import random
import json
import shutil
import os
import pandas as pd
import numpy as np
from typing import List
import copy
import seaborn as sns
cm = sns.light_palette("green", as_cmap=True)

import torch, torch.nn as nn
from torch.distributions import Categorical
from torch.utils.data import IterableDataset, DataLoader
from torch.utils.data._utils import collate

from sts_ml.deck_history import ALL_CARDS_FORMATTED, ALL_RELICS_FORMATTED, card_to_name, card_to_n_upgrades, PAD_TOKEN, detokenize
from sts_ml.helper import count_parameters, torch_to_numpy, numpy_to_torch, save_df, get_available_device
from sts_ml.params import NUM_DATALOADER_WORKERS

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

class StsDataset(IterableDataset):
    PAD_INDEX = 0

    def __init__(self, params: dict, is_train_set: bool = None, tokens: list = None):
        self.is_empty = tokens is None
        self.filepaths = params["train"]["dataset"]
        self.is_train_set = is_train_set
        self.tokens = tokens
        self.params = params
        self.do_relics = params["model"]["input_relics"]
        self.train_test_split = params["train"]["split"]
        
        self.count_down = 0
        self.intra_permutation = []
        self.inter_permutation = []
        assert tokens.index(PAD_TOKEN) == StsDataset.PAD_INDEX

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

    def sample_unpreprocessed(self):
        assert not self.is_empty
        if self.params["train"]["data_sampling"] == "round":
            if len(self.intra_permutation) == 0:
                if len(self.inter_permutation) == 0:
                    self.inter_permutation = np.random.permutation(len(self.filepaths)).tolist()
                data = json.load(open(self.filepaths[self.inter_permutation.pop(0)], "r"))
                self.samples = data["dataset"]
                train_size = int(len(self.samples) * self.train_test_split)
                val_size = len(self.samples) - train_size
                if self.is_train_set:
                    self.intra_permutation = np.random.permutation(train_size)
                else:
                    self.intra_permutation = np.random.permutation(len(self.samples) - train_size) + train_size
                self.intra_permutation = self.intra_permutation.tolist()
                print(f"[Partial dataset] Train ({train_size}) / validation ({val_size}) ratio")
            sample = self.samples[self.intra_permutation.pop(0)]
        elif self.params["train"]["data_sampling"] == "uniform":
            if self.count_down == 0:
                data = json.load(open(random.choice(self.filepaths), "r"))
                self.samples = data["dataset"]
                train_size = int(len(self.samples) * self.train_test_split)
                val_size = len(self.samples) - train_size
                if self.is_train_set:
                    self.samples = self.samples[:train_size]
                else:
                    self.samples = self.samples[train_size:]
                self.count_down = len(self.samples)
                print(f"[Partial dataset] Train ({train_size}) / validation ({val_size}) ratio")
            sample = random.choice(self.samples)
            self.count_down -= 1
        else:
            raise NotImplementedError
        sample = detokenize(sample, self.tokens)
        return sample

    def __iter__(self):
        assert not self.is_empty
        return self.generator()

class ModelAbc(nn.Module):
    DatasetCls = None

    @classmethod
    def load_model(cls, training_dir : str, ckpt : int) -> "ModelAbc":
        params = json.load(open(os.path.join(training_dir, PARAMS_FILENAME), "r"))
        tokens = json.load(open(os.path.join(training_dir, TOKENS_FILENAME), "r"))
        model = cls(params=params, tokens=tokens)
        model.dataset = cls.DatasetCls(params=params, tokens=tokens)
        if ckpt is not None:
            model.load_state_dict(torch.load(os.path.join(training_dir, f"{ckpt}.ckpt")))
        return model

    @classmethod
    def create_model(cls, params: dict):
        data_tokens, train_dataloader, val_dataloader = cls.load_dataloaders(params)
        model = cls(params, tokens=data_tokens)
        return model, train_dataloader, val_dataloader

    @classmethod
    def load_datasets(cls, params):
        data = json.load(open(params["train"]["dataset"][0], "r"))
        data_tokens = data["items"]
        
        if PAD_TOKEN not in data_tokens:
            data_tokens = [PAD_TOKEN] + data_tokens

        assert set(data_tokens).issubset(set(data_tokens) | set([PAD_TOKEN])), set.symmetric_difference(set(data_tokens), set(data_tokens) | set([PAD_TOKEN]))

        train_dataset = cls.DatasetCls(
            params=params,
            tokens=data_tokens,
            is_train_set=True,
        )
        val_dataset = cls.DatasetCls(
            params=params,
            tokens=data_tokens,
            is_train_set=False,
        )

        return data_tokens, train_dataset, val_dataset

    @classmethod
    def dataset_to_dataloader(cls, params: dict, dataset: StsDataset):
        batch_size = params["train"]["batch_size"]
        device = get_available_device()
        dataloader = iter(DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            collate_fn=cls.DatasetCls.collate_fn,
            pin_memory=True,
            pin_memory_device=device,
            num_workers=NUM_DATALOADER_WORKERS if dataset.is_train_set else 0,
        ))
        return dataloader

    @classmethod
    def load_dataloaders(cls, params):

        data_tokens, train_dataset, val_dataset = cls.load_datasets(params)

        train_dataloader = cls.dataset_to_dataloader(params, train_dataset)
        val_dataloader = cls.dataset_to_dataloader(params, val_dataset)

        return data_tokens, train_dataloader, val_dataloader

    def __init__(self, params, tokens=None) -> None:
        super().__init__()

        self.params = params
        self.input_relics = params["model"].get("input_relics", False)
        self.dim = dim = params["model"]["dim"]
        assert PAD_TOKEN in tokens
        self.tokens = tokens
        self.pad_idx = tokens.index(PAD_TOKEN)
        self.just_predicted_one = False

    def token_to_index(self, token : str):
        name = card_to_name(token)
        assert name in self.tokens, name
        index = self.tokens.index(name)
        return index

    def preprocess_one(self, sample: dict):
        preprocessed_sample = self.dataset.preprocess(sample)
        batch = self.DeckDataset.collate_fn([preprocessed_sample])
        # batch = self._to_device(batch)
        return batch

    def _to_device(self, batch: dict):
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(self.device)
        return batch

    def sample_to_tokens(self, sample: dict):
        raise NotImplementedError("Abstract")
    
    def get_attn_weights(self, block_idx: int, head_idx: int):
        assert self.just_predicted_one, "This function should be called right after a call to predict_one"
        assert self.params["model"]["block_class"] == "MHALayer", "This function is intended to investigate MHALayer layer activations"
        assert block_idx < len(self.blocks)
        nheads = self.blocks[0].nheads
        assert head_idx < nheads

        sample = self.last_sample
        sample_idx = 0

        attn_weights_all = torch_to_numpy(self.blocks[block_idx].attn_weights)
        attn_weights = attn_weights_all[sample_idx, head_idx]

        print(f"Attention weights for layer #{block_idx}/{len(self.blocks)-1}, head #{head_idx}/{nheads-1}.")
        print("This shows eg 'the output embedding (at row R) attends this much to input embeddings (at columns C)'")
        
        tokens = self.sample_to_tokens(sample)

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
        
        mha_feat, attn_weights = self.att(feat, feat, feat, need_weights=True, average_attn_weights=False, attn_mask=mask)
        self.attn_weights = attn_weights

        feat = feat + mha_feat
        feat = self.ln1(feat)
        ff_feat = self.ff2(self.relu(self.ff1(feat)))
        feat = feat + ff_feat
        feat = self.ln2(feat)
        return feat

class PoolTimeDimension(nn.Module):
    def __init__(self, dim, hidden_dim=16) -> None:
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.query_nn = nn.Linear(dim, self.hidden_dim)
        self.key_nn = nn.Linear(dim, self.hidden_dim)
        self.output_nn = nn.Linear(self.hidden_dim*self.hidden_dim, dim)
    
    def forward(self, feat):
        bs, seqlen, dim = feat.shape
        assert self.dim == dim

        query = self.query_nn(feat) # bs, seqlen, hidden_dim
        key = self.key_nn(feat) # bs, seqlen, hidden_dim

        query = torch.transpose(query, 1, 2) # -> bs, hidden_dim, seqlen
        dots = torch.matmul(query, key) # -> bs, hidden_dim, hidden_dim
        assert dots.shape == (bs, self.hidden_dim, self.hidden_dim), (dots.shape, bs, self.hidden_dim)

        dots = dots.reshape(bs, self.hidden_dim*self.hidden_dim)
        out = self.output_nn(dots) # -> bs, dim

        return out

class PositionalEncoding(nn.Module):
    """
    cf https://kazemnejad.com/blog/transformer_architecture_positional_encoding/
    """
    def __init__(self, dim: int, do_phases=True):
        super().__init__()

        self.dim = dim

        # frequencies
        exponents = torch.repeat_interleave(torch.arange(0, self.dim, 2) / self.dim, 2)[:self.dim]
        self.w_k = torch.pow(1 / 10000, exponents)
        self.w_k = self.w_k.reshape((1,-1))
        self.w_k = nn.Parameter(self.w_k, requires_grad=False)

        # phases
        self.do_phases = do_phases
        if do_phases:
            self.phi_k = (torch.arange(2) * np.pi/2).repeat(int((self.dim + 1) / 2))[:self.dim]
            self.phi_k = self.phi_k.reshape((1,-1))
            self.phi_k = nn.Parameter(self.phi_k, requires_grad=False)
    
    def forward(self, positions: torch.Tensor):
        positions = positions.reshape((-1,1))
        xs = self.w_k * positions
        if self.do_phases:
            xs += self.phi_k
        return torch.sin(xs)
