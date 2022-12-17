import json
import numpy as np
import torch, torch.nn as nn

from sts_ml.deck_history import ALL_CARDS_FORMATTED, card_to_name

def token_to_index(token : str):
    name = card_to_name(token)
    assert name in ALL_CARDS_FORMATTED, name
    index = ALL_CARDS_FORMATTED.index(name)
    return index

class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        self.dim = dim = 32
        self.embedding = nn.Embedding(len(ALL_CARDS_FORMATTED), dim)
        self.blocks = [Block(dim, 2*dim) for _ in range(2)]
        self.projection = nn.Linear(dim, 2)

        self.opt = torch.optim.Adam(self.parameters(), lr=3e-4)
        self.device = "cpu"
    
    def forward(self, idxes):
        feat = self.embedding(idxes)
        for block in self.blocks:
            feat = block(feat)
        feat = self.projection(feat)
        logits = torch.softmax(feat, dim=2)
        return logits
    
    def learn_on_sample(self, sample : dict):
        logits = self.infer(sample)
        deck_n = len(sample["deck"])
        picked_n = len(sample["cards_picked"])
        skipped_n = len(sample["cards_skipped"])
        loss = - torch.sum(logits[0, deck_n:deck_n+picked_n, 1]) - torch.sum(logits[0, deck_n+picked_n:deck_n+picked_n+skipped_n, 0])
        loss.backward()
        return loss
    
    def learn(self, dataset, batch_size):
        self.opt.zero_grad()
        total_loss = 0.
        for batch_i in range(batch_size):
            sample_idx = np.random.randint(len(dataset))
            sample = dataset[sample_idx]
            total_loss += self.learn_on_sample(sample).item()
        self.opt.step()
        return total_loss
    
    def infer(self, sample : dict):
        deck_idxes = [token_to_index(token) for token in sample["deck"]]
        picked_idxes = [token_to_index(token) for token in sample["cards_picked"]]
        skipped_idxes = [token_to_index(token) for token in sample["cards_skipped"]]
        idxes = deck_idxes + picked_idxes + skipped_idxes
        idxes = np.array(idxes)[np.newaxis, :]
        idxes = torch.from_numpy(idxes).to(self.device)
        logits = self.forward(idxes)
        return logits
    
    def predict(self, sample : dict):
        logits = self.infer(sample)
        deck_n = len(sample["deck"])
        picked_n = len(sample["cards_picked"])
        skipped_n = len(sample["cards_skipped"])
        idx = deck_n
        print(sample["cards_picked"])
        while idx < deck_n + picked_n + skipped_n:
            if idx == deck_n + picked_n:
                print(sample["cards_skipped"])
            print(logits[0,idx,:].argmax().item())
            idx += 1

class Block(nn.Module):
    def __init__(self, dim, ffdim) -> None:
        super().__init__()
        self.dim = dim
        self.att = nn.MultiheadAttention(dim, 2)
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

def main():
    dataset = json.load(open("./draft_dataset.data", "r"))

    model = Model()

    for epoch in range(1000):
        total_loss = model.learn(dataset, 32)
        print(epoch, total_loss)
    
    model.predict(dataset[0])

if __name__ == "__main__":
    main()
