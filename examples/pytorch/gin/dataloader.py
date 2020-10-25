"""
PyTorch compatible dataloader
"""


import math
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
import dgl


# default collate function


def collate(samples):
    # The input `samples` is a list of pairs (graph, label).
    graphs, labels = map(list, zip(*samples))
    for g in graphs:
        # deal with node feats
        for key in g.node_attr_schemes().keys():
            g.ndata[key] = g.ndata[key].float()
        # no edge feats
    batched_graph = dgl.batch(graphs)
    labels = torch.tensor(labels)
    return batched_graph, labels

class GraphDataLoader():
    def __init__(self,
                 dataset,
                 batch_size,
                 device,
                 collate_fn=collate,
                 seed=0,
                 shuffle=True,
                 save_embeddings=False,
                 split_name='fold10',
                 fold_idx=0,
                 split_ratio=0.7):

        self.shuffle = shuffle
        self.seed = seed
        self.kwargs = {'pin_memory': True} if 'cuda' in device.type else {}

        labels = [l for _, l in dataset]

        # if split_name == 'fold10':
        #     train_idx, valid_idx = self._split_fold10(
        #         labels, fold_idx, seed, shuffle)
        # elif split_name == 'rand':
        #     train_idx, valid_idx = self._split_rand(
        #         labels, split_ratio, seed, shuffle)
        # else:
        #     raise NotImplementedError()

        if save_embeddings:
            sampler = None
        else:
            sampler = self.weightedRandomSampler(labels)

        self.loader = DataLoader(
            dataset, sampler=sampler,
            batch_size=batch_size, collate_fn=collate_fn, **self.kwargs)


    def train_valid_loader(self):
        return self.loader

    def _split_fold10(self, labels, fold_idx=0, seed=0, shuffle=True):
        ''' 10 flod '''
        assert 0 <= fold_idx and fold_idx < 10, print(
            "fold_idx must be from 0 to 9.")

        skf = StratifiedKFold(n_splits=10, shuffle=shuffle, random_state=seed)
        idx_list = []
        for idx in skf.split(np.zeros(len(labels)), labels):    # split(x, y)
            idx_list.append(idx)
        train_idx, valid_idx = idx_list[fold_idx]

        print(
            "train_set : test_set = %d : %d",
            len(train_idx), len(valid_idx))

        return train_idx, valid_idx

    def weightedRandomSampler(self, labels):
        #Class Weighting
        labels_unique, counts = np.unique(labels, return_counts=True)
        print('Unique labels: {}'.format(labels_unique))

        class_weights = np.zeros(np.max(labels_unique) + 1, dtype=float)

        for k,c in enumerate(counts):
            class_weights[labels_unique[k]] = sum(counts) / c
        # Assign weight to each input sample
        example_weights = [class_weights[e] for e in labels]
        sampler = WeightedRandomSampler(example_weights, len(labels))
        return sampler


    def _split_rand(self, labels, split_ratio=0.7, seed=0, shuffle=True):
        num_entries = len(labels)
        indices = list(range(num_entries))
        np.random.seed(seed)
        np.random.shuffle(indices)
        split = int(math.floor(split_ratio * num_entries))
        train_idx, valid_idx = indices[:split], indices[split:]

        print(
            "train_set : test_set = %d : %d",
            len(train_idx), len(valid_idx))

        return train_idx, valid_idx

