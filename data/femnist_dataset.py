import json
import os

import albumentations
import numpy as np
import torch

from collections import defaultdict
from pathlib import Path

from albumentations.pytorch import ToTensor
from torch.utils.data import Dataset
import pandas as pd


# taken from LEAF code
def read_dir(data_dir):
    clients = []
    groups = []
    data = defaultdict(lambda : None)

    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    for f in files:
        file_path = os.path.join(data_dir,f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        data.update(cdata['user_data'])

    clients = list(sorted(data.keys()))
    return clients, groups, data


class FEMNISTDataset(Dataset):

    def __init__(self, split, root_dir, args):
        self.root_dir = Path(root_dir) / 'femnist' / split
        clients, _, data = read_dir(self.root_dir)
        assert len(clients) == len(set(clients)), 'duplicate users'
        self.n_groups = len(clients)
        self.groups = list(range(self.n_groups))

        self.num_classes = 62

        self.image_shape = (1, 28, 28)

        agg_X, agg_y, agg_groups = [], [], []
        print("loading femnist")
        for i, client in enumerate(clients):
            client_X, client_y = data[client]['x'], data[client]['y']
            assert len(client_X) == len(client_y), 'malformed user data'
            client_N = len(client_X)
            X_processed = np.array(client_X).reshape((client_N, 28, 28, 1))
            X_processed = (1.0 - X_processed)
            agg_X.append(X_processed)
            agg_y.extend(client_y)
            agg_groups.extend([i] * client_N)
        print("loaded")
        self._len = len(agg_groups)
        self._X, self._y = np.concatenate(agg_X), np.array(agg_y)
        self.group_ids = np.array(agg_groups)

        self.transform = get_transform()

        if split == 'train' and 'learned_groups' in args and args.learned_groups:
            path = Path('output/clusterings/') / 'femnist' / args.clustering_filename
            #df_clusters = pd.read_csv(path)
            cluster_probs = np.load(path)
            self.group_ids = np.argmax(cluster_probs[:,0,:], axis=1)
            print("group ids: ", self.group_ids)
            self.n_groups = np.max(self.group_ids) + 1
            self.groups = list(range(self.n_groups))

        self.group_counts, _ = np.histogram(self.group_ids,
                                            bins=range(self.n_groups + 1),
                                            density=False)



        print("split: ", split)

        print("X sum: ", self._X.sum())

        print("n groups: ", len(clients))
        print("n groups: ", self.n_groups)
        print("Dataset size: ", len(self._y))

        print("Smallest group: ", np.min(self.group_counts))
        print("Largest group: ", np.max(self.group_counts))

    def __len__(self):
        return self._len

    def __getitem__(self, index):
        x = self.transform(**{'image': self._X[index]})['image']
        y = torch.tensor(self._y[index], dtype=torch.long)
        g = torch.tensor(self.group_ids[index], dtype=torch.long)
        return x, y, g

def get_transform():
    transform = albumentations.Compose([
        ToTensor(),
    ])
    return transform
