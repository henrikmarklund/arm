import torch
from pathlib import Path

class Saver:

    def __init__(self, algorithm, device, ckpt_dir):

        self.algorithm = algorithm
        self.device = device
        self.ckpt_dir = Path(ckpt_dir)

    def save(self, epoch, is_best):
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = self.ckpt_dir / f'{epoch}.pkl'
        torch.save(self.algorithm.to('cpu'), ckpt_path)

        if is_best:
            ckpt_path = self.ckpt_dir / f'best.pkl'
            torch.save(self.algorithm, ckpt_path)

        self.algorithm.to(self.device)

