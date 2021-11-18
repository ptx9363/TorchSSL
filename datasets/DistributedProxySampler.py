# copyright: https://github.com/pytorch/pytorch/issues/23430#issuecomment-562350407

import math
from typing import Optional
import torch
from torch.utils.data.distributed import DistributedSampler


class DistributedProxySampler(DistributedSampler):
    """Sampler that restricts data loading to a subset of input sampler indices.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.

    .. note::
        Input sampler is assumed to be of constant size.

    Arguments:
        sampler: Input data sampler.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self, dataset, num_replicas=None, rank=None):
        super().__init__(dataset,
                         num_replicas=num_replicas,
                         rank=rank,
                         shuffle=False)

        dataset_size = max(len(self.dataset), 1024)  # minimal 1024 samples, reduce frequence of the Iterator reset 
        # dataset_size = len(self.dataset)
        
        self.num_samples = math.ceil(dataset_size / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        torch.manual_seed(self.epoch)
        n = len(self.dataset)

        if self.total_size > n:
            indices = list(
                iter(
                    torch.randint(high=n,
                                  size=(self.total_size,),
                                  dtype=torch.int64).tolist()))
        else:
            indices = list(iter(torch.randperm(n).tolist()))[:self.total_size]

        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)