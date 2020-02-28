from torch.utils.data import Sampler
import torch


class UniformRandomSampler(Sampler):
    def __init__(self, data_source, num_samples: int = None):
        self.data_source = data_source
        self._num_samples = num_samples
        self._state = torch.randperm(len(data_source)).tolist()

    @property
    def num_samples(self):
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self):
        n = len(self.data_source)
        if len(self._state) < self.num_samples + 1:
            self._state += torch.randperm(n).tolist()
        output, self._state = (
            self._state[: self.num_samples],
            self._state[self.num_samples :],
        )
        return iter(output)

    def __len__(self):
        return self.num_samples
