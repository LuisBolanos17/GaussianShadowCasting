import torch
from torch.utils.data import Sampler

import numpy as np


class RayImageSampler(Sampler):

    def __init__(self, data_source, N_images=1024,
                 N_iter=None, generator=None):
        self.data_source = data_source
        self.N_images = N_images
        self._N_iter = N_iter
        self.generator = generator

        if self._N_iter is None:
            self._N_iter = len(self.data_source)

        self.sampler = RandIntGenerator(n=len(self.data_source))

    def __iter__(self):

        sampler_iter = iter(self.sampler)
        batch = []
        for i in range(self._N_iter):
            # get idx until we have N_images in batch
            while len(batch) < self.N_images:
                try:
                    idx = next(sampler_iter)
                except StopIteration:
                    sampler_iter = iter(self.sampler)
                    idx = next(sampler_iter)
                batch.append(idx.item())

            # return and clear batch cache
            yield np.sort(batch)
            batch = []

    def __len__(self):
        return self._N_iter


class RayAdjacentImageSampler(RayImageSampler):

    def __init__(self, *args, sample_range=500, **kwargs):
        '''
        sample_range: range to sample other images from
        '''
        super(RayAdjacentImageSampler, self).__init__(*args, **kwargs)
        assert len(self.data_source) > 5000, 'Adjacent sampler is only recommended when the number of training frames are huge'
        self.sample_range = sample_range
    
    def __iter__(self):

        idx = 0
        sampler_iter = iter(self.sampler)
        sample_range = self.sample_range
        for i in range(self._N_iter):

            # get the first idx
            try:
                #idx = next(sampler_iter).item()
                idx = (idx + 1) % len(self.data_source)
            except StopIteration:
                idx = (idx + 1) % len(self.data_source)
                #sampler_iter = iter(self.sampler)
                #idx = next(sampler_iter).item()
            batch = [idx]

            # compute the range that we sample from
            left = idx - sample_range
            right = idx + sample_range
            if left < 0:
                left = 0
                right += sample_range - idx
            if right >= len(self.data_source):
                left += right - len(self.data_source) 
                right = len(self.data_source)
            assert (right - left == sample_range * 2), f'{right-left} : {sample_range * 2}'

            # sample other images, avoid sampling img[idx] twice
            while True:
                adj_idxs = np.random.choice(np.arange(left, right), size=(self.N_images-1), replace=False)
                if idx not in adj_idxs:
                    break
            batch.extend(adj_idxs.tolist())

            # return and clear batch cache
            yield np.sort(batch)


class RandIntGenerator:
    '''
    RandomInt generator that ensures all n data will be
    sampled at least one in every n iteration.
    '''

    def __init__(self, n, generator=None):
        self._n = n
        self.generator = generator

    def __iter__(self):

        if self.generator is None:
            # TODO: this line is buggy for 1.7.0 ... but has to use this for 1.9?
            #       it induces large memory consumptions somehow
            generator = torch.Generator(device=torch.tensor(0.).device)
            #generator = torch.Generator()
            generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
        else:
            generator = self.generator

        yield from torch.randperm(self._n, generator=generator)

    def __len__(self):
        return self._n
