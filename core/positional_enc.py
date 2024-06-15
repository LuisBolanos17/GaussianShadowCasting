import torch
import torch.nn as nn

from typing import Optional, List, Callable

# Positional encoding
class PositionalEncoding(nn.Module):
    def __init__(
        self, 
        input_dims: int , 
        include_input: bool = True,  
        num_freqs: int = 8,
        log_sampling: bool = True,
        periodic_fns: List[Callable] = [torch.sin, torch.cos],
        **kwargs
    ):
        super(PositionalEncoding, self).__init__()
        self.input_dims = input_dims
        self.include_input = include_input
        self.kwargs = kwargs
        self.num_freqs = num_freqs
        self.log_sampling = log_sampling
        self.periodic_fns = periodic_fns
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        fn_names = []
        fn_scales = []
        freq_pows = []
        d = self.input_dims

        out_dim = 0
        if self.include_input:
            #embed_fns.append(lambda x, **kwargs: x)
            #freq_pows.append(-1)
            fn_scales.append(0.)
            out_dim += d

        max_freq = self.num_freqs - 1
        N_freqs = self.num_freqs

        if self.log_sampling:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)

        for freq in freq_bands:
            freq_pow = torch.log2(freq)
            for p_fn in self.periodic_fns:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq.item()))
                fn_names.append(p_fn.__name__)
                freq_pows.append(freq_pow)
                fn_scales.append(freq.item())

                out_dim += d

        self.freq_bands = freq_bands
        self.embed_fns = embed_fns
        self.fn_names = fn_names
        self.out_dims = out_dim
        self.freq_k = torch.tensor(freq_pows).reshape(1, 1, -1, 1)
        self.fn_scales = fn_scales
    
    @property
    def dims(self):
        return self.out_dims

    def forward(self, inputs: torch.Tensor, weights: Optional[torch.Tensor] = None, **kwargs):
        return self._embed(inputs, weights=weights, **kwargs)

    def _embed(self, inputs: torch.Tensor, weights: Optional[torch.Tensor] = None, **kwargs):
        if self.num_freqs == 0:
            assert self.include_input
            return inputs, None
        
        #embedded_ =  torch.cat([fn(inputs) for fn in self.embed_fns], -1) 
        inputs_expand = inputs[..., None, :] 
        n_dims = inputs_expand.dim()
        freq_bands = self.freq_bands.to(inputs.device).reshape((1,) * (n_dims -2) + (-1, 1))
        inputs_bands = inputs_expand * freq_bands
        sin_component = torch.sin(inputs_bands)
        cos_component = torch.cos(inputs_bands)
        embedded = torch.stack([sin_component, cos_component], dim=-2).flatten(start_dim=-3)
        #assert torch.allclose(embedded_, embedded, atol=1e-6)
    
        if weights is not None:
            embedded = embedded * weights
        if self.include_input:
            embedded = torch.cat([inputs, embedded], -1)
        return embedded, None

    def update_threshold(self, *args, **kwargs):
        pass

    def update_tau(self, *args, **kwargs):
        pass

    def get_tau(self):
        return 0.0


class CutoffPositionalEncoding(PositionalEncoding):

    def __init__(
        self, 
        cutoff_dist: float = 0.5,
        dist_inputs: bool = False,
        cutoff_inputs: bool = True, 
        cutoff_dim: int = 24, 
        cut_to_cutoff: bool = False, 
        shift_inputs: bool = False, 
        **kwargs
    ):
        """
        cutoff_inputs: apply cutoff to the none-encoded input as well
        dist_inputs: whether to use an extra 'dist' input to calculate the cutoff
        """

        super().__init__(**kwargs)
        self.dist_inputs = dist_inputs
        self.cutoff_inputs = cutoff_inputs
        self.cut_to_cutoff = cut_to_cutoff
        self.shift_inputs = shift_inputs

        self.cutoff_dim = cutoff_dim

        self.freq_bands = self.freq_bands.view(1, -1, 1).expand(-1, -1, cutoff_dim)
        self.freq_k = torch.log2(self.freq_bands)[..., :len(self.periodic_fns)]

        #self.cutoff_dist = nn.Parameter(torch.ones(cutoff_dim) * self.cutoff_dist)
        self.register_buffer('cutoff_dist', torch.ones(cutoff_dim) * cutoff_dist)

        self.init_tau = 20.
        self.register_buffer('tau', torch.tensor(self.init_tau))

    def get_tau(self):
        return self.tau.item()

    def get_cutoff_dist(self):
        return self.cutoff_dist

    def _embed(self, inputs: torch.Tensor, dists: Optional[torch.Tensor] = None, masks: Optional[torch.Tensor] = None, **kwargs):

        if self.dist_inputs:
            # assume that we have 1-to-1 correspondence between inputs and dists!
            input_size, dist_size = inputs.size(-1), dists.size(-1)
            if input_size >= dist_size:
                expand = input_size // dist_size
                dists = dists[..., None].expand(*dists.shape, expand).flatten(start_dim=-2)
                inputs_freq = self.freq_bands[..., None].expand(-1, -1, -1, expand).flatten(start_dim=-2).to(inputs.device) * inputs[..., None, :]
            else:
                raise NotImplementedError("This should not occur! check why embedding is broken")
        else:
            dists = inputs
            if self.cut_to_cutoff:
                inputs = self.get_cutoff_dist() - inputs
            if self.shift_inputs:
                cutoff_dist = self.get_cutoff_dist()
                # so that the frequencies, after applying cutoff, span [-1, 1]
                shifted = inputs * (2. / cutoff_dist ) - 1.
                #shifted = inputs * 2. - 0.5
                inputs_freq = self.freq_bands.to(inputs.device) * shifted[..., None, :]
            else:
                inputs_freq = self.freq_bands.to(inputs.device) * inputs[..., None, :]

        # compute cutoff weights
        cutoff_dist = self.get_cutoff_dist()
        if not self.dist_inputs:
            v = self.tau * (dists - cutoff_dist)
        else:
            v = self.tau * (dists - cutoff_dist[:, None].expand(-1, expand).flatten(start_dim=-2))
        v = v[..., None, :]
        w = 1. - torch.sigmoid(v)

        # (B, NF, NJ): shaping like the old version for backward consistency
        # stack the sin/cosine encoding and apply weights
        embedded = torch.stack([torch.sin(inputs_freq),
                                torch.cos(inputs_freq)], dim=-2).flatten(start_dim=-3, end_dim=-2)

        if self.include_input and self.cutoff_inputs:
            embedded = torch.cat([inputs[..., None, :], embedded], dim=-2)
            embedded = (embedded * w)
        elif self.include_input:
            embedded = (embedded * w)
            embedded = torch.cat([inputs[..., None, :], embedded], dim=-2)
        else:
            embedded = (embedded * w)
        embedded = embedded.flatten(start_dim=-2)
        return embedded, w

    def update_threshold(
        self, 
        global_step: int, 
        tau_step: int, 
        tau_rate: float,
        alpha_step: int, 
        alpha_target: float,
    ):
        self.update_tau(global_step, tau_step, tau_rate)
        self.update_alpha(global_step, alpha_step, alpha_target)

    def update_tau(self, global_step: int, step: int, rate: float):
        # TODO: makes the starting value not fixed!
        self.tau =  (self.init_tau * torch.ones_like(self.tau) * rate**(global_step / float(step * 1000))).clamp(max=2000.)
