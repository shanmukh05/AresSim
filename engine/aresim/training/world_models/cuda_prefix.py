"""Replay fixed-shape recurrent history on CUDA without per-frame Python launches."""
from __future__ import annotations

import torch
from ...algorithms.dreamerv3.distributions import sample_latent


class CUDAPrefixReplay:
    """Capture inference only; parameters keep their live optimizer-owned storage.

    Padded frames retain the last active state. They consume unused random draws,
    so this backend preserves distributions rather than bitwise RNG trajectories.
    The returned buffers remain valid until the next learner update.
    """

    def __init__(self, model, horizon: int) -> None:
        self.model = model
        self.horizon = horizon
        self.graph = None

    @torch.no_grad()
    def __call__(self, embeddings, actions, lengths):
        batch, steps, width = embeddings.shape
        if steps > self.horizon:
            raise ValueError("replay history exceeds the configured episode limit")
        if self.graph is None:
            self.embeddings = embeddings.new_zeros(batch, self.horizon, width)
            self.actions = actions.new_zeros(batch, self.horizon, actions.shape[-1])
            self.lengths = lengths.clone()
            stream = torch.cuda.Stream(device=embeddings.device)
            stream.wait_stream(torch.cuda.current_stream(embeddings.device))
            with torch.cuda.stream(stream):
                for _ in range(2):
                    self._unroll()
            torch.cuda.current_stream(embeddings.device).wait_stream(stream)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.output = self._unroll()
        elif batch != self.embeddings.shape[0]:
            raise ValueError("CUDA history replay requires a fixed learner batch size")
        self.embeddings.zero_()
        self.actions.zero_()
        self.embeddings[:, :steps].copy_(embeddings)
        self.actions[:, :steps].copy_(actions)
        self.lengths.copy_(lengths)
        self.graph.replay()
        return self.output

    def _unroll(self):
        state = self.model.rssm.initial(len(self.lengths), self.embeddings.device)
        for index in range(self.horizon):
            hidden, posterior = self.model.rssm.posterior_step(
                state, self.actions[:, index], self.embeddings[:, index])
            advanced = hidden, sample_latent(posterior, self.model.config.unimix)
            active = self.lengths > index
            state = tuple(torch.where(active.reshape(-1, *([1] * (new.ndim - 1))), new, old)
                          for new, old in zip(advanced, state, strict=True))
        return state
