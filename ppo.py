# rl/ppo.py
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch as th
import torch.nn as nn
from torch.distributions.normal import Normal


# ---------------------------
# Utility: simple MLP builder
# ---------------------------
def mlp(in_dim: int, hidden: Iterable[int], out_dim: int, act=nn.Tanh) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = in_dim
    for h in hidden:
        layers += [nn.Linear(last, h), act()]
        last = h
    layers += [nn.Linear(last, out_dim)]
    return nn.Sequential(*layers)


# ---------------------------
# Tensor helper
# ---------------------------
def _to_tensor(x: np.ndarray | th.Tensor) -> th.Tensor:
    """Accept numpy or tensor; return 2D float32 tensor [B, D]."""
    if isinstance(x, th.Tensor):
        t = x
    elif isinstance(x, np.ndarray):
        t = th.from_numpy(x).float()
    else:
        t = th.tensor(x, dtype=th.float32)
    if t.dim() == 1:
        t = t.unsqueeze(0)
    return t


# ---------------------------
# PPO Agent
# ---------------------------
class PPOAgent(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: Iterable[int] = (128, 128),
        lr: float = 3e-4,
        init_log_std: float = -0.5,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)

        self.pi = mlp(self.obs_dim, hidden_sizes, self.act_dim, act=nn.Tanh)
        self.v = mlp(self.obs_dim, hidden_sizes, 1, act=nn.Tanh)

        # state-independent log std
        self.log_std = nn.Parameter(th.ones(self.act_dim) * float(init_log_std))

        self.optimizer = th.optim.Adam(self.parameters(), lr=lr)

    # ---- policy sampling for interaction ----
    def select_action(self, obs: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        Returns:
            action (np.float32[act_dim]),
            logp (float),
            value (float)
        """
        obs_t = _to_tensor(obs)
        mu = self.pi(obs_t)
        std = th.exp(self.log_std).expand_as(mu)
        dist = Normal(mu, std)

        a_t = dist.sample()  # [B, act_dim]
        logp = dist.log_prob(a_t).sum(-1)  # [B]
        v = self.v(obs_t).squeeze(-1)  # [B]

        return (
            a_t.squeeze(0).detach().cpu().numpy().astype(np.float32),
            float(logp.detach().cpu().numpy()[0]),
            float(v.detach().cpu().numpy()[0]),
        )

    # ---- evaluate given (obs, act) ----
    def evaluate(self, obs, act) -> Tuple[float, float, float]:
        """
        Compute log-prob, entropy and value for given (obs, act).
        Returns scalar floats for convenience.
        """
        obs_t = _to_tensor(obs)
        act_t = _to_tensor(act)

        mu = self.pi(obs_t)
        std = th.exp(self.log_std).expand_as(mu)
        dist = Normal(mu, std)

        logp = dist.log_prob(act_t).sum(-1)  # [B]
        ent = dist.entropy().sum(-1)  # [B]
        v = self.v(obs_t).squeeze(-1)  # [B]

        return (
            float(logp.detach().cpu().numpy()[0]),
            float(ent.mean().detach().cpu().numpy()),
            float(v.detach().cpu().numpy()[0]),
        )


# ---------------------------
# Rollout Buffer with GAE(λ)
# ---------------------------
@dataclass
class _Ptr:
    idx: int = 0
    path_start: int = 0


class PPOBuffer:
    """
    Stores trajectories and computes GAE(λ) advantages.
    Expected shapes:
      obs:  [T, obs_dim]
      acts: [T, act_dim]
      rew, val, logp: [T]
    """

    def __init__(self, obs_dim: int, act_dim: int, size: int):
        self.obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.acts = np.zeros((size, act_dim), dtype=np.float32)
        self.rew = np.zeros(size, dtype=np.float32)
        self.val = np.zeros(size, dtype=np.float32)
        self.logp = np.zeros(size, dtype=np.float32)

        self.adv = np.zeros(size, dtype=np.float32)
        self.ret = np.zeros(size, dtype=np.float32)

        self._ptr = _Ptr()
        self._max = size

    def store(self, obs, act, rew, val, logp):
        i = self._ptr.idx
        assert i < self._max, "PPOBuffer overflow"
        self.obs[i] = np.asarray(obs, dtype=np.float32)
        self.acts[i] = np.asarray(act, dtype=np.float32)
        self.rew[i] = float(rew)
        self.val[i] = float(val)
        self.logp[i] = float(logp)
        self._ptr.idx += 1

    def finish_path(self, last_val: float, gamma: float, lam: float):
        """
        Compute advantage and returns for the trajectory fragment
        from path_start to current ptr (exclusive).
        """
        ps, pe = self._ptr.path_start, self._ptr.idx
        rews = np.append(self.rew[ps:pe], last_val).astype(np.float32)
        vals = np.append(self.val[ps:pe], last_val).astype(np.float32)

        deltas = rews[:-1] + gamma * vals[1:] - vals[:-1]
        gae = 0.0
        for t in reversed(range(pe - ps)):
            gae = deltas[t] + gamma * lam * gae
            self.adv[ps + t] = gae
        self.ret[ps:pe] = self.adv[ps:pe] + self.val[ps:pe]

        self._ptr.path_start = self._ptr.idx

    def get(self):
        assert self._ptr.idx == self._max, "Buffer not full; cannot get() yet"
        data = dict(
            obs=self.obs,
            acts=self.acts,
            adv=self.adv,
            ret=self.ret,
            logp=self.logp,
        )
        # reset pointer for next epoch
        self._ptr = _Ptr()
        return data


# ---------------------------
# PPO update (minibatch)
# ---------------------------
def ppo_update(
    agent: PPOAgent,
    buf: PPOBuffer,
    clip_ratio: float = 0.2,
    update_epochs: int = 4,
    minibatch_size: int = 1024,
    value_coef: float = 0.5,
    entropy_coef: float = 0.0,
    max_grad_norm: float | None = None,
):
    """
    Minibatch PPO update over the rollout buffer.
    Buffer must return numpy arrays with keys:
        obs [N, obs_dim], acts [N, act_dim], adv [N], ret [N], logp [N]
    """
    data = buf.get()  # numpy
    obs = th.as_tensor(data["obs"], dtype=th.float32)
    acts = th.as_tensor(data["acts"], dtype=th.float32)
    adv = th.as_tensor(data["adv"], dtype=th.float32)
    ret = th.as_tensor(data["ret"], dtype=th.float32)
    logp_old = th.as_tensor(data["logp"], dtype=th.float32)

    # normalize advantages (more stable)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    N = obs.shape[0]
    for _ in range(int(update_epochs)):
        perm = th.randperm(N)
        for start in range(0, N, minibatch_size):
            idx = perm[start : start + minibatch_size]
            o = obs[idx]
            a = acts[idx]
            adv_b = adv[idx]
            ret_b = ret[idx]
            logp_old_b = logp_old[idx]

            mu = agent.pi(o)
            std = th.exp(agent.log_std).expand_as(mu)
            dist = Normal(mu, std)

            logp = dist.log_prob(a).sum(-1)  # [B]
            entropy = dist.entropy().sum(-1)  # [B]
            v = agent.v(o).squeeze(-1)  # [B]

            ratio = th.exp(logp - logp_old_b)  # [B]
            surr1 = ratio * adv_b
            surr2 = th.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv_b
            policy_loss = -th.mean(th.min(surr1, surr2))

            value_loss = th.mean((ret_b - v) ** 2)
            ent_bonus = th.mean(entropy)

            loss = policy_loss + value_coef * value_loss - entropy_coef * ent_bonus

            agent.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if max_grad_norm is not None:
                th.nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
            agent.optimizer.step()
