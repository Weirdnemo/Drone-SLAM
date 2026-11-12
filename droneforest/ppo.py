"""Minimal PPO (own loop, no external RL library)."""
import torch
import torch.nn as nn


def mlp(inp, out, hidden):
    layers, d = [], inp
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ELU()]
        d = h
    layers.append(nn.Linear(d, out))
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, critic_dim=None, hidden=(256, 128, 64), init_std=0.2):
        super().__init__()
        self.critic_dim = critic_dim or obs_dim          # asymmetric actor-critic: critic may see privileged state
        self.actor = mlp(obs_dim, act_dim, hidden)
        self.critic = mlp(self.critic_dim, 1, hidden)
        with torch.no_grad():                       # start near hover: tiny final-layer weights
            self.actor[-1].weight.mul_(0.01); self.actor[-1].bias.zero_()
            self.critic[-1].weight.mul_(0.1); self.critic[-1].bias.zero_()
        self.log_std = nn.Parameter(torch.full((act_dim,), float(torch.log(torch.tensor(init_std)))))

    def dist(self, obs):
        return torch.distributions.Normal(self.actor(obs), self.log_std.exp())

    @torch.no_grad()
    def act(self, obs, cobs=None):
        d = self.dist(obs)
        a = d.sample()
        return a, d.log_prob(a).sum(-1), self.critic(obs if cobs is None else cobs)[:, 0], d.mean

    def evaluate(self, obs, act, cobs=None):
        d = self.dist(obs)
        return d.log_prob(act).sum(-1), d.entropy().sum(-1), self.critic(obs if cobs is None else cobs)[:, 0], d


class PPO:
    def __init__(self, ac, lr=3e-4, clip=0.2, epochs=5, minibatches=4, gamma=0.99, lam=0.95,
                 vf_coef=1.0, ent_coef=0.0, desired_kl=None, max_grad=1.0):
        self.ac, self.lr, self.clip, self.epochs, self.mb = ac, lr, clip, epochs, minibatches
        self.gamma, self.lam, self.vf_coef, self.ent_coef = gamma, lam, vf_coef, ent_coef
        self.desired_kl, self.max_grad = desired_kl, max_grad
        self.opt = torch.optim.Adam(ac.parameters(), lr=lr)

    @torch.no_grad()
    def gae(self, rew, val, last_val, term, timeout_bootstrap):
        """rew,val,term: (H,E); timeout_bootstrap: (H,E) value to add on time-limit truncation."""
        H = rew.shape[0]
        adv = torch.zeros_like(rew)
        last = torch.zeros_like(last_val)
        rew = rew + self.gamma * timeout_bootstrap
        for t in reversed(range(H)):
            nxt_v = last_val if t == H - 1 else val[t + 1]
            nonterm = 1.0 - term[t]
            delta = rew[t] + self.gamma * nxt_v * nonterm - val[t]
            last = delta + self.gamma * self.lam * nonterm * last
            adv[t] = last
        return adv, adv + val

    def update(self, obs, cobs, act, logp, adv, ret, val):
        H, E = adv.shape
        obs, cobs, act = obs.reshape(H * E, -1), cobs.reshape(H * E, -1), act.reshape(H * E, -1)
        logp, adv, ret, val = (x.reshape(-1) for x in (logp, adv, ret, val))
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        N = H * E
        stats = {"pi": 0.0, "vf": 0.0, "kl": 0.0}
        n_upd = 0
        for _ in range(self.epochs):
            perm = torch.randperm(N, device=obs.device)
            for chunk in perm.chunk(self.mb):
                new_logp, ent, v, d = self.ac.evaluate(obs[chunk], act[chunk], cobs[chunk])
                ratio = (new_logp - logp[chunk]).exp()
                pi = -torch.min(ratio * adv[chunk], ratio.clamp(1 - self.clip, 1 + self.clip) * adv[chunk]).mean()
                vf = ((v - ret[chunk]) ** 2).mean()
                loss = pi + self.vf_coef * vf - self.ent_coef * ent.mean()
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), self.max_grad)
                self.opt.step()
                with torch.no_grad():
                    lr_ = new_logp - logp[chunk]
                    kl = ((lr_.exp() - 1) - lr_).mean().item()   # k3 estimator of KL(old||new)
                    if self.desired_kl is None:
                        pass
                    elif kl > 2 * self.desired_kl:
                        self.lr = max(1e-5, self.lr / 1.5)
                    elif kl < 0.5 * self.desired_kl:
                        self.lr = min(1e-3, self.lr * 1.5)
                    for g in self.opt.param_groups:
                        g["lr"] = self.lr
                stats["pi"] += pi.item(); stats["vf"] += vf.item(); stats["kl"] += kl
                n_upd += 1
        return {k: v / n_upd for k, v in stats.items()} | {"lr": self.lr, "std": self.ac.log_std.exp().mean().item()}


def load_init(ac, state):
    """Copy every tensor whose name and shape match (lets a symmetric teacher seed an asymmetric student)."""
    own = ac.state_dict()
    ok = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
    own.update(ok)
    ac.load_state_dict(own)
    return len(ok), len(own)
