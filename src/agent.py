import torch
import os
import torch.nn as nn
import torch.optim as optim
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden_size), nn.Tanh(), nn.Linear(hidden_size, hidden_size), nn.Tanh())
        self.mu = nn.Linear(hidden_size, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))
        self.v = nn.Sequential(nn.Linear(hidden_size, 1))

    def forward(self, x):
        x = self.net(x)
        return self.mu(x), self.log_std.exp(), self.v(x).squeeze(-1)

class PPO:
    def __init__(self, obs_dim, act_dim, cfg):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ActorCritic(obs_dim, act_dim, cfg["hidden_size"]).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg["lr"])
        self.gamma = cfg["gamma"]
        self.lam = cfg["gae_lambda"]
        self.clip = cfg["ppo_clip"]
        self.epochs = cfg["epochs"]
        self.batch_size = cfg["batch_size"]
        self.minibatch_size = cfg["minibatch_size"]
        self.ent_coef = cfg["ent_coef"]
        self.vf_coef = cfg["vf_coef"]

    def select_action(self, obs):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            mu, std, val = self.model(obs_t)
            dist = torch.distributions.Normal(mu, std)
            act = dist.sample()
            logp = dist.log_prob(act).sum(-1)
        return act.cpu().numpy().flatten(), logp.cpu().numpy().item(), val.cpu().numpy().item()

    def compute_gae(self, rewards, vals, dones):
        adv = []
        lastgaelam = 0
        vals = vals + [0]
        for t in reversed(range(len(rewards))):
            nonterminal = 1.0 - float(dones[t])
            delta = rewards[t] + self.gamma * vals[t + 1] * nonterminal - vals[t]
            lastgaelam = delta + self.gamma * self.lam * nonterminal * lastgaelam
            adv.insert(0, lastgaelam)
        returns = [a + b for a, b in zip(adv, vals[:-1])]
        return adv, returns

    def update(self, obs_buf, act_buf, logp_buf, ret_buf, adv_buf):
        obs = torch.tensor(np.array(obs_buf), dtype=torch.float32, device=self.device)
        acts = torch.tensor(np.array(act_buf), dtype=torch.float32, device=self.device)
        old_logp = torch.tensor(np.array(logp_buf), dtype=torch.float32, device=self.device)
        rets = torch.tensor(np.array(ret_buf), dtype=torch.float32, device=self.device)
        advs = torch.tensor(np.array(adv_buf), dtype=torch.float32, device=self.device)
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        dataset_size = len(obs)
        for _ in range(self.epochs):
            idxs = np.arange(dataset_size)
            np.random.shuffle(idxs)
            for start in range(0, dataset_size, self.minibatch_size):
                mb_idx = idxs[start:start + self.minibatch_size]
                mu, std, val = self.model(obs[mb_idx])
                dist = torch.distributions.Normal(mu, std)
                logp = dist.log_prob(acts[mb_idx]).sum(-1)
                ratio = torch.exp(logp - old_logp[mb_idx])
                surr1 = ratio * advs[mb_idx]
                surr2 = torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip) * advs[mb_idx]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = ((val - rets[mb_idx]) ** 2).mean()
                entropy = dist.entropy().sum(-1).mean()
                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))