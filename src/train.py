import argparse
import yaml
import os
import numpy as np
from tqdm import trange
import torch
from env import QControlEnv
from agent import PPO

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main(cfg):
    np.random.seed(cfg.get("seed", 0))
    env_cfg = cfg["env"]
    agent_cfg = cfg["agent"]
    env = QControlEnv(**env_cfg)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    ppo = PPO(obs_dim, act_dim, agent_cfg)
    total_steps = cfg["training"]["total_timesteps"]
    save_path = cfg["training"]["save_path"]
    obs_buf, act_buf, logp_buf, rew_buf, val_buf, done_buf = [], [], [], [], [], []
    obs = env.reset()
    ep_returns = []
    steps = 0
    pbar = trange(total_steps)
    while steps < total_steps:
        act, logp, val = ppo.select_action(obs)
        next_obs, reward, done, info = env.step(act)
        obs_buf.append(obs)
        act_buf.append(act)
        logp_buf.append(logp)
        rew_buf.append(reward)
        val_buf.append(val)
        done_buf.append(done)
        obs = next_obs
        steps += 1
        pbar.update(1)
        if done:
            obs = env.reset()
        if len(obs_buf) >= agent_cfg["batch_size"]:
            last_val = 0 if done else ppo.select_action(obs)[2]
            vals = val_buf.copy()
            vals.append(last_val)
            advs, rets = ppo.compute_gae(rew_buf, vals, done_buf)
            ppo.update(obs_buf, act_buf, logp_buf, rets, advs)
            obs_buf, act_buf, logp_buf, rew_buf, val_buf, done_buf = [], [], [], [], [], []
            ppo.save(save_path)
    pbar.close()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    ppo.save(save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    main(cfg)