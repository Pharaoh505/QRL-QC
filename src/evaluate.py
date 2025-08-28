import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
from env import QControlEnv
from agent import PPO

def expand_state_uniform(state, scale=50):
    probs = np.abs(state) ** 2
    n = len(probs)
    side = int(np.ceil(np.sqrt(n)))
    grid = np.zeros((side, side))
    grid.flat[:n] = probs
    return np.kron(grid, np.ones((scale, scale)))

def run(cfg, model_path):
    env = QControlEnv(
        n_qubits=cfg["env"]["n_qubits"],
        n_controls=cfg["env"]["n_controls"],
        render_mode="rgb_array"
    )

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    agent_cfg = cfg["agent"]
    agent = PPO(obs_dim, act_dim, agent_cfg)
    agent.load(model_path)

    obs = env.reset()
    done = False
    total_reward = 0
    info_dict = {}

    while not done:
        result = agent.select_action(obs)
        action = result[0] if isinstance(result, tuple) else result
        action = np.array(action, dtype=float).flatten()
        action = action[:env.action_space.shape[0]]
        obs, reward, done, info = env.step(action)
        total_reward += reward
        info_dict = info

    final_state = getattr(env, 'state', None)
    if final_state is not None:
        img = expand_state_uniform(final_state)
        plt.imshow(img, cmap='plasma', vmin=0, vmax=1)
        plt.axis('off')
        plt.savefig("finalstate.png", bbox_inches='tight')

    return total_reward, info_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    total_reward, info = run(cfg, args.model)
    print(f"Total reward: {total_reward}, info: {info}")