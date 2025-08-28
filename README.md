# Quantum Reinforcement Learning for Quantum Control


This project looks into how reinforcement learning can be used to control quantum systems, specifically to guide them toward a particular target state as accurately as possible. It matters because improving quantum gate fidelity and control strategies is key for building more reliable quantum computers.

---


## Project Overview
- Implemented a custom environment using [PennyLane](https://pennylane.ai/) and [Gym](https://gymnasium.farama.org/).
- Trained a proximal policy optimization agent using [PyTorch](https://pytorch.org/).
- The agent learns to apply sequences of quantum gates to steer a quantum state toward a target state.



## Problem Formulation
- State: |ψ⟩ ∈ ℂ² (qubit state vector)  
- Action: Choice of quantum gate {RX(θ), RY(θ), RZ(θ)} applied to a wire (qubit) 
- Reward: Fidelity F(|ψ_T⟩, |ψ⟩) = |⟨ψ_T|ψ⟩|²  



## Results
  
| Metric          | Value   |
|-----------------|---------|
| Steps           | 500,000 |
| Average Reward  | 12      |
| Final Fidelity  | 0.70    |

- Steps: The number of training iterations the agent goes through
- Reward: A measure of how successful the agent’s actions are in the environment
- Fidelity: How close the final quantum state is to the desired target state

The agent does not yet reach high fidelity, but it successfully demonstrates how classical reinforcement learning can interact with a quantum simulator.


**Important:** Reward and fidelity may vary slightly on repeated runs due to stochasticity in the agent and quantum environment.



### Visualization
Training generated a visualization of the qubit


![Qubit](src/finalstate.png)


- Dark blue region: about 50% of the agent’s decisions
- Purple region: about 25% of the agent’s decisions
- Orange region: about 25% of the agent’s decisions

Interpretation:

- Each square represents a possible quantum state.
- Color intensity indicates probability: bright colors = more likely states.
- This allows you to see how the agent is controlling the qubit visually.


## Repository Structure
- `src/env.py` – Environment
- `src/agent.py` - PPO agent 
- `src/train.py` – Script to train ppo agent  
- `src/evaluate.py` – Script to test trained models  
- `models/` – Directory where trained models are saved (ignored in git)
- `config.yaml` – Training configuration (you can change the configs)


## Requirements
To run this project, install the dependencies:


```bash
pip install -r requirements.txt
```


Then train the model:


```bash
python src/train.py --config config.yaml
```

(The trained model is saved to `models/control.pth` after training)


And evaluate it:


```bash
python src/evaluate.py --config config.yaml --model models/control.pth
```
