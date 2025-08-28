from pydoc import render_doc
import numpy as np
import matplotlib.pyplot as plt
import pennylane as qml
from gym import spaces, Env

class QControlEnv(Env):
    def __init__(self, n_qubits=1, n_controls=1, max_steps=20, target="X", fidelity_threshold=0.995, **kwargs):
        self.n_qubits = n_qubits
        self.n_controls = n_controls
        self.max_steps = max_steps
        self.step_count = 0
        self.fidelity_threshold = fidelity_threshold
        self.action_space = spaces.Box(low=-np.pi, high=np.pi, shape=(self.n_controls,), dtype=np.float32)
        dim = 2 ** self.n_qubits
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(dim * 2,), dtype=np.float32)
        self.dev = qml.device("default.qubit", wires=self.n_qubits)
        self.target_gate = target
        self._build_target_state()
        self.state = None
        self.render_mode = render_doc

    def _build_target_state(self):
        @qml.qnode(self.dev)
        def circuit():
            if self.target_gate == "X":
                for w in range(self.n_qubits):
                    qml.PauliX(wires=w)
            elif self.target_gate == "H":
                for w in range(self.n_qubits):
                    qml.Hadamard(wires=w)
            elif self.target_gate == "S":
                for w in range(self.n_qubits):
                    qml.PhaseShift(0.5 * np.pi, wires=w)
            return qml.state()
        self.target_state = circuit()

    def _apply_actions_get_state(self, actions):
        @qml.qnode(self.dev)
        def circuit(params):
            for w in range(self.n_qubits):
                angle = params[w % len(params)]
                qml.RX(angle, wires=w)
            return qml.state()
        return circuit(actions)

    def _state_to_obs(self, state):
        real = np.real(state)
        imag = np.imag(state)
        return np.concatenate([real, imag]).astype(np.float32)

    def reset(self):
        self.step_count = 0
        init = np.zeros(2 ** self.n_qubits, dtype=complex)
        init[0] = 1.0 + 0j
        self.state = init
        return self._state_to_obs(self.state)

    def step(self, action):
        self.step_count += 1
        action = np.array(action, dtype=float).flatten()
        new_state = self._apply_actions_get_state(action)
        self.state = new_state
        fidelity = np.abs(np.vdot(self.target_state, self.state)) ** 2
        reward = float(fidelity)
        done = bool(fidelity >= self.fidelity_threshold or self.step_count >= self.max_steps)
        info = {"fidelity": fidelity}
        obs = self._state_to_obs(self.state)
        return obs, reward, done, info

    def render(self):
      if self.render_mode is None:
        return None

      probs = np.abs(self.state) ** 2
      probs = probs / probs.max() if probs.max() > 0 else probs

      n = int(np.ceil(np.sqrt(len(probs))))
      img = np.zeros((n, n))
      img.flat[:len(probs)] = probs

      img_rgb = np.uint8(img * 255)
      img_rgb = np.stack([img_rgb]*3, axis=-1)
      return img_rgb