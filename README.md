# 🐦 Flappy Bird DQN

A reinforcement-learning project that trains a Deep Q-Network (DQN) agent to play Flappy Bird using PyTorch and Gymnasium.

## ✨ Features

- 🧠 DQN policy network implemented with PyTorch
- 💾 Experience replay memory
- 🎲 Epsilon-greedy exploration
- 🎯 Target network for more stable learning
- ⚙️ Hyperparameters configured in YAML
- 📈 Automatic model checkpoints and training logs

## 📁 Project structure

```text
flappy-bird-dqn/
├── agent.py                 # Training and testing workflow
├── dqn.py                   # DQN neural-network model
├── experience_replay.py     # Replay-memory implementation
├── parameters.yaml          # Environment and hyperparameter settings
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
└── .gitignore               # Files excluded from Git
```

Training models and logs are written to `runs/`. This directory is excluded from Git because it contains generated files.

## 🚀 Installation

```bash
git clone https://github.com/YOUR_USERNAME/flappy-bird-dqn.git
cd flappy-bird-dqn
pip install -r requirements.txt
```

## 🏋️ Train the agent

```bash
python agent.py FlappyBird-v0 --train
```

The best model is saved as `runs/FlappyBird-v0.pt`.

## 🎮 Test the agent

Train the model first, then run:

```bash
python agent.py FlappyBird-v0
```

## ⚙️ Hyperparameters

Edit `parameters.yaml` to change the learning rate, discount factor, epsilon schedule, replay-memory size, mini-batch size, and target-network update rate.

## 🛠️ Technologies

- Python
- PyTorch
- Gymnasium
- Flappy Bird Gymnasium
- PyYAML
