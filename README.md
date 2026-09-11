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
python agent.py
```

Training runs for the `train_steps` value in `parameters.yaml` (50,000 by
default) and prints the episode number, episode steps, total steps, reward,
and exploration rate in the terminal. At the end of each session, the model is
saved to `runs/FlappyBird-v0.pt`. A later training session automatically loads
that checkpoint and continues from its weights.

For a short run with a fixed number of episodes:

```bash
python agent.py --episodes 100
```

To override the YAML step limit for one session:

```bash
python agent.py --steps 100000
```

## 🎮 Test the agent

Train the model first, then run a fixed number of environment steps:

```bash
python test.py --steps 1000
```

This opens the game window and prints each completed episode's steps and
reward. Use `--no-render` to run without the window.

## ⚙️ Hyperparameters

Edit `parameters.yaml` to change the learning rate, discount factor, epsilon schedule, replay-memory size, mini-batch size, and target-network update rate.

## 🛠️ Technologies

- Python
- PyTorch
- Gymnasium
- Flappy Bird Gymnasium
- PyYAML
