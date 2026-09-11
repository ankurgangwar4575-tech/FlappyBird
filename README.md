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
├── agent.py                 # Training workflow and checkpoint resume
├── test.py                  # Run a saved model for a fixed number of test steps
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

Example training output:

```text
Train | Episode: 12 | Episode steps: 84 | Total steps: 1035 | Reward: 6.90 | Epsilon: 0.9945
```

Press `Ctrl+C` to end a session early. The current model is saved before the
program exits.

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
reward. The terminal messages are test results only: this command does not
train, update, or save the model.

Example test output:

```text
Test | Episode: 3 | Episode steps: 125 | Total steps: 420 | Reward: 10.50
```

Use `--no-render` to test without opening the game window:

```bash
python test.py --steps 10000 --no-render
```

## ⚙️ Hyperparameters

Edit `parameters.yaml` to change the learning rate, discount factor, epsilon
schedule, replay-memory size, mini-batch size, target-network update rate, and
the number of steps per training session.

```yaml
FlappyBird-v0:
  train_steps: 50000
```

`train_steps` counts environment actions, not episodes. Every training session
stops after this many steps and saves the checkpoint. The next `python agent.py`
run loads that checkpoint and continues training from the saved network weights.

## 📈 Recommended training schedule

Start with 50,000-step sessions and test after every 2–4 sessions:

```bash
python test.py --steps 10000
```

- Aim for **1,000,000 total training steps** first (20 sessions at 50,000 steps).
- Continue to **2,000,000 steps** if test rewards are still improving but the
  bird is inconsistent.
- Consider **3,000,000 steps** only when testing shows continuing improvement.

Do not judge the model from one lucky episode. Compare repeated test runs and
look for consistently higher rewards and longer survival.

## 🛠️ Technologies

- Python
- PyTorch
- Gymnasium
- Flappy Bird Gymnasium
- PyYAML
