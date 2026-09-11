import flappy_bird_gymnasium
import gymnasium as gym
import torch 
import torch.nn as nn
import torch.optim as optim
from dqn import DQN
from experience_replay import ReplayMemory
import itertools
import yaml
import random
import argparse
import os
from collections import deque
# Steps
'''
1. Set env
2. Define DQN
3. Create Experience Replay
4. Multiple episodes DQN train, setup
5. Epsilon greeddy policy + hyper parameter
6. Target Network
7. Train and test
'''

device = torch.device("cpu")
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")

RUNS_DIR="runs"
os.makedirs(RUNS_DIR,exist_ok=True)
class Agent:
    def __init__(self,param_set):
        self.param_set=param_set
        parameters_file = os.path.join(os.path.dirname(__file__), "parameters.yaml")
        with open(parameters_file, "r", encoding="utf-8") as f:
            all_param_set=yaml.safe_load(f)
            params=all_param_set[param_set]

        self.env_id = params["env_id"]
        
        self.alpha=params["alpha"]
        self.gamma=params["gamma"]
        
        self.epsilon_init=params["epsilon_init"]
        self.epsilon_min=params["epsilon_min"]
        self.epsilon_decay=params["epsilon_decay"]
        
        self.replay_memory_size=params["replay_memory_size"]
        self.mini_batch_size=params["mini_batch_size"]
        
        self.network_sync_rate=params["network_sync_rate"]
        self.reward_threshold=params["reward_threshold"]
        self.train_steps=params["train_steps"]
        self.training_goal_steps=params["training_goal_steps"]
        self.best_model_min_improvement=params["best_model_min_improvement"]
        
        
        self.loss_fn=nn.MSELoss()
        self.optimizer=None

        self.LOG_FILE=os.path.join(RUNS_DIR,f"{self.param_set}.log")
        # The best checkpoint is used for testing; the latest checkpoint is used
        # only to resume a later training session.
        self.MODEL_FILE=os.path.join(RUNS_DIR,f"{self.param_set}.pt")
        self.LATEST_FILE=os.path.join(RUNS_DIR,f"{self.param_set}-latest.pt")

    @staticmethod
    def _load_checkpoint(path):
        """Load both new training checkpoints and the project's older state dict."""
        return torch.load(path, map_location=device, weights_only=True)

    @staticmethod
    def _policy_state(checkpoint):
        return checkpoint.get("policy_state_dict", checkpoint)
    def run(self, is_training=True, render=False, max_steps=None, max_episodes=None):
        """Train or evaluate the agent.

        ``max_steps`` limits the total environment steps (useful for testing).
        ``max_episodes`` limits completed episodes (useful for short training runs).
        Training uses ``train_steps`` from parameters.yaml when ``max_steps`` is
        not supplied.
        """
        env = gym.make(self.env_id, render_mode="human" if render else None)

        num_states=env.observation_space.shape[0] # input dim
        num_actions=env.action_space.n # output dim 
        
        policy_dqn=DQN(num_states,num_actions).to(device)
    
        if is_training:
            print(f"Training on {device}")
            memory=ReplayMemory(self.replay_memory_size)
            self.optimizer=optim.Adam(policy_dqn.parameters(),lr=self.alpha)

            epsilon=self.epsilon_init
            lifetime_steps=0
            best_reward=float("-inf")
            recent_rewards=deque(maxlen=100)

            if os.path.exists(self.LATEST_FILE):
                checkpoint=self._load_checkpoint(self.LATEST_FILE)
                policy_dqn.load_state_dict(self._policy_state(checkpoint))
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                epsilon=checkpoint["epsilon"]
                lifetime_steps=checkpoint["lifetime_steps"]
                best_reward=checkpoint["best_mean_reward"]
                recent_rewards.extend(checkpoint.get("recent_rewards", []))
                print(f"Resumed checkpoint from step {lifetime_steps:,}")
            elif os.path.exists(self.MODEL_FILE):
                # Supports the old FlappyBird-v0.pt state-dict format.
                checkpoint=self._load_checkpoint(self.MODEL_FILE)
                policy_dqn.load_state_dict(self._policy_state(checkpoint))
                print("Resumed network weights from the existing best model")
            else:
                print("No saved model found; starting a new training session.")

            target_dqn=DQN(num_states,num_actions).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())
            steps_since_sync=0
        else:
            if not os.path.exists(self.MODEL_FILE):
                raise FileNotFoundError(
                    f"No test model found at {self.MODEL_FILE}. Train the agent first."
                )
            checkpoint=self._load_checkpoint(self.MODEL_FILE)
            policy_dqn.load_state_dict(self._policy_state(checkpoint))
            policy_dqn.eval()

        if is_training and max_steps is None:
            max_steps = self.train_steps

        if is_training:
            if lifetime_steps >= self.training_goal_steps:
                print(
                    f"Training goal already reached: {lifetime_steps:,} / "
                    f"{self.training_goal_steps:,} lifetime steps. "
                    "Increase training_goal_steps in parameters.yaml to continue."
                )
                env.close()
                return

            session_start = lifetime_steps
            session_end = min(
                session_start + max_steps,
                self.training_goal_steps,
            )
            max_steps = session_end - session_start
            print(
                f"Training steps {session_start + 1:,} to {session_end:,} "
                f"(goal: {self.training_goal_steps:,})"
            )

        total_steps = 0
        try:
            for episode in itertools.count():
                if max_episodes is not None and episode >= max_episodes:
                    break

                state, _ = env.reset()
                state=torch.tensor(state,dtype=torch.float,device=device)
            
                episode_rewards=0
                episode_steps = 0
                done = False
            
                while not done:
                    if max_steps is not None and total_steps >= max_steps:
                        done = True
                        break

                    if is_training and random.random()<epsilon:
                        action = env.action_space.sample()
                        action=torch.tensor(action,dtype=torch.long,device=device)
                    else:
                        with torch.no_grad():
                            action=policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()
                    next_state, reward, terminated, truncated, _ = env.step(action.item())
                    done = terminated or truncated
                
                    reward=torch.tensor(reward,dtype=torch.float,device=device)
                    next_state=torch.tensor(next_state,dtype=torch.float,device=device)
                
                    if is_training:
                        memory.append((state, action, reward, next_state, done))
                        lifetime_steps += 1
                        steps_since_sync += 1

                        # DQN learns continuously after enough experiences exist,
                        # rather than only once at the end of an episode.
                        if len(memory) >= self.mini_batch_size:
                            mini_batch=memory.sample(self.mini_batch_size)
                            self.optimize(mini_batch,policy_dqn,target_dqn)

                        if steps_since_sync >= self.network_sync_rate:
                            target_dqn.load_state_dict(policy_dqn.state_dict())
                            steps_since_sync=0

                        epsilon=max(epsilon*self.epsilon_decay,self.epsilon_min)

                    state=next_state
                    episode_steps += 1
                    total_steps += 1
                    
                    
                    episode_rewards+=reward.item()

                if is_training:
                    print(
                        f"Episode {episode + 1}: reward={episode_rewards:.2f}, "
                        f"step={lifetime_steps:,}"
                    )
                else:
                    print(
                        f"Test episode {episode + 1}: reward={episode_rewards:.2f}, "
                        f"step={total_steps:,}"
                    )
            
            
                if is_training:
                    recent_rewards.append(episode_rewards)
                    mean_reward=sum(recent_rewards) / len(recent_rewards)

                    # A rolling mean is more reliable than saving after one lucky
                    # episode. The best checkpoint is never overwritten by a worse
                    # session and is therefore safe to use with test.py.
                    if (len(recent_rewards) == recent_rewards.maxlen
                            and mean_reward >= best_reward + self.best_model_min_improvement):
                        log_msg=(f"Best 100-episode mean reward={mean_reward:.4f} "
                                 f"at lifetime step={lifetime_steps}")
                        with open(self.LOG_FILE,"a") as f:
                            f.write(log_msg+"\n")
                        torch.save(policy_dqn.state_dict(),self.MODEL_FILE)
                        best_reward=mean_reward
                        print(
                            f"New best model saved: mean reward={mean_reward:.2f}, "
                            f"step={lifetime_steps:,}"
                        )

                if max_steps is not None and total_steps >= max_steps:
                    break
        finally:
            if is_training:
                # This checkpoint contains the information needed to continue
                # training, without replacing the best model used by test.py.
                torch.save({
                    "policy_state_dict": policy_dqn.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "epsilon": epsilon,
                    "lifetime_steps": lifetime_steps,
                    "best_mean_reward": best_reward,
                    "recent_rewards": list(recent_rewards),
                }, self.LATEST_FILE)
                print(
                    f"Training session saved at step {lifetime_steps:,}"
                )
                print("Best test model is available at runs/FlappyBird-v0.pt")
            env.close()
    def optimize(self,mini_batch,policy_dqn,target_dqn):
        
        # get experience
        states, actions, rewards, next_states, terminations = zip(*mini_batch)
        states=torch.stack(states)
        actions=torch.stack(actions)
        next_states=torch.stack(next_states)
        rewards=torch.stack(rewards)
        terminations=torch.tensor(terminations).float().to(device)
        
        
        with torch.no_grad():
            target_q=rewards+(1-terminations)*self.gamma*target_dqn(next_states).max(dim=1)[0]
        current_q=policy_dqn(states).gather(dim=1,index=actions.unsqueeze(dim=1)).squeeze()
            
            # loss
        loss=self.loss_fn(current_q,target_q)
            
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        

if __name__=="__main__":
    parser=argparse.ArgumentParser(description="Train the Flappy Bird DQN model")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Optional maximum number of episodes")
    parser.add_argument("--steps", type=int, default=None,
                        help="Override train_steps from parameters.yaml")
    args=parser.parse_args()

    if args.steps is not None and args.steps <= 0:
        parser.error("--steps must be greater than zero")
        
    dql=Agent(param_set="FlappyBird-v0")
    dql.run(is_training=True, max_steps=args.steps, max_episodes=args.episodes)
