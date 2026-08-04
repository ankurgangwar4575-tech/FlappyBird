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
        
        
        self.loss_fn=nn.MSELoss()
        self.optimizer=None

        self.LOG_FILE=os.path.join(RUNS_DIR,f"{self.param_set}.log")
        self.MODEL_FILE=os.path.join(RUNS_DIR,f"{self.param_set}.pt")
    def run(self,is_training=True,render=False):
        env = gym.make(self.env_id, render_mode="human" if render else None)

        num_states=env.observation_space.shape[0] # input dim
        num_actions=env.action_space.n # output dim 
        
        policy_dqn=DQN(num_states,num_actions).to(device)
    
        if is_training:
            memory=ReplayMemory(self.replay_memory_size)
            epsilon=self.epsilon_init
            
            target_dqn=DQN(num_states,num_actions).to(device)
            # copy the wt and bias vals from policy => target
            target_dqn.load_state_dict(policy_dqn.state_dict())
        
            steps=0
            self.optimizer=optim.Adam(policy_dqn.parameters(),lr=self.alpha)
            
            best_reward=float("-inf")
        else:
             policy_dqn.load_state_dict(torch.load(self.MODEL_FILE, map_location=device, weights_only=True))
             policy_dqn.eval()
             
               
        for episode in itertools.count():
            state, _ = env.reset()
            state=torch.tensor(state,dtype=torch.float,device=device)
            
            episode_rewards=0
            done = False
            
            while not done:
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
                    steps+=1

                state=next_state
                    
                    
                episode_rewards+=reward.item()
            if is_training:
                print(f"Episode-->{episode + 1}, total reward -> {episode_rewards} & epsilon -> {epsilon}")
            else:
                print(f"Episode-->{episode + 1}, total reward -> {episode_rewards}")
            
            
            if is_training:
                epsilon=max(epsilon*self.epsilon_decay,self.epsilon_min)
                
                if episode_rewards>best_reward:
                    log_msg=f"Best reward={episode_rewards} for episode={episode+1}"
                    with open(self.LOG_FILE,"a") as f:
                        f.write(log_msg+"\n")
                    
                    torch.save(policy_dqn.state_dict(),self.MODEL_FILE)
                    best_reward=episode_rewards
            if is_training and len(memory) >= self.mini_batch_size:
                mini_batch=memory.sample(self.mini_batch_size)
                
                self.optimize(mini_batch,policy_dqn,target_dqn)
                
                if steps >= self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    steps=0
            # env.close()
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
        
    parser=argparse.ArgumentParser(description="Train or Test model")
    parser.add_argument("hyperparameters",help="")
    parser.add_argument("--train",help="Training mode", action="store_true")
    args=parser.parse_args()
        
    dql=Agent(param_set=args.hyperparameters)
        
    if args.train:
        dql.run(is_training=True)
    else:
        dql.run(is_training=False,render=True)
