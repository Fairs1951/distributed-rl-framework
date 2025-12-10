import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Dict
import gym

class ActorCritic(nn.Module):
    """
    Standard Actor-Critic network for PPO and other RL algorithms.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.actor(state), self.critic(state)

class PPOMemory:
    """
    Buffer to store experience for PPO updates.
    """
    def __init__(self, batch_size: int):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size

    def generate_batches(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[np.ndarray]]:
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]
        return np.array(self.states), np.array(self.actions), np.array(self.probs), \
               np.array(self.vals), np.array(self.rewards), np.array(self.dones), batches

    def store_memory(self, state, action, prob, val, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(prob)
        self.vals.append(val)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states, self.probs, self.vals, self.actions, self.rewards, self.dones = [], [], [], [], [], []

class PPOAgent:
    """
    Proximal Policy Optimization Agent.
    """
    def __init__(self, state_dim: int, action_dim: int, lr: float = 3e-4, gamma: float = 0.99, 
                 gae_lambda: float = 0.95, policy_clip: float = 0.2, batch_size: int = 64, n_epochs: int = 10):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.actor_critic = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        self.memory = PPOMemory(batch_size)

    def choose_action(self, observation: np.ndarray) -> Tuple[int, float, float]:
        state = torch.tensor([observation], dtype=torch.float)
        probs, value = self.actor_critic(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        prob = torch.squeeze(dist.log_prob(action)).item()
        action = torch.squeeze(action).item()
        value = torch.squeeze(value).item()
        return action, prob, value

    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr, batches = self.memory.generate_batches()
            values = torch.tensor(vals_arr)
            advantage = np.zeros(len(reward_arr), dtype=np.float32)
            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount*(reward_arr[k] + self.gamma*vals_arr[k+1]*(1-int(dones_arr[k])) - vals_arr[k])
                    discount *= self.gamma*self.gae_lambda
                advantage[t] = a_t
            advantage = torch.tensor(advantage)

            for batch in batches:
                states = torch.tensor(state_arr[batch], dtype=torch.float)
                old_probs = torch.tensor(old_prob_arr[batch])
                actions = torch.tensor(action_arr[batch])
                probs, critic_value = self.actor_critic(states)
                dist = torch.distributions.Categorical(probs)
                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1-self.policy_clip, 1+self.policy_clip)*advantage[batch]
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()
                returns = advantage[batch] + values[batch]
                critic_loss = (returns-torch.squeeze(critic_value))**2
                critic_loss = critic_loss.mean()
                total_loss = actor_loss + 0.5*critic_loss
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
        self.memory.clear_memory()

if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    agent = PPOAgent(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n)
    print("PPO Agent initialized successfully.")
