import torch as T
import torch.nn.functional as F
import numpy as np
from Files.buffer import ReplayBuffer
from Files.networks import ActorNetwork, CriticNetwork

class DDPGAgent:
    def __init__(self, alpha=1e-4, beta=1e-3, input_dims=[5], n_actions=1,
                 tau=0.005, gamma=0.99, max_size=1000000, batch_size=128, env=None,
                 chkpt_dir='None'):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.env = env


        self.actor = ActorNetwork(alpha, input_dims, max_action=env.action_space.high,
                                  n_actions=n_actions, name='actor_ddpg', chkpt_dir=chkpt_dir)
        self.critic = CriticNetwork(beta, input_dims, n_actions=n_actions,
                                    name='critic_ddpg', chkpt_dir=chkpt_dir)

        self.target_actor = ActorNetwork(alpha, input_dims, max_action=env.action_space.high,
                                         n_actions=n_actions, name='target_actor_ddpg', chkpt_dir=chkpt_dir)
        self.target_critic = CriticNetwork(beta, input_dims, n_actions=n_actions,
                                           name='target_critic_ddpg', chkpt_dir=chkpt_dir)


        self._update_targets(tau=1)

    def choose_action(self, observation, noise=0.1):
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)
        mu, _ = self.actor.forward(state)
        mu = T.tanh(mu) * T.tensor(self.env.action_space.high).to(self.actor.device)
        action = mu.cpu().detach().numpy()[0]

        action = np.clip(action + noise * np.random.randn(*action.shape),
                         self.env.action_space.low, self.env.action_space.high)
        return action

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def _update_targets(self, tau=None):
        if tau is None:
            tau = self.tau

        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau*param.data + (1-tau)*target_param.data)
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau*param.data + (1-tau)*target_param.data)

    def save_models(self):
        print('.... saving models ....')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print('.... loading models ....')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()


    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        states, actions, rewards, states_, dones = self.memory.sample_buffer(self.batch_size)
        states = T.tensor(states, dtype=T.float).to(self.actor.device)
        actions = T.tensor(actions, dtype=T.float).to(self.actor.device)
        rewards = T.tensor(rewards, dtype=T.float).to(self.actor.device)
        states_ = T.tensor(states_, dtype=T.float).to(self.actor.device)
        dones = T.tensor(dones).to(self.actor.device)

        with T.no_grad():
            mu_target, _ = self.target_actor.forward(states_)
            mu_target = T.tanh(mu_target) * T.tensor(self.env.action_space.high).to(self.actor.device)
            q_next = self.target_critic.forward(states_, mu_target).view(-1)
            q_next[dones] = 0.0
            q_target = rewards + self.gamma * q_next

        q_current = self.critic.forward(states, actions).view(-1)
        critic_loss = F.mse_loss(q_current, q_target)
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        mu, _ = self.actor.forward(states)
        mu = T.tanh(mu) * T.tensor(self.env.action_space.high).to(self.actor.device)
        actor_loss = -self.critic.forward(states, mu).mean()
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        self._update_targets()
