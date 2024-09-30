import torch
import torch.nn.functional as F
from torch.optim import Adam
from utils import soft_update, hard_update
from model import GaussianPolicy, QNetwork
# torch.autograd.set_detect_anomaly(True)

class soft_actor_critic_agent(object):
    def __init__(self, num_inputs, action_space, \
                 device, hidden_size, lr, gamma, tau, alpha):

        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        self.device = device 

        # 定义价值网络
        # num_inputs, action_space 表示 环境/action 特征的维度，hiddensize是隐层维度
        self.critic = QNetwork(num_inputs, action_space.shape[0], hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=lr)
        self.critic_target = QNetwork(num_inputs, action_space.shape[0], hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)
        
        # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
        self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optim = Adam([self.log_alpha], lr=lr)

        # 定义策略网络
        self.policy = GaussianPolicy(num_inputs, action_space.shape[0], \
                                         hidden_size, action_space).to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=lr)

    def select_action(self, state, eval=False):
        if len(state)==2: state=state[0]
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if eval == False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def update_parameters(self, memory, batch_size, updates):
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            # Q(S_t, A_t) 的目标
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)

        # ----------------------------- ↓↓↓↓↓ Update Critic Net ↓↓↓↓↓ ------------------------------#
        qf1, qf2 = self.critic(state_batch, action_batch) 
        q_loss = F.mse_loss(qf1, next_q_value) + F.mse_loss(qf2, next_q_value)
        self.critic_optim.zero_grad()
        q_loss.backward()
        self.critic_optim.step()

        # ----------------------------- ↓↓↓↓↓ Update Actor Net ↓↓↓↓↓ ------------------------------#
        # for params in self.critic.parameters(): params.requires_grad = False
        pi, log_pi, _ = self.policy.sample(state_batch)
        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        # ----------------------------- ↓↓↓↓↓ Update alpha ↓↓↓↓↓ ------------------------------#
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        self.alpha = self.log_alpha.exp()
        alpha_tlogs = self.alpha.clone() # For TensorboardX logs

        soft_update(self.critic_target, self.critic, self.tau)    

