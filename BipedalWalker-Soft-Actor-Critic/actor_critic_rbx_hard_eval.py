import gym, time
import numpy as np
import torch
from sac_agent import soft_actor_critic_agent
from replay_memory import ReplayMemory
import torch
# torch.autograd.set_detect_anomaly(True)

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
seed=0
env = gym.make('BipedalWalkerHardcore-v3', render_mode='human')
torch.manual_seed(seed)
np.random.seed(seed)
# env.seed(seed)
max_steps = env._max_episode_steps
print('max_steps: ', max_steps)

batch_size=2

LEARNING_RATE=0.00008 # lr = 0.0001 for BipedalWalker-SAC_lr0001
eval=True  ##
start_steps=10000 ## Steps sampling random actions
replay_size=1000000 ## size of replay buffer
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# alpha=0.2  # relative importance of the entropy
# gamma=0.99  # discount factor
# tau=0.005  # target smoothing coefficient(τ)

agent = soft_actor_critic_agent(env.observation_space.shape[0], env.action_space, \
        device=device, hidden_size=256, lr=LEARNING_RATE, gamma=0.99, tau=0.005, alpha=0.2)



memory = ReplayMemory(replay_size)

print('device: ', device)
print('state dim: ', env.observation_space.shape[0])
print('action dim: ', env.action_space)
print('leraning rate: ', LEARNING_RATE)

def play(env, agent, num_episodes):

    state = env.reset()

    for i_episode in range(num_episodes + 1):

        state = env.reset()
        score = 0
        time_start = time.time()

        while True:

            action = agent.select_action(state, eval=True)
            env.render()
            next_state, reward, done, _, _ = env.step(action)
            score += reward
            state = next_state
            # print(score)

            if done:
                break
def load(agent):
    agent.policy.load_state_dict(
        torch.load('dir_chk/hard/weights_lr00008_rbx_jupyter_actor_final.pth', map_location=torch.device('cpu')))
    agent.critic.load_state_dict(
        torch.load('dir_chk/hard/weights_lr00008_rbx_jupyter_critic_final.pth', map_location=torch.device('cpu')))



load(agent)
play(env=env, agent=agent, num_episodes=5)
env.close()
