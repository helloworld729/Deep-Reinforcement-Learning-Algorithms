import gym
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import time

from collections import deque
from agent import Agent, FloatTensor
from replay_buffer import ReplayMemory, Transition
from torch.autograd import Variable

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
device = torch.device("cuda" if use_cuda else "cpu")

BATCH_SIZE = 64
TAU = 0.005  # 1e-3   # for soft update of target parameters
gamma = 0.99
LEARNING_RATE = 0.001
TARGET_UPDATE = 10

num_episodes = 50
print_every = 1
hidden_dim = 16  ## 12 ## 32 ## 16 ## 64 ## 16
min_eps = 0.01
max_eps_episode = 50

env = gym.make('CartPole-v1', render_mode='agb_array')
# env = gym.wrappers.Monitor(env, directory="monitors", force=True)

space_dim = env.observation_space.shape[0]  # n_spaces
action_dim = env.action_space.n  # n_actions
print('input_dim: ', space_dim, ', output_dim: ', action_dim, ', hidden_dim: ', hidden_dim)

threshold = env.spec.reward_threshold
print('threshold: ', threshold)

agent = Agent(space_dim, action_dim, hidden_dim)


# 退火函数
def epsilon_annealing(i_epsiode, max_episode, min_eps: float):
    ##  if i_epsiode --> max_episode, ret_eps --> min_eps
    ##  if i_epsiode --> 1, ret_eps --> 1
    slope = (min_eps - 1.0) / max_episode
    ret_eps = max(slope * i_epsiode + 1.0, min_eps)
    return ret_eps


def save(directory, filename):
    torch.save(agent.q_local.state_dict(), '%s/%s_local.pth' % (directory, filename))
    torch.save(agent.q_target.state_dict(), '%s/%s_target.pth' % (directory, filename))


def run_episode(env, agent, eps):
    """Play an epsiode and train

    Args:
        env (gym.Env): gym environment (CartPole-v0)
        agent (Agent): agent will train and get action
        eps (float): eps-greedy for exploration

    Returns:
        int: reward earned in this episode
    """
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        if (len(state)) == 2: state = state[0]
        action = agent.get_action(FloatTensor([state]), eps)

        observation, reward, done, truncated, info = env.step(action.item())

        total_reward += reward

        if done:
            reward = -1

        # Store the transition in memory
        agent.replay_memory.push(
            (FloatTensor([state]),
             action,  # action is already a tensor
             FloatTensor([reward]),
             FloatTensor([observation]),
             FloatTensor([done])))

        if len(agent.replay_memory) > BATCH_SIZE:
            batch = agent.replay_memory.sample(BATCH_SIZE)

            agent.learn(batch, gamma)

        state = observation

    return total_reward


def train():
    scores_deque = deque(maxlen=100)
    scores_array = []
    avg_scores_array = []

    time_start = time.time()

    for i_episode in range(num_episodes):
        eps = epsilon_annealing(i_episode, max_eps_episode, min_eps)
        score = run_episode(env, agent, eps)

        scores_deque.append(score)
        scores_array.append(score)

        avg_score = np.mean(scores_deque)
        avg_scores_array.append(avg_score)

        dt = (int)(time.time() - time_start)

        if i_episode % print_every == 0 and i_episode > 0:
            print('Episode: {:5} Score: {:5}  Avg.Score: {:.2f}, eps-greedy: {:5.2f} Time: {:02}:{:02}:{:02}'. \
                  format(i_episode, score, avg_score, eps, dt // 3600, dt % 3600 // 60, dt % 60))

        if len(scores_deque) == scores_deque.maxlen:
            ### 195.0: for cartpole-v0 and 475 for v1
            if np.mean(scores_deque) >= threshold:
                print('\n Environment solved in {:d} episodes!\tAverage Score: {:.2f}'. \
                      format(i_episode, np.mean(scores_deque)))
                break

        if i_episode % TARGET_UPDATE == 0:
            agent.q_target.load_state_dict(agent.q_local.state_dict())

    return scores_array, avg_scores_array


scores, avg_scores = train()


