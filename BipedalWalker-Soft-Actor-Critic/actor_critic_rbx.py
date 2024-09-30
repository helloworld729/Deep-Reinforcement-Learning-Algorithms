import gym
import numpy as np
import torch
from sac_agent import soft_actor_critic_agent
from replay_memory import ReplayMemory
import torch
# torch.autograd.set_detect_anomaly(True)

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
seed=0
env = gym.make('BipedalWalker-v3', render_mode='human')
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


def save(agent, directory, filename, suffix):
    torch.save(agent.policy.state_dict(), '%s/%s_actor_%s.pth' % (directory, filename, suffix))
    torch.save(agent.critic.state_dict(), '%s/%s_critic_%s.pth' % (directory, filename, suffix))


import time
from collections import deque


def sac_train(max_steps):
    total_numsteps = 0
    updates = 0
    num_episodes = 10001
    updates = 0

    time_start = time.time()
    scores_deque = deque(maxlen=100)
    scores_array = []
    avg_scores_array = []

    for i_episode in range(num_episodes):
        episode_reward = 0
        episode_steps = 0
        done = False
        state = env.reset()

        for step in range(max_steps):
            if start_steps > total_numsteps:
                action = env.action_space.sample()  # Sample random action
            else:
                action = agent.select_action(state)  # Sample action from policy

            if len(memory) > batch_size:
                agent.update_parameters(memory, batch_size, updates)
                updates += 1

            next_state, reward, done, truncated, info = env.step(action)  # Step
            # observation, reward, done, truncated, info = env.step(action.item())
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward

            mask = 1.0 if episode_steps == env._max_episode_steps else float(not done)
            if len(state) == 2: state = state[0]
            memory.push(state, action, reward, next_state, mask)  # Append transition to memory

            state = next_state

            if done:
                break

        scores_deque.append(episode_reward)
        scores_array.append(episode_reward)
        avg_score = np.mean(scores_deque)
        avg_scores_array.append(avg_score)

        # if i_episode % 20 == 0 and i_episode > 0:
        #     save(agent, 'dir_chk_lr00008', 'weights', str(i_episode))

        s = (int)(time.time() - time_start)

        print("Ep.: {}, Total Steps: {}, Ep.Steps: {}, Score: {:.2f}, Avg.Score: {:.2f}, Time: {:02}:{:02}:{:02}". \
              format(i_episode, total_numsteps, episode_steps, episode_reward, avg_score, \
                     s // 3600, s % 3600 // 60, s % 60))

        if (avg_score > 300.5):
            print('Solved environment with Avg Score:  ', avg_score)
            break;

    return scores_array, avg_scores_array


scores, avg_scores = sac_train(max_steps=max_steps)