from collections import deque
from enviroment import enviroment
from combatmodel import CombatModel
import torch
import torch.nn as nn
import random
import torch.optim as optim
import numpy as np

VERBOSE = False

# Hyperparameters
gamma = 0.9
epsilon = 1.0
epsilon_min = 0.1
epsilon_decay = 0.995
learning_rate = 0.001
batch_size = 64
target_update = 10

# Replay Memory
combat_memory = deque(maxlen=100000)
noncombat_memory = deque(maxlen=100000)

#InitializeNetworks
env = enviroment()
combat_q_network = CombatModel(env.action_size)
combat_target_network = CombatModel(env.action_size)
combat_target_network.load_state_dict(combat_q_network.state_dict())
optimizer = optim.Adam(combat_q_network.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

max_reward = -float('inf')
save_path_q  = 'model_q_checkpoint.pth'
save_path_target = 'model_target_checkpoint.pth'

if False:
    combat_q_network = torch.load(save_path_q)
    combat_target_network = torch.load(save_path_target)

for episode in range(10000):
    if VERBOSE:
        print(f"Episode: {episode} started!")
    combat_state, non_combat_state = env.start_game()
    done = False
    include = False

    if episode % 5 == 0:
        print(f"Exploration set to 0%. Evaluating")

    while not done:
        if env.in_combat():
            multi_move = env.continue_previous_action
            exploration = np.random.rand()
            if episode % 5 == 0:
                exploration = 1.0
            if exploration < epsilon and not multi_move:
                action = env.select_random_action()
                action_tensor = env.TensorGenerator.move2tensor(action, env.get_hand_ids(), env.get_enemy_ids(), env.get_discard_ids(), env.get_potion_ids())
                combat_next_state, reward, done, include = env.step(action_tensor)
            else:
                q_values = combat_q_network(combat_state)
                mask = env.valid_moves_mask().unsqueeze(0)
                q_values[mask] = -float('inf')
                action_idx = torch.argmax(q_values).item()
                combat_next_state, reward, done, include = env.step(action_idx)
        else:
            env.make_random_action()

        done = env.get_current_hp() <= 0 or 'game_over' in env.last_response
        
        if done and episode % 5 == 0:
            if env.history['cumulative_reward'] > max_reward:
                max_reward = env.history['cumulative_reward']
                torch.save(combat_q_network, save_path_q)
                torch.save(combat_target_network, save_path_target)

        if include and not combat_state is None and not combat_next_state is None:
            combat_memory.append((combat_state, action_idx, reward, combat_next_state, done))
            combat_state = combat_next_state

        if len(combat_memory) >= batch_size:
            batch = random.sample(combat_memory, batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)

            states = tuple(torch.stack(tensors) for tensors in zip(*states))
            actions = torch.LongTensor(actions)
            rewards = torch.FloatTensor(rewards)
            next_states = stacked_states = tuple(torch.stack(tensors) for tensors in zip(*next_states))
            dones = torch.FloatTensor(dones)

            q_values = combat_q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            next_q_values = combat_target_network(next_states).max(1)[0]
            targets = rewards + gamma * next_q_values * (1 - dones)

            loss = nn.MSELoss()(q_values, targets.detach())
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            scheduler.step()

    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    env.log(episode, epsilon)