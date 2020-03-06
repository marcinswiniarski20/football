import random
import os
import time

import gfootball.env as football_env
import numpy as np
import pandas as pd

from PPO.constants import *
from PPO.agent import PPOAgent
from PPO.wrappers import RewardWrapper
from PPO.helpers import get_advantages, get_action_occurance_in_ppo_step, print_in_console

import tensorflow as tf

# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

env = football_env.create_environment(env_name="academy_single_goal_versus_lazy", stacked=True,
                                      logdir='/tmp/fooacademy_3_vs_3tball', rewards='scoring',
                                      write_goal_dumps=False, write_full_episode_dumps=False, render=False)

state = env.reset() / 255.
state_dims = env.observation_space.shape
n_actions = env.action_space.n

dummy_n = np.zeros((1, 1, n_actions))
dummy_1 = np.zeros((1, 1, 1))

agent = PPOAgent(num_actions=n_actions, input_dim=state_dims)
agent.load(FOLDER_NAME, "best.hdf5") # used for loading weights
reward_wrapper = RewardWrapper(env.unwrapped.observation()[0])
max_iters = 500_000
iters = 96001

total_enemy_goals = 0
total_your_goals = 0
last_goals = 0
last_rewards = []
iter_goals = 0
best_reward = 0

steps = 128*iters

losses = pd.DataFrame(columns=['actor_loss', 'critic_loss', 'sum_128_reward', 'goals'])

start_time = time.time()

while iters < max_iters:
    states, actions,  values, masks, rewards, actions_probs, actions_onehot = [], [], [], [], [], [], []

    for itr in range(PPO_STEPS):
        action_dist = agent.actor.predict([np.array([state]), dummy_n, dummy_1, dummy_1, dummy_1], steps=1)
        # print(action_dist)
        q_value = agent.critic.predict([np.array([state])], steps=1)
        if random.random() > EXPLORATION:
            action = np.random.choice(n_actions, p=action_dist[0,:])

            # swaping last two actions in my best model
            if action == 18:
                action = 17
            if action == 17:
                action = 18
        else:
            action = random.randint(0, n_actions - 1)
            print(f"random action {action}")


        action_onehot = np.zeros(n_actions)
        action_onehot[action] = 1

        # time.sleep(0.02) # if you want to see them playing but it's too damn fast

        observation, reward, done, info = env.step(action)

        # own goals count as enemy goals i'm sorry
        if reward >= 1:
            iter_goals += 1
            last_goals += 1
            total_your_goals += 1
        if reward <= -1:
            iter_goals -= 1
            last_goals -= 1
            total_enemy_goals += 1
        mask = not done

        reward = reward_wrapper.transform(reward, env.unwrapped.observation()[0], action)

        states.append(np.array(state))
        actions.append(action)
        actions_onehot.append(action_onehot)
        values.append(q_value)
        masks.append(mask)
        rewards.append(reward)
        last_rewards.append(reward)
        actions_probs.append(action_dist)

        state = observation / 255.
        steps += 1
        if done:
            state = env.reset() / 255.

    total_reward = np.sum(np.array(rewards))

    # print('itr: ' + str(iters) + ', enemy goals=' + str(total_enemy_goals) + ', your goals=' + str(
    #     total_your_goals) + ', total reward=' + str(np.sum(np.array(rewards))))

    actions_choosen = get_action_occurance_in_ppo_step(n_actions, actions)

    q_value = agent.critic.predict(np.array([state]), steps=1)
    values.append(q_value)
    returns, advantages = get_advantages(values, masks, rewards)
    actor_loss = agent.actor.fit([np.array(states), np.array(actions_probs), np.array(advantages),
                                  np.reshape(np.array(rewards), newshape=(-1, 1, 1)), np.array(values[:-1])],
                                 [(np.reshape(np.array(actions_onehot), newshape=(-1, n_actions)))], verbose=0,
                                 shuffle=True, epochs=FIT_EPOCHES)
    critic_loss = agent.critic.fit(np.array(states), [np.reshape(returns, newshape=(-1, 1))], shuffle=True,
                                   epochs=FIT_EPOCHES,
                                   verbose=0)

    print_in_console(iters, critic_loss.history['loss'][7], actor_loss.history['loss'][7], actions_choosen,
                     time.time() - start_time, steps, total_reward, total_your_goals, total_enemy_goals)

    losses.loc[(iters - 1) % 100] = [critic_loss.history['loss'][7], actor_loss.history['loss'][7],
                                     round(total_reward, 4), iter_goals]
    iter_goals = 0

    if iters % 100 == 0:
        avg_reward = round(np.average(np.array(last_rewards)), 4)
        if not os.path.isfile(f'models/{FOLDER_NAME}/losses.csv'):
            losses.to_csv(f'models/{FOLDER_NAME}/losses.csv', float_format='%.4f')
        else:
            losses.to_csv(f'models/{FOLDER_NAME}/losses.csv', mode='a', float_format='%.4f', header=False)
        losses = pd.DataFrame(columns=['actor_loss', 'critic_loss', 'sum_128_reward', 'goals'])
        agent.save(iters, avg_reward, last_goals, FOLDER_NAME)

        if avg_reward > best_reward:
            print('best reward=' + str(avg_reward))
            best_reward = avg_reward
        last_rewards = []
        last_goals = 0
        env.reset()
    iters += 1