import numpy as np
import scipy.stats

from PPO.constants import *


def get_advantages(values, masks, rewards):
    returns = []
    gae = 0
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + GAMMA * values[i + 1] * masks[i] - values[i]
        gae = delta + GAMMA * LAMBDA * masks[i] * gae
        returns.insert(0, gae + values[i])

    adv = np.array(returns) - values[:-1]
    return returns, (adv - np.mean(adv)) / (np.std(adv) + 1e-10)


def get_action_occurance_in_ppo_step(n_actions, actions):
    """
        Printing all action choosen during previous PPO_STEPS in order
        to see if agent is not giving 100% to specific action every time
        index in the list is index of action and value in this index is
        number of times the actions was chosen.
    """
    occurance = []
    for i in range(n_actions):
        occurance.append(actions.count(i))

    return occurance


def print_in_console(iters, critic_loss, actor_loss, actions_choosen, time_elapsed,
                     num_of_steps, reward, your_goals, enemy_goals):

    print(f"=========== {iters} ===========")
    print(f"          {your_goals} : {enemy_goals} ")
    print(f"time elapsed: {round(time_elapsed,1)}s")
    print(f"steps count: {num_of_steps}")
    print(f"critic loss: {round(critic_loss,5)}")
    print(f"actor loss: {round(actor_loss,5)}")
    print(f"total reward: {round(reward,7)}")
    print("actions chosen:")
    print(actions_choosen)
    print("––––––––––––––––––––––––––––––")

