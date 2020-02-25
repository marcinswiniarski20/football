from PPO.constants import *


def distance(x, y):
    return ((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2) ** 0.5


class RewardWrapper:
    """ Reward Wrapper class used to provide own rewarding system

    """
    def __init__(self, observation):
        self.last_observation = observation
        self.agent_index = 0
        self.enemy_index = 1

    def transform(self, old_reward, observation, action):
        new_reward = old_reward
        new_reward += self.reward_for_moving_closer_to_ball_if_enemy_posses_it(observation)
        new_reward += self.reward_for_taking_over_ball(observation)
        self.last_observation = observation
        return new_reward

    def reward_for_moving_closer_to_ball_if_enemy_posses_it(self, observation):
        # if observation['ball_owned_team'] == self.enemy_index: #'if enemy posses it' part
        if observation['active'] == self.last_observation['active']:
            active_player = observation['active']
            active_player_last_position = self.last_observation['left_team'][active_player]
            active_player_position = observation['left_team'][active_player]
            ball_last_position = self.last_observation['ball']
            ball_position = observation['ball']

            if distance(ball_last_position, active_player_last_position) > distance(ball_position,
                                                                                    active_player_position):
                return REWARD_FOR_MOVING_CLOSER_TO_BALL_IF_ENEMY_POSSES_IT
            else:
                return -REWARD_FOR_MOVING_CLOSER_TO_BALL_IF_ENEMY_POSSES_IT
        return 0

    def reward_for_taking_over_ball(self, observation):
        if observation['ball_owned_team'] == self.agent_index and self.last_observation['ball_owned_team'] == self.enemy_index:
            print("Took over a ball")
            return REWARD_FOR_TAKING_OVER_BALL
        return 0
