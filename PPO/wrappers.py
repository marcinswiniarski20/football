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
        self.r = 0.03

    def transform(self, old_reward, observation, action):
        new_reward = 0
        # I don't give negative reward for enemy goal because my model defence was difficult to train with it
        if not old_reward < 0:
            new_reward = old_reward*2

        new_reward += self.reward_for_moving_closer_to_ball_if_enemy_posses_it(observation)
        new_reward += self.reward_for_taking_over_ball(observation)
        new_reward += self.reward_for_moving_ball_closer_to_enemy_goal(observation)
        new_reward += self.reward_for_accurate_shot(observation)
        # new_reward += self.reward_for_close_shot(observation, action)

        # repairing actions which model has given 0%
        # if action in [1,2,3,10,17,11]:
        #     print("repair action")
        #     self.r = self.r*0.996
        #     new_reward += self.r

        self.last_observation = observation
        return new_reward

    def reward_for_close_shot(self, observation, action):  # there are some issues here
        if action == 12 and observation['ball'][0] > 0.3:
            if self.last_observation['ball'][0] < observation['ball'][0]:
                active_player = observation['left_team'][observation['active']]
                if (-0.3 < active_player[1] < 0.3) and active_player[0] > 0.3:
                    print("CLOSE SHOT")
                    return REWARD_FOR_CLOSE_SHOT
        return 0

    def reward_for_accurate_shot(self, observation):
        GOAL_HEIGHT = 0.16
        if observation['ball'][0] > 0.95 and not (self.last_observation['ball'][0] > 0.95):
            if GOAL_HEIGHT / 2 > observation['ball'][1] > -GOAL_HEIGHT / 2:
                print("Accurate shot")
                return REWARD_FOR_ACCURATE_SHOT

        return 0

    def reward_for_moving_closer_to_ball_if_enemy_posses_it(self, observation):
        if observation['ball_owned_team'] == self.enemy_index: #'if enemy posses it' part
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
                    return -REWARD_FOR_MOVING_CLOSER_TO_BALL_IF_ENEMY_POSSES_IT*1.2
        return 0

    def reward_for_moving_ball_closer_to_enemy_goal(self, observation):
        if observation['ball'][0] > -1:  # if on enemy side
            if distance(observation['ball'], [1,0]) < distance(self.last_observation['ball'], [1,0]):
                # print("closer")
                return REWARD_FOR_MOVING_BALL_CLOSER_TO_ENEMY_GOAL
            elif not distance(observation['ball'], [1,0]) == distance(self.last_observation['ball'], [1,0]):
                # print("far")
                return -REWARD_FOR_MOVING_BALL_CLOSER_TO_ENEMY_GOAL*1.2
        return 0

    def reward_for_taking_over_ball(self, observation):
        if observation['ball_owned_team'] == self.agent_index and self.last_observation['ball_owned_team'] == self.enemy_index:
            print("Took over a ball")
            return REWARD_FOR_TAKING_OVER_BALL
        return 0
