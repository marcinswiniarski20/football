# -----------------------
# Agent constants
# -----------------------

ACTOR_LERARNING_RATE = 0.00013
CRITIC_LERARNING_RATE = 0.00014
CLIPPING = 0.12
GAMMA = 0.997  # discouting factor
LAMBDA = 0.95  # advatage estimation discounting factor
FIT_EPOCHES = 8  # number of epoch to fit through batch
ENTROPY_COEF = 0.01  # policy entropy coefficient in the optimization objective (it makes agent explore more)
CRITIC_DISCOUNT = 0.5

# -----------------------
# Training constants
# -----------------------

PPO_STEPS = 128  # amount of steps to update agent
EXPLORATION = 0.0  # force to try even rejected actions
FOLDER_NAME = "academy_single_goal_versus_lazy"

# -----------------------
# RewardWrapper constants
# -----------------------

REWARD_FOR_MOVING_CLOSER_TO_BALL_IF_ENEMY_POSSES_IT = 0.008
REWARD_FOR_TAKING_OVER_BALL = 0.03
REWARD_FOR_MOVING_BALL_CLOSER_TO_ENEMY_GOAL = 0.0002
REWARD_FOR_ACCURATE_SHOT = 0.1
REWARD_FOR_CLOSE_SHOT = 0.2
