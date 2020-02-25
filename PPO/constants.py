# -----------------------
# Agent constants
# -----------------------

ACTOR_LERARNING_RATE = 0.0001
CRITIC_LERARNING_RATE = 0.0002
CLIPPING = 0.2
GAMMA = 0.997  # discouting factor
LAMBDA = 0.95  # advatage estimation discounting factor
NSTEPS = 8  # number of steps per update
ENTROPY_COEF = 0.003  # policy entropy coefficient in the optimization objective
CRITIC_DISCOUNT = 0.5

# -----------------------
# Training constants
# -----------------------

PPO_STEPS = 128  # amount of steps to update agent
# EXPLORATION = 0.1
FOLDER_NAME = "11_vs_11_easy_stochastic"

# -----------------------
# RewardWrapper constants
# -----------------------

REWARD_FOR_MOVING_CLOSER_TO_BALL_IF_ENEMY_POSSES_IT = 0.00005
REWARD_FOR_TAKING_OVER_BALL = 0.004

