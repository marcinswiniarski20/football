from keras.layers import Dense, Conv2D, MaxPooling2D, Input, Dropout, Flatten, Add, Activation
from keras.optimizers import Adam
from keras.models import Model
from keras.initializers import RandomNormal
import keras.backend as K
import numpy as np

from PPO.constants import *


class PPOAgent:
    """ PPO agent class used for training

    Attributes:
        num_actions: number of possible actions in a given enviroment
        input_dim: shape of the input

        image_preprocessor: shared structure between critic and actor. It's used
                            to made of convolution and residual layers to extract
                            valuable information from input SMM image
        critic: outputs value of the given state
        actor: policy head, outputs probabilistic distribution over given actions

        Network architecture heavily inspired by google one used in gfootball paper (similar to impala cnn)
    """

    def __init__(self,
                 num_actions,
                 input_dim,
                 name='ppo'):
        if num_actions < 1:
            raise ValueError('Invalid number of actions')
        self.num_actions = num_actions
        self.input_dim = input_dim

        self.image_preprocessor, self.image_preprocessor_input = self.buid_image_preprocessor()
        self.critic = self.build_critic()
        self.actor = self.build_actor()

    def build_residual(self, input_layer, filters):
        """
            Created residual network used to extract more information from image
        """
        x = Activation('relu')(input_layer)
        x = Conv2D(kernel_size=(3, 3), strides=1, filters=filters, padding='same', kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(x)
        x = Activation('relu')(x)
        x_shortcut = Conv2D(kernel_size=(3, 3), strides=1, filters=filters, padding='same', kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(x)
        x = Add()([input_layer, x_shortcut])
        return x

    def buid_image_preprocessor(self):
        """ Creates shared layer between actor and critic. It's purpose is to transform image information to latent space

            Returns:
                x: last layer
                preprocessor_input: input layer
        """
        preprocessor_input = Input(shape=self.input_dim)
        x = Conv2D(kernel_size=(3, 3), strides=1, filters=32, padding='same', kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(preprocessor_input)
        x = MaxPooling2D(pool_size=(3, 3), strides=2)(x)
        x = self.build_residual(x, 32)
        x = self.build_residual(x, 32)

        for _ in range(3):
            x = Conv2D(kernel_size=(3, 3), strides=1, filters=64, padding='same', kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(x)
            x = MaxPooling2D(pool_size=(3, 3), strides=2)(x)
            x = self.build_residual(x, 64)
            x = self.build_residual(x, 64)

        x = Activation('relu')(x)
        return x, preprocessor_input

    def build_critic(self):
        """
            Creates critic - network used to evaluate q-value of given state
        """
        x = Flatten()(self.image_preprocessor)
        x = Activation('relu')(x)
        x = Dense(256, activation='relu', kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(x)
        output = Dense(1)(x)
        model = Model([self.image_preprocessor_input], [output])
        model.compile(Adam(CRITIC_LERARNING_RATE), loss='mse')
        model.summary()
        return model

    def build_actor(self):
        """
            Creates actor - policy network which estimate best action for given state
        """
        oldpolicy_probs = Input(shape=(1, self.num_actions))
        advantages = Input(shape=(1, 1,))
        rewards = Input(shape=(1, 1,))
        values = Input(shape=(1, 1,))
        x = Flatten()(self.image_preprocessor)
        x = Activation('relu')(x)
        x = Dense(256, activation='relu', kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(x)
        output = Dense(self.num_actions, activation='softmax', name='predictions')(x)
        model = Model(inputs=[self.image_preprocessor_input, oldpolicy_probs, advantages, rewards, values],
                      outputs=[output])
        model.compile(Adam(ACTOR_LERARNING_RATE), loss=[self.ppo_loss(
            oldpolicy_probs=oldpolicy_probs,
            advantages=advantages,
            rewards=rewards,
            values=values)])
        model.summary()
        return model

    def ppo_loss(self, oldpolicy_probs, advantages, rewards, values):
        """
            PPO loss from paper
        """
        def loss(y_true, y_pred):
            newpolicy_probs = y_pred
            ratio = K.exp(K.log(newpolicy_probs + 1e-10) - K.log(oldpolicy_probs + 1e-10))
            p1 = ratio * advantages
            p2 = K.clip(ratio, min_value=1 - CLIPPING, max_value=1 + CLIPPING) * advantages
            actor_loss = -K.mean(K.minimum(p1, p2))
            critic_loss = K.mean(K.square(rewards - values))
            total_loss = CRITIC_DISCOUNT * critic_loss + actor_loss - ENTROPY_COEF * K.mean(
                -(newpolicy_probs * K.log(newpolicy_probs + 1e-10)))
            return total_loss

        return loss

    def load(self, folder, end):
        self.actor.load_weights(f'models/{folder}/model_actor_{end}')
        self.critic.load_weights(f'models/{folder}/model_critic_{end}')

    def save(self, iters, avg_reward, last_goals, name):
        self.actor.save(
            'models/{}/model_actor_{}_{}_{}.hdf5'.format(name, iters, avg_reward,
                                                         last_goals))
        self.critic.save(
            'models/{}/model_critic_{}_{}_{}.hdf5'.format(name, iters, avg_reward,
                                                          last_goals))
