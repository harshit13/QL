# -*- coding: utf-8 -*-
# @Author: harshit
# @Date:   2019-04-28 08:11:55
# @Last Modified by:   harshit
# @Last Modified time: 2019-04-28 10:10:29

from keras.layers import Dense, Input
from keras.models import Model

# 1 Agent y position
# 2-9 Walls x position and height
# 10-11 one hot encoded action #2
NUM_FEATURES = 11


def NN():
    # Network with Input
    #   - agent pos
    #   - all walls heights and pos
    #   - one hot encoded current action
    inputs = Input((NUM_FEATURES,))
    l0 = Dense(
        512, activation='relu', kernel_initializer="random_uniform")(inputs)
    """l1 = Dense(
        1024, activation='relu', kernel_initializer="random_uniform")(l0)"""
    l2 = Dense(
        512, activation='relu', kernel_initializer="random_uniform")(l0)
    output = Dense(1, activation='relu')(l2)

    model = Model(inputs=[inputs], outputs=[output])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])
    model.summary()
    return model
