# -*- coding: utf-8 -*-
# @Author: harshit
# @Date:   2019-04-28 08:11:55
# @Last Modified by:   harshit
# @Last Modified time: 2019-04-30 19:40:12

from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import Adam
from VARS import *


def NN1():
    # Network with Input
    #   - agent pos
    #   - all walls heights and pos
    #   - one hot encoded current action
    inputs = Input((NUM_FEATURES,))
    l0 = Dense(
        512, activation='relu', kernel_initializer="random_uniform")(inputs)
    l1 = Dense(
        512, activation='relu', kernel_initializer="random_uniform")(l0)
    output = Dense(1, activation='linear')(l1)

    model = Model(inputs=[inputs], outputs=[output])
    adm = Adam(lr=0.03)
    model.compile(optimizer=adm, loss='mean_squared_error', metrics=['mse'])
    model.summary()
    return model


def NN2():
    # Network with Input
    #   - agent pos
    #   - all walls heights and pos
    #   - one hot encoded current action
    inputs = Input((NUM_FEATURES,))
    l0 = Dense(
        256, activation='relu', kernel_initializer="random_uniform")(inputs)
    l1 = Dense(
        512, activation='relu', kernel_initializer="random_uniform")(l0)
    l2 = Dense(
        1024, activation='relu', kernel_initializer="random_uniform")(l1)
    l3 = Dense(
        2048, activation='relu', kernel_initializer="random_uniform")(l2)
    l4 = Dense(
        1024, activation='relu', kernel_initializer="random_uniform")(l3)
    output = Dense(1, activation='linear')(l4)

    model = Model(inputs=[inputs], outputs=[output])
    adm = Adam(lr=0.03)
    model.compile(optimizer=adm, loss='mean_squared_error', metrics=['mse'])
    model.summary()
    return model


def NN3():
    # Network with Input
    #   - agent pos
    #   - all walls heights and pos
    #   - one hot encoded current action
    inputs = Input((NUM_FEATURES,))
    l0 = Dense(
        1024, activation='relu', kernel_initializer="random_uniform")(inputs)
    output = Dense(1, activation='linear')(l0)

    model = Model(inputs=[inputs], outputs=[output])
    adm = Adam(lr=0.01)
    model.compile(optimizer=adm, loss='mean_squared_error', metrics=['mse'])
    model.summary()
    return model
