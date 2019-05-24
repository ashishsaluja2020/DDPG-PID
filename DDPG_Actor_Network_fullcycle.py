#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np
import math
from keras.initializers import normal, identity
from keras.models import model_from_json
from keras.models import Sequential, Model
# from keras.engine.training import collect_trainable_weights
from keras.layers import Dense, Flatten, Input, merge, Lambda
from keras.optimizers import Adam
import tensorflow as tensor
import keras.backend as K

HIDDEN1_UNITS = 200
HIDDEN2_UNITS = 200
HIDDEN3_UNITS  = 10


class ActorNetwork(object):
    
    def __init__(self, sess, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE

        K.set_session(sess)
        self.model, self.weights, self.state = self.create_actor_network(state_size, action_size)
        self.target_model, self.target_weights, self.target_state = self.create_actor_network(state_size, action_size)
        self.action_gradient = tensor.placeholder(tensor.float32, [None , action_size])
        self.params_grad = tensor.gradients(self.model.output, self.weights,-self.action_gradient)
        grads = zip(self.params_grad, self.weights)
        self.optimize = tensor.train.AdamOptimizer(LEARNING_RATE).apply_gradients(grads)
        self.sess.run(tensor.global_variables_initializer())
        
    def train(self, states, action_grads):
        self.sess.run(self.optimize, feed_dict = {self.state :states, self.action_gradient : action_grads})
        
        
    def target_train(self):
        actor_weights = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()
        for i in range(len(actor_weights)):
            actor_target_weights[i] = self.TAU * actor_weights[i] + (1 - self.TAU)* actor_target_weights[i]
        self.target_model.set_weights(actor_target_weights)
        
    def create_actor_network(self, state_size, action_size):
        S = Input(shape = (state_size,), name = "actor_input")
        h0 = Dense(HIDDEN1_UNITS, activation='relu')(S)
        h1 = Dense(HIDDEN2_UNITS, activation='relu')(h0)
        h2  =Dense(HIDDEN3_UNITS, activation = 'relu')(h1)
        V = Dense(action_size,activation='relu')(h2)
        model = Model(input = S, output = V, name  = "actor_output")
        return model,model.trainable_weights, S


# In[ ]:





# In[ ]:




