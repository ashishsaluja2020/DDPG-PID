import math
import scipy
import random
import gym
import numpy as np
import control
import matplotlib.pyplot as plt
import tensorflow as tf
import keras.backend as K
from collections import deque
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Lambda, concatenate
from keras.optimizers import Adam
from keras import regularizers


# class DDPG(St, St_next):
#     def __init__(self):
#         self.sess = K.get_session()
#         # update rate for target model
#         self.TAU = 0.01
#         # experience replay 
#         self.memory_buffer = deque(maxlen=50000)
#         # discount rate for q value
#         self.gamma = 0.95
#         # epislon of action selection
#         self.epsilon = 1.0
#         # discount rate for epislon
#         self.epsilon_decay = 0.995
#         # min epislon of e-greedy
#         self.epsilon_min = 0.01
#         # actor learning rate
#         self.actor_lr = 0.0001
#         # critic learning rate
#         self.critic_lr = 0.0001
        
#         # DDPG model
#         self.actor = self._build_actor()
#         self.critic = self._build_critic()

#         # DDPG target model
#         self.target_actor = self._build_actor()
#         self.target_actor.set_weights(self.actor.get_weights())
#         self.target_critic = self._build_critic()
#         self.target_critic.set_weights(self.critic.get_weights())

#         # Gradient function
#         self.get_critic_grad = self.critic_gradient()
#         self.actor_optimizer()

#         def _build_actor(self):
#             inputs = Input(shape=(6,), name='state_input')
#             x = Dense(200, activation='relu6', kernel_regularizers=regularizers.l2(0.01))(inputs)
#             x = Dense(200, activation='relu6', kernel_regularizers=regularizers.l2(0.01))(x)
#             output = Dense(10, activation='relu', kernel_regularizers=regularizers.l2(0.01))(x)

#             model = Model(inputs=inputs, outputs=output)
#             model.compile(loss='mse', optimizer=Adam(lr=self.actor_lr))
#             return model
        
#         def _build_critic(self):
#             inputs = Input(shape=(6,), name='state_input')
#             x = Dense(200, activation='relu6', kernel_regularizers=regularizers.l2(0.01))(inputs)
#             x = Dense(200, activation='relu6', kernel_regularizers=regularizers.l2(0.01))(x)
#             output = Dense(10, activation='relu', kernel_regularizers=regularizers.l2(0.01))(x)

#             model = Model(inputs=inputs, outputs=output)
#             model.compile(loss='mse', optimizer=Adam(lr=self.critic_lr))
#             return model





class environment():
    def reset(self):
        self.kp = 0
        self.ki = 0
        self.kd = 0
        self.curIter = 0

    def env(self, kp, ki, kd, curIter, sys_sel):
        self.kp = kp
        self.ki = ki
        # current iteration: 0~100(default setting by scipy.signal.step())
        self.curIter = curIter
        self.sys_sel = sys_sel
        # variable: Jl & Kt 
        # it varies according to different systems (1~3)
        if self.sys_sel == 1:
            Jl = 2.5*1e-3
            Kt = 1
        elif self.sys_sel == 2:
            Jl = 6.3*1e-3
            Kt = 1
        elif self.sys_sel == 3:
            Jl = 2.5*1e-3
            Kt = 3
        else:
            Jl = 0
            Kt = 0
            print('\tError! system 1~3 is available for now!')
            quit()
        # constant
        Jm = 6.329*1e-4
        Tsigma = 1.1*1e-5
        Tsigman = 1.25*1e-4
        B = 3.0*1e-4
        Ksf = (60*1)/(2500*5*1e-3)
        amplitude = 2000
        # sample period: 100us
        T_period = []
        T_ini = 0
        for _ in range(5000):
            T_period.append(T_ini)
            T_ini += 100*1e-6
            
        '''
        speed loop
        Yspeed(s)/W(s) = B1*kd*s^2 + B1*kp*s + B1*ki / A1*s^3 + (B1*kd+A2)*s^2 + (A3+B1*kp)*s + B1*ki
        '''
        A1 = (Jm + Jl) * Tsigman
        A2 = B * Tsigman + (Jm + Jl)
        A3 = B
        B1 = Kt * Ksf
        # speed loop feedback lti system 
        speed_fb_lti = scipy.signal.lti([B1*self.kd, B1*self.kp, B1*self.ki], [A1, A2+B1*kd, A3+B1*self.kp, B1*self.ki])
        # Angular velocity: Xt
        t1, Xt = scipy.signal.step(speed_fb_lti, T=T_period)
        # Amplitute: 2000
        Xt = amplitude * Xt
        '''
        step response
        Y(s)/e(t) = 1
        '''
        # step response feedback lti system
        step_fb_lti = scipy.signal.lti(1, 1)
        # Input signal: Ht
        t2, Ht = scipy.signal.step(step_fb_lti, T=T_period)
        # Amplitute: 2000
        Ht = amplitude * Ht
        # Tracking error of the output angle of the servo system
        Et = amplitude - Xt

        # current time spot of iteration
        max_len = len(Xt)
        # Integral of absolute error: IAE
        # iae = 0
        # if self.curIter == 0:
        #     iae = 0
        # else:
        #     iae = abs(absolute_Et[self.curIter] - absolute_Et[self.curIter - 1]) * 1
        if self.curIter == 0:
            Xt_last = Xt[0]
            Et_last = Et[0]
            Xt_cur = Xt[self.curIter]
            Et_cur = Et[self.curIter]
            Xt_next = Xt[self.curIter + 1]
            Ht_next = Ht[self.curIter + 1]
            Et_next = Et[self.curIter + 1]
            Ht_a2t = Ht[self.curIter + 2]
            Ht_a3t = Ht[self.curIter + 3]
        elif self.curIter > max_len - 4:
            Xt_last = Xt[self.curIter - 1]
            Et_last = Et[self.curIter - 1]
            Xt_cur = Xt[self.curIter]
            Et_cur = Et[self.curIter]
            Xt_next = Xt[max_len - 1]
            Ht_next = Ht[max_len -1]
            Et_next = Et[max_len -1]
            Ht_a2t = Ht[max_len -1]
            Ht_a3t = Ht[max_len -1]
        else:
            Xt_last = Xt[self.curIter - 1]
            Et_last = Et[self.curIter - 1]
            Xt_cur = Xt[self.curIter]
            Et_cur = Et[self.curIter]
            Xt_next = Xt[self.curIter + 1]
            Ht_next = Ht[self.curIter + 1]
            Et_next = Et[self.curIter + 1]
            Ht_a2t = Ht[self.curIter + 2]
            Ht_a3t = Ht[self.curIter + 3]
        St = [Xt_last, Et_last, Xt_cur, Et_cur, Ht_next, Ht_a2t]
        St_next = [Xt_cur, Et_cur, Xt_next, Et_next, Ht_a2t, Ht_a3t]
       
        # plt.plot(t1, Xt)
        # plt.xlabel('Time(s)')
        # plt.ylabel('Angular velocity')
        # plt.savefig('Angular velocity.jpg')
        # plt.show()
        # plt.plot(t2, Ht)
        # plt.xlabel('Time(s)')
        # plt.ylabel('Tracking signal')
        # plt.savefig('Tracking signal.jpg')
        # plt.show()

        # return St, S(t+1), reward, t1: for plot usage
        return St, St_next, t1
        
environment = environment()
environment.reset()
episodes = 5000
System_selection =[]
Iterations = []
# Initial system status
sys_sel = 1
for iter in range(episodes):
    Iterations.append(iter)
    # sysem disturbance changes every 0.01s
    if iter < 750:
        sys_sel = 1
    elif iter >3750:
        sys_sel = 1
    else:
        if iter % 100 == 0:
            sys_sel = random.randint(1,3)
        else:
            sys_sel = sys_sel
    System_selection.append(sys_sel)
    # Adjust kp & ki from RL network
    St, St_next, t= environment.env(0.5, 2.985, 0.01, iter, sys_sel)
    print(iter, St, St_next, sys_sel)
plt.plot(t, System_selection)
plt.xlabel('Time(s)')
plt.ylabel('System status')
plt.savefig('System status.jpg')
plt.show()