#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 11:52:26 2020

@author: gtsal
"""
import numpy as np
from scipy.optimize import minimize, rosen

#weights of fitness function
w_f_OUT = 2000
w_f_OVERLAP = 2000
w_f_ATTR = 0.1
w_f_SMO = 500
w_f_DIST = 1

# %% Fitness Function 

def ObjectiveFcn(particle,nVars,Stock,Order):

    f=0    
   
    
    # The overall fitness function is obtained by combining the above criteria
    f = (2*w_f_OUT) + (3*w_f_OVERLAP) + (4*w_f_DIST) + (5*w_f_ATTR) + (6*w_f_SMO)
   # print(f)
    return f
# %%
class Simulator:
    def __init__(self, function):
        self.f = function # actual objective function
        self.num_calls = 0 # how many times f has been called
        self.callback_count = 0 # number of times callback has been called, also measures iteration count
        self.list_calls_inp = [] # input of all calls
        self.list_calls_res = [] # result of all calls
        self.decreasing_list_calls_inp = [] # input of calls that resulted in decrease
        self.decreasing_list_calls_res = [] # result of calls that resulted in decrease
        self.list_callback_inp = [] # only appends inputs on callback, as such they correspond to the iterations
        self.list_callback_res = [] # only appends results on callback, as such they correspond to the iterations
    
    def simulate(self, x):
        """Executes the actual simulation and returns the result, while
        updating the lists too. Pass to optimizer without arguments or
        parentheses."""
        result = self.f(x) # the actual evaluation of the function
        if not self.num_calls: # first call is stored in all lists
            self.decreasing_list_calls_inp.append(x)
            self.decreasing_list_calls_res.append(result)
            self.list_callback_inp.append(x)
            self.list_callback_res.append(result)
        elif result < self.decreasing_list_calls_res[-1]:
            self.decreasing_list_calls_inp.append(x)
            self.decreasing_list_calls_res.append(result)
        self.list_calls_inp.append(x)
        self.list_calls_res.append(result)
        self.num_calls += 1
        return result
    
    def callback(self, xk, *_):
        """Callback function that can be used by optimizers of scipy.optimize.
        The third argument "*_" makes sure that it still works when the
        optimizer calls the callback function with more than one argument. Pass
        to optimizer without arguments or parentheses."""
        s1 = ""
        xk = np.atleast_1d(xk)
        # search backwards in input list for input corresponding to xk
        for i, x in reversed(list(enumerate(self.list_calls_inp))):
            x = np.atleast_1d(x)
            if np.allclose(x, xk):
                break
    
        for comp in xk:
            s1 += f"{comp:10.5e}\t"
        s1 += f"{self.list_calls_res[i]:10.5e}"
    
        self.list_callback_inp.append(xk)
        self.list_callback_res.append(self.list_calls_res[i])
    
        if not self.callback_count:
            s0 = ""
            for j, _ in enumerate(xk):
                tmp = f"Comp-{j+1}"
                s0 += f"{tmp:10s}\t"
            s0 += "Objective"
            print(s0)
        print(s1)
        self.callback_count += 1
        
        
ros_sim = Simulator(ObjectiveFcn(1,1,2,3))
minimize(ros_sim.simulate, [0, 0], method='BFGS', callback=ros_sim.callback, options={"disp": True})

print(f"Number of calls to Simulator instance {ros_sim.num_calls}")