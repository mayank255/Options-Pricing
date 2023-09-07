# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 10:37:34 2023

@author: Mayank Srivastav
"""

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# S : Spot price list
# M : number of steps
# T : Time to Expiry
# sigma : volatility
# r : risk free rate of return
# dS =  rdt+sigma*dX
r = 0.0
sigma = 0.4

S0 = 100
M =100
T =2
K = 90

def random_walk(S0,M,T,r,phi):
    S = [S0]
    
    dt = T/M
    for i in range(M):
        x = S[-1]*(1+r*dt+sigma*phi*np.sqrt(dt))
        # print(S[-1],x)
        S.append(x)

    return S
for i in range(10):
    phi = np.random.normal(0,1)
    print(f"iteration {i}")
    S = random_walk(S0,M,T,r,phi)
    print(S)
    plt.plot(S)
    # plt.axis([0,100,0,5000])

plt.show()

