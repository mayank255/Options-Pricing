# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 16:36:40 2023

@author: Mayank Srivastav
"""
import numpy as np
from basketoptions import *
import matplotlib.pyplot as plt
def basket_option_mc_antithetic(S0, T, r, q, sigma, rho_matrix, K, weights, n_simulations=100000):
    n_assets = len(S0)
    
    # Cholesky decomposition
    L = np.linalg.cholesky(rho_matrix)
    
    # Monte Carlo simulation (generate half, as we will use antithetic variates)
    Z = np.random.randn(n_simulations // 2, n_assets)
    correlated_Z = Z @ L.T
    antithetic_Z = -correlated_Z
    
    # Initialize empty arrays for asset prices
    S = np.zeros((n_simulations, n_assets))
    
    for i in range(n_assets):
        # Simulations from the original random variables
        S[:n_simulations//2, i] = S0[i] * np.exp((r - q[i] - 0.5 * sigma[i]**2) * T + sigma[i] * np.sqrt(T) * correlated_Z[:, i])
        # Simulations from the antithetic random variables
        S[n_simulations//2:, i] = S0[i] * np.exp((r - q[i] - 0.5 * sigma[i]**2) * T + sigma[i] * np.sqrt(T) * antithetic_Z[:, i])
    
    # Calculate the weighted sum for each simulation
    weighted_sum = np.dot(S, weights)
    
    # Calculate the payoff for each simulation
    payoffs = np.maximum(weighted_sum - K, 0)
    
    return np.mean(payoffs)
def compute_std_errors(S0, T, r, q, sigma, rho_matrix, K, weights, n_simulations=100000, repetitions=100):
    mc_prices = [basket_option_mc(S0, T, r, q, sigma, rho_matrix, K, weights, n_simulations) for _ in range(repetitions)]
    mc_antithetic_prices = [basket_option_mc_antithetic(S0, T, r, q, sigma, rho_matrix, K, weights, n_simulations) for _ in range(repetitions)]

    mc_std_error = np.std(mc_prices, ddof=1) / np.sqrt(repetitions)
    mc_antithetic_std_error = np.std(mc_antithetic_prices, ddof=1) / np.sqrt(repetitions)

    return mc_std_error, mc_antithetic_std_error

    