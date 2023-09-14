# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 19:05:31 2023

@author: Mayank Srivastav
"""

import numpy as np
import pandas as pd

def basket_option_mc(S0, T, r, q, sigma, rho_matrix, K, weights, n_simulations=100000):
    n_assets = len(S0)
    
    # Cholesky decomposition
    L = np.linalg.cholesky(rho_matrix)
    
    # Monte Carlo simulation
    Z = np.random.randn(n_simulations, n_assets)
    correlated_Z = Z @ L.T
    
    # Initialize empty arrays for asset prices
    S = np.zeros((n_simulations, n_assets))
    
    for i in range(n_assets):
        S[:, i] = S0[i] * np.exp((r - q[i] - 0.5 * sigma[i]**2) * T + sigma[i] * np.sqrt(T) * correlated_Z[:, i])
    
    # Calculate the weighted sum for each simulation
    weighted_sum = np.dot(S, weights)
    
    # Calculate the payoff for each simulation
    payoffs = np.maximum(weighted_sum - K, 0)
    
    return np.mean(payoffs)

def simulate_multiple_times_basket(N_values, params,repetitions=100):
    results = {'N': [], 'Mean Option Price': [], 'Std Dev Option Price': []}

    # Parameters (assuming some values; adjust as needed)
    S0, T, r, q, sigma, rho_matrix, K, weights = params
    
    
    for N in N_values:
        
        option_prices = []
        for rep in range(repetitions):
            option_price= basket_option_mc(S0, T, r, q, sigma, rho_matrix, K,weights, n_simulations=N)
            option_prices.append(option_price)
        
        mean_option_price = np.mean(option_prices)
        std_option_price = np.std(option_prices)
        
        results['N'].append(N)
        results['Mean Option Price'].append(mean_option_price)
        results['Std Dev Option Price'].append(std_option_price)
        
    df = pd.DataFrame(results)
    return df
