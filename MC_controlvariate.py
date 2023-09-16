import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.linalg import cholesky

# Black-Scholes European call option price
def black_scholes(S0, K, T, r, q, sigma):
    d1 = (np.log(S0/K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

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
# Basket option Monte Carlo pricing with control variate

def basket_option_mc_cv(S0, T, r, q, sigma, rho_matrix, K, weights, M):
    # control variate parameter
    S0_y = [110, 100, 107, 103]
    T_y = 1
    r_y = 0.05
    q_y = [0.02, 0.04, 0.03, 0.06]
    sigma_y = [0.2, 0.25, 0.15, 0.22]
    rho_matrix_y = np.array([[1, 0.75, 0.60, 0.50], 
                          [0.75, 1, 0.68, 0.55], 
                          [0.60, 0.68, 1, 0.58], 
                          [0.50, 0.55, 0.58, 1]])
    weights_y = [0.25,0.25,0.15,0.35]
    K_y = 100
    expected_cv = basket_option_mc(S0, T, r, q, sigma, rho_matrix, K, weights, n_simulations=1000000)
    
    N = len(S0)
    dt = T
    discount = np.exp(-r * T)
    
    # Cholesky decomposition for correlated Brownian motion
    L = cholesky(rho_matrix, lower=True)
    
    # Monte Carlo simulation
    increments = np.random.normal(0, 1, (M, N))
    correlated_increments = np.dot(increments, L.T)
    
    stock_paths = np.zeros((M, N))
    stock_paths_y = np.zeros((M, N))
    for i in range(N):
        stock_paths[:, i] = S0[i] * np.exp((r - q[i] - 0.5 * sigma[i]**2) * dt 
                                         + sigma[i] * np.sqrt(dt) * correlated_increments[:, i])
        
    basket_terminal_values = np.dot(stock_paths, weights)
    
    # # Control variate - Using a geometric average option as control
    for i in range(N):
        stock_paths_y[:, i] = S0_y[i] * np.exp((r_y - q_y[i] - 0.5 * sigma_y[i]**2) * dt 
                                         + sigma_y[i] * np.sqrt(dt) * correlated_increments[:, i])
        
    basket_terminal_values_y = np.dot(stock_paths, weights)
    # geo_ave_strike = np.exp(np.sum(np.log(S0) * weights))
    
    
    # bs_geo_ave_price = black_scholes(geo_ave_strike, K, T, r, np.mean(q), sigma[0])  # Assuming same sigma for simplicity
    
    # Control variate method
    covXY = np.cov(basket_terminal_values - K, basket_terminal_values_y - K)[0][1]
    varY = np.var(basket_terminal_values_y - K)
    c = -covXY / varY
    
    basket_option_prices = np.maximum(basket_terminal_values - K, 0)
    control_variate_adjustment = c * (np.maximum(basket_terminal_values_y - K, 0) - expected_cv)
    basket_option_prices_adjusted = basket_option_prices + control_variate_adjustment
    
    return discount * np.mean(basket_option_prices_adjusted)

def compute_std_errors(S0, T, r, q, sigma, rho_matrix, K, weights, n_simulations=100000, repetitions=100):
    mc_prices = [basket_option_mc(S0, T, r, q, sigma, rho_matrix, K, weights, n_simulations) for _ in range(repetitions)]
    mc_cv = [basket_option_mc_cv(S0, T, r, q, sigma, rho_matrix, K, weights, n_simulations) for _ in range(repetitions)]

    mc_std_error = np.std(mc_prices, ddof=1) / np.sqrt(repetitions)
    mc_cv_std_error = np.std(mc_cv, ddof=1) / np.sqrt(repetitions)

    return mc_std_error, mc_cv_std_error
# Given parameters



# price = basket_option_mc_cv(S0, K, T, r, q, sigma, rho_matrix, weights, M)
# print("Basket option price using Control Variate:", price)
