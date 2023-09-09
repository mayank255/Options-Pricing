# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 20:50:14 2023

@author: Mayank Srivastav
"""


import numpy as np
import pandas as pd

import scipy.stats as stats
import matplotlib.pyplot as plt
from numpy import random

def exchange_option_mc(S1_0, S2_0, T, r, q1, q2, sigma1, sigma2, rho, Q1, Q2, n_simulations=100000):
    
    z = np.random.randn(n_simulations, 2)
    z[:, 1] = rho*z[:, 0] + np.sqrt(1-rho**2)*z[:, 1]  # Cholesky decomposition for correlated Brownian motions

    # Simulate asset prices using geometric Brownian motion
    S1_T = S1_0 * np.exp((r - q1 - 0.5 * sigma1**2) * T + sigma1 * np.sqrt(T) * z[:, 0])
    S2_T = S2_0 * np.exp((r - q2 - 0.5 * sigma2**2) * T + sigma2 * np.sqrt(T) * z[:, 1])
    
    # Calculate the payoff for each simulation
    payoffs = np.maximum(Q1*S1_T - Q2*S2_T, 0)
    
    # Calculate the option price
    option_price = np.exp(-r * T) * np.mean(payoffs)
    
    return option_price
def margrabe_option(S1_0, S2_0, T, q1, q2, sigma1, sigma2, rho, Q1, Q2):
    sigma = np.sqrt((sigma1)**2 + (sigma2)**2 -2*rho*sigma1*sigma2)
    
    d1 = (np.log(Q1*S1_0 / (Q2*S2_0)) + (q2 - q1 + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    # print(stats.norm.cdf(d1),stats.norm.cdf(d2))
    V = Q1 * S1_0 * np.exp(-q1 * T) * stats.norm.cdf(d1) - Q2 * S2_0 * np.exp(-q2 * T) * stats.norm.cdf(d2)
    return V
def simulate_multiple_times(N_values, args,repetitions=100):
    results = {'N': [], 'Mean Option Price': [], 'Std Dev Option Price': []}

    # Parameters (assuming some values; adjust as needed)
    print(args)
    S1_0, S2_0, T, r, q1, q2, sigma1, sigma2, rho, Q1, Q2 = args
    
    
    for N in N_values:
        
        option_prices = []
        for rep in range(repetitions):
            option_price= exchange_option_mc(S1_0, S2_0, T, r, q1, q2, sigma1, sigma2, rho, Q1, Q2, N)
            option_prices.append(option_price)
        
        mean_option_price = np.mean(option_prices)
        std_option_price = np.std(option_prices)
        
        results['N'].append(N)
        results['Mean Option Price'].append(mean_option_price)
        results['Std Dev Option Price'].append(std_option_price)
        
    df = pd.DataFrame(results)
    return df
def convergence_study(simulation_sizes, *args, save_path="convergence_study.png"):
    prices = []
    conf_intervals = []
    
    for sim_size in simulation_sizes:
        price, _, interval = exchange_option_mc(*args, n_simulations=sim_size)
        prices.append(price)
        conf_intervals.append(interval)
    
    lower_bounds = [interval[0] for interval in conf_intervals]
    upper_bounds = [interval[1] for interval in conf_intervals]
    
    plt.figure(figsize=(10,6))
    plt.plot(simulation_sizes, prices, '-o', label='Option Price', color='blue')
    plt.fill_between(simulation_sizes, lower_bounds, upper_bounds, color='skyblue', alpha=0.4, label='90% Confidence Interval')
    plt.xscale('log')  # Set x-axis to log scale
    plt.xlabel('Number of Simulations (log scale)')
    plt.ylabel('Option Price')
    plt.title('Convergence Study with 90% Confidence Interval')
    plt.legend()
    plt.grid(True, which="both", ls="--", c='0.65')
    
    # Save the figure at high resolution
    plt.savefig("convergence.png", dpi=300)
    
    plt.show()
# # Example:
# args = [400, 200, 1, 0.05, 0, 0, 0.4, 0.6, 0.5, 0.1, 1]
# S1_0, S2_0, T, r, q1, q2, sigma1, sigma2, rho, Q1, Q2 = args
# exchange_option_price = exchange_option_mc(S1_0, S2_0, T, r, q1, q2, sigma1, sigma2, rho, Q1, Q2)
# print(f"Exchange Option Price (Monte Carlo): {exchange_option_price}")



# # Same parameters as before
# # S1_0, S2_0, T, r, q1, q2, sigma1, sigma2, rho, Q1, Q2 = 400, 50, 1, 0.05, 0, 0, 0.4, 0.6, 0.5, 0.5, 1

# exchange_option_price_analytical = margrabe_option(S1_0, S2_0, T, q1, q2, sigma1, sigma2, rho, Q1, Q2)
# # exchange_option_price_analytical1 = margrabe_optionv1(S1_0, S2_0, T, q1, q2, sigma1, sigma2, rho, Q1, Q2)
# print(f"Exchange Option Price (Margrabe's Formula): {exchange_option_price_analytical}")
# # print(f"Exchange Option Price (Margrabe's 1 Formula): {exchange_option_price_analytical1}")


# sigmas = np.linspace(0.1, 1, 10)
# mc_prices_sigma1 = []
# analytical_prices_sigma1 = []
# mc_prices_sigma2 = []
# analytical_prices_sigma2 = []
# for sigma in sigmas:
#     monte_carlo_price = exchange_option_mc(S1_0, S2_0, T, r, q1, q2, sigma, sigma2, rho, Q1, Q2)
#     analytical_price = margrabe_option(S1_0, S2_0, T, q1, q2, sigma, sigma2, rho, Q1, Q2)
#     mc_prices_sigma1.append(monte_carlo_price)
#     analytical_prices_sigma1.append(analytical_price)

# for sigma in sigmas:
#     monte_carlo_price = exchange_option_mc(S1_0, S2_0, T, r, q1, q2, sigma1, sigma, rho, Q1, Q2)
#     analytical_price = margrabe_option(S1_0, S2_0, T, q1, q2, sigma1, sigma, rho, Q1, Q2)
#     mc_prices_sigma2.append(monte_carlo_price)
#     analytical_prices_sigma2.append(analytical_price)
# plt.figure()
# plt.plot(sigmas, mc_prices_sigma1, '-o', label='MC Price (Sigma1 varied)')
# plt.plot(sigmas, analytical_prices_sigma1, '-o', label='Analytical Price (Sigma1 varied)')
# plt.xlabel('Sigma1')
# plt.ylabel('Option Price')
# plt.legend()
# plt.title('Option Price vs Sigma1')
# plt.grid(True)
# plt.show()
# plt.figure()
# plt.plot(sigmas, mc_prices_sigma2, '-o', label='MC Price (Sigma2 varied)')
# plt.plot(sigmas, analytical_prices_sigma2, '-o', label='Analytical Price (Sigma2 varied)')
# plt.xlabel('Sigma2')
# plt.ylabel('Option Price')
# plt.legend()
# plt.title('Option Price vs Sigma2')
# plt.grid(True)
# plt.show()
# correlations = np.linspace(-1, 1, 20)  # Range of correlations from -1 to 1
# mc_prices_correlation = []
# analytical_prices_correlation = []

# for current_rho in correlations:
#     monte_carlo_price = exchange_option_mc(S1_0, S2_0, T, r, q1, q2, sigma1, sigma2, current_rho, Q1, Q2)
#     analytical_price = margrabe_option(S1_0, S2_0, T, q1, q2, sigma1, sigma2, current_rho, Q1, Q2)
#     mc_prices_correlation.append(monte_carlo_price)
#     analytical_prices_correlation.append(analytical_price)

# plt.figure()
# plt.plot(correlations, mc_prices_correlation, '-o', label='MC Price (Correlation varied)')
# plt.plot(correlations, analytical_prices_correlation, '-o', label='Analytical Price (Correlation varied)')
# plt.xlabel('Correlation')
# plt.ylabel('Option Price')
# plt.legend()
# plt.title('Option Price vs Correlation')
# plt.grid(True)
# plt.show()
# S1_values = np.linspace(S2_0*Q2/Q1, 2*S2_0*Q2/Q1, 20)  # Example range, adjust as needed
# differences = []
# mc_prices_difference = []
# analytical_prices_difference = []

# for current_S1 in S1_values:
#     difference = Q1 * current_S1 - Q2 * S2_0
#     monte_carlo_price = exchange_option_mc(current_S1, S2_0, T, r, q1, q2, sigma1, sigma2, rho, Q1, Q2)
#     analytical_price = margrabe_option(current_S1, S2_0, T, q1, q2, sigma1, sigma2, rho, Q1, Q2)
#     differences.append(difference)
#     mc_prices_difference.append(monte_carlo_price)
#     analytical_prices_difference.append(analytical_price)

# plt.figure()
# plt.plot(differences, mc_prices_difference, '-o', label='MC Price (Q1S1 - Q2S2 varied)')
# plt.plot(differences, analytical_prices_difference, '-o', label='Analytical Price (Q1S1 - Q2S2 varied)')
# plt.xlabel('Q1S1 - Q2S2')
# plt.ylabel('Option Price')
# plt.legend()
# plt.title('Option Price vs Q1S1 - Q2S2')
# plt.grid(True)
# plt.show()


# # ... [rest of the code remains the same] ...

# # Convergence study with 90% CI marks and log x-axis


# # Define the parameters


# # Given exchange_option_mc function ...



# N_values = [10, 100, 1000, 10000]
# df_results = simulate_multiple_times(N_values)
# df_results['Standard Error'] = df_results['Std Dev Option Price'] / np.sqrt(100)

# # Compute 95% confidence intervals
# z = stats.norm.ppf(0.975)  # z-value for 95% CI
# df_results['CI Lower'] = df_results['Mean Option Price'] - z * df_results['Standard Error']
# df_results['CI Upper'] = df_results['Mean Option Price'] + z * df_results['Standard Error']

# # Plot
# plt.figure(figsize=(10, 6))

# # Plotting the line
# plt.plot(df_results['N'], df_results['Mean Option Price'], '-o', color='blue', label='Mean Option Price')

# # Adding the error bars (standard error candles)
# plt.errorbar(df_results['N'], df_results['Mean Option Price'], yerr=df_results['Standard Error'], fmt='o', color='red', ecolor='red', capsize=5, label='Standard Error')

# # Adding the 95% confidence intervals
# plt.fill_between(df_results['N'], df_results['CI Lower'], df_results['CI Upper'], color='yellow', alpha=0.2, label='95% Confidence Interval')

# plt.xscale('log')  # Using a logarithmic x-axis to better distinguish the different N values
# plt.xlabel('N values')
# plt.ylabel('Option Price')
# plt.title('Mean Option Price with Standard Error and 95% CI for Different N values')
# plt.legend()
# plt.tight_layout()
# plt.grid(axis='y')
# plt.show()
