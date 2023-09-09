# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 02:06:13 2023

@author: Mayank Srivastav
"""

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

def exchange_option_mc(S1_0, S2_0, T, r, q1, q2, sigma1, sigma2, rho, Q1, Q2, n_simulations=100000):
    np.random.seed(42)
    z = np.random.randn(n_simulations, 2)
    z[:, 1] = rho*z[:, 0] + np.sqrt(1-rho**2)*z[:, 1]  # Cholesky decomposition for correlated Brownian motions

    # Simulate asset prices using geometric Brownian motion
    S1_T = S1_0 * np.exp((r - q1 - 0.5 * sigma1**2) * T + sigma1 * np.sqrt(T) * z[:, 0])
    S2_T = S2_0 * np.exp((r - q2 - 0.5 * sigma2**2) * T + sigma2 * np.sqrt(T) * z[:, 1])
    
    # Calculate the payoff for each simulation
    payoffs = np.maximum(Q1*S1_T - Q2*S2_T, 0)
    
    # Calculate the option price
    option_price = np.exp(-r * T) * np.mean(payoffs)
    
    # Calculate the variance and confidence interval
    variance = np.var(payoffs)
    conf_interval = stats.norm.interval(0.95, loc=option_price, scale=np.sqrt(variance/n_simulations))
    
    return option_price, variance, conf_interval


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

# ... [rest of the code remains the same] ...

# Convergence study with 90% CI marks and log x-axis


# Define the parameters
S1_0, S2_0, T, r, q1, q2, sigma1, sigma2, rho, Q1, Q2 = 100, 105, 1, 0.05, 0.02, 0.03, 0.2, 0.25, 0.5, 1, 1
params = (S1_0, S2_0, T, r, q1, q2, sigma1, sigma2, rho, Q1, Q2)

# Get option price, variance, and confidence interval
price, variance, conf_interval = exchange_option_mc(*params)
print(f"Exchange Option Price (Monte Carlo): {price}")
print(f"Variance: {variance}")
print(f"95% Confidence Interval: {conf_interval}")

# Convergence study
convergence_study([100, 1000, 10000, 100000, 1000000], *params)

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# ... [Keep the exchange_option_mc function as before] ...

def correlation_study(correlations, *args):
    prices = [exchange_option_mc(*args[:-2], rho, *args[-2:])[0] for rho in correlations]
    
    plt.figure()
    plt.plot(correlations, prices, marker='o')
    plt.xlabel('Correlation (rho)')
    plt.ylabel('Option Price')
    plt.title('Behavior of Exchange Option Price with Correlation')
    plt.grid(True)
    plt.show()

# # Define the parameters
# S1_0, S2_0, T, r, q1, q2, sigma1, sigma2, rho, Q1, Q2 = 100, 105, 1, 0.05, 0.02, 0.03, 0.2, 0.25, 0.5, 1, 1
# params = (S1_0, S2_0, T, r, q1, q2, sigma1, sigma2, rho, Q1, Q2)

# # Study the behavior of the exchange option price with correlation
# correlations = np.linspace(-0, 1, 50)  # Creates 50 points from -1 to 1
# correlation_study(correlations, *params)

