# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 20:50:14 2023

@author: Mayank Srivastav
"""

import numpy as np

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

# Example:
S1_0, S2_0, T, r, q1, q2, sigma1, sigma2, rho, Q1, Q2 = 400, 200, 1, 0.05, 0, 0, 0.4, 0.6, 0.5, 0.1, 1

exchange_option_price = exchange_option_mc(S1_0, S2_0, T, r, q1, q2, sigma1, sigma2, rho, Q1, Q2)
print(f"Exchange Option Price (Monte Carlo): {exchange_option_price}")
import numpy as np
import scipy.stats as stats
def margrabe_option(S1_0, S2_0, T, q1, q2, sigma1, sigma2, rho, Q1, Q2):
    sigma = np.sqrt((sigma1)**2 + (sigma2)**2 -2*rho*sigma1*sigma2)
    
    d1 = (np.log(Q1*S1_0 / (Q2*S2_0)) + (q2 - q1 + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    print(stats.norm.cdf(d1),stats.norm.cdf(d2))
    V = Q1 * S1_0 * np.exp(-q1 * T) * stats.norm.cdf(d1) - Q2 * S2_0 * np.exp(-q2 * T) * stats.norm.cdf(d2)
    return V

# Same parameters as before
# S1_0, S2_0, T, r, q1, q2, sigma1, sigma2, rho, Q1, Q2 = 400, 50, 1, 0.05, 0, 0, 0.4, 0.6, 0.5, 0.5, 1

exchange_option_price_analytical = margrabe_option(S1_0, S2_0, T, q1, q2, sigma1, sigma2, rho, Q1, Q2)
# exchange_option_price_analytical1 = margrabe_optionv1(S1_0, S2_0, T, q1, q2, sigma1, sigma2, rho, Q1, Q2)
print(f"Exchange Option Price (Margrabe's Formula): {exchange_option_price_analytical}")
# print(f"Exchange Option Price (Margrabe's 1 Formula): {exchange_option_price_analytical1}")
import matplotlib.pyplot as plt

sigmas = np.linspace(0.1, 1, 10)
mc_prices_sigma1 = []
analytical_prices_sigma1 = []
mc_prices_sigma2 = []
analytical_prices_sigma2 = []
for sigma in sigmas:
    monte_carlo_price = exchange_option_mc(S1_0, S2_0, T, r, q1, q2, sigma, sigma2, rho, Q1, Q2)
    analytical_price = margrabe_option(S1_0, S2_0, T, q1, q2, sigma, sigma2, rho, Q1, Q2)
    mc_prices_sigma1.append(monte_carlo_price)
    analytical_prices_sigma1.append(analytical_price)

for sigma in sigmas:
    monte_carlo_price = exchange_option_mc(S1_0, S2_0, T, r, q1, q2, sigma1, sigma, rho, Q1, Q2)
    analytical_price = margrabe_option(S1_0, S2_0, T, q1, q2, sigma1, sigma, rho, Q1, Q2)
    mc_prices_sigma2.append(monte_carlo_price)
    analytical_prices_sigma2.append(analytical_price)
plt.figure()
plt.plot(sigmas, mc_prices_sigma1, '-o', label='MC Price (Sigma1 varied)')
plt.plot(sigmas, analytical_prices_sigma1, '-o', label='Analytical Price (Sigma1 varied)')
plt.xlabel('Sigma1')
plt.ylabel('Option Price')
plt.legend()
plt.title('Option Price vs Sigma1')
plt.grid(True)
plt.show()
plt.figure()
plt.plot(sigmas, mc_prices_sigma2, '-o', label='MC Price (Sigma2 varied)')
plt.plot(sigmas, analytical_prices_sigma2, '-o', label='Analytical Price (Sigma2 varied)')
plt.xlabel('Sigma2')
plt.ylabel('Option Price')
plt.legend()
plt.title('Option Price vs Sigma2')
plt.grid(True)
plt.show()
correlations = np.linspace(-1, 1, 20)  # Range of correlations from -1 to 1
mc_prices_correlation = []
analytical_prices_correlation = []

for current_rho in correlations:
    monte_carlo_price = exchange_option_mc(S1_0, S2_0, T, r, q1, q2, sigma1, sigma2, current_rho, Q1, Q2)
    analytical_price = margrabe_option(S1_0, S2_0, T, q1, q2, sigma1, sigma2, current_rho, Q1, Q2)
    mc_prices_correlation.append(monte_carlo_price)
    analytical_prices_correlation.append(analytical_price)

plt.figure()
plt.plot(correlations, mc_prices_correlation, '-o', label='MC Price (Correlation varied)')
plt.plot(correlations, analytical_prices_correlation, '-o', label='Analytical Price (Correlation varied)')
plt.xlabel('Correlation')
plt.ylabel('Option Price')
plt.legend()
plt.title('Option Price vs Correlation')
plt.grid(True)
plt.show()
S1_values = np.linspace(S2_0*Q2/Q1, 2*S2_0*Q2/Q1, 20)  # Example range, adjust as needed
differences = []
mc_prices_difference = []
analytical_prices_difference = []

for current_S1 in S1_values:
    difference = Q1 * current_S1 - Q2 * S2_0
    monte_carlo_price = exchange_option_mc(current_S1, S2_0, T, r, q1, q2, sigma1, sigma2, rho, Q1, Q2)
    analytical_price = margrabe_option(current_S1, S2_0, T, q1, q2, sigma1, sigma2, rho, Q1, Q2)
    differences.append(difference)
    mc_prices_difference.append(monte_carlo_price)
    analytical_prices_difference.append(analytical_price)

plt.figure()
plt.plot(differences, mc_prices_difference, '-o', label='MC Price (Q1S1 - Q2S2 varied)')
plt.plot(differences, analytical_prices_difference, '-o', label='Analytical Price (Q1S1 - Q2S2 varied)')
plt.xlabel('Q1S1 - Q2S2')
plt.ylabel('Option Price')
plt.legend()
plt.title('Option Price vs Q1S1 - Q2S2')
plt.grid(True)
plt.show()
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
