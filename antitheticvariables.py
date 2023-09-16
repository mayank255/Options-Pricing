import numpy as np
import matplotlib.pyplot as plt
from basketoptions import *

def generate_normal_pairs(N):
    """Generate pairs of standard normal random variables."""
    Z = np.random.standard_normal(N)
    return Z, -Z

def basket_payoff(S, K, w):
    """Payoff function for a European call basket option with multiple assets."""
    return max(np.dot(w, S) - K, 0)

def monte_carlo_basket_option_price(S0, K, r, T, sigma, w, rho_matrix, N):
    """Monte Carlo simulation for basket option pricing."""
    M = len(S0)  # Number of assets
    total_payoff = 0.0

    for _ in range(N):
        Z = np.random.multivariate_normal(np.zeros(M), rho_matrix)
        S_T = S0 * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)
        total_payoff += basket_payoff(S_T, K, w)

    return np.exp(-r*T) * total_payoff / N

def monte_carlo_antithetic_basket_option_price(S0, K, r, T, sigma, w, rho_matrix, N):
    """Monte Carlo with antithetic variates for basket option pricing."""
    M = len(S0)  # Number of assets
    total_payoff = 0.0

    for _ in range(N//2):  # N/2 pairs of antithetic variates
        Z, anti_Z = generate_normal_pairs(M)
        S_T = S0 * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)
        total_payoff += basket_payoff(S_T, K, w)

        S_T_anti = S0 * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*anti_Z)
        total_payoff += basket_payoff(S_T_anti, K, w)

    return np.exp(-r*T) * total_payoff / N

def repeated_simulation(S0, K, r, T, sigma, w, rho_matrix, N, repetitions=100):
    mc_std_errors = []
    mc_av_std_errors = []

    for _ in range(repetitions):
        mc_price = basket_option_mc(S0, K, r, T, sigma, w, rho_matrix, N)
        mc_av_price = monte_carlo_antithetic_basket_option_price(S0, K, r, T, sigma, w, rho_matrix, N)

        mc_std_errors.append(mc_price)
        mc_av_std_errors.append(mc_av_price)

    mc_std_error = np.std(mc_std_errors, ddof=1) / np.sqrt(repetitions)
    mc_av_std_error = np.std(mc_av_std_errors, ddof=1) / np.sqrt(repetitions)

    return mc_std_error, mc_av_std_error
S0 = np.array([200, 200, 150])
K = 200
r = 0.05
T = 1
sigma = np.array([0.2, 0.3, 0.25])
w = np.array([0.4, 0.3, 0.3])
rho_matrix = np.array([[1, 0.5, 0.3], [0.5, 1, 0.4], [0.3, 0.4, 1]])

Ns = [10, 100, 1000, 10000]
mc_std_errors = []
mc_av_std_errors = []

for N in Ns:
    mc_std_error, mc_av_std_error = repeated_simulation(S0, K, r, T, sigma, rho_matrix, w, N)
    mc_std_errors.append(mc_std_error)
    mc_av_std_errors.append(mc_av_std_error)

# Plotting
plt.plot(Ns, mc_std_errors, '-o', label='MC Std Error')
plt.plot(Ns, mc_av_std_errors, '-o', label='MC Antithetic Std Error')
plt.xscale('log')
plt.xlabel('Number of Simulations')
plt.ylabel

