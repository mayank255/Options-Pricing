# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 18:34:08 2023

@author: Mayank Srivastav
"""

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import random
import datetime 


# Black-Scholes analytical solution for European call option
def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

# Finite Difference method for European Call option pricing
def finite_difference_call_option_price(S, K, T, r, sigma, M, N):
    # Parameters
    S_max = 2 * K  # Max price of the underlying asset (to cover a reasonable range)
    t = np.linspace(0, T, N + 1)
    dS = S_max / M
    dt = T / N

    # Initialize grid for asset price and option price
    S_values = np.linspace(0, S_max, M + 1)
    V = np.zeros((M + 1, N + 1))

    # Boundary conditions
    V[:, N] = np.maximum(S_values - K, 0)
    # print(V)
    # Main loop: Use explicit finite difference scheme
    for j in range(N-1, -1, -1):
        for i in range(1, M):
            d1 = (np.log(S_values[i] / K) + (r + 0.5 * sigma**2) * (T - t[j])) / (sigma * np.sqrt(T - t[j]))
            d2 = d1 - sigma * np.sqrt(T - t[j])

            V[i, j] =  np.exp(-r * dt) * (0.5 * (V[i+1, j+1] + V[i, j+1]))

    # Interpolate the option price at the initial spot price S
    S_index = np.searchsorted(S_values, S)
    option_price = np.interp(S, S_values, V[:, 0])
    # print(V[:,0], S,K)

    return option_price



# Monte Carlo method for European Call option pricing
def monte_carlo_call_option_price(S, K, T, r, sigma, num_simulations=100, num_steps=int(0.5*365)):
    # num_steps = int(T * 365)  # Number of time steps (assuming daily steps)
    dt = T / num_steps

    option_prices = []

    for _ in range(num_simulations):
        S_t = S
        for _ in range(num_steps):
            # Generate a random daily return (log-normal distribution)
            daily_return = np.random.normal((r - 0.5 * sigma**2) * dt, sigma * np.sqrt(dt))
            S_t *= np.exp(daily_return)

        # Calculate the option payoff at expiration (European Call)
        option_payoff = max(S_t - K, 0)
        option_prices.append(option_payoff)

    # Discount the average option payoff to present value
    option_price = np.exp(-r * T) * np.mean(option_prices)

    return option_price


# Option parameters
spot_price = 100   # Current price of the underlying asset (e.g., stock)
strike_price = 105  # Option's strike price
time_to_expiration = 0.5  # 6 months to expiration
risk_free_rate = 0.05   # 5% annualized risk-free rate
volatility = 0.2       # 20% annualized volatility

# Benchmark value using Black-Scholes analytical formula
benchmark_option_price = black_scholes_call(spot_price, strike_price,
                                            time_to_expiration,
                                            risk_free_rate, volatility)

# Error analysis for Finite Difference method
# M_values = [30000, 40000]  # Different grid sizes
# N = 500  # Number of time steps

# print("Error analysis for Finite Difference method:")
# for M in M_values:
#     option_price_fd = finite_difference_call_option_price(spot_price,
#                                                           strike_price,
#                                                           time_to_expiration,
#                                                           risk_free_rate,
#                                                           volatility, M, N)
#     absolute_error_fd = abs(option_price_fd - benchmark_option_price)
#     print(f"M = {M}: Option Price = {option_price_fd},\
#           Absolute Error = {absolute_error_fd}")

# Error analysis for Monte Carlo method
"""----------------------------num_steps-------------------------------------"""

# # num_simulations_values = [10, 100, 1000, 10000, 100000] # Different numbers of simulations
# num_simulation = 100
# num_steps_values = [1, 10,100,1000,10000]
# computation_time = dict()
# print("\nError analysis for Monte Carlo method:")
# for i in [10, 20, 40, 15,30]:
#     np.random.seed(i)
#     x = []
#     computation_time[i] = dict()
#     for num_steps in num_steps_values:
#         start = datetime.datetime.now()
#         option_price_mc = monte_carlo_call_option_price(spot_price, strike_price, time_to_expiration, risk_free_rate, volatility, num_simulation, num_steps)
#         x.append(option_price_mc)
#         absolute_error_mc = abs(option_price_mc - benchmark_option_price)
#         print(f"Simulations = {num_steps}: Option Price = {option_price_mc}, Absolute Error = {absolute_error_mc}")
#         computation_time[i][num_steps] = datetime.datetime.now()-start
#     plt.plot([0,1,2,3,4], x)
# plt.axhline(y=benchmark_option_price, color='r', linestyle='--', label=f'Analytical Call Price {benchmark_option_price}')
# plt.xlabel('Number of Steps(log scale)')
# plt.ylabel(f'Estimated MC Value')
# plt.legend()
# plt.title('Monte Carlo Simulation: Call Options Convergence')
# plt.show(block = True)

"""---"""
# for i in [10, 20, 40, 15]:
#     np.random.seed(i)
#     x = []
#     computation_time[i] = dict()
#     for num_simulations in num_simulations_values:
#         start = datetime.datetime.now()
#         option_price_mc = monte_carlo_call_option_price(spot_price, strike_price, time_to_expiration, risk_free_rate, volatility, num_simulations)
#         x.append(option_price_mc)
#         absolute_error_mc = abs(option_price_mc - benchmark_option_price)
#         print(f"Simulations = {num_simulations}: Option Price = {option_price_mc}, Absolute Error = {absolute_error_mc}")
#         computation_time[i][num_simulations] = datetime.datetime.now()-start
#     plt.plot([1,2,3,4,5], x)
# plt.axhline(y=benchmark_option_price, color='r', linestyle='--', label=f'Analytical Call Price {benchmark_option_price}')
# plt.xlabel('Number of Simulations(10^x)')
# plt.ylabel(f'Estimated MC Value')
# plt.legend()
# plt.title('Monte Carlo Simulation: Call Options Convergence')
# plt.show(block = True)


"""------------------------------- num_simulations----------------------------------"""
num_simulations_values = [10, 100, 1000, 10000] # Different numbers of simulations
num_steps_values = [10,100,1000]
computation_time = dict()
print("\nError analysis for Monte Carlo method:")
for i in [10, 20, 40, 15]:
    np.random.seed(i)
    x = []
    computation_time[i] = dict()
    for num_simulations in num_simulations_values:
        start = datetime.datetime.now()
        option_price_mc = monte_carlo_call_option_price(spot_price, strike_price, time_to_expiration, risk_free_rate, volatility, num_simulations)
        x.append(option_price_mc)
        absolute_error_mc = abs(option_price_mc - benchmark_option_price)
        print(f"Simulations = {num_simulations}: Option Price = {option_price_mc}, Absolute Error = {absolute_error_mc}")
        computation_time[i][num_simulations] = datetime.datetime.now()-start
    plt.plot([1,2,3,4], x)
plt.axhline(y=benchmark_option_price, color='r', linestyle='--', label=f'Analytical Call Price {benchmark_option_price}')
plt.xlabel('Number of Simulations(logscale)')
plt.ylabel(f'Estimated MC Value')
plt.legend()
plt.title('Monte Carlo Simulation: Call Options Convergence')
plt.show(block = True)
plt.savefig("calloptions.png", dpi=300)
# for i in [10, 20, 40, 15]:
#     np.random.seed(i)
#     x = []
#     computation_time[i] = dict()
#     for num_simulations in num_simulations_values:
#         start = datetime.datetime.now()
#         option_price_mc = monte_carlo_call_option_price(spot_price, strike_price, time_to_expiration, risk_free_rate, volatility, num_simulations)
#         x.append(option_price_mc)
#         absolute_error_mc = abs(option_price_mc - benchmark_option_price)
#         print(f"Simulations = {num_simulations}: Option Price = {option_price_mc}, Absolute Error = {absolute_error_mc}")
#         computation_time[i][num_simulations] = datetime.datetime.now()-start
#     plt.plot([1,2,3,4,5], x)
# plt.axhline(y=benchmark_option_price, color='r', linestyle='--', label=f'Analytical Call Price {benchmark_option_price}')
# plt.xlabel('Number of Simulations(10^x)')
# plt.ylabel(f'Estimated MC Value')
# plt.legend()
# plt.title('Monte Carlo Simulation: Call Options Convergence')
# plt.show(block = True)

"""-------------------generating paths----------------------------"""