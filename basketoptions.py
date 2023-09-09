import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

def basket_option_mc(S0, T, r, q, sigma, rho_matrix, K, n_simulations=100000):
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
    
    # Calculate the worst-off payoff for each simulation
    payoffs = np.maximum(K - np.min(S, axis=1), 0)
    
    return np.mean(payoffs)
def simulate_multiple_times_basket(N_values, params,repetitions=100):
    results = {'N': [], 'Mean Option Price': [], 'Std Dev Option Price': []}

    # Parameters (assuming some values; adjust as needed)
    S0, T, r, q, sigma, rho_matrix, K = params
    
    
    for N in N_values:
        
        option_prices = []
        for rep in range(repetitions):
            option_price= basket_option_mc(S0, T, r, q, sigma, rho_matrix, K, n_simulations=N)
            option_prices.append(option_price)
        
        mean_option_price = np.mean(option_prices)
        std_option_price = np.std(option_prices)
        
        results['N'].append(N)
        results['Mean Option Price'].append(mean_option_price)
        results['Std Dev Option Price'].append(std_option_price)
        
    df = pd.DataFrame(results)
    return df

# # Parameters
# S0 = [100, 101, 102, 103]
# T = 1
# r = 0.05
# q = [0.02, 0.03, 0.03, 0.04]
# sigma = [0.2, 0.25, 0.22, 0.23]
# rho_matrix = np.array([[1, 0.5, 0.3, 0.4], 
#                       [0.5, 1, 0.4, 0.3], 
#                       [0.3, 0.4, 1, 0.5], 
#                       [0.4, 0.3, 0.5, 1]])
# K = 102
# params = (S0, T, r, q, sigma, rho_matrix, K)

# simulation_sizes = [100, 500, 1000, 5000, 10000, 50000, 100000]
# variances = []

# from scipy.stats import linregress

# # ... [The basket option function and other initializations go here] ...

# simulation_sizes = [100, 500, 1000, 5000, 10000, 50000, 100000]
# all_variances = []

# # Repeat the entire process multiple times
# for _ in range(10):
#     variances = []
#     for N in simulation_sizes:
#         estimates = [basket_option_mc(*params, n_simulations=N) for _ in range(10)]
#         var = np.var(estimates)
#         variances.append(var)
#     all_variances.append(variances)

# # Average the variances
# avg_variances = np.mean(all_variances, axis=0)

# # Fit a line to the log-log data
# slope, _, _, _, _ = linregress(np.log(simulation_sizes), np.log(avg_variances))

# plt.figure(figsize=(10, 6))
# plt.loglog(simulation_sizes, avg_variances, 'o-', label=f"Average Variance (Slope={slope:.2f})")
# plt.xlabel('Number of Simulations (log scale)')
# plt.ylabel('Variance (log scale)')
# plt.title('Average Variance vs. Number of Simulations')
# plt.grid(True)
# plt.legend()
# plt.show()


