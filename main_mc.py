# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 03:45:43 2023

@author: Mayank Srivastav
"""

import numpy as np
# from multiassetoptions1 import MultiAssetOption, ExchangeOption, MonteCarloValuedOption, MultiAsset, BasketOption
from project1 import Asset, Option
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from scipy.stats import norm
from multiassetoptions1 import MultiAssetOption, ExchangeOption, MonteCarloValuedOption, MultiAsset, BasketOption
np.random.seed(12)


# # Mean vector and covariance matrix
# mean = [1, 2,100]
# cov_matrix = [[1, 0.5, -0.2], [0.5, 1, 0.6], [-0.2, 0.6, 1]]
# weights = [1/3,1/3, 1/3]

# # Generate a single sample from the multivariate normal distribution
# sample = np.random.multivariate_normal(mean, cov_matrix)

# print("Generated Sample:")
# print(sample)

# # Generate 5 samples
# num_samples = 5
# samples = np.random.multivariate_normal(mean, cov_matrix, size=num_samples)

# print("\nGenerated Samples:")
# print(samples[-1])

"""-----------------------------------------Analytical Call Options-------------------------------"""


"""---------------------------------Analytical ExhcangeOption------------------------------------"""



def exchange_option_analytical(S1, S2, r, T, sigma1, sigma2, rho, K):
    sigma1 = np.sqrt(3.8)*sigma1
    d1 = (np.log(q1*S1 / (q2*S2)) + ( 0.5 * sigma1 ** 2) * T) / (sigma1 * np.sqrt(T))
    
    d2 = d1 - (sigma1 * np.sqrt(T))
    
    # Cumulative distribution function of standard normal distribution
    N = norm.cdf

    exchange_option = q1*S1 * N(d1) - q2*S2 * N(d2)


    return exchange_option

# Example usage:
S1 = 300  # Current price of Asset 1
S2 = 100  # Current price of Asset 2
q1 = 0.5
q2 = 1
r = 0.05  # Risk-free interest rate
T = 1     # Time to expiration in years
sigma1 = 0.4  # Volatility of Asset 1
sigma2 = 0.4  # Volatility of Asset 2
rho = 0.5     # Correlation between asset returns
K = 110  # Strike exchange rate

exchange_option = exchange_option_analytical(S1, S2, r, T, sigma1, sigma2, rho, K)

print("Exchange Option Value:", exchange_option)


"""---------------------Monte Carlo Exchange Option------------------------------------"""
stock1 = Asset("REL", 300, 0, volatility = 0.2,rate = 0.05)
stock2 = Asset("TSLA", 100, 0,volatility = 0.2, rate = 0.05)
option1 = Option('O1', stock1, exercise_price=40, option_type = 'call', maturity_time=2)
option2 = Option('O2', stock2, exercise_price=110, option_type = 'call', maturity_time=2)

num_paths = [100]
number_of_studies = 100
mc_values = dict()

for i in num_paths:

    mc_values[i] = dict()
    for j in range(number_of_studies):
        # option = MultiAssetOption('rainbow_option',[stock1, stock2], maturity_time = 2)
        # montecarlo = MonteCarloValuedOption('montecarlo', [stock1, stock2])
        exoption = ExchangeOption('exchange_options', [stock1, stock2],maturity_time=1)
        
        mc_values[i][j],y = exoption.monte_carlo_value(i,2, interest_rate = 0.05, volatility = 0.20)
df = pd.DataFrame(mc_values)
print("exchange options Options Price")
print(df.describe())


"""--------------------------Monte-Carlo Basket Options-------------------------------------"""
basket_mc_values = dict()
for i in num_paths:

    basket_mc_values[i] = dict()
    for j in range(number_of_studies):
        multiasset2 = MultiAsset([stock1, stock2], covar = np.array([[1,0.9],[0.9,1]]))
        # print(multiasset2.simulate_path(10,5, interest_rate=0.05, volatility = 0.2))
        option = MultiAssetOption('rainbow_option',[stock1, stock2], maturity_time = 1)
        basoption = BasketOption('exchange_options', [stock1, stock2],maturity_time=1)
        # print(stock1.simulate_path(5,5,interest_rate = 0.05, volatility = 0.2))
        # x,y = basoption.monte_carlo_value(i,5, interest_rate = 0.05, volatility = 0.10)
        
        basket_mc_values[i][j],y = basoption.monte_carlo_value(i,2, interest_rate = 0.05, volatility = 0.40)
df = pd.DataFrame(basket_mc_values)
print(df.describe())


data_dict = mc_values
x_data = list(data_dict[100].keys())
y_data = list(data_dict.keys())
z_data = [[data_dict[y][x] for x in x_data] for y in y_data]

# Convert to NumPy arrays
x_data = np.array(x_data)
y_data = np.array(y_data)
z_data = np.array(z_data)

# Create a meshgrid
X, Y = np.meshgrid(x_data, y_data)

# Create a figure and a 3D subplot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create the 3D surface plot
surf = ax.plot_surface(X, Y, z_data, cmap='viridis', edgecolor='none')

# Set labels for the axes
ax.set_xlabel('X Label')
ax.set_ylabel('Series')
ax.set_zlabel('Z Value')

# Show the plot (interactive window)
plt.show()
# In this code, we first extract the x, y, and z data from the dictionary. Then, we use np.meshgrid to create a mesh of x and y coordinates, which allows us to interpolate the z-coordinates and create a continuous surface using plot_surface.

# Make sure you have matplotlib installed (pip install matplotlib) to run this code. The resulting plot will display a continuous surface based on the z-coordinates in the dictionary, with the x-coordinates on the x-axis, the series on the y-axis, and the z-coordinates on the z-axis.





