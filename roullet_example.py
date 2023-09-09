# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 14:58:11 2023

@author: Mayank Srivastav
"""

import random
import matplotlib.pyplot as plt
import numpy
def monte_carlo_roulette_simulation(num_simulations_list, color):
    possible_outcomes = {
        'red': [2*i+1 for i in range(18)],
        'black': [2*i for i in range(18)],
        'green': [37]
    }

    total_numbers = possible_outcomes['red'] + possible_outcomes['black'] + possible_outcomes['green']

    win_probabilities = []
    for num_simulation in num_simulations_list:
        wins = 0
        for i in range(num_simulation):
            # Simulate one game of placing a bet on the specified color
            result = numpy.random.choice(total_numbers)
            if result in possible_outcomes[color]:
                wins += 1

        # Calculate the estimated probability after each simulation
        win_prob = wins / num_simulation
        win_probabilities.append(win_prob)

    return win_probabilities

# Number of simulations (trials)
# num_simulations = [10, 100, 1000, 10000]
# # Run the Monte Carlo simulation
# actual_probability = {'black':round(18/37,4), 'red': round(18/37,4), 'green': round(1/37,4)}
# for color in ['black', 'red', 'green']:
    
#     actual_prob = actual_probability[color]
#     for i in range(5):
#         win_probabilities = monte_carlo_roulette_simulation(num_simulations, color)
#     # Plotting the convergence of the estimated probability
#         plt.plot([1,2,3,4], win_probabilities)
#     plt.axhline(y=actual_prob, color='r', linestyle='--', label=f'Actual Probability {actual_prob}')
#     plt.xlabel('Number of Simulations(10^x)')
#     plt.ylabel(f'Estimated Probability of Winning on {color}')
#     plt.legend()
#     plt.title('Monte Carlo Simulation: Roulette Probability Convergence')
#     plt.show(block = True)
