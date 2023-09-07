# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 03:13:41 2023.

@author: Mayank Srivastav
"""
import random


class FairRoulette():
    """Fair Roulette."""

    def __init__(self):
        self.pockets = []
        for i in range(1, 37):
            self.pockets.append(i)
        self.ball = None
        self.pocket_odds = len(self.pockets)-1

    def spin(self):
        """Spin the Roulette."""
        self.ball = random.choice(self.pockets)

    def bet_pocket(self, pocket, amt):
        """Place bet on the pocket with $amt."""
        if str(pocket) == str(self.ball):
            return amt*self.pocket_odds
        else:
            return -amt

    def __str__(self):
        """Object name."""
        return "Fair Roulette"


"""simulating Playing Roulette."""


def play_roulette(game, num_spins, pocket, bet, tot_pocket=0, print_com=True):
    """Play Roulette."""
    tot_pocket = 0
    for i in range(num_spins):
        game.spin()
        tot_pocket += game.bet_pocket(pocket, bet)
    if print_com:
        print(num_spins, 'spins of', game)
        print('Expected return betting', pocket, '=',\
              str(100*tot_pocket/num_spins)+'%\n')
    return (tot_pocket/num_spins)


random.seed(30)
game = FairRoulette()

for num_spins in (100, 10000, 100000000):
    for i in range(3):
        play_roulette(game, num_spins, 2, 1)
          