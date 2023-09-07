# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 14:44:28 2023

@author: Mayank Srivastav
"""

from project1 import Asset, Option
from math import exp
import numpy as np
from scipy.linalg import cholesky

from numpy import  sqrt, random, mean, log
np.random.seed(10)


class MultiAsset:
    """Class Option on undelying Asset.

    Attributes
    ----------
    name : str
        Name of the Option Instance
    underlying : object Asset
        Asset class instance, as underlying of the option
    exercise_price : float, optional
        Exercise price of the option instance. The default is 0.
    option_type : str
        option type 'call' or 'put;. The default is 'call'.
    maturity_time : float,
        Expiry date of the options in years. The default is 0.

    Methods
    -------
    None



    """
    def __init__(self, basket:list, *, covar=None):
        self.basketassets = basket
        self.covar = covar
        self.basketsize = len(basket)

    @staticmethod
    def cholesky_factorization(matrix):
        eigenvalues = np.linalg.eigvals(matrix)
        # print(eigenvalues)
        if not np.all(eigenvalues > 0):
            raise ValueError("Cholesky factorization failed. The input matrix is not positive definite.")
        try:
            L = cholesky(matrix, lower=True)
            return L
        except np.linalg.LinAlgError:
            raise ValueError("Cholesky factorization failed. The input matrix may not be positive definite.")



    def simulate_path(self, path_length, final_time, *,
                      interest_rate=0, volatility=0):
        """
        Simulate the underlying price path.

        Parameters
        ----------
        path_length : int
            the length of the simulated paths.
        final_time : float
            end time of simulated path.
        interest_rate : float,
            the annualised, continuously-compounded interest rate.
            The default is 0.
        volatility : float,
            the annualised volatility. The default is 0.

        Returns
        -------
        list
            returns another function _simulate_path

        """
        
        del_t = final_time/path_length
        basket_simpaths = list()
        if self.covar is None:
            for asset in self.basketassets:
                self.current_price = asset.current_price
                self.dividend_yield = asset.dividend_yield
                x = self._simulate_path(path_length-1, del_t,
                                         interest_rate=interest_rate,
                                         volatility=self.self_volatility,
                                         simul_path=[self.current_price])
                basket_simpaths.append(x)
            return basket_simpaths
        elif self.covar is not None:
            cholesky = self.cholesky_factorization(self.covar)
            self.current_price = [asset.current_price for asset in self.basketassets]
            self.dividend_yield = [asset.dividend_yield for asset in self.basketassets]
            x = self._simulate_correlated_path(path_length-1, del_t,
                                     interest_rate=interest_rate,
                                     volatility=self.self_volatility,
                                     simul_path=[self.current_price],cholesky=cholesky)
            basket_simpaths.append(x)
            return basket_simpaths

    def _simulate_path(self, rec_path_length, del_t,
                       interest_rate, volatility, simul_path):
        """
        Help function simulatepath to simulate path.

        Parameters
        ----------
        rec_path_length : int
            path length varying with recursion.
        del_t : float
            time step, _time/path_length.
        interest_rate : float,
            the annualised, continuously-compounded interest rate.
            The default is 0.
        volatility : float,
            the annualised volatility. The default is 0.
        simul_path : list
            simulated path of underlying price prior to the recursive step.

        Returns
        -------
        list
            returns itself over the recursion period and at last returns the
            simulated path for underlying prices.

        """
        if rec_path_length == 0:
            return simul_path[-1]

        else:
            r = interest_rate-self.dividend_yield
            mult_fact = (r-volatility ** 2/2) * del_t
            mult_fact =\
                mult_fact + volatility*sqrt(del_t)*random.normal(0, 1, 1)
            mult_fact = exp(mult_fact)
            # print(mult_fact)
            simul_path.append(simul_path[-1]*mult_fact)
            return self._simulate_path(rec_path_length-1, del_t,
                                       interest_rate=self.self_rate,
                                       volatility=self.self_volatility,
                                       simul_path=simul_path)
    def _simulate_correlated_path(self, rec_path_length, del_t,
                       interest_rate, volatility, simul_path, cholesky=cholesky):
        """
        Help function simulatepath to simulate path.

        Parameters
        ----------
        rec_path_length : int
            path length varying with recursion.
        del_t : float
            time step, _time/path_length.
        interest_rate : float,
            the annualised, continuously-compounded interest rate.
            The default is 0.
        volatility : float,
            the annualised volatility. The default is 0.
        simul_path : list
            simulated path of underlying price prior to the recursive step.

        Returns
        -------
        list
            returns itself over the recursion period and at last returns the
            simulated path for underlying prices.

        """
        if rec_path_length == 0:
            return simul_path[-1]

        else:
            rand_fact = list()
            # for asset in self.basketassets:
            r = interest_rate#-self.dividend_yield
            mult_fact = (r-volatility ** 2/2) * del_t
            # print(mult_fact,simul_path[-1],rand_comp)
            rand_comp =\
                volatility*sqrt(del_t)*random.normal(0, 1, self.basketsize)
            # print(mult_fact,simul_path[-1],rand_comp)
            # print(cholesky)
            rand_fact=np.dot(np.transpose(rand_comp),cholesky)
            mult_fact=mult_fact+rand_fact
            mult_fact = np.exp(mult_fact)
            # print(mult_fact,simul_path[-1])
            simul_path.append(simul_path[-1]*mult_fact)
            return self._simulate_correlated_path(rec_path_length-1, del_t,
                                           interest_rate=interest_rate,
                                           volatility=volatility,
                                           simul_path=simul_path,cholesky=cholesky)

        
class MultiAssetOption(MultiAsset):
    """Class Option on undelying Asset.

    Attributes
    ----------
    name : str
        Name of the Option Instance
    underlying : object Asset
        Asset class instance, as underlying of the option
    exercise_price : float, optional
        Exercise price of the option instance. The default is 0.
    option_type : str
        option type 'call' or 'put;. The default is 'call'.
    maturity_time : float,
        Expiry date of the options in years. The default is 0.

    Methods
    -------
    None



    """

    def __init__(self, name, underlyings, *, exercise_price=0,
                 option_type='call', maturity_time=0):
        """
        Initialize the Option attributes.

        Parameters
        ----------
        name : str
            Name of the Option Instance
        underlyings : object Assets
            Asset class instance, as underlying of the option
        exercise_price : float, optional
            Exercise price of the option instance. The default is 0.
        option_type : str
            option type 'call' or 'put;. The default is 'call'.
        maturity_time : float,
            Expiry date of the options in years. The default is 0.

        Raises
        ------
        ValueError
            If option type is not 'call' or 'put, Raise value error option
            type is not valid.

        Returns
        -------
        None.

        """
        self.name = name
        # print(len(underlyings))
        self.underlyings = underlyings
        # self.number_of_underlyings = len(self.underlyings)
        
        # recheck end
        # self.exercise_prices = exercise_prices
        self.maturity_time = maturity_time
        # if option_type in ['call', 'put']:
        #     self.option_type = option_type
        # else:
        #     raise ValueError(f"option type {option_type} not valid."
        #                      "Please keep it consistent with 'call' or 'put'")



class MonteCarloValuedOption(MultiAssetOption):
    """Monte carlo option class.

    Attributes
    ----------
    name : str
        Name of the Option Instance
    underlying : object Asset
        Asset class instance, as underlying of the option
    exercise_price : float, optional
        Exercise price of the option instance. The default is 0.
    option_type : str
        option type 'call' or 'put;. The default is 'call'.
    maturity_time : float,
        Expiry date of the options in years. The default is 0.

    Methods
    -------
    _monte_carlo_value:
        helper function to monte carlo value method, is a recursive
        function which utilized instances _monte_carlo_sim_value and
        computes options average discounted payoff.
    monte_carlo_value:
        calls it helper function to computes
        options average discounted payoff.
    """

    def __init__(self, name, underlyings, *, exercise_price=0,
                 maturity_time=0):
        """

        Initialize attributes of MonteCarloValuedOptions.

        Parameters
        ----------
        name : str
            Name of the Option Instance
        underlying : object Asset
            Asset class instance, as underlying of the option
        exercise_price : float, optional
            Exercise price of the option instance. The default is 0.
        option_type : str
            option type 'call' or 'put;. The default is 'call'.
        maturity_time : float,
            Expiry date of the options in years. The default is 0.

        Raises
        ------
        ValueError
            If option type is not 'call' or 'put, Raise value error option
            type is not valid.

        Returns
        -------
        None.

        """

        MultiAssetOption.__init__(self, name, underlyings,
                                  exercise_price=exercise_price,
                                  maturity_time=maturity_time)
        self.underlying1 = underlyings[0]
        self.underlying2 = underlyings[1]
        self.ratio = 0.5

    def _monte_carlo_value(self, num_paths,
                           path_length,
                           interest_rate,
                           volatility, pay_off, step_payoff,
                           i,step_payoff_list = list()):
        """
        Compute monte carlo options value.

        Parameters
        ----------
        num_paths : int
            The number of simulated paths to construct.
        path_length : TYPE
            The length of the simulated paths.
        interest_rate : float,
            The annualised, continuously-compounded interest rate.
            The default is 0.
        volatility : float,
            The annualised volatility. The default is 0.
        pay_off : float
            Pay off value before the iteration step.
        i : int
            iteration step for recursion over simulation of underlying price.

        Returns
        -------
        function
           The function returns itself over the recursive steps and at the end
           returns the discounted value of average of all the options pay off.

        """
        
        
        if i == 0:
            # print(num_paths)
            # print("pay_off--------------------")
            # print(exp(self.maturity_time*interest_rate)*pay_off/num_paths)
            mean = exp(self.maturity_time*interest_rate)*pay_off/num_paths
            # print(step_payoff_list)
            if len(step_payoff_list) != 0:
                var = sum([(mean-step_payoff)**2  for  step_payoff in step_payoff_list])/num_paths
            else:
                var = 0
                
            return mean, sqrt(var)
        else:
            sim_path1 = \
                self.underlying1.simulate_path(path_length,
                                              self.maturity_time,
                                              interest_rate=self.underlying1.self_rate,
                                              volatility=self.underlying1.self_volatility)

            sim_path2 = \
                self.underlying2.simulate_path(path_length,
                                              self.maturity_time,
                                              interest_rate=self.underlying2.self_rate,
                                              volatility=self.underlying2.self_volatility)
            self.underlying1.price = sim_path1[-1]
            self.underlying2.price = sim_path2[-1]
            # print(sim_path1)
            # print("payoff", pay_off,self._monte_carlo_sim_value(num_paths,
            #                                                 path_length,
            #                                                 interest_rate,
            #                                                 volatility))
            # print(sim_path2[-1], sim_path1[-1], self._monte_carlo_sim_value(num_paths,
            #                                                 path_length,
            #                                                 interest_rate,
            #                                                 volatility))
            step_payoff = self._monte_carlo_sim_value(num_paths,
                                                            path_length,
                                                            interest_rate,
                                                            volatility)
            step_payoff_list.append(step_payoff)
            # print(step_payoff_list)
            # print(pay_off , step_payoff)
            pay_off = pay_off + step_payoff
            

            return self._monte_carlo_value(num_paths,
                                           path_length,
                                           interest_rate,
                                           volatility, pay_off,step_payoff,
                                           i-1,
                                           step_payoff_list = step_payoff_list)

    def monte_carlo_value(self, num_paths, path_length, interest_rate=0,
                          volatility=0):
        """
        Compute monte carlo values of options.

        Parameters
        ----------
        num_paths : int
            The number of simulated paths to construct.
        path_length : TYPE
            The length of the simulated paths.
        interest_rate : float,
            The annualised, continuously-compounded interest rate.
            The default is 0.
        volatility : float,
            The annualised volatility. The default is 0.

        Returns
        -------
        _monte_carlo_value: function
            A recursive function to compute the payoffs over simulated stock
            prices. The average payoff would be the expected options price.

        """
        
        sim_path1 = self.underlying1.simulate_path(path_length,
                                                 self.maturity_time,
                                                 interest_rate=self.underlying1.self_rate,
                                                 volatility=self.underlying1.self_volatility)
        
        sim_path2 = self.underlying2.simulate_path(path_length,
                                                 self.maturity_time,
                                                 interest_rate=self.underlying2.self_rate,
                                                 volatility=self.underlying2.self_volatility)
        self.underlying1.price = sim_path1[-1]
        self.underlying2.price = sim_path2[-1]
        return self._monte_carlo_value(num_paths,
                                       path_length,
                                       interest_rate,
                                       volatility, 0,0,
                                       num_paths)

class ExchangeOption(MonteCarloValuedOption):
    """Class for path independent options.

    Attributes
    ----------
    name : str
        Name of the Option Instance
    underlying : object Asset
        Asset class instance, as underlying of the option
    exercise_price : float, optional
        Exercise price of the option instance. The default is 0.
    option_type : str
        option type 'call' or 'put;. The default is 'call'.
    maturity_time : float,
        Expiry date of the options in years. The default is 0.

    Methods
    -------
    _payoff:
        static method to compute the payoff of the option.
    payoff:
        calls the static method _payoff and passes the instance attributes
        to compute options payoff

    """

    def __init__(self, name, underlyings, *, exercise_prices=0,
                 option_type='call', maturity_time=0):
        """
        Initialize for PathIndependentOptions.

        Parameters
        ----------
        name : str
            Name of the Option Instance
        underlying : object Asset
            Asset class instance, as underlying of the option
        exercise_price : float, optional
            Exercise price of the option instance. The default is 0.
        option_type : str
            option type 'call' or 'put;. The default is 'call'.
        maturity_time : float,
            Expiry date of the options in years. The default is 0.

        Raises
        ------
        ValueError
            If option type is not 'call' or 'put, Raise value error option
            type is not valid.

        Returns
        -------
        None.
        """
        MultiAssetOption.__init__(self, name, exercise_prices,
                                  option_type=option_type,
                                  maturity_time=maturity_time)
        self.underlying1 = underlyings[0]
        self.underlying2 = underlyings[1]
        self.ratio = 0.5


    def payoff(self):
        """
        Compute the payoff.

        Returns
        -------
        function
            Returns function _payoff.

        """
        # print("prices", self.underlying1.price,
        #            self.underlying2.price)
        return max(self.ratio*self.underlying1.price -
                   self.underlying2.price, 0)
    def _monte_carlo_sim_value(self, num_paths,
                               path_length,
                               interest_rate,
                               volatility ):
        return self.payoff()

class BasketOption(MonteCarloValuedOption):
    """Class for path independent options.

    Attributes
    ----------
    name : str
        Name of the Option Instance
    underlying : object Asset
        Asset class instance, as underlying of the option
    exercise_price : float, optional
        Exercise price of the option instance. The default is 0.
    option_type : str
        option type 'call' or 'put;. The default is 'call'.
    maturity_time : float,
        Expiry date of the options in years. The default is 0.

    Methods
    -------
    _payoff:
        static method to compute the payoff of the option.
    payoff:
        calls the static method _payoff and passes the instance attributes
        to compute options payoff

    """

    def __init__(self, name, underlyings, *, exercise_prices=0,
                 option_type='call', maturity_time=0):
        """
        Initialize for PathIndependentOptions.

        Parameters
        ----------
        name : str
            Name of the Option Instance
        underlying : object Asset
            Asset class instance, as underlying of the option
        exercise_price : float, optional
            Exercise price of the option instance. The default is 0.
        option_type : str
            option type 'call' or 'put;. The default is 'call'.
        maturity_time : float,
            Expiry date of the options in years. The default is 0.

        Raises
        ------
        ValueError
            If option type is not 'call' or 'put, Raise value error option
            type is not valid.

        Returns
        -------
        None.
        """
        MultiAssetOption.__init__(self, name, exercise_prices,
                                  option_type=option_type,
                                  maturity_time=maturity_time)
        self.underlying1 = underlyings[0]
        self.underlying2 = underlyings[1]

    def payoff(self):
        """
        Compute the payoff.

        Returns
        -------
        function
            Returns function _payoff.

        """
        # print("prices", self.underlying1.price,
        #            self.underlying2.price)
        return max(0.5*self.underlying1.price -
                   self.underlying2.price, 0)
    def _monte_carlo_sim_value(self, num_paths,
                               path_length,
                               interest_rate,
                               volatility ):
        return self.payoff()

# class BasketOptions(MultiAssetOptions):
    
#     def __init__(self, underlyings)



if __name__ == "__main__":
    stock1 = Asset("REL", 300, 0)
    stock2 = Asset("TSLA", 100, 0)
    option1 = Option('O1', stock1, exercise_price=40, option_type = 'call', maturity_time=2)
    option2 = Option('O2', stock2, exercise_price=110, option_type = 'call', maturity_time=2)
    # sim_return = stock1.simulate_path(100, 100, interest_rate=0.05,
    #                                   volatility=0.4)
    option = MultiAssetOption('rainbow_option',[stock1, stock2], maturity_time = 5)
    montecarlo = MonteCarloValuedOption('montecarlo', [stock1, stock2])
    # exoption = ExchangeOption('exchange_options', [stock1, stock2],maturity_time=5)

    # x, y = exoption.monte_carlo_value(1000,5, interest_rate = 0.05, volatility = 0.10)
    # print(x,y)
    # multiasset = MultiAsset([stock1, stock2])
    # print(multiasset.simulate_path(5,3, interest_rate=0.05, volatility = 0.2))
    multiasset2 = MultiAsset([stock1, stock2], covar = np.array([[5, 2],[1,3]]))
    print(multiasset2.simulate_path(10,5, interest_rate=0.05, volatility = 0.2))
    option = MultiAssetOption('rainbow_option',[stock1, stock2], maturity_time = 5)
    basoption = BasketOption('exchange_options', [stock1, stock2],maturity_time=5)
    # print(stock1.simulate_path(5,5,interest_rate = 0.05, volatility = 0.2))
    x,y = basoption.monte_carlo_value(1000,5, interest_rate = 0.05, volatility = 0.10)
    print(x)
    

