# -*- coding: utf-8 -*-
"""
Created on Sat May 27 18:56:58 2023.

@author: Mayank Srivastav
"""
# Python version: (3, 9, 7)
from math import exp
from numpy import sqrt, random, mean, log
# import random
# random.seed(50)


class Asset:
    """Class object with attributes Str name, float price and dividend.

    Attributes
    ----------
        name : str
            Name of the Asset class instance
        current_price : float
            Current Price of the instace.
        dividend_yield : float
            Dividend Yield of the  instance. The default is 0

    Methods
    -------
        simulate_path:
            Simulates the movement of underlying current price
        _simulate_path:
            Recursive helper function to simulate the path.

    """

    def __init__(self, name, current_price, dividend_yield=0, volatility = 0, rate = 0):
        """
        Initialize attributes for class Asset.

        Parameters
        ----------
        name : str
            Name of the Asset class instance
        current_price : float
            Current Price of the instace.
        dividend_yield : float
            Dividend Yield of the  instance. The default is 0.

        Returns
        -------
        None.

        """
        self.name = name
        self.current_price = current_price
        self.dividend_yield = dividend_yield
        self.self_volatility = volatility
        self.self_rate = rate

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
        S = list()
        S = S.append(self.current_price)
        del_t = final_time/path_length
        # print("i can see the voletility", volatility)

        return self._simulate_path(path_length-1, del_t,
                                   interest_rate=interest_rate,
                                   volatility=volatility,
                                   simul_path=[self.current_price])

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
            return simul_path

        else:
            r = interest_rate-self.dividend_yield
            mult_fact = (r-volatility ** 2/2) * del_t
            mult_fact =\
                mult_fact + volatility*sqrt(del_t)*random.normal(0, 1, 1)
            mult_fact = exp(mult_fact)
            # print(mult_fact)
            simul_path.append(simul_path[-1]*mult_fact)
            return self._simulate_path(rec_path_length-1, del_t,
                                       interest_rate=interest_rate,
                                       volatility=volatility,
                                       simul_path=simul_path)


class Option:
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

    def __init__(self, name, underlying, *, exercise_price=0,
                 option_type='call', maturity_time=0):
        """
        Initialize the Option attributes.

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
        self.name = name
        # recheck start
        self.underlying = underlying
        # recheck end
        self.exercise_price = exercise_price
        self.maturity_time = maturity_time
        if option_type in ['call', 'put']:
            self.option_type = option_type
        else:
            raise ValueError(f"option type {option_type} not valid."
                             "Please keep it consistent with 'call' or 'put'")


class PathIndependentOption(Option):
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

    def __init__(self, name, underlying, *, exercise_price=0,
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
        Option.__init__(self, name, underlying, exercise_price=exercise_price,
                        option_type=option_type, maturity_time=maturity_time)

    @staticmethod
    def _payoff(exercise_price, current_price, option_type):
        """
        Compute the payoff.

        Parameters
        ----------
        exercise_price : float
            The exercise price of the option.
        current_price : float
            Current price of the underlying of the option.
        option_type : str
            Option type 'call' or 'put'

        Raises
        ------
        TypeError
            Raise type error if unrecognizable options type are inserted.
            Accepted values are 'call' or 'put'

        Returns
        -------
        float
            Returns the payoff of the option..

        """
        if option_type == 'call':
            return max(0, current_price - exercise_price)
        elif option_type == 'put':
            return max(0, exercise_price - current_price)
        else:
            raise TypeError(f"option type {option_type} is unrecognizable.")

    def payoff(self):
        """
        Compute the payoff.

        Returns
        -------
        function
            Returns function _payoff.

        """
        return self._payoff(self.exercise_price, self.current_price,
                            self.option_type)


class MonteCarloValuedOption(Option):
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

    def __init__(self, name, underlying, *, exercise_price=0,
                 option_type='call', maturity_time=0):
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
        Option.__init__(self, name, underlying, exercise_price=exercise_price,
                        option_type=option_type, maturity_time=maturity_time)

    def _monte_carlo_value(self, num_paths,
                           path_length,
                           interest_rate,
                           volatility, pay_off,
                           i):
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
            return exp(self.maturity_time*interest_rate)*pay_off/num_paths
        else:
            sim_path = \
                self.underlying.simulate_path(path_length,
                                              self.maturity_time,
                                              interest_rate=interest_rate,
                                              volatility=volatility)
            self.current_price = sim_path[-1]
            pay_off = pay_off + self._monte_carlo_sim_value(num_paths,
                                                            path_length,
                                                            interest_rate,
                                                            volatility)

            return self._monte_carlo_value(num_paths,
                                           path_length,
                                           interest_rate,
                                           volatility, pay_off,
                                           i-1)

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
        sim_path = self.underlying.simulate_path(path_length,
                                                 self.maturity_time,
                                                 interest_rate=interest_rate,
                                                 volatility=volatility)
        self.current_price = sim_path[-1]
        return self._monte_carlo_value(num_paths,
                                       path_length,
                                       interest_rate,
                                       volatility, 0,
                                       num_paths)


class BinomialValuedOption(PathIndependentOption):
    """Class for Binomial options pricing.

    Parent class: PathIndependentOption.

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
    binomial_value:
        The method computed the options value at present time by evaluating
        options value at each node of the binomial tree.
    """

    def __init__(self, name, underlying, *, exercise_price=0,
                 option_type='call', maturity_time=0):
        """

        Initilaize attributes for BinomialValuedOptions.

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

        PathIndependentOption.__init__(self, name, underlying,
                                       exercise_price=exercise_price,
                                       option_type=option_type,
                                       maturity_time=maturity_time)

    def binomial_value(self, n, u=None, d=None, p=None, *,
                       interest_rate=0, volatility=0, method=None):
        """
        Compute Binomial options value.

        Parameters
        ----------
        n : int
            Binomial tree size.
        u : float,
            The up movement factor in the stock price. The default is None.
        d : float,
            The down movement factor in the stock price. The default is None.
        p : float
            Probability for up movement. The default is None.
        interest_rate : float,
            Risk free interest rate. The default is 0.
        volatility : float
            Annualized standard deviation. The default is 0.
        method : str,
            Binomial pricing method, probable values are either equal
            probability of symmetrical or None with given values of u, d, p.
            The default is None.

        Raises
        ------
        ValueError:
            When value of any of the u, d, p are given along with the
            method value raise Value error as for the specified method
            custom values are not accepted
        ValueError:
            When input parameter does not match with any of the values
            raise type error as the method is not recognized
        ValueError:
            When the value of method is none values of u,d, and p must
            be provided. Raise error as All the values of u, d, p must be
            given.
        ValueError:
            When the constraints u>1, 0<d<1, 0<p<1, are not met.
            Raise ValueError, constraints of Binomial method are not met
                             time step too large

        Returns
        -------
        binomial_value : function
            The function returns instance's private function
            _binomial_node_value.

        """
        del_t = self.maturity_time/n
        r = interest_rate-self.underlying.dividend_yield
        cf = exp(r*del_t)
        df = exp(-interest_rate*del_t)

        if method == "symmetrical":
            if u is not None or d is not None or p is not None:
                raise ValueError("Value of u, p, and d should not be given.")
            A = exp(-r*del_t)
            A = A + exp((r + volatility**2)*del_t)
            A = A/2
            u = A + sqrt(A**2 - 1)
            d = A - sqrt(A**2 - 1)
            p = (exp(r*del_t)-d)/(u-d)

        elif method == "equal probability":
            if u is not None or d is not None:
                raise ValueError("Value of u and d should not be given.")
            p = 0.5
            u = cf*(1 + sqrt(exp(del_t*volatility**2) - 1))
            d = 2*cf - u
        elif method is None:
            if u is None or d is None or p is None:
                raise ValueError("All the values of u, d, p must be given.")
            pass
        else:
            raise NotImplementedError(f"{method} is unrecognizable or "
                                      "not implemented.")
        cond = del_t >= 2 * (volatility ** 2) / interest_rate ** 2
        if u <= 1 or d >= 1 or d <= 0 or cond:

            raise ValueError("Constraints of Binomial method are not met."
                             "Time step too large.")
        binomial_value = self._binomial_node_value(n, 0, 0, u, d, p, df)
        return binomial_value


class AmericanOption(BinomialValuedOption):
    """American Option type.

    Parent class: BinomialValuedOption
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
        _binomial_node_value:
            pseduprivet method for American option objects, it computes the
            options value at binomial nodes and is called in parent class
            method binomial_value for computation of the options value for
            American Options.
    """

    def __init__(self, name, underlying, *, exercise_price=0,
                 option_type='call', maturity_time=0):
        """
        Initialize attributes of AmericanOptions.

        Parameters
        ----------
        name : TYPE
            DESCRIPTION.
        underlying : TYPE
            DESCRIPTION.
        * : TYPE
            DESCRIPTION.
        exercise_price : TYPE, optional
            DESCRIPTION. The default is 0.
        option_type : TYPE, optional
            DESCRIPTION. The default is 'call'.
        maturity_time : TYPE, optional
            DESCRIPTION. The default is 0.

        Returns
        -------
        None.

        """
        BinomialValuedOption.__init__(self, name, underlying,
                                      exercise_price=exercise_price,
                                      option_type=option_type,
                                      maturity_time=maturity_time)

    def _binomial_node_value(self, n, i, j, u, d, p, df):
        """
        Compute Binomial Node value.

        Parameters
        ----------
        n : int
            Binomial tree size.
        u : float,
            The up movement factor in the stock price. The default is None.
        d : float,
            The down movement factor in the stock price. The default is None.
        p : float
            Probability for up movement. The default is None.
        i : integer
            Time grid of the binomial tree.
        j : integer
            Price level grid of the binomial tree.

        df : TYPE
            Discount factor, which is exp(-r*del_t) with given  r and del_t.

        Returns
        -------
        TYPE
            The function returns the American options value at node
            (i, j) = (0,0),which is at present time.

        """
        self.current_price = self.underlying.current_price
        self.current_price = self.current_price * u ** j * d ** (i-j)
        if i == n:
            return self.payoff()

        exercise_payoff = self.payoff()
        V_u = self._binomial_node_value(n, i+1, j+1, u, d, p, df)
        V_d = self._binomial_node_value(n, i+1, j, u, d, p, df)
        option_value = max(exercise_payoff, df * (p * V_u + (1 - p) * V_d))

        return option_value


class EuropeanOption(MonteCarloValuedOption, BinomialValuedOption):
    """European Option Pricing Model.

    Parent class: BinomialValuedOption, MonteCarloValuedOption

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
    _binomial_node_value:
        pseduprivet method for European option objects, it computes the
        options value at binomial nodes and is called in parent class
        BinomialValuedOptions method binomial_value for computation of the
        options value for
        European Options.
    _monte_carlo_sim_value:
        pseduprivet method for European option objects, it returns the
        options pay at recursive steps of parent class
        MonteCarloValuedOptions method_monte_carlo_value
        for the computation of the options value for
        European Options.

    """

    def _binomial_node_value(self, n, i, j, u, d, p, df):
        """
        Compute binomial node values.

        Parameters
        ----------
        n : int
            Binomial tree size.
        u : float,
            The up movement factor in the stock price. The default is None.
        d : float,
            The down movement factor in the stock price. The default is None.
        p : float
            Probability for up movement. The default is None.
        i : integer
            Time grid of the binomial tree.
        j : integer
            Price level grid of the binomial tree.
        df : TYPE
            Discount factor, which is exp(-r*del_t) with given  r and del_t.

        Returns
        -------
        TYPE
            The function returns the European options value at node
            (i, j) = (0,0),which is at present time.


        """
        self.current_price = self.underlying.current_price
        self.current_price = self.current_price * u ** j * d ** (i-j)
        if i == n:
            return self.payoff()
        V_u = self._binomial_node_value(n, i+1, j+1, u, d, p, df)
        V_d = self._binomial_node_value(n, i+1, j, u, d, p, df)
        option_value = df * (p * V_u + (1 - p) * V_d)
        # print(V_u, V_d)
        return option_value

    def _monte_carlo_sim_value(self, num_paths,
                               path_length,
                               interest_rate,
                               volatility):
        """
        Compute montecarlo simulation value.

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
        function
            returns the payoff function.

        """
        return self.payoff()


class AsianEuropeanOption(MonteCarloValuedOption):
    """Asian European option class.

    Parent class: MonteCarloValuedOption


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
    averaging_method: str,
        averaging method either 'arithmetic' or geometric.

    Methods
    -------
    _monte_carlo_sim_value:
        pseduprivet method for European option objects, it returns the
        options pay at recursive steps of parent class
        MonteCarloValuedOptions method_monte_carlo_value
        for the computation of the options value for
        Asian Options.

    """

    def __init__(self, name, underlying, *, exercise_price=0,
                 option_type='call', maturity_time=0,
                 averaging_method='arithmetic'):
        """Initialie the attributes."""
        self.averaging_method = averaging_method
        super().__init__(name, underlying, exercise_price=exercise_price,
                         option_type=option_type, maturity_time=maturity_time)
        if averaging_method not in ['arithmetic', 'geometric']:
            raise ValueError("Unrecognized Method")

    def payoff(self):
        """
        Compute options payoff.

        Returns
        -------
        function
            It returns static method _payoff of class PathIndependentOption.

        """
        return PathIndependentOption._payoff(self.exercise_price,
                                             self.current_price,
                                             self.option_type)

    def _monte_carlo_sim_value(self, num_paths, path_length, interest_rate,
                               volatility):
        """
        Compute Monte carlo simulation options value.

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
        function
            The function returns another function payoff which belongs
            to the instance.

        """
        simul_path = self.underlying.simulate_path(path_length,
                                                   self.maturity_time,
                                                   interest_rate=interest_rate,
                                                   volatility=volatility)
        if self.averaging_method == 'arithmetic':
            self.exercise_price = mean(simul_path)
        elif self.averaging_method == 'geometric':
            self.exercise_price = exp(mean(log(simul_path)))
        return self.payoff()


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    print("test is running from main file.")
    stock1 = Asset("REL", 50, 0)
    sim_return = stock1.simulate_path(100, 100, interest_rate=0.05,
                                      volatility=0.2)
    print(sim_return)
# Option value for an American Option using monte carlo method.
    Ameri = AmericanOption("American Option", stock1, exercise_price=40,
                            option_type='put', maturity_time=3)
    Ameri_binomial = Ameri.binomial_value(3, interest_rate=0.05,
                                          volatility=0.5, method='symmetrical')
    print("The American option price through binomial method is: ",Ameri_binomial)
    
    print('-'*50)
    random.seed(42)
    for i in range(10):
        
        
        plt.plot(stock1.simulate_path(1000, 1, interest_rate=0.10,
                                      volatility=0.1))
        plt.xlabel('Simulation Step')
        plt.ylabel(f'Underlying\'s Price')
       
        plt.title('Underlying\'s Price Path Simulation')
    # plt.axhline(y=53, color='r', linestyle='--', label='strikeprice 60')
    plt.legend()
    plt.savefig("sim_return1.png", dpi = 300)
    
    