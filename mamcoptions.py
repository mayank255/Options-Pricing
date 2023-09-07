import numpy as np
from math import exp, sqrt
from scipy.linalg import cholesky
from project1 import Asset

np.random.seed(10)


class MultiAsset:
    def __init__(self, assets, covar=None):
        self.assets = assets
        self.covar = covar

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

    def simulate_paths(self, path_length, final_time, interest_rate, volatility):
        if self.covar is None:
            return [asset.simulate(path_length, final_time, interest_rate, volatility) for asset in self.assets]
        
        cholesky_decomp = cholesky(self.covar, lower=True)
        return self._simulate_correlated_paths(path_length, final_time, interest_rate, volatility, cholesky_decomp)

    def _simulate_correlated_paths(self, path_length, final_time, interest_rate, volatility, cholesky_decomp):
        dt = final_time / path_length
        paths = [[] for _ in self.assets]
        
        for _ in range(path_length):
            drift_factor = (interest_rate - volatility**2 / 2) * dt
            random_vals = np.dot(cholesky_decomp, np.random.normal(0, 1, len(self.assets)))
            growth_factors = np.exp(drift_factor + volatility * sqrt(dt) * random_vals)

            for j, asset in enumerate(self.assets):
                current_price = paths[j][-1] if paths[j] else asset.current_price
                paths[j].append(current_price * growth_factors[j])
                
        return paths

class MultiAssetOption:
    def __init__(self, underlyings, maturity_time):
        self.underlyings = underlyings
        self.maturity_time = maturity_time

    def monte_carlo_value(self, num_paths, path_length, interest_rate, volatility):
        payoffs = [self._monte_carlo_sim_value() for _ in range(num_paths)]
        discount_factor = exp(-self.maturity_time * interest_rate)
        return discount_factor * np.mean(payoffs), np.sqrt(np.var(payoffs))

    def _monte_carlo_sim_value(self):
        raise NotImplementedError("To be implemented by derived classes")

class ExchangeOption(MultiAssetOption):
    def __init__(self, underlyings, ratio):
        super().__init__(underlyings, maturity_time=5)
        self.ratio = ratio

    def _monte_carlo_sim_value(self):
        return max(self.ratio * self.underlyings[0].current_price - self.underlyings[1].current_price, 0)

class BasketOption(MultiAssetOption):
    def _monte_carlo_sim_value(self):
        return max(0.5 * self.underlyings[0].current_price - self.underlyings[1].current_price, 0)

if __name__ == "__main__":
    stock1 = Asset("REL", 300)
    stock2 = Asset("TSLA", 100)
    
    multiasset = MultiAsset([stock1, stock2], covar=np.array([[5, 2], [1, 3]]))
    print(multiasset.simulate_paths(10, 5, 0.05, 0.2))

    basket_option = BasketOption([stock1, stock2], maturity_time=5)
    mean_val, variance = basket_option.monte_carlo_value(1000, 5, 0.05, 0.10)
    print(mean_val)

