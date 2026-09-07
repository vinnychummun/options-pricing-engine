"""Monte Carlo pricer for European options via geometric Brownian motion."""
import numpy as np

def monte_carlo_call(S, K, T, r, sigma, n_sims):
    """Price a European call by simulating terminal stock prices under GBM."""
    Z = np.random.standard_normal(n_sims)
    S_T = S*np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)
    payoffs = np.maximum(S_T - K, 0)
    return np.exp(-r*T) * np.mean(payoffs)

if __name__ == "__main__":
    price = monte_carlo_call(100, 100, 1, 0.05, 0.20, 1_000_000)
    print("Monte Carlo call:", price)
