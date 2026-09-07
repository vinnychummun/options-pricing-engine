"""Black-Scholes closed-form pricer for European options."""
import numpy as np
from scipy.stats import norm

def black_scholes_call(S, K, T, r, sigma):
    """Price a European call option."""
    d1 = (np.log(S/K)+(r + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

def black_scholes_put(S, K, T, r, sigma):
    """Price a European put option."""
    d1 = (np.log(S/K)+ (r + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

if __name__ == "__main__":
    print("Call:", black_scholes_call(100, 100, 1, 0.05, 0.20))
    print("Put:", black_scholes_put(100, 100, 1, 0.05, 0.20))