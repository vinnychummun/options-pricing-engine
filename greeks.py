"""Analytic and finite-difference Greeks for European options under Black-Scholes."""
import numpy as np
from scipy.stats import norm

from black_scholes import black_scholes_call

def call_delta(S, K, T, r, sigma):
    """Analytic delta: sensitivity of call price to the stock price."""
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    return norm.cdf(d1)

def gamma(S, K, T, r, sigma):
    """Analytic gamma: rate of change of delta with the stock price."""
    d1 = (np.log(S/K) + (r+ 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    return norm.pdf(d1)/(S*sigma*np.sqrt(T))

def vega(S, K, T, r, sigma):
    """Analytic vega: sensitivity of price to volatility."""
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    return S*norm.pdf(d1)*np.sqrt(T)

def call_theta(S, K, T, r, sigma):
    """Analytic theta: sensitivity of call price to the passage of time."""
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    term1 = -(S*norm.pdf(d1)*sigma)/(2*np.sqrt(T))
    term2 = -r*K*np.exp(-r*T)*norm.cdf(d2)
    return term1 + term2
def call_rho(S, K, T, r, sigma):
    """Analytic rho: sensitivity of call price to the interest rate."""
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return K*T*np.exp(-r*T)*norm.cdf(d2)

def delta_fd(S, K, T, r, sigma, h=0.01):
    """Delta by central finite difference, for validation against call_delta."""
    up = black_scholes_call(S+h, K, T, r, sigma)
    down = black_scholes_call(S-h, K, T, r, sigma)
    return (up-down)/(2*h)

def vega_fd(S, K, T, r, sigma, h=0.0001):
    """Vega by central finite difference, for validation against vega."""
    up = black_scholes_call(S, K, T, r, sigma+h)
    down = black_scholes_call(S, K, T, r, sigma-h)
    return (up-down)/(2*h)

if __name__ == "__main__":
    S, K, T, r, sigma = 100, 100, 1, 0.05, 0.20
    print("Delta:", call_delta(S, K, T, r, sigma))
    print("Gamma:", gamma(S, K, T, r, sigma))
    print("Vega:", vega(S, K, T, r, sigma))
    print("Theta:", call_theta(S, K, T, r, sigma))
    print("Rho:", call_rho(S, K, T, r, sigma))
    print()
    print("Delta check (analytic vs FD):", call_delta(S, K, T, r, sigma), delta_fd(S, K, T, r, sigma))
    print("Vega check (analytic vs FD):", vega(S, K, T, r, sigma), vega_fd(S, K, T, r, sigma))





                                        


