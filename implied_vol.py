"""Implied volatility solvers (bisection and Newton-Raphson) and a volatility smile."""
import numpy as np

from black_scholes import black_scholes_call
from greeks import vega

def implied_vol_bisection(market_price, S, K, T, r, low=1e-6, high=5.0, tol=1e-8):
    """Recover implied volatility by bisection. Robust: price is monotonic in sigma."""
    for _ in range(200):
        mid = 0.5*(low + high)
        price = black_scholes_call(S, K, T, r, mid)
        if price > market_price:
            high = mid
        else:
            low = mid
        if abs(price - market_price) < tol:
            return mid
    return mid

def implied_vol_newton(market_price, S, K, T, r, guess=0.2, tol=1e-8):
    """Recover implied volatility by Newton-Raphson, using vega as the derivative."""
    sigma = guess
    for _ in range(100):
        price = black_scholes_call(S, K, T, r, sigma)
        v = vega(S, K, T, r, sigma)
        diff = price - market_price
        if abs(diff) < tol:
            return sigma
        sigma = sigma - diff/v
    return sigma

if __name__ == "__main__":
    market = black_scholes_call(100, 100, 1, 0.05, 0.20)
    print("Bisection recovered:", implied_vol_bisection(market, 100, 100, 1, 0.05))
    print("Newton recovered: ", implied_vol_newton(market, 100, 100, 1, 0.05))