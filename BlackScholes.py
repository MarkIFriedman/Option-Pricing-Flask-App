# helper analytics
import numpy as np
from scipy.stats import norm


def ds(K, S, T, vol, r, q=0):
    vol_t = vol * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * vol * vol) * T) / vol_t
    d2 = d1 - vol_t
    return d1, d2


def call(K, S, T, vol, r, q=0):
    disc = np.exp(-r * T)
    pv_k = K * disc
    spot_after_div = S * np.exp(-q * T)

    d1, d2 = ds(K, S, T, vol, r, q)
    c = norm.cdf(d1) * spot_after_div - norm.cdf(d2) * pv_k
    return c


def greeks(K, S, T, vol, r, q=0):
    disc = np.exp(-r * T)
    pv_k = K * disc
    d1, d2 = ds(K, S, T, vol, r, q)
    delta = norm.cdf(d1)
    theta = - S * norm.pdf(d1) * vol / (2 * np.sqrt(T)) - r * pv_k * norm.cdf(d2)
    vega = S * norm.pdf(d1) * np.sqrt(T)
    rho = pv_k * norm.cdf(d2) * T

    return delta, theta, vega, rho


v_call = np.vectorize(call)
v_greeks = np.vectorize(greeks)


# main class
# в качестве параметров выбираем S, T, sigma (vol), r
class DataGen:
    def __init__(self, S_range, T_range, sigma_range, r_range):
        self.S_range = S_range
        self.T_range = T_range
        self.sigma_range = sigma_range
        self.r_range = r_range

    def dataset(self, n_samples, seed=42):
        np.random.seed(seed)
        S = np.random.uniform(*self.S_range, n_samples)
        K = S.copy()  # at the money option (ATM)
        T = np.random.uniform(*self.T_range, n_samples)
        vol = np.random.uniform(*self.sigma_range, n_samples)
        r = np.random.uniform(*self.r_range, n_samples)

        X = np.vstack([S, T, vol, r]).T
        y = v_call(K, S, T, vol, r).reshape((-1, 1))
        dy = np.vstack(v_greeks(K, S, T, vol, r)).T
        return X, y, dy
