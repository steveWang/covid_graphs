from scipy.stats import binom
import numpy as np

# https://en.wikipedia.org/wiki/Group_testing

POOL_SIZES = np.arange(1, 11)

NUM_SAMPLES = 10000
UNIT_SPACE = np.linspace(0, 1, num=NUM_SAMPLES)

# Modeled as binomial CDF.
def pool_positive_rate(pool_size, positive_rate):
    return 1 - (1 - positive_rate) ** pool_size

# Naive strat: test groups of `pool_size`. If positive, test individuals.
# Cutoff is 1 + pN == N, or p = (N-1)/N for p = pool positive rate.
def naive_pool_threshold(pool_size):
    a = pool_positive_rate(pool_size, UNIT_SPACE)
    return np.searchsorted(a, 1 - 1. / pool_size) / NUM_SAMPLES

def naive_pool_savings(pool_size, pool_positive_rate):
    return np.where(pool_size == 1, 0, 1 - (1 / pool_size + pool_positive_rate))

def optimal_naive_pool_size(positive_rate):
    positive_rates = pool_positive_rate(POOL_SIZES, positive_rate)
    savings = naive_pool_savings(POOL_SIZES, positive_rates)
    return np.argmax(savings) + 1

def optimal_naive_pool_cutoffs():
    savings = np.array([optimal_naive_pool_size(p) for p in np.linspace(0, 1, num=NUM_SAMPLES)])
    return (np.where(savings[:-1] != savings[1:])[0] + 1) / NUM_SAMPLES

for n,p in zip(POOL_SIZES[::-1], optimal_naive_pool_cutoffs()):
    print("%d: <= %g%%" % (n, p*100))

