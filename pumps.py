import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import poisson, norm
from scipy.special import expit
from functools import lru_cache


file_numbers = (12, 13, 14, 20, 22, 40)

data = [pd.read_csv('UsageData_{0}.csv'.format(i)) for i in file_numbers]

def plot_power():
    f, plots = plt.subplots(2, 3, sharex='col', sharey='row', figsize=(30, 20))
    for i, df in enumerate(data):
        plots[i%2][int(i/2)].plot(df['timestamp'], df['amount'])

    plt.show()
    plt.close()


def calculate_diffs(df):
    ary = df[['timestamp', 'amount']].values # cast to array to ignore index when subtracting
    d = pd.DataFrame(ary[1:,:] - ary[:-1,:], columns=['timedelta', 'wattdelta'])
    d['wattdelta'] = (d['wattdelta'] * 1000).astype(int)
    return d


def plot_diffs():
    f, plots = plt.subplots(2, 3, sharex='col', sharey='row', figsize=(30, 20))
    for i, df in enumerate(data):
        plots[i%2][int(i/2)].hist(calculate_diffs(df), bins=100)

    plt.show()
    plt.close()


INFINITY = 10000
L = np.arange(-INFINITY, INFINITY)


def probability_n_appliances_get_switched(n, interval_seconds):
    hourly_frequency = 2
    return poisson.pmf(n, hourly_frequency * interval_seconds / 3600)


def pdf_given_n_appliances_get_switched(n):
    if n == 0:
        return pd.Series(L==0, index=L).astype(float)
    else:
        loc = 0
        scale = 100
        return pd.Series(norm.pdf(L, loc, np.sqrt(n) * scale), index=L)


@lru_cache(maxsize=5)
def pdf_given_no_pump(interval_seconds):
    max_number_of_appliances_reasonably_switched = int(interval_seconds / 60)
    return sum([probability_n_appliances_get_switched(n, interval_seconds) *
                pdf_given_n_appliances_get_switched(n)
                for n in range(max_number_of_appliances_reasonably_switched)])


def probability_of_odd_num_pump_switches(interval_seconds):
    pump_daily_frequency = 10
    pump_interval_frequency = interval_seconds * pump_daily_frequency / 24 / 3600
    return (1 - np.exp(-2*pump_interval_frequency)) / 2


def pdf_given_one_pump_switch():
    min_pump_wattage = 100
    max_pump_wattage = 750
    normalization = 0.5 / (max_pump_wattage - min_pump_wattage)
    support = ((np.abs(L) <= max_pump_wattage) & (np.abs(L) >= min_pump_wattage))
    return pd.Series(support * normalization, index=L)


def pdf_from_pump_alone(interval_seconds):
    probability_of_change = probability_of_odd_num_pump_switches(interval_seconds)
    return (probability_of_change * pdf_given_one_pump_switch() +
            (1 - probability_of_change) * pd.Series(L == 0, index=L))


@lru_cache(maxsize=5)
def pdf_given_pump(interval_seconds):
    appliances_pdf = pdf_given_no_pump(interval_seconds)
    pump_pdf = pdf_from_pump_alone(interval_seconds)

    # Recall that the convolution of distributions of random variables is the distribution of their
    # sum.
    return pd.Series(np.convolve(appliances_pdf, pump_pdf)[INFINITY:3*INFINITY], index=L)


def log_likelihood_given_pump(d):
    def likelihood_of_single_measurement(measurement):
        return pdf_given_pump(measurement['timedelta'])[measurement['wattdelta']]

    likelihoods = d.apply(likelihood_of_single_measurement, axis=1)

    return np.log(likelihoods).sum()


def log_likelihood_given_no_pump(d):
    def likelihood_of_single_measurement(measurement):
        return pdf_given_no_pump(measurement['timedelta'])[measurement['wattdelta']]

    likelihoods = d.apply(likelihood_of_single_measurement, axis=1)

    return np.log(likelihoods).sum()


def likelihood_ratio(measurement):
    idx = int(round(INFINITY + measurement['wattdelta']))
    return (pdf_given_pump(measurement['timedelta'])[idx] /
            pdf_given_no_pump(measurement['timedelta'])[idx])


def log_odds_of_pump(d):
    return log_likelihood_given_pump(d) - log_likelihood_given_no_pump(d)
