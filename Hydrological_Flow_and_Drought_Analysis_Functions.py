# --- IMPORT PACKAGES ----------------------

# Floods and droughts:
import os
import numpy as np
import pandas as pd

# For Loading River Cell Data:
import h5py

# Floods
import copy
from scipy.stats import genextreme
from lmoments3 import distr
import warnings  # Suppresses warnings for return period, believed to be scipi bug.

# Droughts
from itertools import groupby

# For testing speed:
import time

# --- FUNCTIONS ---------------------------

# Create a function for calculating flow of return period events:
def calculate_return_events(discharge, method="lmo", return_periods=None):
    """
    :param discharge: List of daily discharges (360 days per year).
    :param method:  "lmo" - Lmoments (slower, more robust);
                    "mle" - mean likelihood estimation (faster, can create huge erroneous flows).
    :param return_periods: List of years that you want return flows calculating for.
    :return: The return period years and flow value for each.
    """

    # Set default return periods to output:
    if return_periods is None:
        return_periods = [3, 5, 10, 25, 50, 100]
    return_periods = np.array(return_periods)

    # Get Annual maximums
    annual_maximums = []
    for year in range(len(discharge) // 360):
        annual_maximums.append(discharge[(year * 360):((year + 1) * 360)].max())

    try:
        # Calculate the curve fits using the desired method (lmo/mle):
        if method == "lmo":
            l_moms = distr.gev.lmom_fit(annual_maximums)
            shape, loc, scale = l_moms.values()

        elif method == "mle":
            # Suppress a warning from the fit function that is understood to be a bug:
            # RuntimeWarning: invalid value encountered in subtract -pex2+logpex2-logex2
            warnings.simplefilter('ignore', RuntimeWarning)

            # Fit the generalized extreme value distribution to the data.
            shape, loc, scale = genextreme.fit(annual_maximums)

            # Reset the warning settings:
            warnings.resetwarnings()

        # Compute the return levels for several return periods.
        return_period_discharges = genextreme.isf(1 / return_periods, shape, loc, scale)
        return_period_discharges = np.round(return_period_discharges, 2)

    # If there is an issue with the calculation (e.g. many 0 values), write the error and return NA/0 cumecs:
    except Exception as e:
        print(e)
        # If the annual maximum values are 90% 0's then set the return flow to 0:
        if len([a for a in annual_maximums if a == 0]) > (0.9*len(annual_maximums)):
            return_period_discharges = [0]*len(return_periods)
        else:
            # else, set the values to NA:
            return_period_discharges = [pd.NA] * len(return_periods)

    return return_periods, return_period_discharges


# Create a function for swapping Nones with 0s
# This is needed for the divisions later.
def remove_None(val):
    if val is None:
        val = 0
    return val


# Define a function for calculating the change in the statistic
def calculate_change(period, reference_period):
    if reference_period != 0 and reference_period is not None:
        change = (period - reference_period) / reference_period * 100
        return change


def aggregate_to_monthly(flow_timeseries):
    monthly_output = []

    # Create an index of 1st day of each month:
    month_starts = np.arange(start=0, stop=len(flow_timeseries) - 29, step=30)

    # Cycle through each month of the dataset:
    for month_start_day in month_starts:
        # Get indexes of each month:
        month_index = np.arange(start=month_start_day, stop=(month_start_day + 30), step=1)

        # Take mean of the monthly flow:
        monthly_output.append(flow_timeseries[month_index].mean())

    return monthly_output


def mean_baseline_flow(monthly_flow_timeseries):
    # Calculate the mean monthly flow (starting with December) for the Baseline period:
    mean_monthly_flow = []
    mean_monthly_flow_std = []

    # Run through January-December
    for calendar_month in range(12):
        # Create indexes of each Jan/Feb/.../Dec
        month_indexes = np.arange(start=calendar_month,
                                  stop=len(monthly_flow_timeseries),
                                  step=12)

        # Extract a list of flows for Jan/Feb/.../Dec
        calendar_month_flow_list = [monthly_flow_timeseries[month] for month in month_indexes]

        # Calculate the standard deviation of the timeseries of calendar months:
        calendar_month_mean_flow_std = np.std(calendar_month_flow_list)

        # Calculate the mean flow for each month (Jan/Feb/.../Dec):
        calendar_month_mean_flow = sum(calendar_month_flow_list) / len(calendar_month_flow_list)

        # Add these values to the lists above:
        mean_monthly_flow.append(calendar_month_mean_flow)
        mean_monthly_flow_std.append(calendar_month_mean_flow_std)

    # Convert the lists to numpy arrays and return them:
    return np.array(mean_monthly_flow), np.array(mean_monthly_flow_std)


def calculate_flow_anomaly(monthly_flow_timeseries, mean_flow_timeseries):
    # Repeat the mean monthly timeseries so that it covers the full period:
    mean_flow_timeseries_repeated = np.tile(mean_flow_timeseries, 100)

    # Crop any end months if needed:
    mean_flow_timeseries_repeated = mean_flow_timeseries_repeated[0:len(monthly_flow_timeseries)]

    # Output anomaly (i.e. monthly flow - mean flow for that month):
    return np.subtract(monthly_flow_timeseries, mean_flow_timeseries_repeated)


def normalise_anomaly(anomaly_to_normalise, monthly_standard_deviations):
    # Repeat the standard deviation timeseries so that it covers the full period:
    monthly_std_repeated = np.tile(monthly_standard_deviations, 100)
    monthly_std_repeated[monthly_std_repeated == 0] = 0.0000001

    # Crop any end months if needed:
    monthly_std_repeated = monthly_std_repeated[0:len(anomaly_to_normalise)]

    return anomaly_to_normalise / monthly_std_repeated


def find_historical_simulation_path(catchment_name, catchment_tracker):
    temp_path = catchment_tracker.loc[int(catchment_name)]["path"]

    if temp_path == "autocal_5params":
        simulation_path = "06_Autocalibration/" + temp_path + "/"
    elif temp_path == "BigOnes-STRlake10-STRriver50":
        simulation_path = "06_Autocalibration/catchments-from-Steve/final-54-catchments-46CATCHMENT-sofar-8-to-go/" + temp_path + "/"
    elif temp_path == "catchment27009-STRlake10-STRriver50-run-at-5km":
        simulation_path = "06_Autocalibration/catchments-from-Steve/final-54-catchments-46CATCHMENT-sofar-8-to-go/" + temp_path + "/"
    elif temp_path == "SmallOnes-STRlake3-STRriver20":
        simulation_path = "06_Autocalibration/catchments-from-Steve/final-54-catchments-46CATCHMENT-sofar-8-to-go/" + temp_path + "/"
    elif temp_path == "N-Ireland-catchments-30CATCHMENTS-StrRiver50":
        simulation_path = "06_Autocalibration/catchments-from-Steve/" + temp_path + "/"
    else:
        simulation_path = "06_Autocalibration/catchments-from-Steve/" + temp_path + "/"

    return simulation_path


# Format the metrics to positive 3dps:
def form(value):
    return abs(round(value, 3))

