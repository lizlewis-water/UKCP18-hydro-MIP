"""
UKCP18 Flow Statistics
Ben Smith
09/01/2023
Newcastle University

--- NOTES -------------------------------
All drought data is for the described period. It is not normalised to the length
of the period (the baseline period is shorter than future periods).

Lower in the script you will need to add / uncomment the chosen output name and file paths.

Flows that cannot have lmoments (i.e. return periods) calculated will be set at 0 cumecs.
This is because many of the instances where the calculation fails is where the flows are 0, so 0 cumecs
flow events makes sense. However! This may mask errors where flows are expected, so best to check this
when the errors are flagged.
This may be due to large numbers of 0 values in the flows. E.g. 02c 33023, which becomes highly ephemeral.

Run this on Blade 4 using:
conda activate \ProgramData\Water_Blade_Programs\BenSmith\env_lmoments
I:
cd "SHETRAN_GB_2021\08_Analysis\03 - Flow Analysis"
python "Hydrological Flow and Drought Analysis - Catchments - LSTM.py" REM 'add any comment you want here'

Ben also has an environment called lmoments on his laptop that works.

--- Lmoments3----------------------------
This analysis requires Lmoments3 - this conflicts with Xarray in my environment. You may need to create an environment
for this potentially removing other modules (e.g. conda remove xarray). Installation instructions are here:
https://anaconda.org/openhydrology/lmoments3 - conda install -c openhydrology lmoments3
This method did not work well, and so 'pip install lmoments3' was used, which did work. You may first have to install
pip (conda install pip), then it may be best to use 'python -m pip install lmoments3' so that it uses the conda python,
not the pip python.

TODO: -------------------------------
  - Consider whether you should be calculating intensity and severity non-standardised and only using
      the standardised datasets to calculate Standardised Severity as an indicator for whether the
      drought is severe or not, rather than for its actual value.
  - Very long droughts exist (i.e. 70 years). These are only recorded in the final period. Consider
      what to do here.
"""

# --- IMPORT PACKAGES ----------------------
from time import sleep
import pickle
# import xarray
from Hydrological_Flow_and_Drought_Analysis_Functions import *
import sys

# --- BEGIN ANALYSIS ---------------------

print("Beginning analysis - if you have an output worksheet open, close it now!")

calculate_flow_stats = False
calculate_return_periods = False
calculate_drought_stats = True

if not calculate_flow_stats and not calculate_drought_stats and not calculate_return_periods:
    print("Check you are making outputs! 'Flow', 'Return Period' and 'Drought' stats are set to False.")

# --- SET FILE PATHS ----------------------

convex = "I:/SHETRAN_GB_2021/"
analysis_path = convex + "08_Analysis/03 - Flow Analysis/"

# Set the path to the Historical Discharges (Use HBV catchment list):
master_folder_historical = convex + "08_Analysis/00 - HBV Outputs for Analysis/Final Models - Monthly Means/OB1980-2010/"

# Read in warming level dates:  [entered manually later]
# warming_levels_path = analysis_path + "Warming_levels_stripped.csv"
# warming_levels = pd.read_csv(warming_levels_path)

# Get a list of the HBV catchments:
master_df = os.listdir(master_folder_historical)
master_df = [str.split(c, ".")[0] for c in master_df]
master_df = pd.DataFrame({"catchment": master_df})
master_df.set_index("catchment", inplace=True)

# --- TEST ---
# master_df = master_df[0:1]
# --- END ---

catchment_list = master_df.index
# code to read LSTM format

#lstm_path = 'I:/LSTM/001/output/lstm_001_cc/bcm_01/test/model_epoch030/test_results.p'
#shetran_path = 'I:/SHETRAN_GB_2021/05_Climate_Change_Simulations/01c_UKCP18rcm_Autocal_UDM_Baseline/bcm_01/39001/output_39001_discharge_sim_regulartimestep.txt'

catchment_area_df = pd.read_csv("I:/CAMELS-GB/data/CAMELS_GB_topographic_attributes.csv", header=0, index_col=0)
# print(catchment_area_df)

# Set date to be used as a baseline for drought calculations:
drought_baseline_date = "1985-2010"

# --- USER INPUTS ------------------------
# --- Choose one of the setups to run - also change the tab name below if needed
# --- Output root name should match between SHETRAN and HBV scripts so that data get added to the same files.

# --- Set model type:
model = "LSTM"  # "SHETRAN" | "HBV" | "LSTM"
model_tab_name = "LSTM"  # 'HBV' | 'SHETRAN-UK Autocalibrated' | 'SHETRAN-UK' | "LSTM"

# --- Set model paths:

# LSTM UKCP18 Baseline:
# >> master_folder_UKCP18 set as lstm_path later in script
output_root_name = "01c_UKCP18_LSTM_UDMbaseline"


# --- CALCULATE HISTORICAL STATISTICS -----
if calculate_flow_stats:
    # Extract flow stats from the historical simulations:
    #   Historical simulations run from 01/01/1980 to 01/01/2011.
    #   We will cut off the first 5 years of the simulation.
    master_df[["hist_q99", "hist_q95", "hist_q05", "hist_q01"]] = pd.NA

    counter = 0
    for catchment in catchment_list:

        counter += 1

        print(f"- {catchment} ({counter}/{len(catchment_list)})")

        try:  # Try/Exception used to account for missing files.
            # LSTM
            lstm_path_historical = 'I:/LSTM/001/output/lstm_001_0502_195340_historical-1980-2010/test/model_epoch030/csv/'
            df = pd.read_csv(lstm_path_historical + str(int(catchment))  + '.csv', header=0)

            catchment_area = catchment_area_df.area.loc[int(catchment)]

            df['LSTM'] = ((df['Discharge_mmd'] / 1000.0) / 86400.0) * (catchment_area * 1000000.0)
            sim_df = df['LSTM']
            # Check that the simulation completed (within 10%), else pass:
            if len(sim_df) < (11324 * 0.9):
                continue

            if max(sim_df) == 0:
                print(f"- Catchment {catchment} skipped as no values greater than 0.")
                continue

            # Calculate flow quantiles for the historical simulation:
            master_df.loc[catchment, ["hist_q99"]] = round(sim_df[365 * 5:].quantile(0.01), 3)  # Low flow
            master_df.loc[catchment, ["hist_q95"]] = round(sim_df[365 * 5:].quantile(0.05), 3)
            master_df.loc[catchment, ["hist_q01"]] = round(sim_df[365 * 5:].quantile(0.99), 3)
            master_df.loc[catchment, ["hist_q05"]] = round(sim_df[365 * 5:].quantile(0.95), 3)  # High flow

        except Exception as e:
            print("EXCEPTION - Catchment: ", catchment, ":")
            print(e)
            continue

# --- SETUP PERIODS & STATISTICS ----------

rcm_list = ["01", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "15"]

# period_list = ["1980-2000", "1980-2010", "1985-2000", "1985-2010", "1985-2015", "1990-2010",  # Baseline periods
#                "2010-2040", "2020-2050", "2030-2060", "2040-2070", "2050-2080",  # Future periods
#                "WL1.5", "WL2.0", "WL2.5", "WL3.0", "WL3.5", "WL4.0"]  # Warming periods - these will need looking up.

date_indexes = {
    # Dates start 01/12/1980. There are 360 days in a climate year.
    # "1980-2000": [0, 360 * 20],
    # "1980-2010": [0, 360 * 30],
    "1985-2000": [360 * 5, 360 * 20],
    "1985-2010": [360 * 5, 360 * 30],
    # "1985-2015": [360 * 5, 360 * 35],
    # "1990-2010": [360 * 10, 360 * 30],

    "2010-2040": [360 * 30, 360 * 60],
    "2020-2050": [360 * 40, 360 * 70],
    "2030-2060": [360 * 50, 360 * 80],
    "2040-2070": [360 * 60, 360 * 90],
    "2050-2080": [360 * 70, 360 * 100],

    # Listed in order of RCP warming period start years.
    # The code reads from the start of the listed year.
    "WL1.5": [2006, 2003, 2007, 2005, 2005, 2006, 2004, 2008, 2004, 2010, 2005, 2006],
    "WL2.0": [2016, 2013, 2018, 2016, 2017, 2018, 2014, 2018, 2015, 2020, 2016, 2019],
    "WL2.5": [2026, 2023, 2028, 2025, 2027, 2029, 2023, 2027, 2025, 2030, 2026, 2030],
    "WL3.0": [2034, 2031, 2037, 2034, 2036, 2038, 2030, 2036, 2034, 2038, 2035, 2038],
    "WL3.5": [2042, 2039, 2044, 2042, 2043, 2047, 2037, 2045, 2042, 2045, 2043, 2046],
    "WL4.0": [2049, 2046, 2051, 2049, 2050, 2055, 2044, 2052, 2050, 2052, 2050, 2054]

}

# Create a list of drought periods (as these won't use all the baseline periods:
# >> IF YOU CHANGE THE PERIODS ABOVE, MAKE SURE YOU CHANGE THESE ACCORDINGLY! <<
# drought_periods = [list(date_indexes.keys())[x] for x in range(len(date_indexes.keys())) if
#                    x in [2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]]
drought_periods = list(date_indexes.keys())

# --- CREATE BLANK OUTPUTS ----------------

output_list = []
output_names = []

tuples = [(r, p) for r in rcm_list for p in date_indexes.keys()]
output_template = pd.DataFrame(columns=tuples, index=catchment_list)
output_template.index.name = 'catchment'
output_template.columns = pd.MultiIndex.from_tuples(tuples)
# df = df.sort_index(1)

if calculate_flow_stats:
    output_Q99 = copy.deepcopy(output_template)
    output_Q95 = copy.deepcopy(output_template)
    output_Q50 = copy.deepcopy(output_template)
    output_Q05 = copy.deepcopy(output_template)
    output_Q01 = copy.deepcopy(output_template)
    output_LTQ95 = copy.deepcopy(output_template)
    output_LTQ99 = copy.deepcopy(output_template)
    output_GTQ05 = copy.deepcopy(output_template)
    output_GTQ01 = copy.deepcopy(output_template)

    # Add outputs to a list for writing:
    output_list.extend([output_Q99, output_Q95, output_Q50, output_Q05, output_Q01,
                        output_LTQ95, output_LTQ99, output_GTQ05, output_GTQ01])  # output_GTbankfull
    output_names.extend(["Q99", "Q95", "Q50", "Q05", "Q01", "LTQ95", "LTQ99", "GTQ05", "GTQ01"])

if calculate_return_periods:
    output_ReturnPeriod_2yr = copy.deepcopy(output_template)
    output_ReturnPeriod_3yr = copy.deepcopy(output_template)
    output_ReturnPeriod_5yr = copy.deepcopy(output_template)
    output_ReturnPeriod_10yr = copy.deepcopy(output_template)
    output_ReturnPeriod_25yr = copy.deepcopy(output_template)
    output_ReturnPeriod_50yr = copy.deepcopy(output_template)
    output_ReturnPeriod_100yr = copy.deepcopy(output_template)

    # Add outputs to a list for writing:
    output_list.extend([output_ReturnPeriod_2yr,
                        output_ReturnPeriod_3yr, output_ReturnPeriod_5yr, output_ReturnPeriod_10yr,
                        output_ReturnPeriod_25yr, output_ReturnPeriod_50yr, output_ReturnPeriod_100yr])
    output_names.extend(["ReturnPeriod_2yr", "ReturnPeriod_3yr", "ReturnPeriod_5yr", "ReturnPeriod_10yr",
                         "ReturnPeriod_25yr", "ReturnPeriod_50yr", "ReturnPeriod_100yr"])

if calculate_drought_stats:
    # Using periods of interests to droughts, calculate standardised metrics of:
    dps = [x for x in range(len(output_template.columns)) if output_template.columns[x][1] in drought_periods]

    output_drought_duration_mean = copy.deepcopy(output_template.iloc[:, dps])  # (not affected by standardisation)
    output_drought_months = copy.deepcopy(output_template.iloc[:, dps])  # (not affected by standardisation)

    output_deficit_max = copy.deepcopy(output_template.iloc[:, dps])
    output_deficit_mean = copy.deepcopy(output_template.iloc[:, dps])
    output_deficit_total = copy.deepcopy(output_template.iloc[:, dps])

    # Calculate the same [standardised] metrics for only the moderate and major droughts:
    output_drought_duration_mean_severe = copy.deepcopy(output_template.iloc[:, dps])
    output_drought_months_severe = copy.deepcopy(output_template.iloc[:, dps])
    output_deficit_mean_severe = copy.deepcopy(output_template.iloc[:, dps])

    # Add outputs to a list for writing:
    output_list.extend([output_drought_duration_mean, output_drought_months, output_drought_months_severe,
                        output_deficit_max, output_deficit_mean, output_deficit_total,
                        output_drought_duration_mean_severe, output_deficit_mean_severe])
    output_names.extend(["drought_duration_mean", "drought_months", "drought_months_severe",
                         "drought_deficit_max", "drought_deficit_mean", "drought_deficit_total",
                         "drought_duration_mean_severe", "drought_deficit_mean_severe"])

# --- CALCULATE FLOW STATISTICS -----------

print("Calculating statistics for catchments:")
test_time = time.time()
# # Create counter for tracking progress:
counter = 0

# Run through each of the catchments that we intended to model:
for catchment in catchment_list:

    counter += 1

    print(f"- {catchment} ({counter}/{len(catchment_list)})")

    # Run through each RCM run:
    print("   rcm:")

    for r in range(len(rcm_list)):

        rcm = rcm_list[r]

        print("   - ", rcm)

        # Try to open the flow output:
        try:
            flow_path = f"I:/LSTM/001/output/lstm_001_cc/bcm_{rcm}/test/model_epoch030/csv/{str(int(catchment))}.csv"
            if os.path.exists(flow_path):
                df = pd.read_csv(flow_path, header=0)

                catchment_area = catchment_area_df.area.loc[int(catchment)]
                df['LSTM'] = ((df['Discharge_mmd'] / 1000.0) / 86400.0) * (catchment_area * 1000000.0)
                flow_df = df['LSTM']

                # Check that the simulation completed 100 years, else skip calculations:
                if len(flow_df) < 36000:
                    print("Catchment ", catchment, " skipped as incomplete.")
                    continue

                if max(flow_df) == 0:
                    print(f"- Catchment {catchment} skipped as no values greater than 0.")
                    continue

                # Sometimes the model outputs are 100yrs x 365 days instead of 100x360.
                # This end period has no driving data, so should be trimmed off.
                flow_df = flow_df[0:min(36000, len(flow_df))]

            else:
                continue

        except Exception as e:
            print("Exception - Catchment: ", catchment, ":")
            print("... ", e)
            continue

        # ---------------------------------------------------------
        # CALCULATE FLOW QUANTILES AND COUNTS OVER/UNDER THRESHOLD:
        # ---------------------------------------------------------

        if calculate_flow_stats or calculate_return_periods:

            # Run through each period:
            for period in date_indexes.keys():

                # print(r, rcm, catchment, period)

                # Get the list of indexes/dates from the period dictionary:
                date_index = date_indexes[period]

                # If it is a warming period, calculate the correct dates for the period (capped at 2080):
                if "WL" in period:
                    date_index = np.arange(360 * (date_index[r] - 1980),
                                           min(360 * (date_index[r] - 1980 + 30), 360 * 100))
                else:
                    # Translate the period string into a list of indexes for the period:
                    date_index = np.arange(date_index[0], date_index[1], 1)

                temp_data = flow_df[date_index]

                if calculate_flow_stats:
                    period_years = len(date_index) / 360

                    # Calculate UKCP18 flow quantiles:
                    output_Q99.loc[catchment, (rcm, period)] = round(temp_data.quantile(0.01), 3)  # Very low flow
                    output_Q95.loc[catchment, (rcm, period)] = round(temp_data.quantile(0.05), 3)  # Low flow
                    output_Q50.loc[catchment, (rcm, period)] = round(temp_data.quantile(0.50), 3)  # Median flow
                    output_Q05.loc[catchment, (rcm, period)] = round(temp_data.quantile(0.95), 3)  # High flow
                    output_Q01.loc[catchment, (rcm, period)] = round(temp_data.quantile(0.99), 3)  # Very high flow

                    # Calculate counts under thresholds from HISTORICAL MODEL:
                    output_LTQ99.loc[catchment, (rcm, period)] = round(remove_None(
                        len(temp_data.loc[temp_data < master_df.loc[catchment, 'hist_q99']])) / period_years, 2)
                    output_LTQ95.loc[catchment, (rcm, period)] = round(remove_None(
                        len(temp_data.loc[temp_data < master_df.loc[catchment, 'hist_q95']])) / period_years, 2)

                    # Calculate counts over thresholds from HISTORICAL MODEL:
                    output_GTQ05.loc[catchment, (rcm, period)] = round(remove_None(
                        len(temp_data.loc[temp_data > master_df.loc[catchment, 'hist_q05']])) / period_years, 2)
                    output_GTQ01.loc[catchment, (rcm, period)] = round(remove_None(
                        len(temp_data.loc[temp_data > master_df.loc[catchment, 'hist_q01']])) / period_years, 2)

                    # Calculate counts under thresholds from OBSERVED DATASET if there is a value:
                    # if not np.isnan(master_df.loc[catchment, 'bankfull_flow']):
                    #     output_GTbankfull.loc[catchment, (rcm, period)] = remove_None(
                    #         len(temp_data.loc[temp_data > master_df.loc[catchment, 'bankfull_flow']])) / period_years

                if calculate_return_periods:
                    # Calculate return periods UKCP18 Data:
                    return_period_years, return_period_flows = calculate_return_events(
                        temp_data, return_periods=[2, 3, 5, 10, 25, 50, 100])
                    output_ReturnPeriod_2yr.loc[catchment, (rcm, period)] = return_period_flows[0]
                    output_ReturnPeriod_3yr.loc[catchment, (rcm, period)] = return_period_flows[1]
                    output_ReturnPeriod_5yr.loc[catchment, (rcm, period)] = return_period_flows[2]
                    output_ReturnPeriod_10yr.loc[catchment, (rcm, period)] = return_period_flows[3]
                    output_ReturnPeriod_25yr.loc[catchment, (rcm, period)] = return_period_flows[4]
                    output_ReturnPeriod_50yr.loc[catchment, (rcm, period)] = return_period_flows[5]
                    output_ReturnPeriod_100yr.loc[catchment, (rcm, period)] = return_period_flows[6]

        # --------------------
        # CEH Drought metrics:
        # --------------------

        if calculate_drought_stats:

            # >>> STEP 1 & 3 <<<
            # (Create long term standardised mean monthly flows)

            # --- Aggregate to monthly:
            monthly_flow = aggregate_to_monthly(flow_df[:36000])  # The Climate data length

            baseline_start = int(date_indexes[drought_baseline_date][0] / 30)
            baseline_stop = int(date_indexes[drought_baseline_date][1] / 30)

            # Calculate the mean monthly flow (i.e. Jan-Dec) for the baseline period:
            mean_monthly_flow_baseline, mean_monthly_flow_baseline_std = mean_baseline_flow(
                monthly_flow[baseline_start:baseline_stop])

            # Calculate flow anomaly by removing mean monthly baseline flow from the full record:
            monthly_flow_anomaly = calculate_flow_anomaly(monthly_flow, mean_monthly_flow_baseline)

            # Normalise the anomaly by dividing by the mean month's standard deviation:
            monthly_flow_anomaly_normalised = normalise_anomaly(monthly_flow_anomaly, mean_monthly_flow_baseline_std)

            # Create list of flow deficits (i.e. negative normalised anomalies):
            monthly_flow_deficit_stnd = [1 if m < 0 else 0 for m in monthly_flow_anomaly_normalised]

            # >>> STEP 2 <<<
            # (Calculate duration, intensity and severity of deficit of standardised timeseries)

            # --- Create data table to hold all the drought data ---

            # Start with columns for lengths for droughts and non-droughts:
            drought_count = [(i, len(list(g))) for i, g in groupby(monthly_flow_deficit_stnd)]
            master_drought_table = pd.DataFrame(drought_count, columns=['drought', 'length'])

            # Add a column with indexes of months when each drought/non-drought ends:
            master_drought_table["month"] = master_drought_table["length"].cumsum()

            # Add columns with indexes for using with Python sub-setting:
            master_drought_table["start_index"] = master_drought_table["month"] - master_drought_table["length"]
            master_drought_table["end_index"] = master_drought_table["start_index"] + master_drought_table["length"]

            # Subset the dataset to only drought periods:
            master_drought_table = master_drought_table[master_drought_table["drought"] == 1]
            master_drought_table = master_drought_table.reset_index()

            # Add columns for intensity and severity:
            master_drought_table[["severity", "severity_class"]] = np.nan  # "intensity_total", "intensity_mean",

            # Run through each period for the catchment and calculate statistics:
            for period in drought_periods:

                # Get indexes from the date_index dictionary:
                date_index = date_indexes[period]

                # If it is a warming period, calculate the correct start/end months for the period.
                # Capped at 2080, converting from days to months:
                if "WL" in period:
                    date_index = [360 / 30 * (date_index[r] - 1980),
                                  min(360 / 30 * (date_index[r] - 1980 + 30), 12 * 100)]
                else:
                    # Translate the period string into a start and end month for the period:
                    date_index = [date_index[0] / 30, date_index[1] / 30]

                # Calculate a factor to multiply short periods by to normalise them up to 30 yrs:
                # (For 30 yr periods, this will be 1).
                thirty_yrs = 360/(date_index[1]-date_index[0])  # Note that this is calculated differently to previous.

                # IMPORTANT: some of the droughts will cross over between periods. This creates terrible results.
                # We need to crop all droughts at their periods, so we will take a subset of the
                # data (only containing data for the period), edit the start and end droughts, if needed,
                # and then do analysis on that.

                # Subset 1. Use filter rows inclusive of those that overlap the thresholds (d_start<p_end + d_end>p_start):
                period_drought_table = master_drought_table[
                    (master_drought_table['start_index'] <= date_index[1]) &
                    (master_drought_table['end_index'] >= date_index[0])
                ].copy()

                # Subset 2. Edit the start of the first drought and the end of the last:
                period_drought_table.at[period_drought_table.index[0], 'start_index'] = max([period_drought_table['start_index'].iloc[0], date_index[0]])
                period_drought_table.at[period_drought_table.index[-1], 'end_index'] = min([period_drought_table['end_index'].iloc[-1], date_index[1]])

                # Subset 3. Calculate new drought durations for the edited rows.
                period_drought_table.at[period_drought_table.index[0], 'length'] = period_drought_table['end_index'].iloc[0] - period_drought_table['start_index'].iloc[0]
                period_drought_table.at[period_drought_table.index[-1], 'length'] = period_drought_table['end_index'].iloc[-1] - period_drought_table['start_index'].iloc[-1]

                # If step 3 means that there is an initial drought with 0 length, remove this:
                period_drought_table = period_drought_table[period_drought_table['length'] != 0]

                # --- Calculate the intensities & severities ---

                # Run through each drought instance:
                for row in period_drought_table.index:

                    # Create index for monthly flow time series based on the drought period:
                    # subset the monthly flow time series based on the drought period:
                    drought_instance = monthly_flow_anomaly_normalised[int(period_drought_table.loc[row, "start_index"]):
                                                                       int(period_drought_table.loc[row, "end_index"])]

                    # Calculate mean intensity (average deficit) and add this to the table:
                    period_drought_table.loc[row, "intensity_mean"] = -drought_instance.mean()

                    # Calculate Severity (total deficit) and add this to the table: (intensity total == mean intensity x duration):
                    period_drought_table.loc[row, "severity"] = -drought_instance.sum()  # period_drought_table.loc[row, "intensity_mean"] * period_drought_table.loc[row, "length"]

                    # Calculate Standardised Severity class:
                    if period_drought_table.loc[row, "severity"] < 4:
                        period_drought_table.loc[row, "severity_class"] = 0
                    if period_drought_table.loc[row, "severity"] >= 4:
                        period_drought_table.loc[row, "severity_class"] = 1
                    if period_drought_table.loc[row, "severity"] >= 8:
                        period_drought_table.loc[row, "severity_class"] = 2

                # >>> STEP 4 & 5 <<<
                # (Statistics; for droughts and severe droughts [Severity > 4 m3/s])

                # --- Calculate Drought Statistics for Periods: ---

                # Subset to severe droughts only:
                period_drought_table_severe = period_drought_table[period_drought_table.severity_class > 0]

                # Add the duration to the output dataset:  [this is not normalised by period length]
                output_drought_duration_mean.loc[catchment, (rcm, period)] = round(period_drought_table["length"].mean(skipna=True), 3)
                output_drought_duration_mean_severe.loc[catchment, (rcm, period)] = round(period_drought_table_severe["length"].mean(skipna=True), 3)

                # Add the counts of drought months:  [these are normalised to 30 years]
                output_drought_months.loc[catchment, (rcm, period)] = round(period_drought_table["length"].sum(skipna=True)*thirty_yrs, 3)
                output_drought_months_severe.loc[catchment, (rcm, period)] = round(period_drought_table_severe["length"].sum(skipna=True)*thirty_yrs, 3)

                # Add deficit statistics to output dataset:  [total deficit is normalised to 30 years]
                output_deficit_total.loc[catchment, (rcm, period)] = round(period_drought_table["severity"].sum(skipna=True)*thirty_yrs, 3)

                output_deficit_max.loc[catchment, (rcm, period)] = round(period_drought_table["severity"].max(skipna=True), 3)
                output_deficit_mean.loc[catchment, (rcm, period)] = round(period_drought_table["severity"].mean(skipna=True), 3)
                output_deficit_mean_severe.loc[catchment, (rcm, period)] = round(period_drought_table_severe["severity"].mean(skipna=True), 3)


print("TIME: ", time.time() - test_time)

# -----------------------------
# WRITE FLOW/DROUGHT STATISTICS
# -----------------------------

print("Writing Excel documents.")


for i in range(len(output_list)):
    output_path = f"{analysis_path}Outputs/01_Catchments/{output_root_name}_{output_names[i]}.xlsx"

    # Check whether to append or write new file:
    if os.path.exists(output_path):
        # Append data. This will overwrite sheets with the same name:
        with pd.ExcelWriter(output_path, mode='a', if_sheet_exists='replace') as writer:
            output_list[i].to_excel(writer, sheet_name=model_tab_name)
    else:
        # Write data to new workbook:
        with pd.ExcelWriter(output_path, mode='w') as writer:
            output_list[i].to_excel(writer, sheet_name=model_tab_name)

