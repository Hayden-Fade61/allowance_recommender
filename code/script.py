import pandas as pd
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from os.path import relpath

# Augmented Dickney-Fuller test for stationarity
# Taken from: https://www.statsmodels.org/stable/examples/notebooks/generated/stationarity_detrending_adf_kpss.html#Stationarity-and-detrending-(ADF/KPSS)
def adf_test(time_series):
    print("Dickey-Fuller Test Results:")
    dftest = adfuller(time_series, autolag="AIC")
    dfoutput = pd.Series(
        dftest[0:4],
        index=[
            "Test Statistic",
            "p-value",
            "#Lags Used",
            "Number of Observations Used",
        ],
    )
    for key, value in dftest[4].items():
        dfoutput["Critical Value (%s)" % key] = value
    print(dfoutput + "\n")

def main():
    # Set output styling
    pd.options.display.max_rows = 400
    sns.set_theme(style="whitegrid")

    # Define column labels for data
    col_labels = ["Date", "Sydney", "Melbourne", "Brisbane", 
                "Adelaide", "Perth", "Hobart", "Darwin", 
                "Canberra", "Australia"]

    # Read raw CPI figures from Qx 19xx onwards
    cpi_data = pd.read_excel(
        relpath("data/640101.xlsx"), 
        sheet_name= "Data1",
        index_col= 0,
        header= None,
        skiprows= 139,  
        usecols= "A:J", 
        names= col_labels
    )

    print(cpi_data.head())
    print(cpi_data.shape)
    # Test raw data for stationarity
    adf_test(cpi_data['Australia'])
    # Produce table showing raw data results

    # Apply first difference to all CPI cols 
    for label in col_labels[1:]:
        cpi_data[label] = cpi_data[label].diff()

    print(cpi_data.head())
    print(cpi_data.shape)

    # Test 1st differences for stationarity
    
main()
