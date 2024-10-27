import pandas as pd
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss
import matplotlib.pyplot as plt
from os.path import relpath

# Augmented Dickey-Fuller test for stationarity
# Taken from: https://www.statsmodels.org/stable/examples/notebooks/generated/stationarity_detrending_adf_kpss.html#Stationarity-and-detrending-(ADF/KPSS)

TEST_BASE_INDEX = [ "Test Statistic", "p-value", "#Lags Used"]

TEST_EXTRA_INDEX = {
    "adf": [
            "Number of Observations Used",
        ]
}

def stationarity_test(time_series: pd.Series, test: str):
    index = TEST_BASE_INDEX + TEST_EXTRA_INDEX.get(test.lower(), [])
    index[0] += f": {test.upper()}"
    index_len = len(index)
    result = adfuller(time_series, autolag="AIC") if test.lower() == "adf" else  kpss(time_series)
    output = pd.Series(
        result[0:index_len],
        index=index
    )
    for key, value in result[index_len].items():
        output[f"Critical Value ({key})"] = value
    return output

def stationarity_tests(time_series):
    adf_res = stationarity_test(time_series, "adf")
    kpss_res = stationarity_test(time_series, "kpss")
    return pd.concat([adf_res, kpss_res])

def main():
    IMG_OUTPUT_DIRECTORY = "/img/"
    # Set output styling
    pd.options.display.max_rows = 400
    sns.set_theme(style="whitegrid")

    # Define column labels for data
    col_labels = ["Date", "Sydney", "Melbourne", "Brisbane", 
                "Adelaide", "Perth", "Hobart", "Darwin", 
                "Canberra", "Australia"]

    # Read raw CPI figures from Qx 19xx onwards
    input_file =  "data/640101.xlsx"
    print(f"Reading {input_file}...")
    cpi_data = pd.read_excel(
        io= input_file, 
        sheet_name= "Data1",
        index_col= 0,
        header= None,
        skiprows= 139,  
        usecols= "A:J", 
        names= col_labels
    )
    print("Read done, conducting tests for stationarity...")
    
    # Test all cols for stationarity
    stationarity_results = []
    for col in col_labels[1:]:
        print(f"Testing {col}...")
        stationarity_results.append(stationarity_tests(cpi_data[col]))
    print("Tests complete. Displaying results...\n")
    # results = pd.DataFrame(stationarity_results, index=col_labels[1:]) 
    print(results)
main()
