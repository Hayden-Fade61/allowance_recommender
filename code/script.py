import pandas as pd
import seaborn as sns
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.arima_model import ARIMA, ARIMAResults
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from collections.abc import Callable

# Adapted from: https://www.statsmodels.org/stable/examples/notebooks/generated/stationarity_detrending_adf_kpss.html#Stationarity-and-detrending-(ADF/KPSS)
def do_stationarity_test(time_series: pd.Series, test: str) -> pd.Series:
    TEST_BASE_INDEX = [ "Test Statistic", "p-value", "#Lags Used"]
    TEST_EXTRA_INDEX = {"adf": ["Number of Observations Used"]}
    index = TEST_BASE_INDEX + TEST_EXTRA_INDEX.get(test.lower(), [])
    index_len = len(index)
    result = adfuller(time_series, autolag="AIC") if test.lower() == "adf" else  kpss(time_series)
    output = pd.Series(
        result[0:index_len],
        index=index
    )
    for key, value in result[index_len].items():
        output[f"Critical Value ({key})"] = value
    return output

def plot_visualisation(
        data: pd.DataFrame | pd.Series, 
        plot_function: Callable[..., Axes],
        chart_title: str | None = None,
        x_label: str | None = None,
        y_label: str | None = None,
        save_fig: bool = False,
        show_fig: bool = True
    ) -> None:
        
        plot_function(data)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(chart_title)
        if save_fig:
            plt.savefig(f"img/{chart_title}")
        if show_fig:
            plt.show()

def make_autocorrelation_plots(data: pd.DataFrame) -> None:
     for col in data.columns:
        plot_visualisation(
            data = data[col], 
            plot_function = plot_acf,
            chart_title = f"{col} CPI Autocorrelation",
            x_label = "Lag [1Q]",
            y_label = "ACF",
            save_fig=True,
            show_fig=False
        )
        plot_visualisation(
            data = data[col], 
            plot_function = plot_pacf,
            chart_title = f"{col} CPI Partial Autocorrelation",
            x_label = "Lag [1Q]",
            y_label = "PACF",
            save_fig=True,
            show_fig=False
        )

def test_for_stationarity(data: pd.DataFrame) -> None:
    adf_results_list = []
    kpss_results_list = []
    for col in data.columns:
        print(f"Testing {col}...")
        adf_results_list.append(do_stationarity_test(data[col], "adf"))
        kpss_results_list.append(do_stationarity_test(data[col], "kpss"))
    print("Tests done. Displaying results...")
    adf = pd.DataFrame(adf_results_list, index = data.columns)
    kpss = pd.DataFrame(kpss_results_list, index = data.columns)
    print(f"ADF Testing\n{adf}\n")
    print(f"KPSS Testing\n{kpss}\n")

def difference_data(data: pd.DataFrame | pd.Series, order: int) -> pd.DataFrame | pd.Series:
    differenced = data.copy()
    for col in data.columns:
        differenced[col] = np.append(
            [np.nan] * order, 
            np.diff(data[col], n = order)
        )
    return differenced

def main() -> None:
    pd.options.display.max_rows = 400
    CPI_DATA_COLUMNS = ["Date", "Sydney", "Melbourne", "Brisbane", 
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
        names= CPI_DATA_COLUMNS
    )
    
    # Test all cols for stationarity
    print("Read done, testing raw data for stationarity...")
    test_for_stationarity(cpi_data)
    
    # Produce CPI line plot
    print("Plotting all CPI line graphs...")
    sns.set_theme(style="whitegrid")
    plot_visualisation(
        data = cpi_data, 
        plot_function = sns.lineplot,
        chart_title = "QoQ Australian national and state capital CPIs over time",
        x_label = "Year",
        y_label = "CPI",
        save_fig = False,
        show_fig = False
    )

    # Take first difference of all data cols
    cpi_first_diff = difference_data(data = cpi_data, order = 1)
    print(cpi_first_diff)
        
main()