import pandas as pd
import seaborn as sns
import numpy as np
from sys import exit
from os import path, mkdir
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from collections.abc import Callable
from sklearn.model_selection import train_test_split
from pmdarima.arima import ndiffs
import warnings
from statsmodels.tools.sm_exceptions import InterpolationWarning

# Adapted from: https://www.statsmodels.org/stable/examples/notebooks/generated/stationarity_detrending_adf_kpss.html#Stationarity-and-detrending-(ADF/KPSS)
def do_stationarity_tests(time_series: pd.Series, test: str) -> pd.Series:
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
        chart_folder: str = "",
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
            folder_path = f"img/{chart_folder}/"
            img_name = f"{folder_path}{chart_title}"

            if path.exists(folder_path):
                plt.savefig(img_name)
            else:
                try:
                    mkdir(folder_path)
                    plt.savefig(img_name)
                except FileExistsError:
                    print(f"{img_name} already exists")
                except PermissionError:
                    print(f"Insufficient permissions to create folder {folder_path}")
        
        if show_fig:
            plt.show()
        plt.clf()

def make_autocorrelation_plots(data: pd.DataFrame, folder_path: str = "") -> None:
     for col in data.columns:
        plot_visualisation(
            data = data[col], 
            plot_function = plot_acf,
            chart_folder= folder_path,
            chart_title = f"{col} CPI Autocorrelation",
            x_label = "Lag [1Q]",
            y_label = "ACF",
            save_fig=True,
            show_fig=False
        )
        plot_visualisation(
            data = data[col], 
            plot_function = plot_pacf,
            chart_folder= folder_path,
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
        adf_results_list.append(do_stationarity_tests(data[col], "adf"))
        kpss_results_list.append(do_stationarity_tests(data[col], "kpss"))
    print("Tests done. Displaying results...")
    adf = pd.DataFrame(adf_results_list, index = data.columns)
    kpss = pd.DataFrame(kpss_results_list, index = data.columns)
    print(f"ADF Testing\n{adf}\n")
    print(f"KPSS Testing\n{kpss}\n")

def difference_data(data: pd.DataFrame | pd.Series, order: list[int]) -> pd.DataFrame | pd.Series:
    i = 0
    differenced = data.copy(deep=True)
    for col in data.columns:
        differenced[col] = np.append(
            [np.nan] * order[i], 
            np.diff(data[col], n = order[i])
        )
        i += 1
    return differenced.dropna(axis=0)

def main() -> None:
    sns.set_theme(style ="whitegrid")
    warnings.simplefilter(action = 'ignore', category = InterpolationWarning)

    IMAGE_ROOT = "/img"
    if not path.exists(IMAGE_ROOT):
        try:
            mkdir(IMAGE_ROOT)
        except PermissionError:
            print("Insufficient permissions to create /img directory")
    
    # Read raw CPI figures from Qx 19xx onwards
    input_file =  "data/640101.xlsx"
    CPI_DATA_COLUMNS = ["Date", "Sydney", "Melbourne", "Brisbane", "Perth", "Australia"]
    print(f"Reading {input_file}...")
    try:
        cpi_data = pd.read_excel(
            io= input_file, 
            sheet_name= "Data1",
            index_col= 0,
            header= None,
            skiprows= 139,  
            usecols= "A:D, F, J", 
            names= CPI_DATA_COLUMNS
        )
        cpi_data.index = pd.DatetimeIndex(data=cpi_data.index, freq="QS-DEC")
    except FileNotFoundError:
        print(f"{input_file} does not exist")
        exit(1)
    print("Read done, testing raw data for stationarity...")
    
    # Test all cols for stationarity
    # test_for_stationarity(cpi_data)
    # make_autocorrelation_plots(cpi_data, "raw_acf")

    # Produce CPI line plot
    # print("Plotting all CPI line graphs...")
    # plot_visualisation(
    #     data = cpi_data, 
    #     plot_function = sns.lineplot,
    #     chart_title = "QoQ Australian national and state capital CPIs over time",
    #     x_label = "Year",
    #     y_label = "CPI",
    #     save_fig = True,
    #     show_fig = False
    # )

    # Estimate difference order with pmdarima
    estimated_orders = []
    for col in CPI_DATA_COLUMNS[1:]:
        kpss_order = ndiffs(cpi_data[col], test = "kpss", max_d = 4)
        adf_order = ndiffs(cpi_data[col], test = "adf", max_d = 4)
        order =  adf_order if adf_order <= kpss_order else kpss_order
        estimated_orders.append(order) 
    differenced_df = difference_data(cpi_data, estimated_orders)
    print(f"Estimated difference orders: {estimated_orders}")
    # test_for_stationarity(differenced_df) 
    # make_autocorrelation_plots(differenced_df, "1st_diff_acf")
    model = ARIMA(endog = cpi_data['Australia'], order=(2,1,1))
    res: ARIMAResults
    res = model.fit()
    print(res.specification)
    print(res.summary())
main()