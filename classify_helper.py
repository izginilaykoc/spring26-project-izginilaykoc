import pandas as pd

import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
from datetime import datetime

def get_historical_weather(start_date,variables,coordinates,get_forecast_data = True,**additional_api_params):
    """
    A function to retrieve historical data (archive or forecast). The latest available data is from two days ago (e.g. If you fetch data on April 24th you will get the data until April 22th).
    * start_date: Start date of data to be retrieved. Note that you may encounter null values if this date is too early.
    * variables: Hourly weather parameters you want to retrieve
    * coordinates: List of coordinates you query. It should be a list of lists. Inner lists are pairs of latitude and longitude
    * get_forecast_data: If true, it will retrieve history of forecast data. Otherwise it will return the historical weather data.
    * **additional_api_params: You can use additional api parameters if you want. Use carefully.
    """
    cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)

    end_date = str(datetime.now()-pd.DateOffset(days=2))[:10]

    all_dataframes = []

    actual_past_url = "https://archive-api.open-meteo.com/v1/archive"
    forecast_past_url = "https://historical-forecast-api.open-meteo.com/v1/forecast"

    if get_forecast_data:
        url = forecast_past_url
    else:
        url = actual_past_url

    for i,coordinate in enumerate(coordinates):
        params = {
            "latitude": coordinate[0],
            "longitude": coordinate[1],
            "start_date": start_date,
            "end_date": end_date,
            "hourly": variables,
            "timezone":"GMT+03",
        }

        for k,v in additional_api_params.items():
            params[k] = v

        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]
        hourly = response.Hourly()
        variable_values = [hourly.Variables(i_var).ValuesAsNumpy() for i_var in range(len(variables))]
        hourly_data = {"dt": pd.date_range(
            start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
            end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
            freq = pd.Timedelta(seconds = hourly.Interval()),
            inclusive = "left"
        )}
        for i_var,variable in enumerate(variables):
            hourly_data[f"location_{str(i).zfill(3)} {variable}"] = variable_values[i_var]
        hourly_dataframe = pd.DataFrame(data = hourly_data)
        hourly_dataframe["dt"] = hourly_dataframe["dt"].dt.tz_convert('Europe/Istanbul')

        hourly_dataframe= hourly_dataframe.set_index("dt")


        all_dataframes.append(hourly_dataframe)

    all_dataframes = pd.concat(all_dataframes,axis=1)
    all_dataframes = all_dataframes.reset_index()

    return all_dataframes



def get_weather_forecast(forecast_days,past_days,variables,coordinates,**additional_api_params):
    """
    A function to retrieve forecast data (future or past). The latest available data is from two days ago (e.g. If you fetch data on April 24th you will get the data until April 22th).
    * forecast_days: Number of the future days (including today) to be retrieved.
    * past_days: Number of the past days (including today) to be retrieved.
    * variables: Hourly weather parameters you want to retrieve.
    * coordinates: List of coordinates you query. It should be a list of lists. Inner lists are pairs of latitude and longitude.
    * **additional_api_params: You can use additional api parameters if you want. Use carefully.
    """
    cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)

    end_date = str(datetime.now()-pd.DateOffset(days=2))[:10]

    all_dataframes = []

    url = "https://api.open-meteo.com/v1/forecast"

    for i,coordinate in enumerate(coordinates):
        params = {
            "latitude": coordinate[0],
            "longitude": coordinate[1],
            "forecast_days": forecast_days,
            "past_days": past_days,
            "hourly": variables,
            "timezone":"GMT+03",
        }

        for k,v in additional_api_params.items():
            params[k] = v

        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]
        hourly = response.Hourly()
        variable_values = [hourly.Variables(i_var).ValuesAsNumpy() for i_var in range(len(variables))]
        hourly_data = {"dt": pd.date_range(
            start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
            end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
            freq = pd.Timedelta(seconds = hourly.Interval()),
            inclusive = "left"
        )}
        for i_var,variable in enumerate(variables):
            hourly_data[f"location_{str(i).zfill(3)} {variable}"] = variable_values[i_var]
        hourly_dataframe = pd.DataFrame(data = hourly_data)
        hourly_dataframe["dt"] = hourly_dataframe["dt"].dt.tz_convert('Europe/Istanbul')

        hourly_dataframe= hourly_dataframe.set_index("dt")


        all_dataframes.append(hourly_dataframe)

    all_dataframes = pd.concat(all_dataframes,axis=1)
    all_dataframes = all_dataframes.reset_index()

    return all_dataframes


def get_imbalance_data():
    """
    Retrieves the hourly system imbalance data from the provided data source.
    Returns a DataFrame with columns including:
      - dt (datetime): combined date and hour
      - date (str): date string
      - hour (int): hour of day (0-23)
      - net (float): net imbalance (downRegulation - upRegulation total)
      - system_direction (str): 'Positive' if net >= 50, 'Negative' if net <= -50, 'Neutral' otherwise.
        Note: The most recent imbalance data may not be available due to EPIAS reporting delays
        (typically 6-12 hours, but variable). Your code should handle missing recent data gracefully.
    """
    imbalance_url='https://docs.google.com/spreadsheets/d/1ow3xkQS56qnUY78dxxoPdRJ4lXSCJtLrcV-Gwepovu0/export?format=csv&gid=206544102'
    imbalance_data = pd.read_csv(imbalance_url)
    imbalance_data["dt"] = pd.to_datetime(
        imbalance_data["date"].astype(str) + " " + imbalance_data["hour"].astype(str).str.zfill(2) + ":00:00"
    )
    return imbalance_data
