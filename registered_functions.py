from datetime import datetime
import os
import requests


def get_current_temperature(location: str) -> float:
    """
    Simulates getting the current temperature at a location.
    """

    return str(12.0)+" F" # Replace with actual temperature retrieval


def get_3_hour_forecast(city_name: str):
    """Obtain the three-hour interval temperature forecast for the next five
    days for a given city using the OpenWeatherMap API. This forecast provides
    temperature data at three-hour intervals for a total of 40 data points per
    day, covering a five-day period.
    """
    url = f"http://api.openweathermap.org/data/2.5/forecast"
    params = {
        "q": city_name,
        "appid": os.environ["OPENWEATHERMAP_API_KEY"],
        "units": "imperial",  # Use 'imperial' for Fahrenheit
    }

    response = requests.get(url, params=params)
    data = response.json()

    # Extract and print the forecast data
    str_list = []
    forecast_data = data["list"]
    for forecast in forecast_data:
        dt = datetime.fromtimestamp(forecast["dt"])
        temp = forecast["main"]["temp"]
        weather_description = forecast["weather"][0]["description"]
        print(f"Time: {dt}, Temp: {temp}°F, Weather: {weather_description}")
        str_list.append([dt, temp])

    return str(str_list)


def get_current_weather(location: str) -> str:
    """
    Fetches the weather for a given location and returns a dictionary

    Parameters:
    - location: the search term to find current weather information
    Returns:
        The current weather for that location
    """
    api_key = os.environ["OPENWEATHERMAP_API_KEY"]
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {"q": location, "appid": api_key, "units": "imperial"}
    response = requests.get(base_url, params=params)
    weather_data = response.json()
    return weather_data


# print(get_current_weather("Rio Rancho, New Mexico"))
# print(get_3_hour_forecast("San Francisco, California"))

REGISTERED_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_temperature",
            "description": "Get the current temperature for a specific location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and full state, e.g., Rio Rancho, New Mexico",
                    },
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_3_hour_forecast",
            "description": """Obtain the three-hour interval temperature forecast for the next five
                            days for a given city using the OpenWeatherMap API. This forecast provides
                            temperature data at three-hour intervals for a total of 40 data points per
                            day, covering a five-day period.
                            """,
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and full state, e.g., Portland, Oregon.",
                    },
                },
                "required": ["location"],
            },
        },
    },
]
