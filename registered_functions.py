from datetime import datetime
import os
import requests


def update_weather(location):
    """
    Get current weather conditions for a given location
    using the OpenWeatherMap API.
    """
    
    api_key = os.environ["OPENWEATHERMAP_API_KEY"]
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {"q": location, "appid": api_key, "units": "imperial"}
    response = requests.get(base_url, params=params)
    weather_data = response.json()

    if response.status_code != 200:
        return f"Error fetching weather data: {weather_data.get('message', 'Unknown error')}"

    lon = weather_data["coord"]["lon"]
    lat = weather_data["coord"]["lat"]
    main = weather_data["weather"][0]["main"]
    feels_like = weather_data["main"]["feels_like"]
    temp_min = weather_data["main"]["temp_min"]
    temp_max = weather_data["main"]["temp_max"]
    pressure = weather_data["main"]["pressure"]
    visibility = weather_data["visibility"]
    wind_speed = weather_data["wind"]["speed"]
    wind_deg = weather_data["wind"]["deg"]
    sunrise = datetime.fromtimestamp(weather_data["sys"]["sunrise"]).strftime('%H:%M:%S')
    sunset = datetime.fromtimestamp(weather_data["sys"]["sunset"]).strftime('%H:%M:%S')
    temp = weather_data["main"]["temp"]
    humidity = weather_data["main"]["humidity"]
    condition = weather_data["weather"][0]["description"]

    return f"""**Weather in {location}:**
- **Coordinates:** (lon: {lon}, lat: {lat})
- **Temperature:** {temp:.2f}°F (Feels like: {feels_like:.2f}°F)
- **Min Temperature:** {temp_min:.2f}°F, **Max Temperature:** {temp_max:.2f}°F
- **Humidity:** {humidity}%
- **Condition:** {condition.capitalize()}
- **Pressure:** {pressure} hPa
- **Visibility:** {visibility} meters
- **Wind Speed:** {wind_speed} m/s, **Wind Direction:** {wind_deg}°
- **Sunrise:** {sunrise}, **Sunset:** {sunset}"""



def update_weather_forecast(location: str) -> str:
    """ Fetches the weather forecast for a given location and returns a formatted string
    Parameters:
    - location: the search term to find weather information
    Returns:
    A formatted string containing the weather forecast data
    """

    api_key = os.environ["OPENWEATHERMAP_API_KEY"]
    base_url = "http://api.openweathermap.org/data/2.5/forecast"
    params = {
        "q": location,
        "appid": api_key,
        "units": "imperial",
        "cnt": 40  # Request 40 data points (5 days * 8 three-hour periods)
    }
    response = requests.get(base_url, params=params)
    weather_data = response.json()
    if response.status_code != 200:
        return f"Error fetching weather data: {weather_data.get('message', 'Unknown error')}"

    # Organize forecast data per date
    forecast_data = {}
    for item in weather_data['list']:
        dt_txt = item['dt_txt']  # 'YYYY-MM-DD HH:MM:SS'
        date_str = dt_txt.split(' ')[0]  # 'YYYY-MM-DD'
        time_str = dt_txt.split(' ')[1]  # 'HH:MM:SS'
        forecast_data.setdefault(date_str, [])
        forecast_data[date_str].append({
            'time': time_str,
            'temp': item['main']['temp'],
            'feels_like': item['main']['feels_like'],
            'humidity': item['main']['humidity'],
            'pressure': item['main']['pressure'],
            'wind_speed': item['wind']['speed'],
            'wind_deg': item['wind']['deg'],
            'condition': item['weather'][0]['description'],
            'visibility': item.get('visibility', 'N/A'),  # sometimes visibility may be missing
        })

    # Process data to create daily summaries
    daily_summaries = {}
    for date_str, forecasts in forecast_data.items():
        temps = [f['temp'] for f in forecasts]
        feels_likes = [f['feels_like'] for f in forecasts]
        humidities = [f['humidity'] for f in forecasts]
        pressures = [f['pressure'] for f in forecasts]
        wind_speeds = [f['wind_speed'] for f in forecasts]
        conditions = [f['condition'] for f in forecasts]

        min_temp = min(temps)
        max_temp = max(temps)
        avg_temp = sum(temps) / len(temps)
        avg_feels_like = sum(feels_likes) / len(feels_likes)
        avg_humidity = sum(humidities) / len(humidities)
        avg_pressure = sum(pressures) / len(pressures)
        avg_wind_speed = sum(wind_speeds) / len(wind_speeds)

        # Find the most common weather condition
        condition_counts = Counter(conditions)
        most_common_condition = condition_counts.most_common(1)[0][0]

        daily_summaries[date_str] = {
            'min_temp': min_temp,
            'max_temp': max_temp,
            'avg_temp': avg_temp,
            'avg_feels_like': avg_feels_like,
            'avg_humidity': avg_humidity,
            'avg_pressure': avg_pressure,
            'avg_wind_speed': avg_wind_speed,
            'condition': most_common_condition,
        }

    # Build the formatted string
    city_name = weather_data['city']['name']
    ret_str = f"**5-Day Weather Forecast for {city_name}:**\n"

    for date_str in sorted(daily_summaries.keys()):
        summary = daily_summaries[date_str]
        ret_str += f"\n**{date_str}:**\n"
        ret_str += f"- **Condition:** {summary['condition'].capitalize()}\n"
        ret_str += f"- **Min Temperature:** {summary['min_temp']:.2f}°F\n"
        ret_str += f"- **Max Temperature:** {summary['max_temp']:.2f}°F\n"
        ret_str += f"- **Average Temperature:** {summary['avg_temp']:.2f}°F (Feels like {summary['avg_feels_like']:.2f}°F)\n"
        ret_str += f"- **Humidity:** {summary['avg_humidity']:.0f}%\n"
        ret_str += f"- **Pressure:** {summary['avg_pressure']:.0f} hPa\n"
        ret_str += f"- **Wind Speed:** {summary['avg_wind_speed']:.2f} m/s\n"

    return ret_str



# print(get_current_weather("Rio Rancho, New Mexico"))
# print(get_3_hour_forecast("San Francisco, California"))

REGISTERED_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "update_weather",
            "description": "Get the current weather information for a specific location",
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
            "name": "update_weather_forecast",
            "description": """Obtain the weather forecast for the next five
                            days for a given city using the OpenWeatherMap API. This forecast provides
                            weather data for the next 5 days.
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