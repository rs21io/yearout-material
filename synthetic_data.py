import pandas as pd
import numpy as np
from datetime import timedelta, datetime


# Function to generate synthetic data for each component
def _generate_synthetic_data(dates):
    # Define random seed for reproducibility
    np.random.seed(0)

    # Generate random data for each component
    data = {
        "Temperature_Pressure_Relief_Valve": np.random.choice(
            [0, 1], size=len(dates)
        ),  # 0 = OK, 1 = Faulty
        "Outlet_Nipple_Assembly": np.random.normal(
            loc=80, scale=10, size=len(dates)
        ),  # Temperature in °F
        "Inlet_Nipple": np.random.normal(
            loc=50, scale=5, size=len(dates)
        ),  # Temperature in °F
        "Upper_Element": np.random.normal(
            loc=150, scale=20, size=len(dates)
        ),  # Wattage (Watts)
        "Lower_Element": np.random.normal(
            loc=150, scale=20, size=len(dates)
        ),  # Wattage (Watts)
        "Anode_Rod": np.random.normal(
            loc=7, scale=1.5, size=len(dates)
        ),  # Length in inches
        "Drain_Valve": np.random.choice(
            [0, 1], size=len(dates)
        ),  # 0 = Closed, 1 = Open
        "Upper_Thermostat": np.random.normal(
            loc=120, scale=10, size=len(dates)
        ),  # Temperature in °F
        "Lower_Thermostat": np.random.normal(
            loc=120, scale=10, size=len(dates)
        ),  # Temperature in °F
        "Operating_Time": np.random.randint(
            1, 25, size=len(dates)
        ),  # Operating time in hours
    }

    # Inject an anomaly in the Upper Thermostat values around the midpoint
    midpoint_index = len(dates) // 2
    anomaly_range = (midpoint_index - 5, midpoint_index + 5)

    # Create a spike in Upper Thermostat values
    data["Upper_Thermostat"][anomaly_range[0] : anomaly_range[1]] = np.random.normal(
        loc=200, scale=5, size=anomaly_range[1] - anomaly_range[0]
    )

    return pd.DataFrame(data, index=dates)


def _generate_date_range(start_date, end_date, freq="D"):
    return pd.date_range(start=start_date, end=end_date, freq=freq)

def get_synthetic_data():
    """ """
    # Generate the dataset
    start_date = datetime(2023, 10, 1)
    end_date = datetime(2024, 10, 1)
    dates = _generate_date_range(start_date, end_date)
    now = datetime.now()

    # Create a DataFrame with synthetic data
    synthetic_dataset = _generate_synthetic_data(dates)
    synthetic_dataset["time"] = [
        now - timedelta(hours=5 * i) for i in range(synthetic_dataset.shape[0])
    ]
    
    return synthetic_dataset
