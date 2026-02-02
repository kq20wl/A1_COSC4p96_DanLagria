import GetData as gd
import numpy as np
import pandas as pd
# purpose is to use numpy and pandas to provide statistics on the data preparation functions

def get_data_statistics(data):
    # Get basic statistics
    stats = {
        "mean": np.mean(data, axis=0),
        "std": np.std(data, axis=0),
        "min": np.min(data, axis=0),
        "max": np.max(data, axis=0),
    }
    return stats

def display_statistics(stats):
    # Convert stats to DataFrame for better visualization
    df = pd.DataFrame(stats)
    print(df)
