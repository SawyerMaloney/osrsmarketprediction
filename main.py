import requests
import time
from typing import Dict, Any, Optional
import json
import numpy as np
import matplotlib.pyplot as plt

class RuneScapePricesAPI:
    """
        Client for querying RuneScape real-time price data and preparing that data for training
        
        item_timeseries:    time x items x item
                            7000 x 7 x 6
    """

    item_ids = {"Zulrah's Scales": "12934", "Super Restore": "3024", "Prayer Potion": "2434", "Shark": "385", "Nature Rune": "561", "Death Rune": "560", "Blood Rune": "565"}
    item_timeseries = []

    def __init__(self, base_url: str = "https://prices.runescape.wiki/api/v1/osrs",
                 user_agent: str = "Sawyer Maloney | ML Training | sawyerdmaloney@gmail.com"):
        """
        :param base_url: Base URL for the API.
        :param user_agent: User agent for requests headers.
        """
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": user_agent,
            "Accept": "application/json"
        })

    def get_price(self, timestamp: int, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Fetch real-time price for a given item.

        timestamp - unix timestamp in seconds
        return - Parsed JSON with price data, or None if error.
        """

        # make sure timestamp is valid
        timestamp = int(timestamp - (timestamp % 3600))

        params = {
            # These would depend on how the real-time prices API expects parameters.
            "timestamp": timestamp,
        }
        # merge extra params
        params.update(kwargs)

        try:
            response = self.session.get(self.base_url + "/1h", params=params, timeout=10)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"Error during request: {e}")
            return None
        print(f"requested {response.url}")
        try:
            data = response.json()
        except Exception as e:
            print(f"Error decoding JSON: {e}")
            return None

        return data, timestamp
    
    def remove_nontracked_items(self, data, timestamp):
        """
            remove non-tracked items from hourly data returned from api call
            returns dict of tracked item ids and their (avgHigh, avgLow, highVol, lowVol)

            data - dictionary returned from get_price
            timestamp -unix timestamp in seconds
            return - dict of item ids and values
        """
        
        tracked_items = {}
        for key in data["data"].keys():
            if key in self.item_ids.values():
                tracked_items[key] = data["data"][key]

        return tracked_items
    
    def request_num_hours(self, timestamp, num_hours, remove_untracked=True):
        """
            Request a certain of hours back from (and including) timestamp

            timestamp - unix timestamp
            num_hours - int 
        """

        timestamps = [timestamp - (_ * 3600) for _ in range(num_hours)]
        timeseries = []

        print(f"querying timestamps: {timestamps}")

        for ts in timestamps:
            data, _ts = self.get_price(ts)
            if remove_untracked:
                data = self.remove_nontracked_items(data, ts)
            timeseries.append(self.organize_into_array(data, remove_untracked))

        return timeseries

    def organize_into_array(self, data, remove_untracked=True):
        """
            Organize data from remove_untracked into an array for training
        """
        if remove_untracked:
            keys = sorted(self.item_ids.values())
            d = []
            for key in keys:
                try:
                    d.append(list(data[key].values()))
                except Exception as e:
                    print(f"error in organize_into_array. key {key}")
                    print(e)
            return d
        else:
            print(data.keys())
            print(data["data"])
            keys = sorted(data["data"].keys())
            with open("keys.txt", "w") as f:
                f.write(str(keys))
            d = []
            for key in keys:
                print(f"Appending data from key {key}.")
                d.append(data["data"][key])
            return d

    
    def get_current_time(self):
        # returns the current time set back at least one hour (3600 seconds)
        # so that the api endpoint will work
        now = time.time()
        now = (now - 3600) - (now % 3600)
        return int(now)
    
    def get_data(self):
        result, timestamp = self.get_price(self.get_current_time())
        with open("results.json", "w") as f:
            json.dump(result, f)
        with open("cleaned_results.json", "w") as f:
            json.dump(self.remove_nontracked_items(result, timestamp), f)

        self.item_timeseries = self.request_num_hours(self.get_current_time(), 24, remove_untracked=False)
        with open("timeseries.json", "w") as f:
            json.dump(self.item_timeseries, f)

    def load_data(self):
        with open("timeseries.json", "r") as f:
            self.item_timeseries = json.load(f)

    def convert_timeseries_to_numpy(self):
        self.item_timeseries = np.array(self.item_timeseries)

    def clean_data(self, data=""):
        """
            Remove columns that have an abnormal number of items
            Remove all None values and replace them with np.nan value
            (Recurvsively :))
        """
        if data == "":
            data = self.item_timeseries

        # print(f"clean_data run on:")
        # print(data)
        length = len(self.item_ids.values())
        self.item_timeseries = [timestep for timestep in self.item_timeseries if len(timestep) == length] 
        self.item_timeseries = self.remove_none_vals(self.item_timeseries)


    def remove_none_vals(self, data):
        if isinstance(data, list):
            return [self.remove_none_vals(d) for d in data]
        elif data is None:
            return 0
        else:
            return data

    def save_numpy(self, filename="timeseries.npy"):
        np.save(filename, self.item_timeseries)

    def load_numpy(self, filename="timeseries.npy"):
        try:
            self.item_timeseries = np.load(filename)
        except Exception as e:
            print(f"failed to load from numpy binary {filename} with error {e}")

if __name__ == "__main__":
    api = RuneScapePricesAPI(user_agent="MyRuneApp/1.0")
    api.get_data()