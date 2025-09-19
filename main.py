import requests
import time
from typing import Dict, Any, Optional
import json

class RuneScapePricesAPI:
    """
    Client for querying RuneScape real-time price data.
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
                _ = data["data"][key]
                _["timestamp"] = timestamp
                tracked_items[key] = _

        return tracked_items
    
    def request_num_hours(self, timestamp, num_hours):
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
            data = self.remove_nontracked_items(data, ts)
            timeseries.append(self.organize_into_array(data))

        return timeseries

    def organize_into_array(self, data):
        """
            Organize data from remove_untracked into an array for training
        """
        keys = sorted(self.item_ids.values())
        d = []
        for key in keys:
            try:
                d.append(list(data[key].values()))
            except Exception as e:
                print(f"error in organize_into_array. key {key}")
                print(e)
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
        print(self.organize_into_array(self.remove_nontracked_items(result, timestamp)))

        self.item_timeseries = self.request_num_hours(self.get_current_time(), 7000)
        with open("timeseries.json", "w") as f:
            json.dump(self.item_timeseries, f)


if __name__ == "__main__":
    api = RuneScapePricesAPI(user_agent="MyRuneApp/1.0")
    api.get_data()