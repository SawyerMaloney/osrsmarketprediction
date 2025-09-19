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

    def get_price(self, item_id: int, timestamp: int, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Fetch real-time price for a given item.

        :param item_id: The ID of the item to query.
        :param kwargs: Any extra parameters the API supports.
        :return: Parsed JSON with price data, or None if error.
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
        except ValueError as e:
            print(f"Error decoding JSON: {e}")
            return None

        # TODO: adapt this based on the actual response structure
        # Example: suppose the response has something like {"price": {...}}
        return data, timestamp
    
    def remove_nontracked_items(self, data, timestamp):
        # remove non-tracked items from hourly data returned from api call
        # returns dict of tracked item ids and their (avgHigh, avgLow, highVol, lowVol)
        tracked_items = {}
        for key in data["data"].keys():
            if key in self.item_ids.values():
                _ = data["data"][key]
                _["timestamp"] = timestamp
                tracked_items[key] = _

        return tracked_items


def main():
    api = RuneScapePricesAPI(user_agent="MyRuneApp/1.0")
    item_id = 12934  # example: Abyssal whip, just illustrative
    result, timestamp = api.get_price(item_id, int(time.time()) - 7200)
    with open("results.json", "w") as f:
        json.dump(result, f)
    with open("cleaned_results.json", "w") as f:
        json.dump(api.remove_nontracked_items(result, timestamp), f)

if __name__ == "__main__":
    main()
